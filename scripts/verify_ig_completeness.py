#!/usr/bin/env python3
"""
IG Completeness Axiom Verification for Phase 3b
=================================================

Verifies that Integrated Gradients attributions satisfy the completeness
axiom (Sundararajan et al., 2017 ICML):

    sum_i IG_i(x) = F(x) - F(baseline)

For each seed x fold x direction:
1. Load saved DeepSurv model (deepsurv_model.pt)
2. Load saved IG attributions (ig_attributions.csv)
3. Load held-out patient expression data
4. Compute F(x) for each held-out patient and F(baseline) = F(0)
5. Compare sum(IG) to F(x) - F(baseline)
6. Report absolute and relative error

No retraining or recomputing IG. Purely a verification step.

Acceptance criteria (standard in IG literature):
  - Mean relative error < 5%
  - >= 95% of patients within 5% relative error

Usage:
    cd ~/hpchome/oscc-biomarker-elasticdeepsurv
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/verify_ig_completeness.py

    # Single seed:
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/verify_ig_completeness.py --seed 42

Output:
    results/ig_completeness_verification/
        detailed_results.csv
        summary.json
        verification_<timestamp>.log
"""

import sys
import os
import argparse
import logging
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from sklearn.model_selection import StratifiedKFold

from src.data.data_loader import load_cohort
from src.models.elastic_deepsurv import ElasticDeepSurv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logging.getLogger('src.models.deepsurv').setLevel(logging.ERROR)
logging.getLogger('src.models.elastic_deepsurv').setLevel(logging.ERROR)
logging.getLogger('src.data.data_loader').setLevel(logging.WARNING)


# =============================================================================
# Configuration — must match Phase 3 / 3b exactly
# =============================================================================

DATA_DIR = "data/processed"
PHASE3_DIR = Path("results/phase3_cross_fitted")
OUTPUT_DIR = Path("results/ig_completeness_verification")

SEEDS = [42, 123, 456]
N_FOLDS = 5

# Model hyperparameters (must match Phase 3 for architecture reconstruction)
DEEPSURV_PARAMS = {
    'tcga_to_orien': {
        'hidden_sizes': [96, 64],
        'dropout': 0.29,
        'activation': 'relu',
        'batch_norm': False,
        'l1_ratio': 0.3,
        'alpha': 1.92e-3,
    },
    'orien_to_tcga': {
        'hidden_sizes': [128, 96],
        'dropout': 0.35,
        'activation': 'relu',
        'batch_norm': True,
        'l1_ratio': 0.9,
        'alpha': 5.60e-3,
    },
}

# Acceptance criteria
RELATIVE_ERROR_THRESHOLD = 0.05  # 5%
PATIENT_PASS_RATE = 0.95  # 95% of patients must be within threshold


# =============================================================================
# Helper functions
# =============================================================================

def create_strata(y_event, y_time):
    """Create stratification labels (matches Phase 3)."""
    time_quartiles = pd.qcut(y_time, q=4, labels=False, duplicates='drop')
    strata = y_event.astype(int) * 4 + time_quartiles
    return strata


def load_deepsurv_model(
    model_path: Path,
    direction: str,
    n_features: int,
    device: str,
) -> ElasticDeepSurv:
    """
    Load a saved DeepSurv model checkpoint.

    Args:
        model_path: Path to deepsurv_model.pt (state_dict)
        direction: 'tcga_to_orien' or 'orien_to_tcga'
        n_features: Number of input genes
        device: 'cuda' or 'cpu'

    Returns:
        Loaded model in eval mode
    """
    params = DEEPSURV_PARAMS[direction]

    model = ElasticDeepSurv(
        n_features=n_features,
        hidden_sizes=params['hidden_sizes'],
        dropout=params['dropout'],
        l1_ratio=params['l1_ratio'],
        alpha=params['alpha'],
    )

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def verify_completeness(
    model: ElasticDeepSurv,
    ig_attributions: pd.DataFrame,
    X_held_out: np.ndarray,
    sample_ids: List[str],
    device: str,
) -> Dict:
    """
    Verify the completeness axiom for saved IG attributions.

    Completeness axiom: sum_i IG_i(x) = F(x) - F(baseline)
    Baseline = zero vector (population mean after z-scoring)

    Args:
        model: Loaded DeepSurv model
        ig_attributions: Saved IG DataFrame (n_patients x n_genes)
        X_held_out: Held-out patient expression (n_patients x n_genes)
        sample_ids: Sample IDs for held-out patients
        device: 'cuda' or 'cpu'

    Returns:
        Dictionary with convergence statistics
    """
    # Align patients: IG file may have different ordering
    ig_sample_ids = ig_attributions.index.tolist()

    # Find common patients and maintain order
    common_ids = [sid for sid in ig_sample_ids if sid in sample_ids]
    if len(common_ids) < len(ig_sample_ids):
        logger.warning(
            f"  {len(ig_sample_ids) - len(common_ids)} IG patients "
            f"not found in expression data"
        )

    # Get expression data for common patients in IG order
    id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
    X_aligned = np.array([X_held_out[id_to_idx[sid]] for sid in common_ids])
    ig_aligned = ig_attributions.loc[common_ids].values

    n_patients = len(common_ids)
    n_genes = ig_aligned.shape[1]

    # Compute F(x) for each held-out patient
    X_tensor = torch.from_numpy(X_aligned.astype(np.float32)).to(device)
    with torch.no_grad():
        f_x = model(X_tensor).squeeze().cpu().numpy()

    # Compute F(baseline) = F(zero vector)
    baseline = torch.zeros(1, n_genes).to(device)
    with torch.no_grad():
        f_baseline = model(baseline).squeeze().cpu().numpy()
    f_baseline = float(f_baseline)

    # Sum of IG attributions per patient
    ig_sum = ig_aligned.sum(axis=1)  # (n_patients,)

    # Expected: F(x) - F(baseline)
    expected_diff = f_x - f_baseline  # (n_patients,)

    # Absolute error
    abs_error = np.abs(ig_sum - expected_diff)

    # Relative error (handle near-zero denominators)
    denom = np.abs(expected_diff)
    # For patients where |F(x) - F(baseline)| is very small,
    # relative error is undefined/inflated. Use a floor.
    denom_floor = np.maximum(denom, 1e-6)
    rel_error = abs_error / denom_floor

    # For patients where expected_diff is truly near zero,
    # flag them separately
    near_baseline_mask = denom < 0.01
    n_near_baseline = int(near_baseline_mask.sum())

    # Acceptance: proportion within threshold
    within_threshold = rel_error < RELATIVE_ERROR_THRESHOLD
    pct_within = float(np.mean(within_threshold))

    # Excluding near-baseline patients
    if n_patients - n_near_baseline > 0:
        pct_within_excl = float(
            np.mean(within_threshold[~near_baseline_mask])
        )
    else:
        pct_within_excl = np.nan

    converged = pct_within >= PATIENT_PASS_RATE

    return {
        'n_patients': n_patients,
        'n_genes': n_genes,
        'f_baseline': f_baseline,
        'mean_abs_error': float(np.mean(abs_error)),
        'max_abs_error': float(np.max(abs_error)),
        'median_abs_error': float(np.median(abs_error)),
        'mean_rel_error': float(np.mean(rel_error)),
        'median_rel_error': float(np.median(rel_error)),
        'max_rel_error': float(np.max(rel_error)),
        'pct_within_5pct': float(pct_within),
        'pct_within_5pct_excl_near_baseline': float(pct_within_excl),
        'n_near_baseline': n_near_baseline,
        'converged': bool(converged),
        'mean_ig_sum': float(np.mean(ig_sum)),
        'mean_expected_diff': float(np.mean(expected_diff)),
        'std_expected_diff': float(np.std(expected_diff)),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify IG Completeness Axiom on Phase 3b Outputs"
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help="Run for a specific seed only (default: all seeds)"
    )
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUTPUT_DIR / f"verification_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 70)
    logger.info("IG COMPLETENESS AXIOM VERIFICATION")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Threshold: {RELATIVE_ERROR_THRESHOLD*100:.0f}% relative error")
    logger.info(f"Pass rate: {PATIENT_PASS_RATE*100:.0f}% of patients")

    # Load data
    logger.info("\nLoading cohort data...")
    tcga_expr, tcga_surv = load_cohort('tcga', DATA_DIR)
    orien_expr, orien_surv = load_cohort('orien', DATA_DIR)

    tcga_X = tcga_expr.values.T.astype(np.float32)
    tcga_event = tcga_surv['event'].values.astype(np.float32)
    tcga_time = tcga_surv['time'].values.astype(np.float32)
    tcga_ids = tcga_surv.index.tolist()

    orien_X = orien_expr.values.T.astype(np.float32)
    orien_event = orien_surv['event'].values.astype(np.float32)
    orien_time = orien_surv['time'].values.astype(np.float32)
    orien_ids = orien_surv.index.tolist()

    gene_names = tcga_expr.index.tolist()
    n_features = len(gene_names)

    logger.info(f"  TCGA: {tcga_X.shape[0]} patients, {n_features} genes")
    logger.info(f"  ORIEN: {orien_X.shape[0]} patients, {n_features} genes")

    # Reconstruct fold splits
    seeds_to_run = [args.seed] if args.seed else SEEDS

    directions_config = [
        ('tcga_to_orien', 'tcga', 'orien',
         tcga_X, tcga_event, tcga_time, tcga_ids,
         orien_X, orien_event, orien_time, orien_ids),
        ('orien_to_tcga', 'orien', 'tcga',
         orien_X, orien_event, orien_time, orien_ids,
         tcga_X, tcga_event, tcga_time, tcga_ids),
    ]

    all_results = []

    for seed in seeds_to_run:
        logger.info(f"\n{'='*70}")
        logger.info(f"SEED {seed}")
        logger.info(f"{'='*70}")

        tcga_strata = create_strata(tcga_event, tcga_time)
        orien_strata = create_strata(orien_event, orien_time)

        tcga_skf = StratifiedKFold(
            n_splits=N_FOLDS, shuffle=True, random_state=seed
        )
        orien_skf = StratifiedKFold(
            n_splits=N_FOLDS, shuffle=True, random_state=seed
        )

        tcga_folds = list(tcga_skf.split(tcga_X, tcga_strata))
        orien_folds = list(orien_skf.split(orien_X, orien_strata))

        for fold_idx in range(N_FOLDS):
            fold_num = fold_idx + 1

            for (direction_name, train_cohort, test_cohort,
                 src_X, src_event, src_time, src_ids,
                 tgt_X, tgt_event, tgt_time, tgt_ids) in directions_config:

                fold_dir = (
                    PHASE3_DIR / f"seed_{seed}" / f"fold_{fold_num}"
                    / direction_name
                )

                model_path = fold_dir / "deepsurv_model.pt"
                ig_path = fold_dir / "ig_attributions.csv"

                if not model_path.exists() or not ig_path.exists():
                    logger.warning(
                        f"  Missing files for seed={seed} fold={fold_num} "
                        f"{direction_name}"
                    )
                    continue

                # Get held-out test indices
                if train_cohort == 'tcga':
                    test_idx = orien_folds[fold_idx][1]
                else:
                    test_idx = tcga_folds[fold_idx][1]

                X_test = tgt_X[test_idx]
                sample_ids_test = [tgt_ids[i] for i in test_idx]

                # Load model
                model = load_deepsurv_model(
                    model_path, direction_name, n_features, device
                )

                # Load IG
                ig_df = pd.read_csv(ig_path, index_col=0)

                # Verify completeness
                result = verify_completeness(
                    model, ig_df, X_test, sample_ids_test, device
                )
                result['seed'] = seed
                result['fold'] = fold_num
                result['direction'] = direction_name

                status = "PASS" if result['converged'] else "FAIL"
                logger.info(
                    f"  seed={seed} fold={fold_num} {direction_name}: "
                    f"{status} | "
                    f"mean_rel_err={result['mean_rel_error']:.4f} | "
                    f"median_rel_err={result['median_rel_error']:.6f} | "
                    f"pct_within_5%={result['pct_within_5pct']:.4f} | "
                    f"near_baseline={result['n_near_baseline']}"
                )

                all_results.append(result)

                # Free GPU memory
                del model
                if device == 'cuda':
                    torch.cuda.empty_cache()

    # =========================================================================
    # Aggregate
    # =========================================================================
    if not all_results:
        logger.warning("No results computed.")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_DIR / "detailed_results.csv", index=False)

    n_pass = int(df['converged'].sum())
    n_total = len(df)

    logger.info(f"\n{'='*70}")
    logger.info("AGGREGATE RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"  Total fold-direction pairs: {n_total}")
    logger.info(f"  Passed ({PATIENT_PASS_RATE*100:.0f}% within {RELATIVE_ERROR_THRESHOLD*100:.0f}%): {n_pass}/{n_total}")
    logger.info(f"  Failed: {n_total - n_pass}/{n_total}")
    logger.info(f"")
    logger.info(f"  Mean absolute error:  {df['mean_abs_error'].mean():.6f} +/- {df['mean_abs_error'].std():.6f}")
    logger.info(f"  Mean relative error:  {df['mean_rel_error'].mean():.4f} +/- {df['mean_rel_error'].std():.4f}")
    logger.info(f"  Median relative error: {df['median_rel_error'].mean():.6f} +/- {df['median_rel_error'].std():.6f}")
    logger.info(f"  Pct within 5%:        {df['pct_within_5pct'].mean():.4f} +/- {df['pct_within_5pct'].std():.4f}")
    logger.info(f"  Near-baseline patients: {df['n_near_baseline'].mean():.1f} +/- {df['n_near_baseline'].std():.1f}")

    # Per-direction
    for direction in df['direction'].unique():
        dir_df = df[df['direction'] == direction]
        n_dir_pass = int(dir_df['converged'].sum())
        logger.info(f"\n  {direction} (n={len(dir_df)}, pass={n_dir_pass}):")
        logger.info(f"    Mean abs error:   {dir_df['mean_abs_error'].mean():.6f}")
        logger.info(f"    Mean rel error:   {dir_df['mean_rel_error'].mean():.4f}")
        logger.info(f"    Pct within 5%:    {dir_df['pct_within_5pct'].mean():.4f}")
        logger.info(f"    Near-baseline:    {dir_df['n_near_baseline'].mean():.1f}")

    # Save summary
    summary = {
        'n_total': n_total,
        'n_passed': n_pass,
        'n_failed': n_total - n_pass,
        'threshold_relative_error': RELATIVE_ERROR_THRESHOLD,
        'threshold_patient_pass_rate': PATIENT_PASS_RATE,
        'aggregate': {
            'mean_abs_error': float(df['mean_abs_error'].mean()),
            'mean_rel_error': float(df['mean_rel_error'].mean()),
            'median_rel_error': float(df['median_rel_error'].mean()),
            'pct_within_5pct': float(df['pct_within_5pct'].mean()),
            'near_baseline_patients_mean': float(df['n_near_baseline'].mean()),
        },
    }
    with open(OUTPUT_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*70}")
    if n_pass == n_total:
        logger.info("ALL PAIRS PASSED COMPLETENESS VERIFICATION")
    else:
        logger.info(f"WARNING: {n_total - n_pass} PAIRS FAILED")
        failed = df[~df['converged']]
        for _, row in failed.iterrows():
            logger.info(
                f"  FAILED: seed={int(row['seed'])} fold={int(row['fold'])} "
                f"{row['direction']} "
                f"(pct_within={row['pct_within_5pct']:.4f})"
            )
    logger.info(f"{'='*70}")
    logger.info(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
