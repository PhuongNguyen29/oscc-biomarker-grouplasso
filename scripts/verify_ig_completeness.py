#!/usr/bin/env python3
"""
IG Completeness Axiom Verification for Phase 7 Bidirectional Design
====================================================================

Verifies that Integrated Gradients attributions satisfy the completeness
axiom (Sundararajan et al., 2017 ICML):

    sum_j IG_j(x) = F(x) - F(baseline)

For each direction (TCGA→ORIEN, ORIEN→TCGA):
1. Load saved DeepSurv model from sweep results
2. Load saved IG attributions (train + test cohorts)
3. Compute F(x) for each patient and F(baseline) = F(0)
4. Compare sum(IG) to F(x) - F(baseline)
5. Report absolute and relative error

Adapted from verify_ig_completeness.py for the corrected pipeline:
  - Phase 7 bidirectional (no cross-fitting, no seeds/folds)
  - 25-gene union set with retuned DeepSurv parameters
  - IG on both train and test cohorts per direction

Acceptance criteria (standard in IG literature):
  - Mean relative error < 5%
  - >= 95% of patients within 5% relative error

Usage:
    cd ~/hpchome/oscc-biomarker-grouplasso
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/verify_ig_completeness_phase7.py

Output:
    results/ig_completeness_verification_phase7/
        detailed_results.csv
        summary.json
"""

import sys
import os
import logging
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List

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
# Configuration
# =============================================================================

DATA_DIR = "data/processed"
SWEEP_DIR = Path("results/sensitivity_sweep/mult0.6_pi60")
VALIDATION_DIR = SWEEP_DIR / "validation"
DS_DIR = VALIDATION_DIR / "deepsurv"
TUNING_DIR = SWEEP_DIR / "tuning"
OUTPUT_DIR = Path("results/ig_completeness_verification_phase7")

# Union gene list
UNION_GENES_FILE = SWEEP_DIR / "union_genes.txt"

# Acceptance criteria
RELATIVE_ERROR_THRESHOLD = 0.05  # 5%
PATIENT_PASS_RATE = 0.95  # 95% of patients must be within threshold


# =============================================================================
# Helpers
# =============================================================================

def load_union_genes(filepath):
    """Load union gene list."""
    with open(filepath, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


def load_deepsurv_params(tuning_dir, cohort):
    """Load retuned DeepSurv parameters for a cohort."""
    param_file = tuning_dir / f"deepsurv_{cohort}_best_25genes.json"
    if not param_file.exists():
        # Try alternate naming
        param_file = tuning_dir / f"deepsurv_{cohort}_best.json"
    with open(param_file) as f:
        data = json.load(f)
    return data['best_params']


def load_deepsurv_model(model_path, params, n_features, device):
    """Load a saved DeepSurv model checkpoint."""
    model = ElasticDeepSurv(
        n_features=n_features,
        hidden_sizes=params['hidden_sizes'],
        dropout=params['dropout'],
        activation=params['activation'],
        batch_norm=params['batch_norm'],
        l1_ratio=params['l1_ratio'],
        alpha=params['alpha'],
    )
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def subset_expression(X_full, all_gene_names, selected_genes):
    """Subset expression matrix to selected genes."""
    gene_indices = [all_gene_names.index(g) for g in selected_genes]
    return X_full[:, gene_indices]


def verify_completeness(model, ig_attributions, X_data, sample_ids, device):
    """
    Verify the completeness axiom for saved IG attributions.

    Completeness axiom: sum_j IG_j(x) = F(x) - F(baseline)
    Baseline = zero vector (population mean after z-scoring)
    """
    # Align patients
    ig_sample_ids = ig_attributions.index.tolist()
    common_ids = [sid for sid in ig_sample_ids if sid in sample_ids]
    if len(common_ids) < len(ig_sample_ids):
        logger.warning(
            f"  {len(ig_sample_ids) - len(common_ids)} IG patients "
            f"not found in expression data"
        )

    # Get expression data for common patients in IG order
    id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
    X_aligned = np.array([X_data[id_to_idx[sid]] for sid in common_ids])
    ig_aligned = ig_attributions.loc[common_ids].values

    n_patients = len(common_ids)
    n_genes = ig_aligned.shape[1]

    # Compute F(x) for each patient
    X_tensor = torch.from_numpy(X_aligned.astype(np.float32)).to(device)
    with torch.no_grad():
        f_x = model(X_tensor).squeeze().cpu().numpy()

    # Compute F(baseline) = F(zero vector)
    baseline = torch.zeros(1, n_genes).to(device)
    with torch.no_grad():
        f_baseline = model(baseline).squeeze().cpu().numpy()
    f_baseline = float(f_baseline)

    # Sum of IG attributions per patient
    ig_sum = ig_aligned.sum(axis=1)

    # Expected: F(x) - F(baseline)
    expected_diff = f_x - f_baseline

    # Absolute error
    abs_error = np.abs(ig_sum - expected_diff)

    # Relative error
    denom = np.abs(expected_diff)
    denom_floor = np.maximum(denom, 1e-6)
    rel_error = abs_error / denom_floor

    # Near-baseline patients
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
    logger.info("IG COMPLETENESS AXIOM VERIFICATION — Phase 7 Bidirectional")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Threshold: {RELATIVE_ERROR_THRESHOLD*100:.0f}% relative error")
    logger.info(f"Pass rate: {PATIENT_PASS_RATE*100:.0f}% of patients")
    logger.info(f"Sweep dir: {SWEEP_DIR}")

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    logger.info("\nLoading cohort data...")
    tcga_expr, tcga_surv = load_cohort('tcga', DATA_DIR)
    orien_expr, orien_surv = load_cohort('orien', DATA_DIR)

    all_gene_names = tcga_expr.index.tolist()

    tcga_X_full = tcga_expr.values.T.astype(np.float32)
    tcga_ids = tcga_surv.index.tolist()

    orien_X_full = orien_expr.values.T.astype(np.float32)
    orien_ids = orien_surv.index.tolist()

    logger.info(f"  TCGA: {tcga_X_full.shape[0]} patients, {tcga_X_full.shape[1]} genes")
    logger.info(f"  ORIEN: {orien_X_full.shape[0]} patients, {orien_X_full.shape[1]} genes")

    # Load union genes
    union_genes = load_union_genes(UNION_GENES_FILE)
    n_features = len(union_genes)
    logger.info(f"  Union genes: {n_features}")

    # Subset to union genes
    tcga_X_union = subset_expression(tcga_X_full, all_gene_names, union_genes)
    orien_X_union = subset_expression(orien_X_full, all_gene_names, union_genes)

    # -------------------------------------------------------------------------
    # Directions config
    # -------------------------------------------------------------------------
    directions = [
        {
            'name': 'tcga_to_orien',
            'train_cohort': 'tcga',
            'X_train': tcga_X_union,
            'train_ids': tcga_ids,
            'X_test': orien_X_union,
            'test_ids': orien_ids,
        },
        {
            'name': 'orien_to_tcga',
            'train_cohort': 'orien',
            'X_train': orien_X_union,
            'train_ids': orien_ids,
            'X_test': tcga_X_union,
            'test_ids': tcga_ids,
        },
    ]

    all_results = []

    for d in directions:
        direction = d['name']
        train_cohort = d['train_cohort']

        logger.info(f"\n{'='*70}")
        logger.info(f"DIRECTION: {direction}")
        logger.info(f"{'='*70}")

        # Load model
        model_path = DS_DIR / f"deepsurv_model_{direction}.pt"
        if not model_path.exists():
            logger.error(f"  Model not found: {model_path}")
            continue

        params = load_deepsurv_params(TUNING_DIR, train_cohort)
        model = load_deepsurv_model(model_path, params, n_features, device)
        logger.info(f"  Model loaded: {model_path}")

        # Check both train and test IG files
        for split, X_data, sample_ids in [
            ('train', d['X_train'], d['train_ids']),
            ('test', d['X_test'], d['test_ids']),
        ]:
            ig_path = DS_DIR / f"deepsurv_ig_{direction}_{split}.csv"
            if not ig_path.exists():
                logger.warning(f"  IG file not found: {ig_path}")
                continue

            ig_df = pd.read_csv(ig_path, index_col=0)
            logger.info(f"\n  Split: {split} ({ig_df.shape[0]} patients)")

            result = verify_completeness(
                model, ig_df, X_data, sample_ids, device
            )
            result['direction'] = direction
            result['split'] = split

            status = "PASS" if result['converged'] else "FAIL"
            logger.info(
                f"  {status} | "
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
    logger.info(f"  Total direction-split pairs: {n_total}")
    logger.info(f"  Passed (>={PATIENT_PASS_RATE*100:.0f}% within "
                f"{RELATIVE_ERROR_THRESHOLD*100:.0f}%): {n_pass}/{n_total}")
    logger.info(f"")
    logger.info(f"  Mean relative error:  {df['mean_rel_error'].mean():.4f} "
                f"+/- {df['mean_rel_error'].std():.4f}")
    logger.info(f"  Median relative error: {df['median_rel_error'].mean():.6f} "
                f"+/- {df['median_rel_error'].std():.6f}")
    logger.info(f"  Pct within 5%:        {df['pct_within_5pct'].mean():.4f} "
                f"+/- {df['pct_within_5pct'].std():.4f}")

    # Per direction
    for direction in df['direction'].unique():
        dir_df = df[df['direction'] == direction]
        logger.info(f"\n  {direction}:")
        for _, row in dir_df.iterrows():
            logger.info(
                f"    {row['split']}: mean_rel_err={row['mean_rel_error']:.4f}, "
                f"pct_within_5%={row['pct_within_5pct']:.4f}, "
                f"converged={row['converged']}"
            )

    # Per split (for dissertation reporting)
    logger.info(f"\n  --- By split (for dissertation) ---")
    for split in ['train', 'test']:
        split_df = df[df['split'] == split]
        if len(split_df) > 0:
            logger.info(
                f"  {split}: mean_rel_err={split_df['mean_rel_error'].mean():.4f} "
                f"+/- {split_df['mean_rel_error'].std():.4f}, "
                f"pct_within_5%={split_df['pct_within_5pct'].mean():.4f} "
                f"+/- {split_df['pct_within_5pct'].std():.4f}"
            )

    # Save summary
    summary = {
        'design': 'Phase 7 bidirectional, 25-gene union set',
        'sweep_config': 'mult=0.6x, pi>=0.60',
        'n_total': n_total,
        'n_passed': n_pass,
        'n_failed': n_total - n_pass,
        'threshold_relative_error': RELATIVE_ERROR_THRESHOLD,
        'threshold_patient_pass_rate': PATIENT_PASS_RATE,
        'aggregate': {
            'mean_rel_error': float(df['mean_rel_error'].mean()),
            'std_rel_error': float(df['mean_rel_error'].std()),
            'median_rel_error': float(df['median_rel_error'].mean()),
            'pct_within_5pct': float(df['pct_within_5pct'].mean()),
        },
        'per_direction_split': [
            {k: v for k, v in row.items()}
            for _, row in df.iterrows()
        ],
    }
    with open(OUTPUT_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\n{'='*70}")
    if n_pass == n_total:
        logger.info("ALL PAIRS PASSED COMPLETENESS VERIFICATION")
    else:
        logger.info(f"WARNING: {n_total - n_pass} PAIRS FAILED")
        failed = df[~df['converged']]
        for _, row in failed.iterrows():
            logger.info(
                f"  FAILED: {row['direction']} {row['split']} "
                f"(pct_within={row['pct_within_5pct']:.4f})"
            )
    logger.info(f"{'='*70}")
    logger.info(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
