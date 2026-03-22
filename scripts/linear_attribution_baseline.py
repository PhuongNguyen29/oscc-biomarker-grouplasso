#!/usr/bin/env python3
"""
Linear Attribution Baseline: Cox Coefficient Comparison
========================================================

Purpose:
    Quantify the extent to which DeepSurv IG attributions deviate from
    a linear Cox attribution baseline, measuring the contribution of
    nonlinear model components to patient-level gene attributions.

Method:
    For each seed x fold x direction in Phase 3:
    1. Load the fitted elastic-net Cox model (cox_model.pkl).
    2. Extract the coefficient vector beta at the best alpha.
    3. Compute linear attribution for each held-out patient:
           A_ij^Cox = beta_i * x_ij
       (analytically exact for a linear model with zero baseline).
    4. Load existing DeepSurv IG from Phase 3b (ig_attributions.csv).
    5. Compute comparison metrics.

    No retraining is needed. The Cox model was already fitted with
    appropriate regularization (inner 5-fold CV for lambda selection)
    during Phase 3.

Metrics (reported for all genes and 64 consensus genes):
    1. Normalized deviation: mean(|IG_ds - A_cox|) / mean(|IG_ds|)
    2. Frobenius normalized deviation: ||IG_ds - A_cox||_F / ||IG_ds||_F
    3. Gene-wise Pearson correlation (mean across genes)
    4. Sign concordance (proportion of same-sign patient-gene pairs)
    5. Spearman rank correlation of gene-wise mean |attribution|

Design:
    3 seeds x 5 folds x 2 directions = 30 comparisons
    All from existing saved files — zero computation beyond loading and
    matrix arithmetic.

References:
    - Sundararajan et al. (2017) ICML — Integrated Gradients
    - Simon et al. (2011) JSS — Regularization paths for Cox
    - Katzman et al. (2018) BMC Med Res Methodol — DeepSurv

Usage:
    cd ~/hpchome/oscc-biomarker-elasticdeepsurv
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/linear_attribution_baseline_cox.py

    # Single seed:
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/linear_attribution_baseline_cox.py --seed 42

Output:
    results/linear_attribution_baseline_cox/
        seed_42/
            fold_1/
                tcga_to_orien/
                    cox_attributions.csv
                    comparison_metrics.json
                orien_to_tcga/
                    ...
            ...
        summary_all_genes.json
        summary_consensus_genes.json
        detailed_results.csv
"""

import sys
import os
import argparse
import logging
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from scipy import stats

from sklearn.model_selection import StratifiedKFold

from src.data.data_loader import load_cohort

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logging.getLogger('src.data.data_loader').setLevel(logging.WARNING)


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = "data/processed"
PHASE3_DIR = Path("results/phase3_cross_fitted")
OUTPUT_DIR = Path("results/linear_attribution_baseline_cox")

SEEDS = [42, 123, 456]
N_FOLDS = 5

CONSENSUS_GENE_FILE = Path(
    "results/experiment_mad3000_union_guardrail/phase4_stability/"
    "consensus/cox_consensus_pi080.txt"
)


# =============================================================================
# Helper functions
# =============================================================================

def create_strata(y_event, y_time):
    """
    Create stratification labels for StratifiedKFold.
    Matches Phase 3 exactly.
    """
    time_quartiles = pd.qcut(y_time, q=4, labels=False, duplicates='drop')
    strata = y_event.astype(int) * 4 + time_quartiles
    return strata


def extract_cox_coefficients(cox_pkl_path: Path, gene_names: List[str]) -> np.ndarray:
    """
    Load a saved Cox model and extract the coefficient vector at best alpha.

    Args:
        cox_pkl_path: Path to cox_model.pkl
        gene_names: Gene names (for verification)

    Returns:
        beta: coefficient vector (n_genes,)
    """
    with open(cox_pkl_path, 'rb') as f:
        saved = pickle.load(f)

    cox_model = saved['model']
    best_alpha = saved['best_alpha']

    # Find the alpha index closest to best_alpha
    alpha_idx = np.argmin(np.abs(cox_model.alphas_ - best_alpha))
    beta = cox_model.coef_[:, alpha_idx]

    # Verify shape
    assert len(beta) == len(gene_names), (
        f"Cox coefficient length {len(beta)} != gene count {len(gene_names)}"
    )

    return beta


def compute_cox_attributions(
    beta: np.ndarray,
    X_held_out: np.ndarray,
    sample_ids: List[str],
    gene_names: List[str],
) -> pd.DataFrame:
    """
    Compute linear attribution for held-out patients using Cox coefficients.

    For a linear model with zero baseline (population mean after z-scoring):
        A_ij = beta_i * x_ij

    This is analytically exact — no numerical integration needed.

    Args:
        beta: Cox coefficient vector (n_genes,)
        X_held_out: Held-out patient expression (n_patients, n_genes)
        sample_ids: Sample IDs
        gene_names: Gene names

    Returns:
        DataFrame (n_patients x n_genes) of linear attributions
    """
    # Attribution = beta * x (element-wise, broadcasting beta across patients)
    attributions = X_held_out * beta[np.newaxis, :]

    attr_df = pd.DataFrame(
        attributions,
        index=sample_ids,
        columns=gene_names,
    )
    attr_df.index.name = 'sampleID'

    return attr_df


# =============================================================================
# Comparison metrics
# =============================================================================

def compute_comparison_metrics(
    ig_deepsurv: pd.DataFrame,
    attr_cox: pd.DataFrame,
    gene_subset: List[str] = None,
    label: str = "all_genes",
) -> Dict:
    """
    Compute comparison metrics between DeepSurv IG and Cox linear attribution.

    Args:
        ig_deepsurv: DeepSurv IG attributions (n_patients x n_genes)
        attr_cox: Cox linear attributions (same shape)
        gene_subset: Optional gene list to restrict to
        label: Label for this metric set

    Returns:
        Dictionary of metric values
    """
    # Align patients and genes
    common_patients = ig_deepsurv.index.intersection(attr_cox.index)
    common_genes = ig_deepsurv.columns.intersection(attr_cox.columns)

    if gene_subset is not None:
        common_genes = common_genes.intersection(gene_subset)

    if len(common_patients) == 0 or len(common_genes) == 0:
        logger.warning(f"No common patients/genes for {label}")
        return {'label': label, 'n_patients': 0, 'n_genes': 0}

    ds = ig_deepsurv.loc[common_patients, common_genes].values
    cox = attr_cox.loc[common_patients, common_genes].values

    deviation = ds - cox

    # Metric 1: Element-wise normalized deviation
    mean_abs_dev = np.mean(np.abs(deviation))
    mean_abs_ds = np.mean(np.abs(ds))
    mean_abs_cox = np.mean(np.abs(cox))
    norm_deviation = mean_abs_dev / mean_abs_ds if mean_abs_ds > 0 else np.nan

    # Metric 2: Frobenius normalized deviation
    frob_dev = np.linalg.norm(deviation, 'fro')
    frob_ds = np.linalg.norm(ds, 'fro')
    frob_norm_dev = frob_dev / frob_ds if frob_ds > 0 else np.nan

    # Metric 3: Gene-wise Pearson correlation (mean across genes)
    gene_pearson = []
    for j in range(ds.shape[1]):
        if np.std(ds[:, j]) > 1e-10 and np.std(cox[:, j]) > 1e-10:
            r, _ = stats.pearsonr(ds[:, j], cox[:, j])
            gene_pearson.append(r)
    mean_gene_pearson = np.mean(gene_pearson) if gene_pearson else np.nan
    median_gene_pearson = np.median(gene_pearson) if gene_pearson else np.nan

    # Metric 4: Sign concordance
    sign_ds = np.sign(ds)
    sign_cox = np.sign(cox)
    nonzero_mask = (sign_ds != 0) & (sign_cox != 0)
    if nonzero_mask.sum() > 0:
        sign_concordance = np.mean(sign_ds[nonzero_mask] == sign_cox[nonzero_mask])
    else:
        sign_concordance = np.nan

    # Metric 5: Spearman rank correlation of gene-wise mean |attribution|
    gene_mean_abs_ds = np.mean(np.abs(ds), axis=0)
    gene_mean_abs_cox = np.mean(np.abs(cox), axis=0)
    if len(gene_mean_abs_ds) > 2:
        spearman_rho, spearman_p = stats.spearmanr(
            gene_mean_abs_ds, gene_mean_abs_cox
        )
    else:
        spearman_rho, spearman_p = np.nan, np.nan

    # Additional descriptive: Cox sparsity
    n_nonzero_cox = np.sum(np.any(cox != 0, axis=0))

    return {
        'label': label,
        'n_patients': int(len(common_patients)),
        'n_genes': int(len(common_genes)),
        'normalized_deviation': float(norm_deviation),
        'frobenius_normalized_deviation': float(frob_norm_dev),
        'mean_gene_pearson_r': float(mean_gene_pearson),
        'median_gene_pearson_r': float(median_gene_pearson),
        'sign_concordance': float(sign_concordance),
        'spearman_rank_rho': float(spearman_rho),
        'spearman_rank_p': float(spearman_p),
        'mean_abs_ig_deepsurv': float(mean_abs_ds),
        'mean_abs_attr_cox': float(mean_abs_cox),
        'mean_abs_deviation': float(mean_abs_dev),
        'n_nonzero_cox_genes': int(n_nonzero_cox),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Linear Attribution Baseline: Cox Coefficient Comparison"
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help="Run for a specific seed only (default: all seeds)"
    )
    args = parser.parse_args()

    # Setup output directory and logging
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUTPUT_DIR / f"cox_baseline_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 70)
    logger.info("LINEAR ATTRIBUTION BASELINE: COX COEFFICIENT COMPARISON")
    logger.info("=" * 70)
    logger.info(f"Phase 3 results: {PHASE3_DIR}")
    logger.info(f"Output: {OUTPUT_DIR}")

    # -------------------------------------------------------------------------
    # Load data (same as Phase 3, lines 777-788)
    # -------------------------------------------------------------------------
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
    gene_names_o = orien_expr.index.tolist()

    logger.info(f"  TCGA: {tcga_X.shape[0]} patients, {tcga_X.shape[1]} genes")
    logger.info(f"  ORIEN: {orien_X.shape[0]} patients, {orien_X.shape[1]} genes")

    assert list(gene_names) == list(gene_names_o), "Gene names must match"
    logger.info(f"  Genes aligned: {len(gene_names)}")

    # -------------------------------------------------------------------------
    # Load consensus gene list
    # -------------------------------------------------------------------------
    consensus_genes = None
    if CONSENSUS_GENE_FILE.exists():
        with open(CONSENSUS_GENE_FILE, 'r') as f:
            consensus_genes = [line.strip() for line in f if line.strip()]
        logger.info(f"  Consensus genes loaded: {len(consensus_genes)}")
    else:
        logger.warning(f"  Consensus gene file not found: {CONSENSUS_GENE_FILE}")

    # -------------------------------------------------------------------------
    # Process each seed x fold x direction
    # -------------------------------------------------------------------------
    seeds_to_run = [args.seed] if args.seed else SEEDS

    directions = [
        ('tcga_to_orien', 'tcga', 'orien',
         tcga_X, tcga_event, tcga_time, tcga_ids,
         orien_X, orien_event, orien_time, orien_ids),
        ('orien_to_tcga', 'orien', 'tcga',
         orien_X, orien_event, orien_time, orien_ids,
         tcga_X, tcga_event, tcga_time, tcga_ids),
    ]

    all_fold_metrics = []

    for seed in seeds_to_run:
        logger.info(f"\n{'='*70}")
        logger.info(f"SEED {seed}")
        logger.info(f"{'='*70}")

        # Reconstruct fold splits (same as Phase 3 lines 600-610)
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
            logger.info(f"\n  --- Fold {fold_num}/{N_FOLDS} ---")

            for (direction_name, train_cohort, test_cohort,
                 src_X, src_event, src_time, src_ids,
                 tgt_X, tgt_event, tgt_time, tgt_ids) in directions:

                logger.info(f"    Direction: {direction_name}")

                out_dir = OUTPUT_DIR / f"seed_{seed}" / f"fold_{fold_num}" / direction_name
                out_dir.mkdir(parents=True, exist_ok=True)

                # Get held-out test indices
                if train_cohort == 'tcga':
                    test_idx = orien_folds[fold_idx][1]
                else:
                    test_idx = tcga_folds[fold_idx][1]

                X_test = tgt_X[test_idx]
                sample_ids_test = [tgt_ids[i] for i in test_idx]

                # ----- Step 1: Load Cox model and extract coefficients -----
                cox_pkl_path = (
                    PHASE3_DIR / f"seed_{seed}" / f"fold_{fold_num}"
                    / direction_name / "cox_model.pkl"
                )

                if not cox_pkl_path.exists():
                    logger.warning(f"    Cox model not found: {cox_pkl_path}")
                    continue

                beta = extract_cox_coefficients(cox_pkl_path, gene_names)
                n_nonzero = np.sum(np.abs(beta) > 1e-10)
                logger.info(
                    f"    Cox: {n_nonzero}/{len(beta)} nonzero coefficients, "
                    f"max |beta|={np.max(np.abs(beta)):.6f}"
                )

                # ----- Step 2: Compute Cox linear attributions -----
                attr_cox = compute_cox_attributions(
                    beta, X_test, sample_ids_test, gene_names
                )
                attr_cox.to_csv(out_dir / "cox_attributions.csv")

                # ----- Step 3: Load DeepSurv IG -----
                ig_path = (
                    PHASE3_DIR / f"seed_{seed}" / f"fold_{fold_num}"
                    / direction_name / "ig_attributions.csv"
                )

                if not ig_path.exists():
                    logger.warning(f"    DeepSurv IG not found: {ig_path}")
                    continue

                ig_deepsurv = pd.read_csv(ig_path, index_col=0)
                logger.info(f"    DeepSurv IG: {ig_deepsurv.shape}")

                # ----- Step 4: Compute comparison metrics -----
                # All genes
                metrics_all = compute_comparison_metrics(
                    ig_deepsurv, attr_cox,
                    gene_subset=None,
                    label="all_genes",
                )
                metrics_all['seed'] = seed
                metrics_all['fold'] = fold_num
                metrics_all['direction'] = direction_name
                all_fold_metrics.append(metrics_all)

                logger.info(
                    f"    All genes: "
                    f"norm_dev={metrics_all['normalized_deviation']:.4f}, "
                    f"pearson={metrics_all['mean_gene_pearson_r']:.4f}, "
                    f"sign_conc={metrics_all['sign_concordance']:.4f}, "
                    f"spearman={metrics_all['spearman_rank_rho']:.4f}"
                )

                # Consensus genes
                if consensus_genes is not None:
                    metrics_cons = compute_comparison_metrics(
                        ig_deepsurv, attr_cox,
                        gene_subset=consensus_genes,
                        label="consensus_64",
                    )
                    metrics_cons['seed'] = seed
                    metrics_cons['fold'] = fold_num
                    metrics_cons['direction'] = direction_name
                    all_fold_metrics.append(metrics_cons)

                    logger.info(
                        f"    Consensus 64: "
                        f"norm_dev={metrics_cons['normalized_deviation']:.4f}, "
                        f"pearson={metrics_cons['mean_gene_pearson_r']:.4f}, "
                        f"sign_conc={metrics_cons['sign_concordance']:.4f}, "
                        f"spearman={metrics_cons['spearman_rank_rho']:.4f}"
                    )

                # Save per-fold metrics
                fold_metrics = {'all_genes': metrics_all}
                if consensus_genes is not None:
                    fold_metrics['consensus_64'] = metrics_cons
                with open(out_dir / "comparison_metrics.json", 'w') as f:
                    json.dump(fold_metrics, f, indent=2)

    # =========================================================================
    # Aggregate results
    # =========================================================================
    if not all_fold_metrics:
        logger.warning("No metrics computed.")
        return

    df = pd.DataFrame(all_fold_metrics)
    df.to_csv(OUTPUT_DIR / "detailed_results.csv", index=False)

    logger.info(f"\n{'='*70}")
    logger.info("AGGREGATE RESULTS")
    logger.info(f"{'='*70}")

    metric_cols = [
        'normalized_deviation',
        'frobenius_normalized_deviation',
        'mean_gene_pearson_r',
        'median_gene_pearson_r',
        'sign_concordance',
        'spearman_rank_rho',
    ]

    for label_val in df['label'].unique():
        subset = df[df['label'] == label_val]
        logger.info(f"\n  --- {label_val} (n={len(subset)} fold-direction pairs) ---")

        summary = {}
        for col in metric_cols:
            vals = subset[col].dropna()
            if len(vals) > 0:
                summary[col] = {
                    'mean': float(vals.mean()),
                    'std': float(vals.std()),
                    'min': float(vals.min()),
                    'max': float(vals.max()),
                    'median': float(vals.median()),
                }
                logger.info(
                    f"    {col}: "
                    f"{vals.mean():.4f} +/- {vals.std():.4f} "
                    f"[{vals.min():.4f}, {vals.max():.4f}]"
                )

        # Per-direction breakdown
        for direction in subset['direction'].unique():
            dir_subset = subset[subset['direction'] == direction]
            logger.info(f"\n    {direction} (n={len(dir_subset)}):")
            for col in metric_cols:
                vals = dir_subset[col].dropna()
                if len(vals) > 0:
                    logger.info(
                        f"      {col}: {vals.mean():.4f} +/- {vals.std():.4f}"
                    )

        # Additional: Cox sparsity summary
        if 'n_nonzero_cox_genes' in subset.columns:
            vals = subset['n_nonzero_cox_genes']
            logger.info(
                f"\n    Cox nonzero genes: {vals.mean():.0f} +/- {vals.std():.0f} "
                f"[{vals.min():.0f}, {vals.max():.0f}]"
            )

        # Save summary
        summary_file = OUTPUT_DIR / f"summary_{label_val}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"    Saved: {summary_file}")

    logger.info(f"\n{'='*70}")
    logger.info("COX ATTRIBUTION BASELINE ANALYSIS COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Results: {OUTPUT_DIR}")
    logger.info(f"Log: {log_file}")


if __name__ == "__main__":
    main()
