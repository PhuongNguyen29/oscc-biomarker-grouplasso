#!/usr/bin/env python3
"""
Linear Attribution Baseline for Phase 7 Bidirectional Design
=============================================================

Quantify the extent to which DeepSurv IG attributions deviate from
a linear Cox attribution baseline on the 25-gene union set.

For each direction (TCGA→ORIEN, ORIEN→TCGA):
1. Load the fitted Cox model from Phase 7 sweep (cox_model_*_union.pkl)
2. Extract the coefficient vector beta at the best alpha
3. Compute linear attribution: A_ij^Cox = beta_j * x_ij
4. Load existing DeepSurv IG from Phase 7 sweep
5. Compute comparison metrics on the TEST cohort
   (train cohort comparison also computed for completeness)

Adapted from linear_attribution_baseline_cox.py:
  - Phase 7 bidirectional (no cross-fitting, no seeds/folds)
  - 25-gene union set
  - Cox models from sweep validation (with retuned l1_ratio)

Metrics:
    1. Normalized deviation: mean(|IG_ds - A_cox|) / mean(|IG_ds|)
    2. Frobenius normalized deviation: ||IG_ds - A_cox||_F / ||IG_ds||_F
    3. Gene-wise Pearson correlation (mean across genes)
    4. Sign concordance (proportion of same-sign patient-gene pairs)
    5. Spearman rank correlation of gene-wise mean |attribution|

Usage:
    cd ~/hpchome/oscc-biomarker-grouplasso
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/linear_baseline_phase7.py

Output:
    results/linear_attribution_baseline_phase7/
        detailed_results.csv
        summary.json
"""

import sys
import os
import logging
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from scipy import stats

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
SWEEP_DIR = Path("results/sensitivity_sweep/mult0.6_pi60")
VALIDATION_DIR = SWEEP_DIR / "validation"
COX_DIR = VALIDATION_DIR / "cox"
DS_DIR = VALIDATION_DIR / "deepsurv"
OUTPUT_DIR = Path("results/linear_attribution_baseline_phase7")

UNION_GENES_FILE = SWEEP_DIR / "union_genes.txt"


# =============================================================================
# Helpers
# =============================================================================

def load_union_genes(filepath):
    """Load union gene list."""
    with open(filepath, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


def subset_expression(X_full, all_gene_names, selected_genes):
    """Subset expression matrix to selected genes."""
    gene_indices = [all_gene_names.index(g) for g in selected_genes]
    return X_full[:, gene_indices]


def extract_cox_coefficients(cox_pkl_path, gene_names):
    """Load a saved Cox model and extract coefficient vector at best alpha."""
    with open(cox_pkl_path, 'rb') as f:
        saved = pickle.load(f)

    cox_model = saved['model']
    best_alpha = saved['best_alpha']

    alpha_idx = np.argmin(np.abs(cox_model.alphas_ - best_alpha))
    beta = cox_model.coef_[:, alpha_idx]

    assert len(beta) == len(gene_names), (
        f"Cox coefficient length {len(beta)} != gene count {len(gene_names)}"
    )

    return beta


def compute_cox_attributions(beta, X_data, sample_ids, gene_names):
    """
    Compute linear attribution: A_ij = beta_j * x_ij

    Analytically exact for a linear model with zero baseline.
    """
    attributions = X_data * beta[np.newaxis, :]
    attr_df = pd.DataFrame(attributions, index=sample_ids, columns=gene_names)
    attr_df.index.name = 'sampleID'
    return attr_df


def compute_comparison_metrics(ig_deepsurv, attr_cox, label="25_gene_union"):
    """
    Compute comparison metrics between DeepSurv IG and Cox linear attribution.
    """
    # Align patients and genes
    common_patients = ig_deepsurv.index.intersection(attr_cox.index)
    common_genes = ig_deepsurv.columns.intersection(attr_cox.columns)

    if len(common_patients) == 0 or len(common_genes) == 0:
        logger.warning(f"No common patients/genes for {label}")
        return {'label': label, 'n_patients': 0, 'n_genes': 0}

    ds = ig_deepsurv.loc[common_patients, common_genes].values
    cox = attr_cox.loc[common_patients, common_genes].values
    deviation = ds - cox

    # Metric 1: Element-wise normalized deviation
    mean_abs_dev = np.mean(np.abs(deviation))
    mean_abs_ds = np.mean(np.abs(ds))
    norm_deviation = mean_abs_dev / mean_abs_ds if mean_abs_ds > 0 else np.nan

    # Metric 2: Frobenius normalized deviation
    frob_dev = np.linalg.norm(deviation, 'fro')
    frob_ds = np.linalg.norm(ds, 'fro')
    frob_norm_dev = frob_dev / frob_ds if frob_ds > 0 else np.nan

    # Metric 3: Gene-wise Pearson correlation
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

    # Cox sparsity
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
        'mean_abs_attr_cox': float(np.mean(np.abs(cox))),
        'mean_abs_deviation': float(mean_abs_dev),
        'n_nonzero_cox_genes': int(n_nonzero_cox),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUTPUT_DIR / f"linear_baseline_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 70)
    logger.info("LINEAR ATTRIBUTION BASELINE — Phase 7 Bidirectional (25 genes)")
    logger.info("=" * 70)
    logger.info(f"Sweep dir: {SWEEP_DIR}")
    logger.info(f"Output: {OUTPUT_DIR}")

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

    # Load union genes
    union_genes = load_union_genes(UNION_GENES_FILE)
    n_features = len(union_genes)
    logger.info(f"  Union genes: {n_features}")

    # Subset to union genes
    tcga_X_union = subset_expression(tcga_X_full, all_gene_names, union_genes)
    orien_X_union = subset_expression(orien_X_full, all_gene_names, union_genes)

    logger.info(f"  TCGA: {tcga_X_union.shape}")
    logger.info(f"  ORIEN: {orien_X_union.shape}")

    # -------------------------------------------------------------------------
    # Directions
    # -------------------------------------------------------------------------
    directions = [
        {
            'name': 'tcga_to_orien',
            'cox_model_file': 'cox_model_tcga_to_orien_union.pkl',
            'X_train': tcga_X_union,
            'train_ids': tcga_ids,
            'X_test': orien_X_union,
            'test_ids': orien_ids,
        },
        {
            'name': 'orien_to_tcga',
            'cox_model_file': 'cox_model_orien_to_tcga_union.pkl',
            'X_train': orien_X_union,
            'train_ids': orien_ids,
            'X_test': tcga_X_union,
            'test_ids': tcga_ids,
        },
    ]

    all_metrics = []

    for d in directions:
        direction = d['name']
        logger.info(f"\n{'='*70}")
        logger.info(f"DIRECTION: {direction}")
        logger.info(f"{'='*70}")

        # Load Cox model
        cox_path = COX_DIR / d['cox_model_file']
        if not cox_path.exists():
            logger.error(f"  Cox model not found: {cox_path}")
            continue

        beta = extract_cox_coefficients(cox_path, union_genes)
        n_nonzero = np.sum(np.abs(beta) > 1e-10)
        logger.info(f"  Cox: {n_nonzero}/{len(beta)} nonzero coefficients")

        # Process train and test splits
        for split, X_data, sample_ids in [
            ('train', d['X_train'], d['train_ids']),
            ('test', d['X_test'], d['test_ids']),
        ]:
            # Load DeepSurv IG
            ig_path = DS_DIR / f"deepsurv_ig_{direction}_{split}.csv"
            if not ig_path.exists():
                logger.warning(f"  IG file not found: {ig_path}")
                continue

            ig_deepsurv = pd.read_csv(ig_path, index_col=0)
            logger.info(f"\n  Split: {split} ({ig_deepsurv.shape[0]} patients)")

            # Compute Cox linear attributions
            attr_cox = compute_cox_attributions(
                beta, X_data, sample_ids, union_genes
            )

            # Compute comparison metrics
            metrics = compute_comparison_metrics(ig_deepsurv, attr_cox)
            metrics['direction'] = direction
            metrics['split'] = split
            metrics['n_nonzero_cox_genes'] = int(n_nonzero)

            logger.info(
                f"  norm_dev={metrics['normalized_deviation']:.4f}, "
                f"pearson={metrics['mean_gene_pearson_r']:.4f}, "
                f"sign_conc={metrics['sign_concordance']:.4f}, "
                f"spearman={metrics['spearman_rank_rho']:.4f}"
            )

            all_metrics.append(metrics)

            # Save per-direction attributions
            out_subdir = OUTPUT_DIR / direction
            out_subdir.mkdir(parents=True, exist_ok=True)
            attr_cox.to_csv(out_subdir / f"cox_attributions_{split}.csv")

    # =========================================================================
    # Aggregate
    # =========================================================================
    if not all_metrics:
        logger.warning("No metrics computed.")
        return

    df = pd.DataFrame(all_metrics)
    df.to_csv(OUTPUT_DIR / "detailed_results.csv", index=False)

    metric_cols = [
        'normalized_deviation',
        'frobenius_normalized_deviation',
        'mean_gene_pearson_r',
        'median_gene_pearson_r',
        'sign_concordance',
        'spearman_rank_rho',
    ]

    logger.info(f"\n{'='*70}")
    logger.info("AGGREGATE RESULTS")
    logger.info(f"{'='*70}")

    # Overall
    logger.info(f"\n  Overall (n={len(df)} direction-split pairs):")
    summary = {}
    for col in metric_cols:
        vals = df[col].dropna()
        if len(vals) > 0:
            summary[col] = {
                'mean': float(vals.mean()),
                'std': float(vals.std()),
                'min': float(vals.min()),
                'max': float(vals.max()),
            }
            logger.info(f"    {col}: {vals.mean():.4f} +/- {vals.std():.4f}")

    # Per direction
    for direction in df['direction'].unique():
        dir_df = df[df['direction'] == direction]
        logger.info(f"\n  {direction}:")
        for _, row in dir_df.iterrows():
            logger.info(
                f"    {row['split']}: norm_dev={row['normalized_deviation']:.4f}, "
                f"pearson={row['mean_gene_pearson_r']:.4f}, "
                f"sign_conc={row['sign_concordance']:.4f}"
            )

    # Test-only summary (most relevant for dissertation)
    test_df = df[df['split'] == 'test']
    if len(test_df) > 0:
        logger.info(f"\n  --- TEST SPLIT ONLY (for dissertation) ---")
        for col in metric_cols:
            vals = test_df[col].dropna()
            if len(vals) > 0:
                logger.info(f"    {col}: {vals.mean():.4f} +/- {vals.std():.4f}")

    # Save summary
    full_summary = {
        'design': 'Phase 7 bidirectional, 25-gene union set',
        'sweep_config': 'mult=0.6x, pi>=0.60',
        'n_direction_split_pairs': len(df),
        'aggregate': summary,
        'per_direction_split': df.to_dict(orient='records'),
    }
    with open(OUTPUT_DIR / "summary.json", 'w') as f:
        json.dump(full_summary, f, indent=2, default=str)

    logger.info(f"\n{'='*70}")
    logger.info("LINEAR ATTRIBUTION BASELINE COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
