#!/usr/bin/env python3
"""
Recompute aggregate metrics for the linear attribution baseline,
restricted to fold-direction pairs where Cox retained >= N nonzero genes.

Usage:
    cd ~/hpchome/oscc-biomarker-elasticdeepsurv
    python scripts/linear_baseline_restricted_summary.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

OUTPUT_DIR = Path("results/linear_attribution_baseline_cox")
THRESHOLD = 20  # minimum nonzero Cox genes for inclusion

# -------------------------------------------------------------------------
# Load detailed results
# -------------------------------------------------------------------------
df = pd.read_csv(OUTPUT_DIR / "detailed_results.csv")
print(f"Total fold-direction pairs: {len(df)}")
print(f"Labels present: {df['label'].unique()}")
print()

# -------------------------------------------------------------------------
# Metrics to report
# -------------------------------------------------------------------------
metric_cols = [
    'normalized_deviation',
    'frobenius_normalized_deviation',
    'mean_gene_pearson_r',
    'median_gene_pearson_r',
    'sign_concordance',
    'spearman_rank_rho',
]

# -------------------------------------------------------------------------
# For each label (all_genes, consensus_64), compute:
#   1. Full aggregate (all 30 pairs)
#   2. Restricted aggregate (>= threshold nonzero Cox genes)
# -------------------------------------------------------------------------
for label_val in df['label'].unique():
    subset = df[df['label'] == label_val].copy()
    n_total = len(subset)

    # Apply threshold
    restricted = subset[subset['n_nonzero_cox_genes'] >= THRESHOLD].copy()
    n_restricted = len(restricted)

    # Identify which pairs were excluded
    excluded = subset[subset['n_nonzero_cox_genes'] < THRESHOLD]

    print("=" * 70)
    print(f"  {label_val.upper()}")
    print("=" * 70)
    print(f"  Total pairs: {n_total}")
    print(f"  Pairs with >= {THRESHOLD} nonzero Cox genes: {n_restricted}")
    print(f"  Pairs excluded: {n_total - n_restricted}")
    print()

    # --- Sparsity distribution ---
    print(f"  Cox nonzero gene distribution (all pairs):")
    print(f"    mean: {subset['n_nonzero_cox_genes'].mean():.1f}")
    print(f"    median: {subset['n_nonzero_cox_genes'].median():.1f}")
    print(f"    min: {subset['n_nonzero_cox_genes'].min()}")
    print(f"    max: {subset['n_nonzero_cox_genes'].max()}")
    print()

    # --- Full aggregate ---
    print(f"  FULL AGGREGATE (n={n_total}):")
    full_summary = {}
    for col in metric_cols:
        vals = subset[col].dropna()
        if len(vals) > 0:
            full_summary[col] = {
                'mean': round(float(vals.mean()), 4),
                'std': round(float(vals.std()), 4),
                'n': int(len(vals)),
            }
            print(f"    {col}: {vals.mean():.4f} +/- {vals.std():.4f} (n={len(vals)})")
    print()

    # --- Restricted aggregate ---
    print(f"  RESTRICTED AGGREGATE (>= {THRESHOLD} nonzero Cox genes, n={n_restricted}):")
    restricted_summary = {}
    if n_restricted > 0:
        for col in metric_cols:
            vals = restricted[col].dropna()
            if len(vals) > 0:
                restricted_summary[col] = {
                    'mean': round(float(vals.mean()), 4),
                    'std': round(float(vals.std()), 4),
                    'min': round(float(vals.min()), 4),
                    'max': round(float(vals.max()), 4),
                    'median': round(float(vals.median()), 4),
                    'n': int(len(vals)),
                }
                print(f"    {col}: {vals.mean():.4f} +/- {vals.std():.4f} [{vals.min():.4f}, {vals.max():.4f}] (n={len(vals)})")
        print()

        # Per-direction breakdown (restricted only)
        for direction in restricted['direction'].unique():
            dir_sub = restricted[restricted['direction'] == direction]
            print(f"    {direction} (n={len(dir_sub)}):")
            for col in metric_cols:
                vals = dir_sub[col].dropna()
                if len(vals) > 0:
                    print(f"      {col}: {vals.mean():.4f} +/- {vals.std():.4f}")
            print(f"      Cox nonzero genes: {dir_sub['n_nonzero_cox_genes'].mean():.0f} +/- {dir_sub['n_nonzero_cox_genes'].std():.0f} [{dir_sub['n_nonzero_cox_genes'].min()}, {dir_sub['n_nonzero_cox_genes'].max()}]")
            print()
    else:
        print("    No pairs meet the threshold.")
    print()

    # --- Save restricted summary ---
    output = {
        'threshold': THRESHOLD,
        'n_total': n_total,
        'n_restricted': n_restricted,
        'full_aggregate': full_summary,
        'restricted_aggregate': restricted_summary,
    }
    out_file = OUTPUT_DIR / f"summary_{label_val}_restricted_ge{THRESHOLD}.json"
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {out_file}")
    print()

# -------------------------------------------------------------------------
# Print the list of included/excluded pairs for transparency
# -------------------------------------------------------------------------
print("=" * 70)
print("INCLUDED PAIRS (>= 20 nonzero Cox genes)")
print("=" * 70)
all_genes_df = df[df['label'] == 'all_genes'].copy()
included = all_genes_df[all_genes_df['n_nonzero_cox_genes'] >= THRESHOLD]
for _, row in included.iterrows():
    print(f"  seed={int(row['seed'])} fold={int(row['fold'])} {row['direction']}: {int(row['n_nonzero_cox_genes'])} nonzero")
print()

print("EXCLUDED PAIRS (< 20 nonzero Cox genes)")
print("-" * 70)
excluded = all_genes_df[all_genes_df['n_nonzero_cox_genes'] < THRESHOLD]
for _, row in excluded.iterrows():
    print(f"  seed={int(row['seed'])} fold={int(row['fold'])} {row['direction']}: {int(row['n_nonzero_cox_genes'])} nonzero")
