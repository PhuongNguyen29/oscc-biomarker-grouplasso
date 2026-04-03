"""
Unsupervised Gene Filtering by Interquartile Range (IQR)

Applies a cohort-specific median IQR filter:
    1. Compute IQR per gene within each cohort (unsupervised)
    2. Retain genes with IQR >= cohort-specific median
    3. Intersect retained genes across TCGA and ORIEN

This follows the independent filtering principle (Bourgon et al., 2010,
PNAS) and uses the robust variability metric and default quantile
threshold implemented in the Bioconductor genefilter package.

Rationale:
    - IQR is robust to outliers (Hampel, 1974, JASA)
    - Median (50th percentile) is the default cutoff in genefilter
    - Filtering on expression variability is independent of survival
      outcome, so it does not inflate type I error or introduce
      outcome leakage
    - Cohort-specific thresholds account for distributional differences
      between TCGA and ORIEN

References:
    - Bourgon et al. (2010) PNAS — Independent filtering increases
      detection power for high-throughput experiments
    - Bioconductor genefilter package — IQR as default variability
      metric with 0.5 quantile cutoff
    - Hampel (1974) JASA — Robust estimation via influence functions

Usage:
    cd ~/hpchome/oscc-biomarker-grouplasso
    python scripts/filter_genes_iqr.py

    # Verify output:
    wc -l data/processed/iqr_filtered/retained_genes.txt
    cat data/processed/iqr_filtered/filter_metadata.json

Output:
    data/processed/iqr_filtered/
        ├── retained_genes.txt          # One Ensembl ID per line
        ├── retained_genes_detailed.csv # Gene ID + IQR values
        ├── iqr_values_tcga.csv         # All genes, sorted by IQR
        ├── iqr_values_orien.csv        # All genes, sorted by IQR
        └── filter_metadata.json        # Thresholds, counts, rationale
"""

import sys
import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# =============================================================================
# Setup
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = "data/processed"
OUTPUT_DIR = "data/processed/iqr_filtered"

# Expression file names (from data_loader.py FILE_MAP)
TCGA_EXPR_FILE = "tcga_preprocessed_v4.csv"
ORIEN_EXPR_FILE = "orien_preprocessed_v4.csv"


# =============================================================================
# IQR Filter
# =============================================================================

def compute_iqr_filter(
    tcga_expr: pd.DataFrame,
    orien_expr: pd.DataFrame,
    output_dir: Path,
) -> tuple:
    """
    Unsupervised IQR-based gene filtering.

    Computes IQR per gene per cohort using expression values only
    (no survival outcome). Retains genes with IQR >= cohort-specific
    median in both cohorts, then intersects.

    Args:
        tcga_expr: DataFrame, shape (n_genes, n_samples), genes as rows
        orien_expr: DataFrame, shape (n_genes, n_samples), genes as rows
        output_dir: Path to save results

    Returns:
        retained_genes: Sorted list of Ensembl IDs passing the filter
        metadata: Dict with filter statistics
    """
    logger.info("Computing IQR filter (above-median in both cohorts)...")

    # Verify gene sets match
    assert set(tcga_expr.index) == set(orien_expr.index), \
        "Gene sets differ between TCGA and ORIEN expression matrices"

    n_total = len(tcga_expr)

    # ------------------------------------------------------------------
    # Step 1: Compute IQR per gene (rows=genes, columns=samples)
    # ------------------------------------------------------------------
    tcga_iqr = tcga_expr.quantile(0.75, axis=1) - tcga_expr.quantile(0.25, axis=1)
    orien_iqr = orien_expr.quantile(0.75, axis=1) - orien_expr.quantile(0.25, axis=1)

    tcga_median_iqr = float(tcga_iqr.median())
    orien_median_iqr = float(orien_iqr.median())

    logger.info(f"  Total genes: {n_total}")
    logger.info(f"  TCGA IQR:  median={tcga_median_iqr:.4f}, "
                f"mean={tcga_iqr.mean():.4f}, "
                f"range=[{tcga_iqr.min():.4f}, {tcga_iqr.max():.4f}]")
    logger.info(f"  ORIEN IQR: median={orien_median_iqr:.4f}, "
                f"mean={orien_iqr.mean():.4f}, "
                f"range=[{orien_iqr.min():.4f}, {orien_iqr.max():.4f}]")

    # ------------------------------------------------------------------
    # Step 2: Retain genes above cohort-specific median
    # ------------------------------------------------------------------
    tcga_above = set(tcga_iqr[tcga_iqr >= tcga_median_iqr].index)
    orien_above = set(orien_iqr[orien_iqr >= orien_median_iqr].index)

    logger.info(f"  TCGA above median: {len(tcga_above)} genes")
    logger.info(f"  ORIEN above median: {len(orien_above)} genes")

    # ------------------------------------------------------------------
    # Step 3: Intersect across cohorts
    # ------------------------------------------------------------------
    intersection = tcga_above & orien_above
    retained_genes = sorted(intersection)
    n_retained = len(retained_genes)

    logger.info(f"  Intersection (retained): {n_retained} genes")
    logger.info(f"  Removed: {n_total - n_retained} genes "
                f"({100 * (n_total - n_retained) / n_total:.1f}%)")

    # ------------------------------------------------------------------
    # Step 4: Save outputs
    # ------------------------------------------------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4a. Simple gene list (one Ensembl ID per line)
    genes_txt = output_dir / "retained_genes.txt"
    with open(genes_txt, 'w') as f:
        for gene in retained_genes:
            f.write(f"{gene}\n")
    logger.info(f"  Saved gene list: {genes_txt}")

    # 4b. Detailed CSV with IQR values for retained genes
    detailed_df = pd.DataFrame({
        'gene_id': retained_genes,
        'tcga_iqr': [tcga_iqr[g] for g in retained_genes],
        'orien_iqr': [orien_iqr[g] for g in retained_genes],
    })
    detailed_df = detailed_df.sort_values('tcga_iqr', ascending=False)
    detailed_csv = output_dir / "retained_genes_detailed.csv"
    detailed_df.to_csv(detailed_csv, index=False)
    logger.info(f"  Saved detailed CSV: {detailed_csv}")

    # 4c. Full IQR values for all genes (for reproducibility)
    tcga_iqr_df = tcga_iqr.sort_values(ascending=False).to_frame('IQR')
    tcga_iqr_df.index.name = 'gene_id'
    tcga_iqr_csv = output_dir / "iqr_values_tcga.csv"
    tcga_iqr_df.to_csv(tcga_iqr_csv)
    logger.info(f"  Saved TCGA IQR values: {tcga_iqr_csv}")

    orien_iqr_df = orien_iqr.sort_values(ascending=False).to_frame('IQR')
    orien_iqr_df.index.name = 'gene_id'
    orien_iqr_csv = output_dir / "iqr_values_orien.csv"
    orien_iqr_df.to_csv(orien_iqr_csv)
    logger.info(f"  Saved ORIEN IQR values: {orien_iqr_csv}")

    # 4d. Metadata JSON
    metadata = {
        'filter_method': 'IQR',
        'filter_rule': 'Above cohort-specific median IQR in both cohorts, intersected',
        'quantile_threshold': 0.50,
        'references': [
            'Bourgon et al. (2010) PNAS — Independent filtering principle',
            'Bioconductor genefilter — IQR with 0.5 quantile default',
        ],
        'timestamp': datetime.now().isoformat(),
        'input_genes': n_total,
        'tcga_samples': int(tcga_expr.shape[1]),
        'orien_samples': int(orien_expr.shape[1]),
        'tcga_median_iqr': tcga_median_iqr,
        'orien_median_iqr': orien_median_iqr,
        'tcga_above_median': len(tcga_above),
        'orien_above_median': len(orien_above),
        'tcga_only': len(tcga_above - orien_above),
        'orien_only': len(orien_above - tcga_above),
        'intersection': n_retained,
        'removed': n_total - n_retained,
        'removed_pct': round(100 * (n_total - n_retained) / n_total, 1),
        'output_files': {
            'gene_list': str(genes_txt),
            'detailed_csv': str(detailed_csv),
            'tcga_iqr': str(tcga_iqr_csv),
            'orien_iqr': str(orien_iqr_csv),
        },
    }

    metadata_file = output_dir / "filter_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Saved metadata: {metadata_file}")

    return retained_genes, metadata


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("IQR Gene Filter — Unsupervised Variability Filtering")
    logger.info("=" * 70)

    data_dir = Path(DATA_DIR)

    # Load expression matrices
    logger.info("Loading expression data...")
    tcga_path = data_dir / TCGA_EXPR_FILE
    orien_path = data_dir / ORIEN_EXPR_FILE

    if not tcga_path.exists():
        raise FileNotFoundError(f"TCGA expression file not found: {tcga_path}")
    if not orien_path.exists():
        raise FileNotFoundError(f"ORIEN expression file not found: {orien_path}")

    tcga_expr = pd.read_csv(tcga_path, index_col=0)
    orien_expr = pd.read_csv(orien_path, index_col=0)

    logger.info(f"  TCGA: {tcga_expr.shape[0]} genes × {tcga_expr.shape[1]} samples")
    logger.info(f"  ORIEN: {orien_expr.shape[0]} genes × {orien_expr.shape[1]} samples")

    # Run filter
    retained_genes, metadata = compute_iqr_filter(
        tcga_expr=tcga_expr,
        orien_expr=orien_expr,
        output_dir=OUTPUT_DIR,
    )

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("FILTER SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Input:     {metadata['input_genes']} genes")
    logger.info(f"  Retained:  {metadata['intersection']} genes")
    logger.info(f"  Removed:   {metadata['removed']} genes ({metadata['removed_pct']}%)")
    logger.info(f"  TCGA median IQR threshold:  {metadata['tcga_median_iqr']:.4f}")
    logger.info(f"  ORIEN median IQR threshold: {metadata['orien_median_iqr']:.4f}")
    logger.info(f"  Output:    {OUTPUT_DIR}/")
    logger.info("=" * 70)
    logger.info("Done.")
