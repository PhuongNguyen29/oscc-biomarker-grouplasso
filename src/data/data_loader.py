"""
Data Loader for v4 Pipeline (Phase 2+)

Loads Phase 1 preprocessed data (ComBat-corrected, IQR-filtered, z-scored,
clipped) from CSV files produced by phase1_data_preprocessing_v4.R.

Input files (in data/processed/):
    - tcga_corrected_preprocessed_v4.csv  : genes x samples, IQR-retained, z-scored, clipped
    - orien_corrected_preprocessed_v4.csv : genes x samples, IQR-retained, z-scored, clipped
    - surv_tcga_aligned_v4.csv            : sampleID, time, event
    - surv_orien_aligned_v4.csv           : sampleID, time, event
    - common_genes_v4.txt                 : 14,675 common genes (post TPM filter)

NOTE ON THE TWO EXPRESSION FILES ON DISK
----------------------------------------
`*_corrected_preprocessed_v4.csv` holds the 4,835 IQR-retained genes.
`*_zscored_full14675_v4.csv` holds the full 14,675-gene universe.
Both are z-scored and clipped, and a gene present in both has IDENTICAL values
(z-scoring is per gene, so the filter changes membership, not values -- verified
max |diff| = 0). The full matrix is retained because Phase G's top ladder row
needs it. FILE_MAP points at the RETAINED set; anything wanting the full
universe must ask for it explicitly.

SAMPLE ORDERING (added after audit)
-----------------------------------
Downstream code does:
    X = expr.loc[genes].values.T
    y = surv['event'].values
which is only correct if expression COLUMNS and survival ROWS are in the same
order. The previous version checked set membership and returned both objects
unsorted, so a reordering anywhere upstream would have silently paired the wrong
patient with the wrong outcome -- producing plausible but meaningless results.
load_cohort() now REORDERS survival to expression column order and asserts it.

Usage:
    # Load one cohort
    expr, surv = load_cohort('tcga')

    # Load both cohorts
    data = load_both_cohorts()

    # Create PyTorch dataset ready for training
    dataset = create_survival_dataset('tcga')
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

from src.data.dataset import SurvivalDataset

logger = logging.getLogger(__name__)

# Default paths relative to project root
DEFAULT_PROCESSED_DIR = "data/processed"

# v4 file naming convention
FILE_MAP = {
    'tcga': {
        'expression': 'tcga_corrected_preprocessed_v4.csv',
        'survival': 'surv_tcga_aligned_v4.csv',
        'clinical': 'clinical_tcga_aligned_v4.csv',
    },
    'orien': {
        'expression': 'orien_corrected_preprocessed_v4.csv',
        'survival': 'surv_orien_aligned_v4.csv',
        'clinical': 'clinical_orien_aligned_v4.csv',
    },
    'common_genes': 'common_genes_v4.txt',
}


def load_cohort(
    cohort: str,
    processed_dir: str = DEFAULT_PROCESSED_DIR
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load expression and survival data for one cohort.

    Args:
        cohort: 'tcga' or 'orien'
        processed_dir: Path to processed data directory

    Returns:
        expression: pd.DataFrame, shape (n_genes, n_samples), genes as rows
        survival: pd.DataFrame, index=sampleID, columns=['time', 'event']

    Raises:
        ValueError: If cohort is not 'tcga' or 'orien'
        FileNotFoundError: If data files are missing
    """
    cohort = cohort.lower()
    if cohort not in ('tcga', 'orien'):
        raise ValueError(f"cohort must be 'tcga' or 'orien', got '{cohort}'")

    processed_dir = Path(processed_dir)
    files = FILE_MAP[cohort]

    # Load expression (genes × samples)
    expr_path = processed_dir / files['expression']
    if not expr_path.exists():
        raise FileNotFoundError(f"Expression file not found: {expr_path}")

    expression = pd.read_csv(expr_path, index_col=0)

    # Load survival
    surv_path = processed_dir / files['survival']
    if not surv_path.exists():
        raise FileNotFoundError(f"Survival file not found: {surv_path}")

    survival = pd.read_csv(surv_path)

    # Set sampleID as index for survival (required by SurvivalDataset)
    if 'sampleID' in survival.columns:
        survival = survival.set_index('sampleID')
    elif survival.columns[0] == 'Unnamed: 0' or survival.columns[0] == '':
        # Handle case where first column is a row index from R
        survival = survival.set_index(survival.columns[0])
        if 'sampleID' in survival.columns:
            survival = survival.set_index('sampleID')

    # Drop any extra unnamed index columns from R export
    survival = survival[['time', 'event']]

    # ------------------------------------------------------------------
    # ENFORCE SAMPLE ORDER -- do not merely check set membership.
    #
    # Callers pair expr.values.T row i with surv.iloc[i], so the two must be
    # in the SAME ORDER, not merely contain the same IDs. Set equality cannot
    # detect a permutation, and a permutation would pair the wrong patient
    # with the wrong outcome while every count and shape still looked right.
    # ------------------------------------------------------------------
    missing = [s for s in expression.columns if s not in survival.index]
    if missing:
        raise ValueError(
            f"{cohort.upper()}: {len(missing)} expression samples have no "
            f"survival record (first few: {missing[:5]}). Re-run Phase 1; "
            f"expression and survival outputs must cover the same patients."
        )
    extra = [s for s in survival.index if s not in set(expression.columns)]
    if extra:
        logger.warning(
            f"{cohort.upper()}: {len(extra)} survival records have no expression "
            f"column and are dropped (first few: {extra[:5]})"
        )

    survival = survival.loc[expression.columns]        # reorder, not just subset

    assert list(expression.columns) == list(survival.index), \
        f"{cohort}: expression/survival order still mismatched after reindex"

    if survival[['time', 'event']].isna().any().any():
        n_na = int(survival[['time', 'event']].isna().any(axis=1).sum())
        raise ValueError(f"{cohort.upper()}: {n_na} samples have NA time/event")

    logger.info(
        f"Loaded {cohort.upper()}: {expression.shape[0]} genes × "
        f"{expression.shape[1]} samples, "
        f"{int(survival['event'].sum())} events "
        f"({survival['event'].mean():.1%})"
    )

    return expression, survival


def load_both_cohorts(
    processed_dir: str = DEFAULT_PROCESSED_DIR
) -> Dict[str, pd.DataFrame]:
    """
    Load expression and survival data for both cohorts.

    Returns:
        dict with keys:
            'tcga_expr', 'orien_expr': Expression DataFrames (genes × samples)
            'surv_tcga', 'surv_orien': Survival DataFrames (index=sampleID)
    """
    tcga_expr, surv_tcga = load_cohort('tcga', processed_dir)
    orien_expr, surv_orien = load_cohort('orien', processed_dir)

    # Verify gene alignment
    if not tcga_expr.index.equals(orien_expr.index):
        common = sorted(set(tcga_expr.index) & set(orien_expr.index))
        logger.warning(
            f"Gene indices differ. Using {len(common)} common genes "
            f"(TCGA: {len(tcga_expr)}, ORIEN: {len(orien_expr)})"
        )
        tcga_expr = tcga_expr.loc[common]
        orien_expr = orien_expr.loc[common]

    logger.info(
        f"Both cohorts loaded: {tcga_expr.shape[0]} genes, "
        f"TCGA={tcga_expr.shape[1]} samples, "
        f"ORIEN={orien_expr.shape[1]} samples"
    )

    return {
        'tcga_expr': tcga_expr,
        'orien_expr': orien_expr,
        'surv_tcga': surv_tcga,
        'surv_orien': surv_orien,
    }


def load_clinical(
    cohort: str,
    processed_dir: str = DEFAULT_PROCESSED_DIR
) -> pd.DataFrame:
    """
    Load clinical data for one cohort (for Phase 5+ downstream analysis).

    Args:
        cohort: 'tcga' or 'orien'
        processed_dir: Path to processed data directory

    Returns:
        clinical: pd.DataFrame with clinical variables
    """
    cohort = cohort.lower()
    if cohort not in ('tcga', 'orien'):
        raise ValueError(f"cohort must be 'tcga' or 'orien', got '{cohort}'")

    processed_dir = Path(processed_dir)
    clin_path = processed_dir / FILE_MAP[cohort]['clinical']

    if not clin_path.exists():
        raise FileNotFoundError(f"Clinical file not found: {clin_path}")

    clinical = pd.read_csv(clin_path, index_col=0)
    logger.info(f"Loaded {cohort.upper()} clinical: {clinical.shape}")

    return clinical


def create_survival_dataset(
    cohort: str,
    processed_dir: str = DEFAULT_PROCESSED_DIR
) -> SurvivalDataset:
    """
    Load data and create a PyTorch SurvivalDataset for one cohort.

    Args:
        cohort: 'tcga' or 'orien'
        processed_dir: Path to processed data directory

    Returns:
        SurvivalDataset ready for DataLoader or direct use
    """
    expression, survival = load_cohort(cohort, processed_dir)
    dataset = SurvivalDataset(expression, survival)
    return dataset


def get_n_features(processed_dir: str = DEFAULT_PROCESSED_DIR) -> int:
    """
    Get the number of features (genes) without loading full expression data.
    Reads the common genes file.

    Returns:
        Number of common genes (14,675 for v4 -- protein-coding, TPM-filtered,
        common to both cohorts). NOT the retained IQR set, which is 4,835.
    """
    genes_path = Path(processed_dir) / FILE_MAP['common_genes']
    if genes_path.exists():
        with open(genes_path, 'r') as f:
            n_genes = sum(1 for line in f if line.strip())
        return n_genes
    # The previous fallback read the header only (nrows=0) and returned
    # len(expr), which is ALWAYS 0 because no gene rows were read. It also
    # answered a different question -- the expression matrix holds the RETAINED
    # genes, not the common-gene universe. Raise instead of returning a wrong
    # number silently.
    raise FileNotFoundError(
        f"Common genes file not found: {genes_path}. This function reports the "
        f"size of the common-gene universe; for the retained-set size, read "
        f"data/processed/iqr_filtered_v4_prez/retained_genes.txt, and for a "
        f"loaded matrix use expression.shape[0]."
    )


def validate_v4_data(processed_dir: str = DEFAULT_PROCESSED_DIR) -> bool:
    """
    Validate that all required v4 files exist and are consistent.
    Run this before starting Phase 2 to catch data issues early.

    Returns:
        True if all checks pass
    """
    processed_dir = Path(processed_dir)
    all_ok = True

    logger.info("=" * 60)
    logger.info("Validating v4 preprocessed data...")

    # Check file existence
    required_files = [
        FILE_MAP['tcga']['expression'],
        FILE_MAP['tcga']['survival'],
        FILE_MAP['orien']['expression'],
        FILE_MAP['orien']['survival'],
        FILE_MAP['common_genes'],
    ]

    for fname in required_files:
        fpath = processed_dir / fname
        if fpath.exists():
            logger.info(f"  OK: {fname}")
        else:
            logger.error(f"  MISSING: {fname}")
            all_ok = False

    if not all_ok:
        logger.error("Missing files. Run phase1_data_preprocessing_v4.R first.")
        return False

    # Load and validate content
    data = load_both_cohorts(str(processed_dir))

    # Check dimensions
    tcga_genes = data['tcga_expr'].shape[0]
    orien_genes = data['orien_expr'].shape[0]
    assert tcga_genes == orien_genes, (
        f"Gene count mismatch: TCGA={tcga_genes}, ORIEN={orien_genes}"
    )

    # Check expression values are z-scored (mean ≈ 0, std ≈ 1)
    for cohort_name, expr_key in [('TCGA', 'tcga_expr'), ('ORIEN', 'orien_expr')]:
        expr = data[expr_key]
        gene_means = expr.mean(axis=1)
        gene_stds = expr.std(axis=1)
        mean_of_means = gene_means.mean()
        mean_of_stds = gene_stds.mean()

        if abs(mean_of_means) > 0.1:
            logger.warning(
                f"  {cohort_name} mean of gene means = {mean_of_means:.4f} "
                f"(expected ~0). Data may not be z-scored."
            )
            all_ok = False
        if abs(mean_of_stds - 1.0) > 0.2:
            logger.warning(
                f"  {cohort_name} mean of gene stds = {mean_of_stds:.4f} "
                f"(expected ~1). Data may not be z-scored."
            )
            all_ok = False

    # Check clipping (values should be within ±3)
    for cohort_name, expr_key in [('TCGA', 'tcga_expr'), ('ORIEN', 'orien_expr')]:
        expr = data[expr_key]
        max_val = expr.max().max()
        min_val = expr.min().min()
        if max_val > 3.01 or min_val < -3.01:
            logger.warning(
                f"  {cohort_name} range [{min_val:.2f}, {max_val:.2f}] "
                f"exceeds ±3 SD. Clipping may not have been applied."
            )
            all_ok = False

    # Check survival data
    for cohort_name, surv_key in [('TCGA', 'surv_tcga'), ('ORIEN', 'surv_orien')]:
        surv = data[surv_key]
        n_events = int(surv['event'].sum())
        n_total = len(surv)
        if n_events == 0:
            logger.error(f"  {cohort_name}: No events in survival data!")
            all_ok = False
        if surv['time'].min() <= 0:
            logger.warning(f"  {cohort_name}: Survival times include values <= 0")

    # Check sample alignment
    for cohort_name, expr_key, surv_key in [
        ('TCGA', 'tcga_expr', 'surv_tcga'),
        ('ORIEN', 'orien_expr', 'surv_orien')
    ]:
        # load_cohort() now guarantees identical ORDER, not just identical
        # membership. Check the stronger property here.
        if list(data[expr_key].columns) != list(data[surv_key].index):
            logger.error(
                f"  {cohort_name}: expression columns and survival index are "
                f"not in the same order -- load_cohort() should have enforced "
                f"this"
            )
            all_ok = False
        else:
            logger.info(f"  OK: {cohort_name} expression/survival order matches")

    if all_ok:
        logger.info("All validation checks passed.")
    else:
        logger.warning("Some validation checks failed. Review warnings above.")

    logger.info("=" * 60)
    return all_ok


# =============================================================================
# Self-test
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("Testing v4 Data Loader")
    print("=" * 60)

    # 1. Validate data
    print("\n--- Validation ---")
    valid = validate_v4_data()

    if not valid:
        print("Validation failed. Fix issues before proceeding.")
        exit(1)

    # 2. Test single cohort loading
    print("\n--- Single Cohort Loading ---")
    tcga_expr, surv_tcga = load_cohort('tcga')
    print(f"TCGA expression: {tcga_expr.shape} (genes × samples)")
    print(f"TCGA survival: {surv_tcga.shape}")
    print(f"TCGA expression range: [{tcga_expr.min().min():.2f}, {tcga_expr.max().max():.2f}]")
    print(f"TCGA survival columns: {surv_tcga.columns.tolist()}")
    print(f"TCGA survival index name: {surv_tcga.index.name}")

    # 3. Test both cohorts
    print("\n--- Both Cohorts ---")
    data = load_both_cohorts()
    print(f"Gene alignment: {data['tcga_expr'].index.equals(data['orien_expr'].index)}")

    # 4. Test PyTorch dataset creation
    print("\n--- PyTorch Dataset ---")
    dataset = create_survival_dataset('tcga')
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Feature shape: {sample['features'].shape}")
    print(f"Expected features: {tcga_expr.shape[0]}")
    print(f"Time: {sample['time']:.2f}, Event: {sample['event']:.0f}")

    # 5. Verify dataset dimensions match
    assert sample['features'].shape[0] == tcga_expr.shape[0], \
        f"Feature dimension mismatch: {sample['features'].shape[0]} vs {tcga_expr.shape[0]}"

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
