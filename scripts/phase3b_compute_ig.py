"""
Phase 3b: Compute Integrated Gradients on Saved Phase 3 Models

Loads saved deepsurv_model.pt checkpoints from Phase 3 cross-fitted
evaluation and computes Integrated Gradients (IG) attributions for
the held-out fold patients in each direction.

No retraining occurs. Models are loaded in eval mode and IG is computed
using the Captum library with Gauss-Legendre quadrature.

Design:
    3 seeds × 5 folds × 2 directions = 30 IG computations
    Each produces a (n_held_out, n_genes) attribution matrix

    Direction 1 (TCGA→ORIEN): TCGA-trained model → IG on ORIEN held-out
    Direction 2 (ORIEN→TCGA): ORIEN-trained model → IG on TCGA held-out

Configuration:
    n_steps=300 (v4 document specification for Phase 3 IG)
    Baseline: zero vector (population mean after z-scoring)
    Method: Gauss-Legendre quadrature

References:
    - Sundararajan et al. (2017) ICML — Integrated Gradients
    - Captum library (Kokhlikyan et al., 2020)

Usage:
    cd ~/hpchome/oscc-biomarker-elasticdeepsurv
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/phase3b_compute_ig.py

    # Single seed:
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/phase3b_compute_ig.py --seed 42

    # Specific direction only:
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/phase3b_compute_ig.py --direction tcga_to_orien

Output:
    results/phase3_cross_fitted/seed_42/fold_1/tcga_to_orien/
        ig_attributions.csv          (n_patients × n_genes, sampleID as index)
    ...
    results/phase3_cross_fitted/seed_42/aggregated/
        tcga_to_orien_deepsurv_ig_all.csv   (all ORIEN patients, out-of-fold IG)
        orien_to_tcga_deepsurv_ig_all.csv   (all TCGA patients, out-of-fold IG)
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
PHASE3_DIR = Path("results/phase3_cross_fitted")
SEEDS = [42, 123, 456]
N_FOLDS = 5

# IG configuration — v4 document specifies 300 steps for Phase 3
IG_N_STEPS = 300
IG_BATCH_SIZE = 32  # patients per IG batch (GPU memory management)

# Model hyperparameters (must match Phase 3 training)

def load_phase2_deepsurv_params(phase2_dir="results/phase2_tuning"):
    """Load best DeepSurv hyperparameters from Phase 2 tuning output."""
    phase2_path = Path(phase2_dir)
    tuning_dirs = sorted(phase2_path.glob("deepsurv_tuning_*"))
    if not tuning_dirs:
        raise FileNotFoundError(f"No deepsurv tuning results found in {phase2_path}")
    latest = tuning_dirs[-1]
    results_file = latest / "all_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    with open(results_file) as f:
        all_results = json.load(f)
    logger.info(f"Loaded DeepSurv Phase 2 params from: {results_file}")
    return {
        cohort: all_results[cohort]['best_params']
        for cohort in ['tcga', 'orien']
    }

_phase2_params = load_phase2_deepsurv_params()

# Map direction to training cohort's params
# tcga_to_orien: model trained on TCGA → use TCGA params
# orien_to_tcga: model trained on ORIEN → use ORIEN params
MODEL_PARAMS = {
    'tcga_to_orien': _phase2_params['tcga'],
    'orien_to_tcga': _phase2_params['orien'],
}

DIRECTIONS = ['tcga_to_orien', 'orien_to_tcga']


# =============================================================================
# Core IG computation
# =============================================================================

def compute_ig_for_fold(
    model_path: Path,
    risk_scores_path: Path,
    expr_data: pd.DataFrame,
    direction: str,
    device: str,
    n_features: int,
) -> pd.DataFrame:
    """
    Load a saved model checkpoint and compute IG for held-out patients.

    Args:
        model_path: Path to deepsurv_model.pt (state_dict)
        risk_scores_path: Path to deepsurv_risk_scores.csv (defines held-out patients)
        expr_data: Full expression DataFrame (genes × samples) for the TARGET cohort
        direction: 'tcga_to_orien' or 'orien_to_tcga'
        device: 'cuda' or 'cpu'
        n_features: Number of genes (must match model input dimension)

    Returns:
        DataFrame: (n_held_out, n_genes) with sampleID as index, gene IDs as columns
    """
    params = MODEL_PARAMS[direction]

    # Reconstruct model architecture
    model = ElasticDeepSurv(
        n_features=n_features,
        hidden_sizes=params['hidden_sizes'],
        dropout=params['dropout'],
        activation=params['activation'],
        batch_norm=params['batch_norm'],
        l1_ratio=params['l1_ratio'],
        alpha=params['alpha'],
    )

    # Load saved weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Identify held-out patients from risk scores file
    risk_df = pd.read_csv(risk_scores_path)
    held_out_ids = risk_df['sampleID'].tolist()

    # Get expression data for held-out patients
    # expr_data is (genes × samples), we need (samples × genes)
    available_ids = [sid for sid in held_out_ids if sid in expr_data.columns]
    if len(available_ids) != len(held_out_ids):
        logger.warning(f"  {len(held_out_ids) - len(available_ids)} held-out patients "
                       f"not found in expression data")

    X_held_out = expr_data[available_ids].values.T.astype(np.float32)  # (n_patients, n_genes)
    gene_names = expr_data.index.tolist()

    # Setup IG
    from captum.attr import IntegratedGradients
    ig = IntegratedGradients(model)

    # Zero baseline (population mean after z-scoring)
    baseline = torch.zeros(1, n_features).to(device)

    # Compute IG in batches
    all_attributions = []
    X_tensor = torch.from_numpy(X_held_out).to(device)

    for start in range(0, len(X_held_out), IG_BATCH_SIZE):
        end = min(start + IG_BATCH_SIZE, len(X_held_out))
        batch_input = X_tensor[start:end]
        batch_baseline = baseline.expand(batch_input.shape[0], -1)

        with torch.no_grad():
            pass  # IG needs gradients, but we use no_grad for non-IG ops

        attr = ig.attribute(
            batch_input,
            baselines=batch_baseline,
            n_steps=IG_N_STEPS,
            method='gausslegendre',
        )
        all_attributions.append(attr.cpu().detach().numpy())

    attributions = np.concatenate(all_attributions, axis=0)  # (n_patients, n_genes)

    # Create DataFrame with sampleID index and gene columns
    attr_df = pd.DataFrame(
        attributions,
        index=available_ids,
        columns=gene_names,
    )
    attr_df.index.name = 'sampleID'

    return attr_df


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 3b: Compute IG on saved Phase 3 models"
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help="Run for a specific seed only (default: all seeds)"
    )
    parser.add_argument(
        '--direction', type=str, default=None,
        choices=['tcga_to_orien', 'orien_to_tcga'],
        help="Run for a specific direction only (default: both)"
    )
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Log to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = PHASE3_DIR / f"phase3b_ig_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(file_handler)

    logger.info("Phase 3b: Compute IG Attributions on Saved Models")
    logger.info(f"Device: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"IG n_steps: {IG_N_STEPS}")
    logger.info(f"IG method: gausslegendre")

    # Load expression data
    logger.info("Loading expression data...")
    tcga_expr, tcga_surv = load_cohort('tcga', DATA_DIR)
    orien_expr, orien_surv = load_cohort('orien', DATA_DIR)

    gene_names = tcga_expr.index.tolist()
    n_features = len(gene_names)
    logger.info(f"Genes: {n_features}")

    # Verify gene order
    if list(orien_expr.index) != gene_names:
        raise ValueError("Gene order mismatch between TCGA and ORIEN")

    # Map direction to target cohort expression data
    # tcga_to_orien: model trained on TCGA, IG computed on ORIEN held-out
    # orien_to_tcga: model trained on ORIEN, IG computed on TCGA held-out
    target_expr = {
        'tcga_to_orien': orien_expr,
        'orien_to_tcga': tcga_expr,
    }

    # Determine which seeds and directions to run
    seeds_to_run = [args.seed] if args.seed else SEEDS
    directions_to_run = [args.direction] if args.direction else DIRECTIONS

    logger.info(f"Seeds: {seeds_to_run}")
    logger.info(f"Directions: {directions_to_run}")

    # =================================================================
    # Compute IG for each seed × fold × direction
    # =================================================================

    for seed in seeds_to_run:
        seed_dir = PHASE3_DIR / f"seed_{seed}"
        if not seed_dir.exists():
            logger.warning(f"Seed directory not found: {seed_dir}, skipping")
            continue

        logger.info(f"")
        logger.info(f"{'='*70}")
        logger.info(f"SEED {seed}")
        logger.info(f"{'='*70}")

        for direction in directions_to_run:
            logger.info(f"")
            logger.info(f"  --- {direction} ---")

            fold_attr_dfs = []

            for fold in range(1, N_FOLDS + 1):
                fold_dir = seed_dir / f"fold_{fold}" / direction
                model_path = fold_dir / "deepsurv_model.pt"
                risk_path = fold_dir / "deepsurv_risk_scores.csv"

                if not model_path.exists():
                    logger.warning(f"  Fold {fold}: model not found at {model_path}")
                    continue
                if not risk_path.exists():
                    logger.warning(f"  Fold {fold}: risk scores not found at {risk_path}")
                    continue

                logger.info(f"  Fold {fold}: computing IG...")

                attr_df = compute_ig_for_fold(
                    model_path=model_path,
                    risk_scores_path=risk_path,
                    expr_data=target_expr[direction],
                    direction=direction,
                    device=device,
                    n_features=n_features,
                )

                logger.info(f"    {attr_df.shape[0]} patients, "
                            f"mean |IG| range: [{attr_df.abs().mean().min():.6f}, "
                            f"{attr_df.abs().mean().max():.6f}]")

                # Save per-fold attributions
                attr_df.to_csv(fold_dir / "ig_attributions.csv")
                fold_attr_dfs.append(attr_df)

            # Aggregate across folds (concatenate held-out patients)
            if fold_attr_dfs:
                aggregated = pd.concat(fold_attr_dfs, axis=0)

                # Verify no duplicate patients
                if aggregated.index.duplicated().any():
                    n_dup = aggregated.index.duplicated().sum()
                    logger.warning(f"  {n_dup} duplicate sampleIDs in aggregated IG!")

                agg_dir = seed_dir / "aggregated"
                agg_dir.mkdir(parents=True, exist_ok=True)
                agg_file = agg_dir / f"{direction}_deepsurv_ig_all.csv"
                aggregated.to_csv(agg_file)

                logger.info(f"  Aggregated: {aggregated.shape[0]} patients × "
                            f"{aggregated.shape[1]} genes → {agg_file.name}")

                # Quick summary stats
                mean_abs = aggregated.abs().mean()
                logger.info(f"  Mean |IG| summary: "
                            f"median={mean_abs.median():.6f}, "
                            f"max={mean_abs.max():.6f}, "
                            f"top-5 genes by mean |IG|:")
                top5 = mean_abs.nlargest(5)
                for gene, val in top5.items():
                    logger.info(f"    {gene}: {val:.6f}")

    # =================================================================
    # Summary
    # =================================================================

    logger.info(f"")
    logger.info(f"{'='*70}")
    logger.info(f"PHASE 3b COMPLETE")
    logger.info(f"{'='*70}")

    # Check what was computed
    for seed in seeds_to_run:
        for direction in directions_to_run:
            agg_file = PHASE3_DIR / f"seed_{seed}" / "aggregated" / f"{direction}_deepsurv_ig_all.csv"
            if agg_file.exists():
                df = pd.read_csv(agg_file, index_col=0, nrows=0)
                logger.info(f"  seed_{seed}/{direction}: {agg_file.name} exists "
                            f"({df.shape[1]} gene columns)")
            else:
                logger.warning(f"  seed_{seed}/{direction}: aggregated file MISSING")

    logger.info(f"\nLog saved to {log_file}")
