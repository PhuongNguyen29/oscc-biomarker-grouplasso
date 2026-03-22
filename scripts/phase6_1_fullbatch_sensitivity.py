"""
Phase 6.1: Full-Batch vs Mini-Batch Training Sensitivity Analysis

Purpose:
    Validate that mini-batch Cox loss approximation does not materially
    affect ElasticDeepSurv results.

Design:
    Re-train ElasticDeepSurv with full-batch gradient descent
    (batch_size = n_train) for folds 1, 3, 5 of seed 42, both directions.

    Compare to existing mini-batch results (from Phase 3):
      (a) C-index difference on held-out fold
      (b) Spearman rank correlation of IG gene rankings
      (c) Convergence behavior: loss curves and early stopping epoch

Acceptance criteria (from pipeline v4):
    - C-index difference < 0.02 across all folds
    - Gene ranking correlation > 0.90 across all folds

References:
    - Katzman et al. (2018) BMC Med Res Methodol — DeepSurv
    - Sundararajan et al. (2017) ICML — Integrated Gradients
    - Bottou (2010) COMPSTAT — mini-batch vs full-batch SGD
    - Harrell et al. (1982) JAMA — C-index

Usage:
    cd ~/hpchome/oscc-biomarker-elasticdeepsurv
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/phase6_1_fullbatch_sensitivity.py

    # Or via nohup:
    PYTHONPATH="${PWD}:${PYTHONPATH}" nohup python scripts/phase6_1_fullbatch_sensitivity.py \
        > results/phase6_1_nohup.log 2>&1 &

Output:
    results/phase6_1_fullbatch/
        fold_1/
            tcga_to_orien/
                fullbatch_risk_scores.csv
                fullbatch_ig_attributions.csv
                fullbatch_model.pt
                fullbatch_training_history.json
                comparison.json
            orien_to_tcga/
                ...
        fold_3/ ...
        fold_5/ ...
        phase6_1_summary.json
        phase6_1_YYYYMMDD_HHMMSS.log
"""

import sys
import os
import logging
import json
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from scipy.stats import spearmanr

from sklearn.model_selection import StratifiedKFold, train_test_split
from sksurv.metrics import concordance_index_censored
from captum.attr import IntegratedGradients

from src.data.data_loader import load_cohort
from src.models.elastic_deepsurv import ElasticDeepSurv, ElasticDeepSurvTrainer
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# Setup
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose sub-module logging
logging.getLogger('src.models.deepsurv').setLevel(logging.ERROR)
logging.getLogger('src.models.elastic_deepsurv').setLevel(logging.ERROR)
logging.getLogger('src.utils.batch_samplers').setLevel(logging.ERROR)
logging.getLogger('src.data.dataset').setLevel(logging.ERROR)
logging.getLogger('src.data.data_loader').setLevel(logging.WARNING)

# =============================================================================
# Configuration — IDENTICAL to Phase 3
# =============================================================================

SEED = 42
N_FOLDS = 5
TARGET_FOLDS = [0, 1, 2, 3, 4]  # fold_idx 0=fold_1, 2=fold_3, 4=fold_5
DATA_DIR = "data/processed"
PHASE3_DIR = "results/phase3_cross_fitted"
RESULTS_DIR = "results/phase6_1_fullbatch"

# Training config — same as Phase 3
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 20
SCHEDULER_PATIENCE = 10
WEIGHT_DECAY = 0.0

# IG config — same as Phase 3b
IG_N_STEPS = 300

# Phase 2 best hyperparameters — IDENTICAL to Phase 3
DEEPSURV_PARAMS = {
    'tcga': {
        'hidden_sizes': [96, 64],
        'alpha': 1.92e-03,
        'l1_ratio': 0.3,
        'learning_rate': 4.6e-04,
        'dropout': 0.29,
        'batch_size': 24,       # mini-batch (for reference only)
        'activation': 'relu',
        'batch_norm': False,
    },
    'orien': {
        'hidden_sizes': [128, 96],
        'alpha': 5.60e-03,
        'l1_ratio': 0.9,
        'learning_rate': 4.6e-04,
        'dropout': 0.35,
        'batch_size': 64,       # mini-batch (for reference only)
        'activation': 'relu',
        'batch_norm': True,
    },
}


# =============================================================================
# Helper: strata (identical to Phase 3)
# =============================================================================

def create_strata(y_event, y_time):
    """Create event x time quartile stratification labels."""
    time_quartiles = pd.qcut(y_time, q=4, labels=False, duplicates='drop')
    return y_event.astype(int) * 4 + time_quartiles


# =============================================================================
# Helper: lightweight dataset for full-batch loading
# =============================================================================

class FullBatchDataset:
    """Dataset that yields all data in a single batch."""
    def __init__(self, X, y_time, y_event, sample_ids, gene_names):
        self.X = X.astype(np.float32)
        self.y_time = y_time.astype(np.float32)
        self.y_event = y_event.astype(np.float32)
        self.sample_ids = sample_ids
        self.gene_names = gene_names
        self.n_features = X.shape[1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'features': torch.from_numpy(self.X[idx]),
            'time': torch.tensor(self.y_time[idx], dtype=torch.float32),
            'event': torch.tensor(self.y_event[idx], dtype=torch.float32),
            'sample_id': self.sample_ids[idx],
        }


def make_fullbatch_loader(dataset):
    """Create a DataLoader that yields one batch with all samples."""
    return DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,  # No need to shuffle — single batch
        num_workers=0,
        pin_memory=True,
    )


# =============================================================================
# Full-batch training
# =============================================================================

def train_fullbatch_deepsurv(
    X_train, y_event_train, y_time_train, sample_ids_train,
    X_test, sample_ids_test,
    gene_names,
    params: Dict,
    device: str,
    train_seed: int,
) -> Tuple[np.ndarray, ElasticDeepSurv, Dict]:
    """
    Train ElasticDeepSurv with full-batch gradient descent.

    Identical to Phase 3 train_deepsurv_and_predict EXCEPT:
      - batch_size = len(train_ds) for training loader
      - batch_size = len(es_ds) for early stopping loader
      - No StratifiedBatchSampler (unnecessary with single batch)

    Returns:
        risk_scores: Test set predictions
        model: Trained model (CPU)
        history: Training history with per-epoch metrics
    """
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(train_seed)

    n_features = X_train.shape[1]

    # --- Internal ES split: IDENTICAL to Phase 3 (same seed, same strata) ---
    strata_es = create_strata(y_event_train, y_time_train)
    tr_idx, es_idx = train_test_split(
        np.arange(len(X_train)),
        test_size=0.1,
        stratify=strata_es,
        random_state=train_seed,
    )

    logger.info(f"      Train split: {len(tr_idx)} train, {len(es_idx)} ES")

    train_ds = FullBatchDataset(
        X_train[tr_idx], y_time_train[tr_idx], y_event_train[tr_idx],
        [sample_ids_train[i] for i in tr_idx], gene_names,
    )
    es_ds = FullBatchDataset(
        X_train[es_idx], y_time_train[es_idx], y_event_train[es_idx],
        [sample_ids_train[i] for i in es_idx], gene_names,
    )

    # FULL-BATCH loaders — single batch containing all samples
    train_loader = make_fullbatch_loader(train_ds)
    es_loader = make_fullbatch_loader(es_ds)

    logger.info(f"      Full-batch train loader: batch_size={len(train_ds)}")
    logger.info(f"      Full-batch ES loader: batch_size={len(es_ds)}")

    # --- Create model: IDENTICAL architecture ---
    model = ElasticDeepSurv(
        n_features=n_features,
        hidden_sizes=params['hidden_sizes'],
        dropout=params['dropout'],
        activation=params['activation'],
        batch_norm=params['batch_norm'],
        l1_ratio=params['l1_ratio'],
        alpha=params['alpha'],
    )

    trainer = ElasticDeepSurvTrainer(
        model=model,
        learning_rate=params['learning_rate'],
        weight_decay=WEIGHT_DECAY,
        scheduler_patience=SCHEDULER_PATIENCE,
        device=device,
    )

    # --- Training loop: IDENTICAL logic to Phase 3 ---
    best_cindex = 0.0
    best_epoch = 0
    patience_counter = 0
    best_state = None

    history = {
        'train_loss': [],
        'es_cindex': [],
        'epoch': [],
    }

    for epoch in range(MAX_EPOCHS):
        train_loss, train_cox, train_penalty = trainer.train_epoch(train_loader)
        _, _, _, es_cindex = trainer.evaluate(es_loader)
        trainer.scheduler.step(es_cindex)

        history['train_loss'].append(float(train_loss))
        history['es_cindex'].append(float(es_cindex))
        history['epoch'].append(epoch + 1)

        if es_cindex > best_cindex:
            best_cindex = es_cindex
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    history['best_epoch'] = best_epoch
    history['best_es_cindex'] = float(best_cindex)
    history['total_epochs'] = epoch + 1

    logger.info(
        f"      Full-batch: best epoch={best_epoch}, "
        f"ES C-index={best_cindex:.4f}, total epochs={epoch+1}"
    )

    # --- Predict on test data ---
    model.eval()
    model.to(device)
    X_test_tensor = torch.from_numpy(X_test.astype(np.float32)).to(device)

    with torch.no_grad():
        log_hazards = model(X_test_tensor)
        risk_scores = torch.exp(log_hazards).squeeze().cpu().numpy()

    model.cpu()
    return risk_scores, model, history


# =============================================================================
# IG computation (matches Phase 3b)
# =============================================================================

def compute_ig_attributions(
    model: ElasticDeepSurv,
    X: np.ndarray,
    device: str,
    n_steps: int = 300,
) -> np.ndarray:
    """
    Compute Integrated Gradients for the given samples.

    Uses zero-vector baseline (population mean after z-scoring).
    Returns attributions matrix (n_samples x n_features).

    Reference: Sundararajan et al. (2017) ICML
    """
    model.eval()
    model.to(device)

    X_tensor = torch.from_numpy(X.astype(np.float32)).to(device)
    baseline = torch.zeros_like(X_tensor[0:1]).to(device)  # single zero vector

    # Captum IG — attribute w.r.t. the scalar log-hazard output
    def forward_fn(x):
        return model(x).squeeze(-1)  # (batch,) for Captum

    ig = IntegratedGradients(forward_fn)

    # Compute per-sample attributions
    # Process in chunks to manage GPU memory
    chunk_size = 64
    all_attrs = []

    for start in range(0, len(X_tensor), chunk_size):
        end = min(start + chunk_size, len(X_tensor))
        chunk = X_tensor[start:end]

        # Expand baseline to match chunk size
        baseline_expanded = baseline.expand(chunk.shape[0], -1)

        attrs = ig.attribute(
            chunk,
            baselines=baseline_expanded,
            n_steps=n_steps,
            internal_batch_size=chunk.shape[0],
        )
        all_attrs.append(attrs.cpu().detach().numpy())

    model.cpu()
    return np.vstack(all_attrs)


# =============================================================================
# Comparison logic
# =============================================================================

def compare_results(
    fb_risk_scores: np.ndarray,
    mb_risk_df: pd.DataFrame,
    fb_ig_attrs: np.ndarray,
    mb_ig_df: pd.DataFrame,
    y_event_test: np.ndarray,
    y_time_test: np.ndarray,
    gene_names: List[str],
    fb_history: Dict,
) -> Dict:
    """
    Compare full-batch vs mini-batch results.

    Returns dict with all comparison metrics.
    """
    results = {}

    # --- (a) C-index comparison ---
    event_bool = y_event_test.astype(bool)

    fb_ci, _, _, _, _ = concordance_index_censored(
        event_bool, y_time_test, fb_risk_scores
    )
    mb_ci, _, _, _, _ = concordance_index_censored(
        event_bool, y_time_test, mb_risk_df['risk_score'].values
    )

    ci_diff = abs(fb_ci - mb_ci)

    results['fullbatch_cindex'] = float(fb_ci)
    results['minibatch_cindex'] = float(mb_ci)
    results['cindex_difference'] = float(ci_diff)
    results['cindex_acceptable'] = bool(ci_diff < 0.02)

    logger.info(
        f"        C-index: FB={fb_ci:.4f}, MB={mb_ci:.4f}, "
        f"|diff|={ci_diff:.4f} ({'PASS' if ci_diff < 0.02 else 'FAIL'})"
    )

    # --- (b) Spearman correlation of IG gene rankings ---
    # Full-batch: mean |IG| per gene
    fb_mean_abs_ig = np.mean(np.abs(fb_ig_attrs), axis=0)
    fb_gene_ranking = np.argsort(-fb_mean_abs_ig)  # descending

    # Mini-batch: mean |IG| per gene (from saved csv)
    # mb_ig_df has sampleID as first col, then gene columns
    mb_ig_values = mb_ig_df.drop(columns=['sampleID']).values
    mb_gene_names_from_csv = mb_ig_df.drop(columns=['sampleID']).columns.tolist()

    # Ensure gene order matches
    assert mb_gene_names_from_csv == gene_names, (
        "Gene name mismatch between full-batch and mini-batch IG files! "
        f"First 3 FB: {gene_names[:3]}, First 3 MB: {mb_gene_names_from_csv[:3]}"
    )

    mb_mean_abs_ig = np.mean(np.abs(mb_ig_values), axis=0)

    # Spearman on the mean |IG| values directly (rank correlation)
    rho, pval = spearmanr(fb_mean_abs_ig, mb_mean_abs_ig)

    results['ig_spearman_rho'] = float(rho)
    results['ig_spearman_pval'] = float(pval)
    results['ig_ranking_acceptable'] = bool(rho > 0.90)

    logger.info(
        f"        IG Spearman: rho={rho:.4f}, p={pval:.2e} "
        f"({'PASS' if rho > 0.90 else 'FAIL'})"
    )

    # Also report top-100 and top-500 overlap for interpretability
    fb_top100 = set(np.argsort(-fb_mean_abs_ig)[:100])
    mb_top100 = set(np.argsort(-mb_mean_abs_ig)[:100])
    top100_overlap = len(fb_top100 & mb_top100)

    fb_top500 = set(np.argsort(-fb_mean_abs_ig)[:500])
    mb_top500 = set(np.argsort(-mb_mean_abs_ig)[:500])
    top500_overlap = len(fb_top500 & mb_top500)

    results['ig_top100_overlap'] = top100_overlap
    results['ig_top500_overlap'] = top500_overlap

    logger.info(
        f"        IG top-k overlap: top-100={top100_overlap}/100, "
        f"top-500={top500_overlap}/500"
    )

    # --- (c) Convergence ---
    results['fullbatch_best_epoch'] = fb_history['best_epoch']
    results['fullbatch_total_epochs'] = fb_history['total_epochs']
    results['fullbatch_best_es_cindex'] = fb_history['best_es_cindex']

    # Overall pass/fail
    results['overall_acceptable'] = bool(
        results['cindex_acceptable'] and results['ig_ranking_acceptable']
    )

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results_base = Path(RESULTS_DIR)
    results_base.mkdir(parents=True, exist_ok=True)

    # Log to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        results_base / f"phase6_1_{timestamp}.log"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 70)
    logger.info("Phase 6.1: Full-Batch vs Mini-Batch Sensitivity Analysis")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Target folds: {[f+1 for f in TARGET_FOLDS]} (seed {SEED})")
    logger.info(f"Results: {results_base}")
    logger.info("")

    # --- Load data (identical to Phase 3) ---
    logger.info("Loading data...")
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

    logger.info(
        f"TCGA: {tcga_X.shape[0]} samples x {tcga_X.shape[1]} genes, "
        f"{int(tcga_event.sum())} events"
    )
    logger.info(
        f"ORIEN: {orien_X.shape[0]} samples x {orien_X.shape[1]} genes, "
        f"{int(orien_event.sum())} events"
    )

    # --- Regenerate folds (identical to Phase 3) ---
    tcga_strata = create_strata(tcga_event, tcga_time)
    orien_strata = create_strata(orien_event, orien_time)

    tcga_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    orien_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    tcga_folds = list(tcga_skf.split(tcga_X, tcga_strata))
    orien_folds = list(orien_skf.split(orien_X, orien_strata))

    # --- Run target folds ---
    all_comparisons = []

    for fold_idx in TARGET_FOLDS:
        fold_num = fold_idx + 1
        logger.info("")
        logger.info(f"{'='*60}")
        logger.info(f"  FOLD {fold_num}")
        logger.info(f"{'='*60}")

        tcga_train_idx, tcga_test_idx = tcga_folds[fold_idx]
        orien_train_idx, orien_test_idx = orien_folds[fold_idx]

        fold_dir = results_base / f"fold_{fold_num}"

        # Phase 3 mini-batch results directory
        mb_fold_dir = Path(PHASE3_DIR) / f"seed_{SEED}" / f"fold_{fold_num}"

        for train_cohort, test_cohort in [('tcga', 'orien'), ('orien', 'tcga')]:
            direction = f"{train_cohort}_to_{test_cohort}"
            logger.info("")
            logger.info(f"    --- {direction} ---")

            out_dir = fold_dir / direction
            out_dir.mkdir(parents=True, exist_ok=True)

            # Select train/test data based on direction
            if train_cohort == 'tcga':
                X_train = tcga_X[tcga_train_idx]
                y_event_train = tcga_event[tcga_train_idx]
                y_time_train = tcga_time[tcga_train_idx]
                sample_ids_train = [tcga_ids[i] for i in tcga_train_idx]
                X_test = orien_X[orien_test_idx]
                y_event_test = orien_event[orien_test_idx]
                y_time_test = orien_time[orien_test_idx]
                sample_ids_test = [orien_ids[i] for i in orien_test_idx]
            else:
                X_train = orien_X[orien_train_idx]
                y_event_train = orien_event[orien_train_idx]
                y_time_train = orien_time[orien_train_idx]
                sample_ids_train = [orien_ids[i] for i in orien_train_idx]
                X_test = tcga_X[tcga_test_idx]
                y_event_test = tcga_event[tcga_test_idx]
                y_time_test = tcga_time[tcga_test_idx]
                sample_ids_test = [tcga_ids[i] for i in tcga_test_idx]

            params = DEEPSURV_PARAMS[train_cohort]
            train_seed = SEED + fold_idx  # identical to Phase 3

            logger.info(
                f"      Train: n={len(X_train)}, events={int(y_event_train.sum())}"
            )
            logger.info(
                f"      Test: n={len(X_test)}, events={int(y_event_test.sum())}"
            )
            logger.info(
                f"      Mini-batch size was: {params['batch_size']}, "
                f"full-batch size: {len(X_train)}"
            )

            # === Step 1: Full-batch training ===
            logger.info("      Training full-batch ElasticDeepSurv...")
            fb_risks, fb_model, fb_history = train_fullbatch_deepsurv(
                X_train, y_event_train, y_time_train, sample_ids_train,
                X_test, sample_ids_test,
                gene_names, params, device, train_seed,
            )

            # Save full-batch results
            fb_risk_df = pd.DataFrame({
                'sampleID': sample_ids_test,
                'risk_score': fb_risks,
                'time': y_time_test,
                'event': y_event_test.astype(int),
            })
            fb_risk_df.to_csv(out_dir / "fullbatch_risk_scores.csv", index=False)
            torch.save(fb_model.state_dict(), out_dir / "fullbatch_model.pt")

            with open(out_dir / "fullbatch_training_history.json", 'w') as f:
                json.dump(fb_history, f, indent=2)

            # === Step 2: Compute IG on test set (same patients as Phase 3b) ===
            logger.info(
                f"      Computing IG attributions (n_steps={IG_N_STEPS}) "
                f"on {len(X_test)} test samples..."
            )
            fb_ig_attrs = compute_ig_attributions(
                fb_model, X_test, device, n_steps=IG_N_STEPS,
            )

            fb_ig_df = pd.DataFrame(fb_ig_attrs, columns=gene_names)
            fb_ig_df.insert(0, 'sampleID', sample_ids_test)
            fb_ig_df.to_csv(
                out_dir / "fullbatch_ig_attributions.csv", index=False
            )

            # === Step 3: Load mini-batch results ===
            mb_risk_path = mb_fold_dir / direction / "deepsurv_risk_scores.csv"
            mb_ig_path = mb_fold_dir / direction / "ig_attributions.csv"

            if not mb_risk_path.exists():
                logger.error(
                    f"      Mini-batch risk scores not found: {mb_risk_path}"
                )
                continue
            if not mb_ig_path.exists():
                logger.error(
                    f"      Mini-batch IG attributions not found: {mb_ig_path}"
                )
                continue

            mb_risk_df = pd.read_csv(mb_risk_path)
            mb_ig_df = pd.read_csv(mb_ig_path)

            # === Step 4: Compare ===
            logger.info("      Comparing full-batch vs mini-batch...")
            comparison = compare_results(
                fb_risk_scores=fb_risks,
                mb_risk_df=mb_risk_df,
                fb_ig_attrs=fb_ig_attrs,
                mb_ig_df=mb_ig_df,
                y_event_test=y_event_test,
                y_time_test=y_time_test,
                gene_names=gene_names,
                fb_history=fb_history,
            )

            comparison['fold'] = fold_num
            comparison['direction'] = direction
            comparison['train_cohort'] = train_cohort
            comparison['n_train'] = len(X_train)
            comparison['n_test'] = len(X_test)
            comparison['minibatch_batch_size'] = params['batch_size']
            comparison['fullbatch_batch_size'] = len(X_train)

            # Save per-run comparison
            with open(out_dir / "comparison.json", 'w') as f:
                json.dump(comparison, f, indent=2,
                          default=lambda o: float(o)
                          if isinstance(o, (np.floating, np.integer)) else o)

            all_comparisons.append(comparison)

    # =================================================================
    # Overall summary
    # =================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 6.1 — OVERALL SUMMARY")
    logger.info("=" * 70)

    if not all_comparisons:
        logger.error("No comparisons completed!")
        sys.exit(1)

    # Summary table
    logger.info("")
    logger.info(
        f"  {'Fold':>5} {'Direction':>20} {'MB CI':>8} {'FB CI':>8} "
        f"{'|diff|':>8} {'Spear':>8} {'Pass':>6}"
    )
    logger.info("  " + "-" * 65)

    all_pass = True
    ci_diffs = []
    spearman_rhos = []

    for c in all_comparisons:
        status = "YES" if c['overall_acceptable'] else "NO"
        if not c['overall_acceptable']:
            all_pass = False

        ci_diffs.append(c['cindex_difference'])
        spearman_rhos.append(c['ig_spearman_rho'])

        logger.info(
            f"  {c['fold']:>5} {c['direction']:>20} "
            f"{c['minibatch_cindex']:>8.4f} {c['fullbatch_cindex']:>8.4f} "
            f"{c['cindex_difference']:>8.4f} {c['ig_spearman_rho']:>8.4f} "
            f"{status:>6}"
        )

    logger.info("  " + "-" * 65)
    logger.info(
        f"  {'':>5} {'MEAN':>20} "
        f"{'':>8} {'':>8} "
        f"{np.mean(ci_diffs):>8.4f} {np.mean(spearman_rhos):>8.4f}"
    )
    logger.info(
        f"  {'':>5} {'MAX':>20} "
        f"{'':>8} {'':>8} "
        f"{np.max(ci_diffs):>8.4f} {np.min(spearman_rhos):>8.4f}"
    )

    logger.info("")

    if all_pass:
        logger.info(
            "  CONCLUSION: Mini-batch approximation is ACCEPTABLE. "
            "All 6 runs meet both criteria."
        )
    else:
        n_fail = sum(1 for c in all_comparisons if not c['overall_acceptable'])
        logger.info(
            f"  CONCLUSION: {n_fail}/{len(all_comparisons)} runs FAILED "
            "acceptance criteria. Results diverge — report both and discuss."
        )

    # Convergence comparison
    logger.info("")
    logger.info("  Convergence (full-batch):")
    for c in all_comparisons:
        logger.info(
            f"    Fold {c['fold']} {c['direction']}: "
            f"best epoch={c['fullbatch_best_epoch']}, "
            f"total epochs={c['fullbatch_total_epochs']}, "
            f"ES C-index={c['fullbatch_best_es_cindex']:.4f}"
        )

    # Save overall summary
    summary = {
        'timestamp': timestamp,
        'seed': SEED,
        'target_folds': [f + 1 for f in TARGET_FOLDS],
        'device': device,
        'n_runs': len(all_comparisons),
        'all_acceptable': all_pass,
        'mean_cindex_difference': float(np.mean(ci_diffs)),
        'max_cindex_difference': float(np.max(ci_diffs)),
        'mean_spearman_rho': float(np.mean(spearman_rhos)),
        'min_spearman_rho': float(np.min(spearman_rhos)),
        'acceptance_criteria': {
            'cindex_threshold': 0.02,
            'spearman_threshold': 0.90,
        },
        'per_run': all_comparisons,
    }

    with open(results_base / "phase6_1_summary.json", 'w') as f:
        json.dump(
            summary, f, indent=2,
            default=lambda o: float(o)
            if isinstance(o, (np.floating, np.integer)) else o,
        )

    logger.info("")
    logger.info(f"Summary saved to: {results_base / 'phase6_1_summary.json'}")
    logger.info("Phase 6.1 complete.")
