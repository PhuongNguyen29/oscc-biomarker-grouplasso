"""
Phase 3: Cross-Fitted Bidirectional Evaluation (Stage A)

Produces out-of-fold risk scores for every patient where no survival
outcomes from a held-out patient influence their prediction.

Design:
    3 seeds × 5 folds × 2 directions × 2 methods = 60 training runs

    Direction 1 (TCGA→ORIEN): Train on TCGA_train_i → predict ORIEN_fold_i
    Direction 2 (ORIEN→TCGA): Train on ORIEN_train_i → predict TCGA_fold_i

    Methods: Elastic net Cox (scikit-survival) + ElasticDeepSurv (PyTorch)

    IG attributions are NOT computed here (separate Phase 3b script).
    Model weights are saved for later IG computation.

References:
    - Stone (1974) — Cross-validation
    - Katzman et al. (2018) BMC Med Res Methodol — DeepSurv
    - Simon et al. (2011) JSS — Regularization paths for Cox
    - Royston & Altman (2013) BMC Med Res Methodol — Stratified CV

Usage:
    cd ~/hpchome/oscc-biomarker-elasticdeepsurv
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/phase3_cross_fitted.py

    # Single seed:
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/phase3_cross_fitted.py --seed 42

Output:
    results/phase3_cross_fitted/
        seed_42/
            fold_1/
                tcga_to_orien/
                    cox_risk_scores.csv
                    cox_model.pkl
                    deepsurv_risk_scores.csv
                    deepsurv_model.pt
                orien_to_tcga/
                    ...
            ...
            aggregated/
                tcga_to_orien_cox_all_scores.csv
                tcga_to_orien_deepsurv_all_scores.csv
                orien_to_tcga_cox_all_scores.csv
                orien_to_tcga_deepsurv_all_scores.csv
            summary.json
        seed_123/ ...
        seed_456/ ...
        overall_summary.json
"""

import sys
import os
import argparse
import logging
import json
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

from src.data.data_loader import load_cohort
from src.data.dataset import SurvivalDataset
from src.models.elastic_deepsurv import ElasticDeepSurv, ElasticDeepSurvTrainer
from src.utils.batch_samplers import StratifiedBatchSampler
from torch.utils.data import DataLoader

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
# Configuration — Fixed hyperparameters from Phase 2
# =============================================================================

SEEDS = [42, 123, 456]
N_FOLDS = 5
DATA_DIR = "data/processed"
RESULTS_BASE = "results/phase3_cross_fitted"

# ElasticDeepSurv training config (fixed, not tuned)
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 20
SCHEDULER_PATIENCE = 10
WEIGHT_DECAY = 0.0

# Phase 2 best hyperparameters
DEEPSURV_PARAMS = {
    'tcga': {
        'hidden_sizes': [96, 64],
        'alpha': 1.92e-03,
        'l1_ratio': 0.3,
        'learning_rate': 4.6e-04,
        'dropout': 0.29,
        'batch_size': 24,
        'activation': 'relu',
        'batch_norm': False,
    },
    'orien': {
        'hidden_sizes': [128, 96],
        'alpha': 5.60e-03,
        'l1_ratio': 0.9,
        'learning_rate': 4.6e-04,
        'dropout': 0.35,
        'batch_size': 64,
        'activation': 'relu',
        'batch_norm': True,
    },
}

COX_PARAMS = {
    'tcga': {'l1_ratio': 0.7},
    'orien': {'l1_ratio': 1.0},
}

# Cox regularization path
COX_N_ALPHAS = 100
COX_INNER_CV_FOLDS = 5  # Inner CV for lambda selection


# =============================================================================
# Helper functions
# =============================================================================

def create_strata(y_event, y_time):
    """Create event × time quartile stratification labels."""
    time_quartiles = pd.qcut(y_time, q=4, labels=False, duplicates='drop')
    return y_event.astype(int) * 4 + time_quartiles


class FoldDataset:
    """Lightweight dataset for a CV fold."""
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


def make_loader(dataset, batch_size, shuffle=True):
    """
    Create DataLoader with StratifiedBatchSampler.

    Wraps the sampler to drop any batch with only 1 sample,
    which would cause BatchNorm to fail during training.
    """
    sampler = _SafeBatchSampler(
        events=dataset.y_event,
        batch_size=batch_size,
        min_events_per_batch=2,
        shuffle=shuffle,
    )
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=True)


class _SafeBatchSampler:
    """
    Wraps StratifiedBatchSampler but filters out batches with only 1 sample.
    Regenerates batches each epoch (via __iter__) to preserve shuffling.
    """
    def __init__(self, events, batch_size, min_events_per_batch=2, shuffle=True):
        self.inner = StratifiedBatchSampler(
            events=events,
            batch_size=batch_size,
            min_events_per_batch=min_events_per_batch,
            shuffle=shuffle,
        )

    def __iter__(self):
        for batch in self.inner:
            if len(batch) > 1:
                yield batch

    def __len__(self):
        # Approximate — may be off by 1 due to filtering
        return len(self.inner)


def bootstrap_cindex(y_event, y_time, risk_scores, n_bootstrap=1000, seed=42):
    """
    Compute C-index with bootstrap 95% CI.

    Reference: Harrell et al. (1996), Efron (1979) bootstrap percentile method.

    Returns:
        dict with point_estimate, ci_lower, ci_upper, se
    """
    rng = np.random.RandomState(seed)
    n = len(y_event)

    # Point estimate
    event_bool = y_event.astype(bool)
    ci, _, _, _, _ = concordance_index_censored(event_bool, y_time, risk_scores)

    # Bootstrap
    boot_cindices = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            bci, _, _, _, _ = concordance_index_censored(
                event_bool[idx], y_time[idx], risk_scores[idx]
            )
            boot_cindices.append(bci)
        except Exception:
            continue

    boot_cindices = np.array(boot_cindices)

    return {
        'point_estimate': ci,
        'ci_lower': float(np.percentile(boot_cindices, 2.5)),
        'ci_upper': float(np.percentile(boot_cindices, 97.5)),
        'se': float(np.std(boot_cindices)),
        'n_bootstrap': len(boot_cindices),
    }


# =============================================================================
# Cox training and prediction
# =============================================================================

def train_cox_and_predict(
    X_train, y_event_train, y_time_train,
    X_test, sample_ids_test,
    l1_ratio: float,
    inner_cv_seed: int,
) -> Tuple[np.ndarray, object, float]:
    """
    Train elastic net Cox on training data, select lambda by inner CV,
    predict risk scores for test data.

    Returns:
        risk_scores: Predictions for test samples
        model: Fitted CoxnetSurvivalAnalysis
        best_alpha: Selected lambda value
    """
    # Structured array for sksurv
    y_train = np.array(
        [(bool(e), t) for e, t in zip(y_event_train, y_time_train)],
        dtype=[('event', bool), ('time', float)]
    )

    # Fit full regularization path
    model = CoxnetSurvivalAnalysis(
        l1_ratio=l1_ratio,
        n_alphas=COX_N_ALPHAS,
        normalize=False,
        tol=1e-7,
        max_iter=100000,
    )
    model.fit(X_train, y_train)

    # Inner CV to select best lambda
    # Key: pass the OUTER model's alphas to inner models so paths match exactly.
    # This ensures we evaluate the same lambda values across all inner folds.
    outer_alphas = model.alphas_

    strata_inner = create_strata(y_event_train, y_time_train)
    skf_inner = StratifiedKFold(
        n_splits=COX_INNER_CV_FOLDS, shuffle=True, random_state=inner_cv_seed
    )

    # Score each outer alpha across inner folds
    alpha_scores = {i: [] for i in range(len(outer_alphas))}

    for inner_train_idx, inner_valid_idx in skf_inner.split(X_train, strata_inner):
        X_inner_train = X_train[inner_train_idx]
        X_inner_valid = X_train[inner_valid_idx]
        y_inner_train = y_train[inner_train_idx]
        y_inner_valid = y_train[inner_valid_idx]

        # Fit inner model using the outer model's alpha path
        inner_model = CoxnetSurvivalAnalysis(
            l1_ratio=l1_ratio,
            alphas=outer_alphas,   # Force same lambda path as outer model
            normalize=False,
            tol=1e-7,
            max_iter=100000,
        )
        try:
            inner_model.fit(X_inner_train, y_inner_train)
        except Exception as e:
            logger.warning(f"    Inner CV fold fit failed: {e}")
            continue

        for i, alpha_val in enumerate(inner_model.alphas_):
            try:
                pred = inner_model.predict(X_inner_valid, alpha=alpha_val)
                ci, _, _, _, _ = concordance_index_censored(
                    y_inner_valid['event'], y_inner_valid['time'], pred
                )
                alpha_scores[i].append(ci)
            except Exception:
                continue

    # Select best alpha by mean inner CV C-index
    best_alpha = None
    best_mean_ci = 0.0
    for i, scores in alpha_scores.items():
        if len(scores) >= 3:  # Need at least 3 inner folds
            mean_ci = np.mean(scores)
            if mean_ci > best_mean_ci:
                best_mean_ci = mean_ci
                best_alpha = outer_alphas[i]

    if best_alpha is None:
        # Fallback: use the outer model's score method (evaluates at last alpha)
        best_alpha = outer_alphas[-1]
        logger.warning("Inner CV could not select alpha, using smallest lambda")

    # Predict on test data at best alpha
    risk_scores = model.predict(X_test, alpha=best_alpha)

    return risk_scores, model, float(best_alpha)


# =============================================================================
# DeepSurv training and prediction
# =============================================================================

def train_deepsurv_and_predict(
    X_train, y_event_train, y_time_train,
    sample_ids_train,
    X_test, sample_ids_test,
    gene_names,
    params: Dict,
    device: str,
    train_seed: int,
) -> Tuple[np.ndarray, ElasticDeepSurv]:
    """
    Train ElasticDeepSurv on training data with early stopping,
    predict risk scores for test data.

    Uses a 10% internal validation split from training data for
    early stopping (stratified by event).

    Returns:
        risk_scores: Predictions for test samples
        model: Trained ElasticDeepSurv model (on CPU, for saving)
    """
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(train_seed)

    n_features = X_train.shape[1]

    # Internal validation split for early stopping (10% of training data)
    from sklearn.model_selection import train_test_split
    strata_es = create_strata(y_event_train, y_time_train)
    tr_idx, es_idx = train_test_split(
        np.arange(len(X_train)),
        test_size=0.1,
        stratify=strata_es,
        random_state=train_seed,
    )

    train_ds = FoldDataset(
        X_train[tr_idx], y_time_train[tr_idx], y_event_train[tr_idx],
        [sample_ids_train[i] for i in tr_idx], gene_names
    )
    es_ds = FoldDataset(
        X_train[es_idx], y_time_train[es_idx], y_event_train[es_idx],
        [sample_ids_train[i] for i in es_idx], gene_names
    )

    train_loader = make_loader(train_ds, params['batch_size'], shuffle=True)
    es_loader = make_loader(es_ds, params['batch_size'], shuffle=False)

    # Create model
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

    # Training with early stopping
    best_cindex = 0.0
    best_epoch = 0
    patience_counter = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        trainer.train_epoch(train_loader)
        _, _, _, es_cindex = trainer.evaluate(es_loader)
        trainer.scheduler.step(es_cindex)

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

    logger.debug(f"    DeepSurv: best epoch={best_epoch}, ES C-index={best_cindex:.4f}")

    # Predict on test data
    model.eval()
    model.to(device)
    X_test_tensor = torch.from_numpy(X_test.astype(np.float32)).to(device)

    with torch.no_grad():
        log_hazards = model(X_test_tensor)
        risk_scores = torch.exp(log_hazards).squeeze().cpu().numpy()

    # Move model to CPU for saving
    model.cpu()

    return risk_scores, model


# =============================================================================
# Single fold execution
# =============================================================================

def run_fold(
    fold_idx: int,
    train_cohort: str,
    test_cohort: str,
    X_train, y_event_train, y_time_train, sample_ids_train,
    X_test, y_event_test, y_time_test, sample_ids_test,
    gene_names,
    fold_dir: Path,
    device: str,
    seed: int,
) -> Dict:
    """Run both methods for one fold, one direction."""

    direction = f"{train_cohort}_to_{test_cohort}"
    out_dir = fold_dir / direction
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'direction': direction,
        'train_cohort': train_cohort,
        'test_cohort': test_cohort,
        'n_train': len(X_train),
        'n_train_events': int(y_event_train.sum()),
        'n_test': len(X_test),
        'n_test_events': int(y_event_test.sum()),
    }

    # --- Cox ---
    logger.info(f"      Cox {direction}...")
    cox_l1_ratio = COX_PARAMS[train_cohort]['l1_ratio']

    cox_risks, cox_model, cox_alpha = train_cox_and_predict(
        X_train, y_event_train, y_time_train,
        X_test, sample_ids_test,
        l1_ratio=cox_l1_ratio,
        inner_cv_seed=seed,
    )

    # Save Cox results
    cox_df = pd.DataFrame({
        'sampleID': sample_ids_test,
        'risk_score': cox_risks,
        'time': y_time_test,
        'event': y_event_test.astype(int),
    })
    cox_df.to_csv(out_dir / "cox_risk_scores.csv", index=False)

    with open(out_dir / "cox_model.pkl", 'wb') as f:
        pickle.dump({'model': cox_model, 'best_alpha': cox_alpha}, f)

    # Cox C-index for this fold
    try:
        cox_ci, _, _, _, _ = concordance_index_censored(
            y_event_test.astype(bool), y_time_test, cox_risks
        )
    except Exception:
        cox_ci = 0.5

    results['cox_cindex'] = cox_ci
    results['cox_best_alpha'] = cox_alpha

    # Count nonzero genes
    alpha_idx = np.argmin(np.abs(cox_model.alphas_ - cox_alpha))
    cox_n_nonzero = int(np.sum(cox_model.coef_[:, alpha_idx] != 0))
    results['cox_n_nonzero'] = cox_n_nonzero

    # --- DeepSurv ---
    logger.info(f"      DeepSurv {direction}...")
    ds_params = DEEPSURV_PARAMS[train_cohort]

    ds_risks, ds_model = train_deepsurv_and_predict(
        X_train, y_event_train, y_time_train,
        sample_ids_train,
        X_test, sample_ids_test,
        gene_names,
        ds_params,
        device,
        train_seed=seed + fold_idx,
    )

    # Save DeepSurv results
    ds_df = pd.DataFrame({
        'sampleID': sample_ids_test,
        'risk_score': ds_risks,
        'time': y_time_test,
        'event': y_event_test.astype(int),
    })
    ds_df.to_csv(out_dir / "deepsurv_risk_scores.csv", index=False)

    torch.save(ds_model.state_dict(), out_dir / "deepsurv_model.pt")

    # DeepSurv C-index for this fold
    try:
        ds_ci, _, _, _, _ = concordance_index_censored(
            y_event_test.astype(bool), y_time_test, ds_risks
        )
    except Exception:
        ds_ci = 0.5

    results['deepsurv_cindex'] = ds_ci

    logger.info(
        f"      → Cox CI={cox_ci:.4f} ({cox_n_nonzero} genes), "
        f"DeepSurv CI={ds_ci:.4f}"
    )

    return results


# =============================================================================
# Run all folds for one seed
# =============================================================================

def run_seed(
    seed: int,
    tcga_X, tcga_event, tcga_time, tcga_ids,
    orien_X, orien_event, orien_time, orien_ids,
    gene_names,
    results_base: Path,
    device: str,
) -> Dict:
    """Run all 5 folds × 2 directions for one seed."""

    seed_dir = results_base / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"")
    logger.info(f"{'='*70}")
    logger.info(f"SEED: {seed}")
    logger.info(f"{'='*70}")

    # Create stratified folds for each cohort
    tcga_strata = create_strata(tcga_event, tcga_time)
    orien_strata = create_strata(orien_event, orien_time)

    tcga_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    orien_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    tcga_folds = list(tcga_skf.split(tcga_X, tcga_strata))
    orien_folds = list(orien_skf.split(orien_X, orien_strata))

    all_fold_results = []

    for fold_idx in range(N_FOLDS):
        logger.info(f"")
        logger.info(f"  --- Fold {fold_idx+1}/{N_FOLDS} ---")

        fold_dir = seed_dir / f"fold_{fold_idx+1}"

        # TCGA fold split
        tcga_train_idx, tcga_test_idx = tcga_folds[fold_idx]
        # ORIEN fold split
        orien_train_idx, orien_test_idx = orien_folds[fold_idx]

        # --- Direction 1: TCGA → ORIEN ---
        # Train on TCGA_train, predict ORIEN_fold
        d1_results = run_fold(
            fold_idx=fold_idx,
            train_cohort='tcga',
            test_cohort='orien',
            X_train=tcga_X[tcga_train_idx],
            y_event_train=tcga_event[tcga_train_idx],
            y_time_train=tcga_time[tcga_train_idx],
            sample_ids_train=[tcga_ids[i] for i in tcga_train_idx],
            X_test=orien_X[orien_test_idx],
            y_event_test=orien_event[orien_test_idx],
            y_time_test=orien_time[orien_test_idx],
            sample_ids_test=[orien_ids[i] for i in orien_test_idx],
            gene_names=gene_names,
            fold_dir=fold_dir,
            device=device,
            seed=seed,
        )
        all_fold_results.append(d1_results)

        # --- Direction 2: ORIEN → TCGA ---
        # Train on ORIEN_train, predict TCGA_fold
        d2_results = run_fold(
            fold_idx=fold_idx,
            train_cohort='orien',
            test_cohort='tcga',
            X_train=orien_X[orien_train_idx],
            y_event_train=orien_event[orien_train_idx],
            y_time_train=orien_time[orien_train_idx],
            sample_ids_train=[orien_ids[i] for i in orien_train_idx],
            X_test=tcga_X[tcga_test_idx],
            y_event_test=tcga_event[tcga_test_idx],
            y_time_test=tcga_time[tcga_test_idx],
            sample_ids_test=[tcga_ids[i] for i in tcga_test_idx],
            gene_names=gene_names,
            fold_dir=fold_dir,
            device=device,
            seed=seed,
        )
        all_fold_results.append(d2_results)

    # --- Aggregate out-of-fold scores ---
    logger.info(f"")
    logger.info(f"  Aggregating out-of-fold scores...")

    agg_dir = seed_dir / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-direction scores
    directions = ['tcga_to_orien', 'orien_to_tcga']
    methods = ['cox', 'deepsurv']

    cindex_results = {}

    for direction in directions:
        for method in methods:
            all_scores = []

            for fold_idx in range(N_FOLDS):
                fold_dir = seed_dir / f"fold_{fold_idx+1}" / direction
                scores_file = fold_dir / f"{method}_risk_scores.csv"
                fold_scores = pd.read_csv(scores_file)
                all_scores.append(fold_scores)

            # Concatenate all fold scores
            combined = pd.concat(all_scores, ignore_index=True)
            combined.to_csv(
                agg_dir / f"{direction}_{method}_all_scores.csv", index=False
            )

            # Compute C-index with bootstrap CI
            ci_result = bootstrap_cindex(
                combined['event'].values,
                combined['time'].values,
                combined['risk_score'].values,
                n_bootstrap=1000,
                seed=seed,
            )

            key = f"{direction}_{method}"
            cindex_results[key] = ci_result

            logger.info(
                f"    {key}: C-index = {ci_result['point_estimate']:.4f} "
                f"(95% CI: {ci_result['ci_lower']:.4f} - {ci_result['ci_upper']:.4f})"
            )

    # --- Summary ---
    summary = {
        'seed': seed,
        'n_folds': N_FOLDS,
        'tcga_n': len(tcga_ids),
        'tcga_events': int(tcga_event.sum()),
        'orien_n': len(orien_ids),
        'orien_events': int(orien_event.sum()),
        'cindex_results': cindex_results,
        'per_fold_results': all_fold_results,
    }

    with open(seed_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o)

    return summary


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Cross-Fitted Evaluation")
    parser.add_argument(
        '--seed', type=int, default=None,
        help="Run single seed (42, 123, or 456). Default: run all 3."
    )
    args = parser.parse_args()

    # Determine seeds
    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = SEEDS

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Results directory
    results_base = Path(RESULTS_BASE)
    results_base.mkdir(parents=True, exist_ok=True)

    # Log to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        results_base / f"phase3_{timestamp}.log"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(file_handler)

    logger.info(f"Phase 3: Cross-Fitted Bidirectional Evaluation")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Device: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Results: {results_base}")
    logger.info(f"")

    # Load both cohorts
    logger.info("Loading data...")
    tcga_expr, tcga_surv = load_cohort('tcga', DATA_DIR)
    orien_expr, orien_surv = load_cohort('orien', DATA_DIR)

    # Prepare arrays (samples × genes)
    tcga_X = tcga_expr.values.T.astype(np.float32)
    tcga_event = tcga_surv['event'].values.astype(np.float32)
    tcga_time = tcga_surv['time'].values.astype(np.float32)
    tcga_ids = tcga_surv.index.tolist()

    orien_X = orien_expr.values.T.astype(np.float32)
    orien_event = orien_surv['event'].values.astype(np.float32)
    orien_time = orien_surv['time'].values.astype(np.float32)
    orien_ids = orien_surv.index.tolist()

    gene_names = tcga_expr.index.tolist()

    logger.info(f"TCGA: {tcga_X.shape[0]} samples × {tcga_X.shape[1]} genes, "
                f"{int(tcga_event.sum())} events")
    logger.info(f"ORIEN: {orien_X.shape[0]} samples × {orien_X.shape[1]} genes, "
                f"{int(orien_event.sum())} events")

    # Run all seeds
    all_summaries = {}

    for seed in seeds:
        try:
            summary = run_seed(
                seed=seed,
                tcga_X=tcga_X, tcga_event=tcga_event,
                tcga_time=tcga_time, tcga_ids=tcga_ids,
                orien_X=orien_X, orien_event=orien_event,
                orien_time=orien_time, orien_ids=orien_ids,
                gene_names=gene_names,
                results_base=results_base,
                device=device,
            )
            all_summaries[seed] = summary
        except Exception as e:
            logger.error(f"Seed {seed} failed: {e}")
            import traceback
            traceback.print_exc()

    # --- Overall summary across seeds ---
    logger.info(f"")
    logger.info(f"{'='*70}")
    logger.info(f"OVERALL SUMMARY — PHASE 3")
    logger.info(f"{'='*70}")

    directions = ['tcga_to_orien', 'orien_to_tcga']
    methods = ['cox', 'deepsurv']

    overall_results = {}

    for direction in directions:
        for method in methods:
            key = f"{direction}_{method}"
            seed_cindices = []

            for seed, summary in all_summaries.items():
                ci = summary['cindex_results'][key]['point_estimate']
                seed_cindices.append(ci)

            mean_ci = np.mean(seed_cindices)
            std_ci = np.std(seed_cindices)
            spread = max(seed_cindices) - min(seed_cindices)

            overall_results[key] = {
                'mean_cindex': mean_ci,
                'std_across_seeds': std_ci,
                'spread': spread,
                'per_seed': {str(s): c for s, c in zip(all_summaries.keys(), seed_cindices)},
            }

            stability = "STABLE" if spread < 0.03 else "UNSTABLE"
            logger.info(
                f"  {key:>30}: {mean_ci:.4f} ± {std_ci:.4f} "
                f"(spread={spread:.4f}, {stability})"
            )

    logger.info(f"")

    # Method comparison
    logger.info(f"  Method Comparison:")
    for direction in directions:
        cox_ci = overall_results[f"{direction}_cox"]['mean_cindex']
        ds_ci = overall_results[f"{direction}_deepsurv"]['mean_cindex']
        diff = ds_ci - cox_ci
        logger.info(
            f"    {direction}: Cox={cox_ci:.4f}, DeepSurv={ds_ci:.4f}, "
            f"Δ={diff:+.4f}"
        )

    logger.info(f"{'='*70}")

    # Save overall summary
    overall_file = results_base / "overall_summary.json"
    with open(overall_file, 'w') as f:
        json.dump(
            overall_results, f, indent=2,
            default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o
        )

    logger.info(f"\nOverall summary saved to: {overall_file}")
    logger.info("Phase 3 complete.")
