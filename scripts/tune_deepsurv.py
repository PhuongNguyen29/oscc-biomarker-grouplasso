"""
Phase 2 — ElasticDeepSurv Hyperparameter Tuning

Tunes architecture and regularization hyperparameters for ElasticDeepSurv
using Optuna TPE with 5-fold stratified CV.

Search spaces are cohort-specific (TCGA vs ORIEN) to account for
different sample sizes and event rates. Alpha ranges were calibrated
by diagnostic trials to achieve penalty/Cox ratio in [0.1, 1.0].

References:
    - Katzman et al. (2018) BMC Med Res Methodol — DeepSurv
    - Zou & Hastie (2005) JRSS-B — Elastic net
    - Yuan & Lin (2006) JRSS-B — Group lasso
    - Bergstra et al. (2011) NeurIPS — TPE algorithm
    - Royston & Altman (2013) BMC Med Res Methodol — Stratified CV

Usage:
    cd ~/hpchome/oscc-biomarker-elasticdeepsurv
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/tune_deepsurv.py

    # Run only one cohort:
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/tune_deepsurv.py --cohort tcga
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/tune_deepsurv.py --cohort orien

Output:
    results/phase2_tuning/deepsurv_tuning_<timestamp>/
        ├── deepsurv_tuning.log
        ├── tcga_best_params.json
        ├── orien_best_params.json
        └── all_results.json
"""

import sys
import os
import argparse
import logging
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data.data_loader import load_cohort
from src.data.dataset import SurvivalDataset
from src.models.elastic_deepsurv import ElasticDeepSurv, ElasticDeepSurvTrainer
from src.utils.batch_samplers import StratifiedBatchSampler

# =============================================================================
# Setup
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose logging from sub-modules during tuning
logging.getLogger('src.models.deepsurv').setLevel(logging.WARNING)
logging.getLogger('src.models.elastic_deepsurv').setLevel(logging.WARNING)
logging.getLogger('src.utils.batch_samplers').setLevel(logging.WARNING)
logging.getLogger('src.data.dataset').setLevel(logging.WARNING)

# Suppress Optuna internal logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# Configuration
# =============================================================================

N_TRIALS = 50
N_FOLDS = 5
RANDOM_SEED = 42
DATA_DIR = "data/processed"

# Training configuration (fixed, not tuned)
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 20
SCHEDULER_PATIENCE = 10
WEIGHT_DECAY = 0.0           # Elastic net handles regularization
WEIGHT_INIT = 'kaiming_uniform'

# Cohort-specific search spaces
# Alpha ranges calibrated from diagnostic trials on 4,448 IQR-filtered genes:
#   TCGA alpha=1e-3 → P/C=0.21 (epoch 1), P/C=0.92 (epoch 30)
#   ORIEN alpha=5e-3 → P/C=0.77 (epoch 1), P/C=1.08 (epoch 30)
SEARCH_SPACES = {
    'tcga': {
        'n_layers': [1, 2],
        'hidden_size_min': 64,
        'hidden_size_max': 256,
        'dropout_min': 0.2,
        'dropout_max': 0.4,
        'batch_size_choices': [16, 24],
        'alpha_min': 2e-4,
        'alpha_max': 1e-2,
    },
    'orien': {
        'n_layers': [2, 3],
        'hidden_size_min': 128,
        'hidden_size_max': 256,
        'dropout_min': 0.3,
        'dropout_max': 0.5,
        'batch_size_choices': [32, 64],
        'alpha_min': 3e-4,
        'alpha_max': 1.5e-2,
    },
}

# Common search space (both cohorts)
L1_RATIO_CHOICES = [0.3,0.4, 0.5,0.6,0.7,0.8, 0.9]
ACTIVATION_CHOICES = ['relu', 'elu']
LR_MIN = 1e-4
LR_MAX = 1e-3


# =============================================================================
# Stratified CV folds (event status + survival time quartiles)
# =============================================================================

def create_strata(y_event, y_time):
    """
    Create stratification labels from event status × survival time quartiles.

    Reference:
        Royston & Altman (2013) BMC Medical Research Methodology
    """
    time_quartiles = pd.qcut(y_time, q=4, labels=False, duplicates='drop')
    strata = y_event.astype(int) * 4 + time_quartiles
    return strata


# =============================================================================
# Dataset helpers
# =============================================================================

class FoldDataset:
    """Lightweight dataset for a CV fold subset."""

    def __init__(self, X, y_time, y_event, gene_names):
        self.X = X.astype(np.float32)
        self.y_time = y_time.astype(np.float32)
        self.y_event = y_event.astype(np.float32)
        self.gene_names = gene_names
        self.n_features = X.shape[1]
        self.sample_ids = [f"sample_{i}" for i in range(len(X))]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'features': torch.from_numpy(self.X[idx]),
            'time': torch.tensor(self.y_time[idx], dtype=torch.float32),
            'event': torch.tensor(self.y_event[idx], dtype=torch.float32),
            'sample_id': self.sample_ids[idx],
        }


def make_loader(fold_dataset, batch_size, shuffle=True):
    """Create DataLoader with StratifiedBatchSampler."""
    sampler = StratifiedBatchSampler(
        events=fold_dataset.y_event,
        batch_size=batch_size,
        min_events_per_batch=2,
        shuffle=shuffle,
    )
    return DataLoader(
        fold_dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )


# =============================================================================
# Single fold training
# =============================================================================

def train_single_fold(
    X_train, y_time_train, y_event_train,
    X_valid, y_time_valid, y_event_valid,
    gene_names,
    params: Dict,
    device: str,
) -> Dict:
    """
    Train ElasticDeepSurv on one CV fold and return validation C-index.

    Args:
        X_train, y_time_train, y_event_train: Training data
        X_valid, y_time_valid, y_event_valid: Validation data
        gene_names: List of gene names
        params: Hyperparameter dict
        device: 'cuda' or 'cpu'

    Returns:
        dict with valid_cindex, best_epoch, train_cindex, etc.
    """
    n_features = X_train.shape[1]

    # Create datasets and loaders
    train_ds = FoldDataset(X_train, y_time_train, y_event_train, gene_names)
    valid_ds = FoldDataset(X_valid, y_time_valid, y_event_valid, gene_names)

    train_loader = make_loader(train_ds, params['batch_size'], shuffle=True)
    valid_loader = make_loader(valid_ds, params['batch_size'], shuffle=False)

    # Build architecture
    hidden_sizes = params['hidden_sizes']

    # Create model
    model = ElasticDeepSurv(
        n_features=n_features,
        hidden_sizes=hidden_sizes,
        dropout=params['dropout'],
        activation=params['activation'],
        batch_norm=params['batch_norm'],
        weight_init=WEIGHT_INIT,
        l1_ratio=params['l1_ratio'],
        alpha=params['alpha'],
    )

    # Create trainer
    trainer = ElasticDeepSurvTrainer(
        model=model,
        learning_rate=params['learning_rate'],
        weight_decay=WEIGHT_DECAY,
        scheduler_patience=SCHEDULER_PATIENCE,
        device=device,
    )

    # Training loop with early stopping
    best_valid_cindex = 0.0
    best_epoch = 0
    best_train_cindex = 0.0
    patience_counter = 0
    best_model_state = None

    for epoch in range(MAX_EPOCHS):
        # Train
        train_total, train_cox, train_penalty = trainer.train_epoch(train_loader)

        # Evaluate
        _, _, _, train_cindex = trainer.evaluate(train_loader)
        _, _, _, valid_cindex = trainer.evaluate(valid_loader)

        # LR scheduler
        trainer.scheduler.step(valid_cindex)

        # Early stopping
        if valid_cindex > best_valid_cindex:
            best_valid_cindex = valid_cindex
            best_train_cindex = train_cindex
            best_epoch = epoch + 1
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            break

    # Get sparsity from best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    sparsity_info = model.get_sparsity_info()

    return {
        'valid_cindex': best_valid_cindex,
        'train_cindex': best_train_cindex,
        'best_epoch': best_epoch,
        'total_epochs': epoch + 1,
        'sparsity': sparsity_info['sparsity_ratio'],
    }


# =============================================================================
# Optuna objective
# =============================================================================

def create_objective(
    X: np.ndarray,
    y_event: np.ndarray,
    y_time: np.ndarray,
    gene_names: List[str],
    cohort: str,
    device: str,
):
    """Create Optuna objective for ElasticDeepSurv tuning."""

    space = SEARCH_SPACES[cohort]

    # Pre-compute stratification and folds (same for all trials)
    strata = create_strata(y_event, y_time)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_indices = list(skf.split(X, strata))

    def objective(trial):
        # --- Sample hyperparameters ---

        # Architecture
        n_layers = trial.suggest_categorical('n_layers', space['n_layers'])

        hidden_sizes = []
        for i in range(n_layers):
            # Each layer size sampled independently, but constrained to
            # be <= previous layer (funnel architecture)
            if i == 0:
                size = trial.suggest_int(
                    'hidden_size_0',
                    space['hidden_size_min'],
                    space['hidden_size_max'],
                    step=32,
                )
            else:
                # Subsequent layers: size <= previous, min 32
                prev_size = hidden_sizes[-1]
                size = trial.suggest_int(
                    f'hidden_size_{i}',
                    32,
                    prev_size,
                    step=32,
                )
            hidden_sizes.append(size)

        # Regularization
        alpha = trial.suggest_float(
            'alpha', space['alpha_min'], space['alpha_max'], log=True
        )
        l1_ratio = trial.suggest_categorical('l1_ratio', L1_RATIO_CHOICES)

        # Training
        dropout = trial.suggest_float(
            'dropout', space['dropout_min'], space['dropout_max']
        )
        learning_rate = trial.suggest_float(
            'learning_rate', LR_MIN, LR_MAX, log=True
        )
        batch_size = trial.suggest_categorical(
            'batch_size', space['batch_size_choices']
        )

        # Discrete choices
        activation = trial.suggest_categorical('activation', ACTIVATION_CHOICES)
        batch_norm = trial.suggest_categorical('batch_norm', [True, False])

        params = {
            'hidden_sizes': hidden_sizes,
            'alpha': alpha,
            'l1_ratio': l1_ratio,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'activation': activation,
            'batch_norm': batch_norm,
        }

        # --- Cross-validation ---
        fold_results = []

        for fold_idx, (train_idx, valid_idx) in enumerate(fold_indices):
            try:
                result = train_single_fold(
                    X[train_idx], y_time[train_idx], y_event[train_idx],
                    X[valid_idx], y_time[valid_idx], y_event[valid_idx],
                    gene_names,
                    params,
                    device,
                )
                fold_results.append(result)
            except Exception as e:
                logger.warning(
                    f"  Trial {trial.number+1}, fold {fold_idx+1} failed: {e}"
                )
                fold_results.append({
                    'valid_cindex': 0.5,
                    'train_cindex': 0.5,
                    'best_epoch': 0,
                    'total_epochs': 0,
                    'sparsity': 0.0,
                })

        # Aggregate
        mean_valid = np.mean([f['valid_cindex'] for f in fold_results])
        std_valid = np.std([f['valid_cindex'] for f in fold_results])
        mean_train = np.mean([f['train_cindex'] for f in fold_results])
        mean_epoch = np.mean([f['best_epoch'] for f in fold_results])
        mean_sparsity = np.mean([f['sparsity'] for f in fold_results])

        # Store in trial
        trial.set_user_attr('std_cindex', std_valid)
        trial.set_user_attr('mean_train_cindex', mean_train)
        trial.set_user_attr('mean_best_epoch', mean_epoch)
        trial.set_user_attr('mean_sparsity', mean_sparsity)
        trial.set_user_attr('hidden_sizes', hidden_sizes)
        trial.set_user_attr('fold_results', fold_results)

        logger.info(
            f"  Trial {trial.number+1:>3}: "
            f"arch={hidden_sizes}, "
            f"α={alpha:.2e}, l1={l1_ratio}, "
            f"lr={learning_rate:.1e}, "
            f"do={dropout:.2f}, bs={batch_size}, "
            f"act={activation}, bn={batch_norm} → "
            f"CV={mean_valid:.4f}±{std_valid:.4f} "
            f"(train={mean_train:.4f}, ep={mean_epoch:.0f}, "
            f"spar={mean_sparsity:.1%})"
        )

        return mean_valid

    return objective


# =============================================================================
# Run tuning for one cohort
# =============================================================================

def tune_deepsurv_for_cohort(
    cohort: str,
    results_dir: Path,
    device: str,
) -> Dict:
    """Run Optuna tuning for ElasticDeepSurv on one cohort."""

    logger.info("=" * 70)
    logger.info(f"ELASTICDEEPSURV TUNING: {cohort.upper()}")
    logger.info("=" * 70)

    space = SEARCH_SPACES[cohort]
    logger.info(f"Search space:")
    logger.info(f"  n_layers: {space['n_layers']}")
    logger.info(f"  hidden_size: [{space['hidden_size_min']}, {space['hidden_size_max']}]")
    logger.info(f"  dropout: [{space['dropout_min']}, {space['dropout_max']}]")
    logger.info(f"  batch_size: {space['batch_size_choices']}")
    logger.info(f"  alpha: [{space['alpha_min']:.1e}, {space['alpha_max']:.1e}]")
    logger.info(f"  l1_ratio: {L1_RATIO_CHOICES}")
    logger.info(f"  learning_rate: [{LR_MIN:.1e}, {LR_MAX:.1e}]")
    logger.info(f"  activation: {ACTIVATION_CHOICES}")
    logger.info(f"  batch_norm: [True, False]")

    # Load data
    logger.info("Loading data...")
    expression, survival = load_cohort(cohort, DATA_DIR)

    # Prepare matrices (samples × genes)
    # Apply IQR filter
    iqr_gene_file = Path(DATA_DIR) / "iqr_filtered" / "retained_genes.txt"
    if not iqr_gene_file.exists():
        raise FileNotFoundError(
            f"IQR filtered gene list not found: {iqr_gene_file}. "
            f"Run scripts/filter_genes_iqr.py first."
        )
    with open(iqr_gene_file) as f:
        iqr_genes = [line.strip() for line in f if line.strip()]
    
    # Subset expression to IQR-filtered genes
    n_before = expression.shape[0]
    expression = expression.loc[expression.index.isin(iqr_genes)]
    logger.info(f"IQR filter: {n_before} → {expression.shape[0]} genes")

    # Prepare matrices (samples × genes)
    X = expression.values.T.astype(np.float32)
    y_event = survival['event'].values.astype(np.float32)
    y_time = survival['time'].values.astype(np.float32)
    gene_names = expression.index.tolist()
    n_features = X.shape[1]

    logger.info(f"X shape: {X.shape} (samples × genes)")
    logger.info(f"Events: {int(y_event.sum())}/{len(y_event)} ({y_event.mean():.1%})")
    logger.info(f"Device: {device}")
    logger.info("")

    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        study_name=f"deepsurv_{cohort}",
    )

    # Run optimization
    objective = create_objective(X, y_event, y_time, gene_names, cohort, device)

    start_time = datetime.now()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    elapsed = datetime.now() - start_time

    # Best result
    best = study.best_trial

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"BEST RESULT: {cohort.upper()}")
    logger.info(f"  Architecture: {best.user_attrs['hidden_sizes']}")
    logger.info(f"  Alpha: {best.params['alpha']:.6f}")
    logger.info(f"  L1 ratio: {best.params['l1_ratio']}")
    logger.info(f"  Learning rate: {best.params['learning_rate']:.6f}")
    logger.info(f"  Dropout: {best.params['dropout']:.3f}")
    logger.info(f"  Batch size: {best.params['batch_size']}")
    logger.info(f"  Activation: {best.params['activation']}")
    logger.info(f"  Batch norm: {best.params['batch_norm']}")
    logger.info(f"  Mean CV C-index: {best.value:.4f} ± {best.user_attrs['std_cindex']:.4f}")
    logger.info(f"  Mean train C-index: {best.user_attrs['mean_train_cindex']:.4f}")
    logger.info(f"  Mean best epoch: {best.user_attrs['mean_best_epoch']:.0f}")
    logger.info(f"  Mean sparsity: {best.user_attrs['mean_sparsity']:.1%}")
    logger.info(f"  Total time: {elapsed}")
    logger.info("=" * 70)

    # Compile results
    results = {
        'cohort': cohort,
        'best_params': {
            'hidden_sizes': best.user_attrs['hidden_sizes'],
            'alpha': best.params['alpha'],
            'l1_ratio': best.params['l1_ratio'],
            'learning_rate': best.params['learning_rate'],
            'dropout': best.params['dropout'],
            'batch_size': best.params['batch_size'],
            'activation': best.params['activation'],
            'batch_norm': best.params['batch_norm'],
        },
        'best_mean_cindex': best.value,
        'best_std_cindex': best.user_attrs['std_cindex'],
        'best_mean_train_cindex': best.user_attrs['mean_train_cindex'],
        'best_mean_epoch': best.user_attrs['mean_best_epoch'],
        'best_mean_sparsity': best.user_attrs['mean_sparsity'],
        'n_features': n_features,
        'n_samples': len(y_event),
        'n_events': int(y_event.sum()),
        'elapsed_seconds': elapsed.total_seconds(),
        'fold_results': best.user_attrs['fold_results'],
        'all_trial_summary': [],
    }

    # Store all trials summary
    for t in study.trials:
        trial_info = {
            'trial': t.number,
            'mean_cindex': t.value if t.value is not None else 0.0,
            'hidden_sizes': t.user_attrs.get('hidden_sizes', []),
            'alpha': t.params.get('alpha', 0),
            'l1_ratio': t.params.get('l1_ratio', 0),
            'learning_rate': t.params.get('learning_rate', 0),
            'dropout': t.params.get('dropout', 0),
            'batch_size': t.params.get('batch_size', 0),
            'activation': t.params.get('activation', ''),
            'batch_norm': t.params.get('batch_norm', False),
            'std_cindex': t.user_attrs.get('std_cindex', 0),
            'mean_train_cindex': t.user_attrs.get('mean_train_cindex', 0),
            'mean_best_epoch': t.user_attrs.get('mean_best_epoch', 0),
            'mean_sparsity': t.user_attrs.get('mean_sparsity', 0),
        }
        results['all_trial_summary'].append(trial_info)

    # Save per-cohort results
    out_file = results_dir / f"{cohort}_best_params.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else o)
    logger.info(f"Results saved to: {out_file}")

    # Print top 10 trials
    logger.info("")
    logger.info(f"Top 10 trials for {cohort.upper()}:")
    logger.info(
        f"  {'#':>3} | {'CV CI':>7} | {'Arch':>15} | {'Alpha':>9} | "
        f"{'L1':>4} | {'LR':>8} | {'DO':>5} | {'BS':>3} | "
        f"{'Act':>4} | {'BN':>5} | {'Epoch':>5}"
    )
    logger.info(f"  {'-'*3} | {'-'*7} | {'-'*15} | {'-'*9} | "
                f"{'-'*4} | {'-'*8} | {'-'*5} | {'-'*3} | "
                f"{'-'*4} | {'-'*5} | {'-'*5}")

    sorted_trials = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)
    for t in sorted_trials[:10]:
        logger.info(
            f"  {t.number+1:>3} | {t.value:>7.4f} | "
            f"{str(t.user_attrs.get('hidden_sizes', [])):>15} | "
            f"{t.params.get('alpha', 0):>9.2e} | "
            f"{t.params.get('l1_ratio', 0):>4.1f} | "
            f"{t.params.get('learning_rate', 0):>8.1e} | "
            f"{t.params.get('dropout', 0):>5.2f} | "
            f"{t.params.get('batch_size', 0):>3} | "
            f"{t.params.get('activation', ''):>4} | "
            f"{str(t.params.get('batch_norm', '')):>5} | "
            f"{t.user_attrs.get('mean_best_epoch', 0):>5.0f}"
        )

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ElasticDeepSurv Hyperparameter Tuning")
    parser.add_argument(
        '--cohort', type=str, default='both',
        choices=['tcga', 'orien', 'both'],
        help="Which cohort to tune (default: both)"
    )
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/phase2_tuning/deepsurv_tuning_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Log to file
    file_handler = logging.FileHandler(results_dir / "deepsurv_tuning.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(file_handler)

    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Device: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Configuration:")
    logger.info(f"  N_TRIALS: {N_TRIALS}")
    logger.info(f"  N_FOLDS: {N_FOLDS}")
    logger.info(f"  MAX_EPOCHS: {MAX_EPOCHS}")
    logger.info(f"  EARLY_STOPPING_PATIENCE: {EARLY_STOPPING_PATIENCE}")
    logger.info(f"  RANDOM_SEED: {RANDOM_SEED}")

    # Determine which cohorts to run
    if args.cohort == 'both':
        cohorts = ['tcga', 'orien']
    else:
        cohorts = [args.cohort]

    all_results = {}

    for cohort in cohorts:
        try:
            results = tune_deepsurv_for_cohort(cohort, results_dir, device)
            all_results[cohort] = results
        except Exception as e:
            logger.error(f"Tuning failed for {cohort}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL SUMMARY — ELASTICDEEPSURV")
    logger.info("=" * 70)

    for cohort, res in all_results.items():
        bp = res['best_params']
        logger.info(
            f"  {cohort.upper():>5}: "
            f"arch={bp['hidden_sizes']}, "
            f"α={bp['alpha']:.2e}, "
            f"l1={bp['l1_ratio']}, "
            f"lr={bp['learning_rate']:.1e}, "
            f"do={bp['dropout']:.2f}, "
            f"bs={bp['batch_size']}, "
            f"act={bp['activation']}, "
            f"bn={bp['batch_norm']}"
        )
        logger.info(
            f"         C-index={res['best_mean_cindex']:.4f} ± "
            f"{res['best_std_cindex']:.4f}, "
            f"epoch={res['best_mean_epoch']:.0f}, "
            f"time={res['elapsed_seconds']/60:.1f}min"
        )

    logger.info("=" * 70)

    # Save combined results
    combined_file = results_dir / "all_results.json"
    with open(combined_file, 'w') as f:
        json.dump(
            all_results, f, indent=2,
            default=lambda o: float(o) if isinstance(o, np.floating)
            else int(o) if isinstance(o, np.integer) else o
        )
    logger.info(f"\nAll results saved to: {combined_file}")
    logger.info("Done.")
