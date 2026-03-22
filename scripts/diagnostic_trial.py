"""
Phase 2 Diagnostic Trial — ElasticDeepSurv on v4 Preprocessed Data

Purpose:
    Verify that search space boundaries are reasonable for 16,353 genes
    before committing GPU time to full Optuna tuning (50 trials).

    Trains ONE ElasticDeepSurv model per cohort with mid-range hyperparameters
    for 30 epochs. No Optuna, no hyperparameter search — just a sanity check.

What to check from the output:
    1. Penalty-to-Cox-loss ratio at epoch 1 and epoch 30
       - Healthy: 0.1 to 1.0
       - If > 5: alpha is too high (model over-regularized)
       - If < 0.01: alpha is too low (no regularization effect)

    2. Gradient norms
       - Should not trigger explosion/clipping warnings frequently
       - Occasional clipping warnings are acceptable

    3. Training and validation C-index at epoch 30
       - Training C-index should be above 0.5 (model is learning)
       - If training ~0.5 after 30 epochs: model not learning
       - If training >0.75 but validation ~0.5: overfitting

    4. Sparsity at epoch 30
       - Expected: 10-40% of genes suppressed
       - If 0%: alpha too low
       - If >90%: alpha too high

Usage:
    cd ~/hpchome/oscc-biomarker-elasticdeepsurv
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/diagnostic_trial.py

References:
    - Katzman et al. (2018) BMC Med Res Methodol — DeepSurv architecture
    - Zou & Hastie (2005) JRSS-B — Elastic net regularization
    - Yuan & Lin (2006) JRSS-B — Group lasso
"""

import sys
import os
import logging
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Imports from project
from src.data.data_loader import load_cohort, create_survival_dataset
from src.data.dataset import SurvivalDataset
from src.models.elastic_deepsurv import ElasticDeepSurv, ElasticDeepSurvTrainer
from src.utils.batch_samplers import StratifiedBatchSampler


# =============================================================================
# Configuration
# =============================================================================

# Mid-range hyperparameters for diagnostic (not optimized)
CONFIGS = {
    'tcga': {
        'hidden_sizes': [128],        # 1 layer, moderate compression
        'dropout': 0.3,
        'batch_norm': True,
        'activation': 'relu',
        'batch_size': 24,
        'alpha': 1e-2,                # Mid-range for TCGA
        'l1_ratio': 0.5,
        'learning_rate': 3e-4,
        'n_epochs': 30,
        'early_stopping_patience': None,  # No early stopping for diagnostic
    },
    'orien': {
        'hidden_sizes': [128, 64],    # 2 layers
        'dropout': 0.4,
        'batch_norm': True,
        'activation': 'relu',
        'batch_size': 64,
        'alpha': 1e-1,                # Mid-range for ORIEN
        'l1_ratio': 0.5,
        'learning_rate': 3e-4,
        'n_epochs': 30,
        'early_stopping_patience': None,
    },
}

VALID_SIZE = 0.2       # 80/20 stratified split
RANDOM_SEED = 42
DATA_DIR = "data/processed"


# =============================================================================
# Helper functions
# =============================================================================

def create_stratified_dataloader(dataset, batch_size, shuffle=True):
    """
    Create a DataLoader with StratifiedBatchSampler.

    Ensures minimum 2 events per batch for valid Cox loss computation.
    """
    events = dataset.y_event

    sampler = StratifiedBatchSampler(
        events=events,
        batch_size=batch_size,
        min_events_per_batch=2,
        shuffle=shuffle,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,      # Avoid multiprocessing issues
        pin_memory=True,
    )
    return loader


def split_dataset(dataset, valid_size=0.2, seed=42):
    """
    Split a SurvivalDataset into train/valid by stratified event status.

    Returns two SurvivalDataset-like objects with .X, .y_time, .y_event, .sample_ids
    """
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(dataset))
    train_idx, valid_idx = train_test_split(
        indices,
        test_size=valid_size,
        stratify=dataset.y_event,
        random_state=seed,
    )

    # Create lightweight wrapper datasets
    train_ds = SubsetDataset(dataset, train_idx)
    valid_ds = SubsetDataset(dataset, valid_idx)

    logger.info(
        f"Split: train={len(train_ds)} ({train_ds.y_event.sum():.0f} events), "
        f"valid={len(valid_ds)} ({valid_ds.y_event.sum():.0f} events)"
    )
    return train_ds, valid_ds


class SubsetDataset:
    """
    Subset of a SurvivalDataset that preserves the interface needed by
    StratifiedBatchSampler and ElasticDeepSurvTrainer.
    """
    def __init__(self, parent, indices):
        self.X = parent.X[indices]
        self.y_time = parent.y_time[indices]
        self.y_event = parent.y_event[indices]
        self.sample_ids = [parent.sample_ids[i] for i in indices]
        self.gene_names = parent.gene_names
        self.n_features = parent.n_features

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'features': torch.from_numpy(self.X[idx].astype(np.float32)),
            'time': torch.tensor(self.y_time[idx], dtype=torch.float32),
            'event': torch.tensor(self.y_event[idx], dtype=torch.float32),
            'sample_id': self.sample_ids[idx],
        }


# =============================================================================
# Main diagnostic function
# =============================================================================

def run_diagnostic(cohort: str):
    """
    Run diagnostic trial for one cohort.

    Args:
        cohort: 'tcga' or 'orien'
    """
    config = CONFIGS[cohort]

    logger.info("=" * 70)
    logger.info(f"DIAGNOSTIC TRIAL: {cohort.upper()}")
    logger.info("=" * 70)
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")

    # ----- Device -----
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ----- Load data -----
    logger.info("Loading data...")
    dataset = create_survival_dataset(cohort, DATA_DIR)
    n_features = dataset.n_features
    logger.info(f"Features: {n_features}, Samples: {len(dataset)}")

    # ----- Train/valid split -----
    train_ds, valid_ds = split_dataset(dataset, VALID_SIZE, RANDOM_SEED)

    # ----- Create data loaders with stratified batching -----
    train_loader = create_stratified_dataloader(
        train_ds, config['batch_size'], shuffle=True
    )
    valid_loader = create_stratified_dataloader(
        valid_ds, config['batch_size'], shuffle=False
    )

    # ----- Create model -----
    model = ElasticDeepSurv(
        n_features=n_features,
        hidden_sizes=config['hidden_sizes'],
        dropout=config['dropout'],
        activation=config['activation'],
        batch_norm=config['batch_norm'],
        l1_ratio=config['l1_ratio'],
        alpha=config['alpha'],
    )

    n_params = sum(p.numel() for p in model.parameters())
    first_layer_params = n_features * config['hidden_sizes'][0]
    logger.info(f"Total parameters: {n_params:,}")
    logger.info(f"First layer parameters: {first_layer_params:,}")
    logger.info(
        f"Events-per-parameter ratio: "
        f"{train_ds.y_event.sum() / n_params:.6f}"
    )

    # ----- Create trainer -----
    trainer = ElasticDeepSurvTrainer(
        model=model,
        learning_rate=config['learning_rate'],
        weight_decay=0.0,   # No weight decay — elastic net handles regularization
        device=device,
    )

    # ----- Training loop (manual, to capture per-epoch diagnostics) -----
    logger.info("")
    logger.info("Starting diagnostic training...")
    logger.info("-" * 70)
    logger.info(
        f"{'Epoch':>5} | {'Total':>8} {'Cox':>8} {'Penalty':>8} "
        f"{'P/C Ratio':>9} | {'Train CI':>8} {'Valid CI':>8} | "
        f"{'Sparsity':>8}"
    )
    logger.info("-" * 70)

    results = {
        'cohort': cohort,
        'config': config,
        'n_features': n_features,
        'n_train': len(train_ds),
        'n_valid': len(valid_ds),
        'n_train_events': int(train_ds.y_event.sum()),
        'n_valid_events': int(valid_ds.y_event.sum()),
        'n_params': n_params,
        'epochs': [],
    }

    for epoch in range(config['n_epochs']):
        # Train one epoch
        train_total, train_cox, train_penalty = trainer.train_epoch(train_loader)

        # Evaluate
        _, _, _, train_cindex = trainer.evaluate(train_loader)
        _, _, _, valid_cindex = trainer.evaluate(valid_loader)

        # Sparsity
        sparsity_info = model.get_sparsity_info()
        sparsity = sparsity_info['sparsity_ratio']

        # Penalty-to-Cox ratio
        if train_cox > 0:
            penalty_cox_ratio = train_penalty / train_cox
        else:
            penalty_cox_ratio = float('inf')

        # Learning rate
        current_lr = trainer.optimizer.param_groups[0]['lr']

        # Step scheduler
        trainer.scheduler.step(valid_cindex)

        # Store epoch results
        epoch_result = {
            'epoch': epoch + 1,
            'train_total_loss': train_total,
            'train_cox_loss': train_cox,
            'train_penalty': train_penalty,
            'penalty_cox_ratio': penalty_cox_ratio,
            'train_cindex': train_cindex,
            'valid_cindex': valid_cindex,
            'sparsity': sparsity,
            'learning_rate': current_lr,
        }
        results['epochs'].append(epoch_result)

        # Log every epoch (only 30 epochs, so this is fine)
        logger.info(
            f"{epoch+1:>5} | {train_total:>8.4f} {train_cox:>8.4f} "
            f"{train_penalty:>8.4f} {penalty_cox_ratio:>9.4f} | "
            f"{train_cindex:>8.4f} {valid_cindex:>8.4f} | "
            f"{sparsity:>7.1%}"
        )

    # ----- Summary -----
    logger.info("-" * 70)
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"DIAGNOSTIC SUMMARY: {cohort.upper()}")
    logger.info("=" * 70)

    ep1 = results['epochs'][0]
    ep_last = results['epochs'][-1]
    best_valid = max(e['valid_cindex'] for e in results['epochs'])
    best_epoch = max(results['epochs'], key=lambda e: e['valid_cindex'])['epoch']

    logger.info(f"")
    logger.info(f"  Penalty/Cox ratio  — Epoch 1: {ep1['penalty_cox_ratio']:.4f}, "
                f"Epoch {config['n_epochs']}: {ep_last['penalty_cox_ratio']:.4f}")
    logger.info(f"    → Healthy range: 0.1 to 1.0")
    if ep1['penalty_cox_ratio'] > 5:
        logger.warning(f"    ⚠ Ratio > 5 at epoch 1: alpha may be too HIGH")
    elif ep1['penalty_cox_ratio'] < 0.01:
        logger.warning(f"    ⚠ Ratio < 0.01 at epoch 1: alpha may be too LOW")
    else:
        logger.info(f"    ✓ Ratio is in healthy range")

    logger.info(f"")
    logger.info(f"  Train C-index      — Epoch 1: {ep1['train_cindex']:.4f}, "
                f"Epoch {config['n_epochs']}: {ep_last['train_cindex']:.4f}")
    logger.info(f"  Valid C-index      — Epoch 1: {ep1['valid_cindex']:.4f}, "
                f"Epoch {config['n_epochs']}: {ep_last['valid_cindex']:.4f}")
    logger.info(f"  Best valid C-index — {best_valid:.4f} at epoch {best_epoch}")
    if ep_last['train_cindex'] < 0.52:
        logger.warning(f"    ⚠ Training C-index near 0.5: model may not be learning")
    elif ep_last['train_cindex'] > 0.75 and ep_last['valid_cindex'] < 0.52:
        logger.warning(f"    ⚠ Large train/valid gap: model may be overfitting")
    else:
        logger.info(f"    ✓ Model shows learning signal")

    logger.info(f"")
    logger.info(f"  Sparsity           — Epoch 1: {ep1['sparsity']:.1%}, "
                f"Epoch {config['n_epochs']}: {ep_last['sparsity']:.1%}")
    if ep_last['sparsity'] < 0.01:
        logger.warning(f"    ⚠ Near-zero sparsity: alpha may be too LOW")
    elif ep_last['sparsity'] > 0.9:
        logger.warning(f"    ⚠ Very high sparsity (>90%): alpha may be too HIGH")
    else:
        logger.info(f"    ✓ Sparsity in reasonable range")

    logger.info(f"")
    logger.info(f"  Learning rate      — Start: {config['learning_rate']}, "
                f"End: {ep_last['learning_rate']:.2e}")
    logger.info("=" * 70)

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/phase2_tuning/diagnostic_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Also log to file
    file_handler = logging.FileHandler(results_dir / "diagnostic.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(file_handler)

    logger.info(f"Results will be saved to: {results_dir}")

    # Run diagnostics for both cohorts
    all_results = {}

    for cohort in ['tcga', 'orien']:
        try:
            results = run_diagnostic(cohort)
            all_results[cohort] = results
        except Exception as e:
            logger.error(f"Diagnostic failed for {cohort}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    # Convert to serializable format
    output = {}
    for cohort, res in all_results.items():
        output[cohort] = {
            'config': res['config'],
            'n_features': res['n_features'],
            'n_train': res['n_train'],
            'n_valid': res['n_valid'],
            'n_train_events': res['n_train_events'],
            'n_valid_events': res['n_valid_events'],
            'n_params': res['n_params'],
            'epoch_1': res['epochs'][0],
            'epoch_last': res['epochs'][-1],
            'best_valid_cindex': max(e['valid_cindex'] for e in res['epochs']),
        }

    results_file = results_dir / "diagnostic_results.json"
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")
    logger.info("Done.")
