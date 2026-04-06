#!/usr/bin/env python3
"""
Sensitivity Sweep: Retune + Validate Across All (mult, pi) Combinations
=========================================================================

For each (alpha_multiplier, pi_threshold) combination from Phase 4:
  Stage 1: Retune Cox l1_ratio and DeepSurv hyperparameters on that gene set
  Stage 2: Phase 7 bidirectional validation with retuned hyperparameters
  Stage 3: Phase 8 external validation on GSE41613 and GSE42743

This eliminates the concern that mult=0.6x/pi>=0.60 was favored by
hyperparameters tuned on a different feature space (4,448 genes).

Design decisions:
  - Cox retuning: l1_ratio search via Optuna (50 trials), alpha selected
    by inner 5-fold CV within Phase 7 (same as original pipeline)
  - DeepSurv retuning: full hyperparameter search via Optuna (100 trials),
    search spaces adapted for small gene sets (10-74 features)
  - Phase 7: bidirectional validation (train TCGA→test ORIEN and vice versa)
  - Phase 8: 100-seed repeated 3-fold ridge Cox on GSE41613 and GSE42743

References:
    - Meinshausen & Bühlmann (2010) JRSS-B — Stability selection
    - Katzman et al. (2018) BMC Med Res Methodol — DeepSurv
    - Bergstra et al. (2011) NeurIPS — TPE (Optuna)
    - Verweij & Van Houwelingen (1994) Stat Med — CV partial likelihood

Usage:
    cd ~/hpchome/oscc-biomarker-grouplasso
    PYTHONPATH="${PWD}:${PYTHONPATH}" nohup python scripts/sweep_sensitivity.py \
        > results/sensitivity_sweep/sweep_nohup.log 2>&1 &

Output:
    results/sensitivity_sweep/
        ├── mult0.3_pi60/
        │   ├── tuning/ (cox + deepsurv best params per cohort)
        │   ├── validation/ (phase7: models, IG, C-indices)
        │   └── external/ (phase8: GEO C-indices)
        ├── ...
        ├── mult0.8_pi60/
        └── sweep_comparison.csv

Author: Phuong Nguyen
Date: April 2026
"""

import sys
import os
import argparse
import logging
import json
import pickle
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from captum.attr import IntegratedGradients

from src.data.data_loader import load_cohort
from src.models.elastic_deepsurv import ElasticDeepSurv, ElasticDeepSurvTrainer
from src.utils.batch_samplers import StratifiedBatchSampler
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress verbose sub-module logging
logging.getLogger("src.models.deepsurv").setLevel(logging.ERROR)
logging.getLogger("src.models.elastic_deepsurv").setLevel(logging.ERROR)
logging.getLogger("src.utils.batch_samplers").setLevel(logging.ERROR)
logging.getLogger("src.data.dataset").setLevel(logging.ERROR)
logging.getLogger("src.data.data_loader").setLevel(logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# Global Configuration
# =============================================================================

SEED = 42
DATA_DIR = "data/processed"
EXTERNAL_DATA_DIR = Path("data/external")
ANNOTATION_PATH = Path("data/raw/protein_coding_genes_annotation.csv")
PHASE4_DIR = "results/phase4_stability_randomized_lasso"

SWEEP_OUTPUT_DIR = Path("results/sensitivity_sweep")

# All (mult, pi) combinations to evaluate
SWEEP_CONFIGS = [
    (0.3, 0.60), (0.3, 0.70), (0.3, 0.80),
    (0.4, 0.60), (0.4, 0.70),
    (0.5, 0.60), (0.5, 0.70),
    (0.6, 0.60), (0.6, 0.70),
    (0.7, 0.60),
    (0.8, 0.60),
]

# --- Cox Tuning ---
COX_N_TRIALS = 50
COX_L1_RATIO_CHOICES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
COX_N_ALPHAS = 100

# --- DeepSurv Tuning ---
DS_N_TRIALS = 100
DS_N_FOLDS = 5
DS_MAX_EPOCHS = 100
DS_EARLY_STOPPING = 20
DS_SCHEDULER_PATIENCE = 10
DS_WEIGHT_DECAY = 0.0
DS_WEIGHT_INIT = "kaiming_uniform"
DS_L1_RATIO_CHOICES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DS_ACTIVATION_CHOICES = ["relu", "elu"]
DS_LR_MIN = 1e-4
DS_LR_MAX = 1e-3

# --- Phase 7 ---
P7_COX_INNER_CV_FOLDS = 5
P7_MAX_EPOCHS = 100
P7_EARLY_STOPPING = 20
P7_SCHEDULER_PATIENCE = 10
P7_IG_N_STEPS = 300
P7_IG_BATCH_SIZE = 32

# --- Phase 8 ---
P8_SEEDS = list(range(42, 142))  # 100 seeds for repeated CV
P8_N_FOLDS = 3
P8_ALPHAS_GENES = np.logspace(-1, 4, 50)
P8_ALPHAS_CLINICAL = np.logspace(-4, 4, 50)


# =============================================================================
# Adapted DeepSurv Search Spaces for Small Gene Sets
# =============================================================================

def get_deepsurv_search_space(n_features: int, cohort: str) -> Dict:
    """
    Adapt DeepSurv search space based on number of input features.

    With 10-74 features (vs 4,448 in Phase 2), we need:
      - Smaller architectures to avoid overparameterization
      - Wider alpha range (regularization calibration differs)
      - Adjusted hidden size ranges

    The hidden_size_max is capped at 4× n_features to prevent
    extreme overparameterization (rule of thumb for shallow networks).
    """
    # Cap hidden size at max(4× features, 64) to avoid trivial networks
    # but prevent extreme overparameterization
    hidden_max = min(max(4 * n_features, 64), 256)
    hidden_min = 32

    if cohort == "tcga":
        return {
            "n_layers": [1, 2],
            "hidden_size_min": hidden_min,
            "hidden_size_max": hidden_max,
            "dropout_min": 0.2,
            "dropout_max": 0.5,
            "batch_size_choices": [16, 24],
            "alpha_min": 1e-4,
            "alpha_max": 5e-2,
        }
    else:  # orien
        return {
            "n_layers": [1, 2, 3],
            "hidden_size_min": hidden_min,
            "hidden_size_max": hidden_max,
            "dropout_min": 0.2,
            "dropout_max": 0.5,
            "batch_size_choices": [32, 64],
            "alpha_min": 1e-4,
            "alpha_max": 5e-2,
        }


# =============================================================================
# Shared Utilities
# =============================================================================

def create_strata(y_event, y_time):
    """Create event × time quartile stratification labels."""
    time_quartiles = pd.qcut(y_time, q=4, labels=False, duplicates="drop")
    return y_event.astype(int) * 4 + time_quartiles


class FoldDataset:
    """Lightweight dataset for DataLoader."""
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
            "features": torch.from_numpy(self.X[idx]),
            "time": torch.tensor(self.y_time[idx], dtype=torch.float32),
            "event": torch.tensor(self.y_event[idx], dtype=torch.float32),
            "sample_id": self.sample_ids[idx],
        }


class _SafeBatchSampler:
    def __init__(self, events, batch_size, min_events_per_batch=2, shuffle=True):
        self.inner = StratifiedBatchSampler(
            events=events, batch_size=batch_size,
            min_events_per_batch=min_events_per_batch, shuffle=shuffle,
        )
    def __iter__(self):
        for batch in self.inner:
            if len(batch) > 1:
                yield batch
    def __len__(self):
        return len(self.inner)


def make_loader(dataset, batch_size, shuffle=True):
    sampler = _SafeBatchSampler(
        events=dataset.y_event, batch_size=batch_size,
        min_events_per_batch=2, shuffle=shuffle,
    )
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=True)


def subset_expression(X_full, all_gene_names, selected_genes):
    """Subset expression matrix to selected genes, preserving order."""
    gene_indices = [all_gene_names.index(g) for g in selected_genes]
    return X_full[:, gene_indices]


def load_phase4_genes(phase4_dir, alpha_mult, pi_thresh):
    """Load per-cohort stable gene lists from Phase 4."""
    phase4_path = Path(phase4_dir)
    consensus_dir = phase4_path / "consensus"
    pi_label = int(pi_thresh * 100)

    stable_genes = {}
    for cohort in ["tcga", "orien"]:
        fname = consensus_dir / f"{cohort}_stable_mult{alpha_mult}_pi{pi_label:02d}.csv"
        if not fname.exists():
            raise FileNotFoundError(f"Phase 4 stable gene file not found: {fname}")
        df = pd.read_csv(fname)
        stable_genes[cohort] = df["gene"].tolist()

    union_genes = sorted(set(stable_genes["tcga"]) | set(stable_genes["orien"]))
    overlap_genes = sorted(set(stable_genes["tcga"]) & set(stable_genes["orien"]))

    return {
        "tcga": stable_genes["tcga"],
        "orien": stable_genes["orien"],
        "union": union_genes,
        "overlap": overlap_genes,
    }


# =============================================================================
# STAGE 1: Cox Tuning
# =============================================================================

def evaluate_cox_cv(X, y_event, y_time, l1_ratio, n_folds=5, seed=SEED):
    """Evaluate elastic net Cox with a given l1_ratio using 5-fold CV."""
    y_struct = np.array(
        [(bool(e), t) for e, t in zip(y_event, y_time)],
        dtype=[("event", bool), ("time", float)],
    )
    strata = create_strata(y_event, y_time)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_results = []
    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, strata)):
        try:
            model = CoxnetSurvivalAnalysis(
                l1_ratio=l1_ratio, n_alphas=COX_N_ALPHAS,
                normalize=False, tol=1e-7, max_iter=100000,
            )
            model.fit(X[train_idx], y_struct[train_idx])
        except Exception:
            fold_results.append({"best_cindex": 0.5, "best_alpha": None, "n_nonzero": 0})
            continue

        best_cindex, best_alpha, best_nz = 0.0, None, 0
        for alpha_val in model.alphas_:
            try:
                pred = model.predict(X[valid_idx], alpha=alpha_val)
                ci, _, _, _, _ = concordance_index_censored(
                    y_struct[valid_idx]["event"], y_struct[valid_idx]["time"], pred
                )
                if ci > best_cindex:
                    best_cindex = ci
                    best_alpha = alpha_val
                    coef_idx = np.where(model.alphas_ == alpha_val)[0][0]
                    best_nz = int(np.sum(model.coef_[:, coef_idx] != 0))
            except Exception:
                continue

        fold_results.append({
            "best_cindex": best_cindex,
            "best_alpha": float(best_alpha) if best_alpha else None,
            "n_nonzero": best_nz,
        })

    cindices = [f["best_cindex"] for f in fold_results]
    return {
        "mean_cindex": np.mean(cindices),
        "std_cindex": np.std(cindices),
        "mean_nonzero": np.mean([f["n_nonzero"] for f in fold_results]),
        "fold_results": fold_results,
    }


def tune_cox(X, y_event, y_time, cohort_label, out_dir):
    """Tune Cox l1_ratio via Optuna."""
    logger.info(f"    Tuning Cox for {cohort_label} ({X.shape[1]} genes)...")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    def objective(trial):
        l1_ratio = trial.suggest_categorical("l1_ratio", COX_L1_RATIO_CHOICES)
        result = evaluate_cox_cv(X, y_event, y_time, l1_ratio)
        trial.set_user_attr("std_cindex", result["std_cindex"])
        trial.set_user_attr("mean_nonzero", result["mean_nonzero"])
        trial.set_user_attr("fold_results", result["fold_results"])
        return result["mean_cindex"]

    study.optimize(objective, n_trials=COX_N_TRIALS, show_progress_bar=False)

    best = study.best_trial
    best_l1 = best.params["l1_ratio"]

    # Get median alpha from best trial's folds
    fold_alphas = [
        f["best_alpha"] for f in best.user_attrs["fold_results"]
        if f["best_alpha"] is not None
    ]
    recommended_alpha = float(np.median(fold_alphas)) if fold_alphas else None

    results = {
        "best_l1_ratio": best_l1,
        "best_mean_cindex": best.value,
        "best_std_cindex": best.user_attrs["std_cindex"],
        "recommended_alpha": recommended_alpha,
        "n_features": X.shape[1],
    }

    out_file = out_dir / f"cox_{cohort_label}_best.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"      Best l1_ratio={best_l1}, CV C-index={best.value:.4f}")
    return results


# =============================================================================
# STAGE 1: DeepSurv Tuning
# =============================================================================

def train_single_fold_ds(
    X_train, y_time_train, y_event_train,
    X_valid, y_time_valid, y_event_valid,
    gene_names, params, device,
):
    """Train ElasticDeepSurv on one CV fold."""
    n_features = X_train.shape[1]
    train_ds = FoldDataset(X_train, y_time_train, y_event_train,
                           [f"s{i}" for i in range(len(X_train))], gene_names)
    valid_ds = FoldDataset(X_valid, y_time_valid, y_event_valid,
                           [f"s{i}" for i in range(len(X_valid))], gene_names)

    train_loader = make_loader(train_ds, params["batch_size"], shuffle=True)
    valid_loader = make_loader(valid_ds, params["batch_size"], shuffle=False)

    model = ElasticDeepSurv(
        n_features=n_features,
        hidden_sizes=params["hidden_sizes"],
        dropout=params["dropout"],
        activation=params["activation"],
        batch_norm=params["batch_norm"],
        weight_init=DS_WEIGHT_INIT,
        l1_ratio=params["l1_ratio"],
        alpha=params["alpha"],
    )

    trainer = ElasticDeepSurvTrainer(
        model=model,
        learning_rate=params["learning_rate"],
        weight_decay=DS_WEIGHT_DECAY,
        scheduler_patience=DS_SCHEDULER_PATIENCE,
        device=device,
    )

    best_valid_cindex, best_epoch, best_train_cindex = 0.0, 0, 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(DS_MAX_EPOCHS):
        trainer.train_epoch(train_loader)
        _, _, _, train_ci = trainer.evaluate(train_loader)
        _, _, _, valid_ci = trainer.evaluate(valid_loader)
        trainer.scheduler.step(valid_ci)

        if valid_ci > best_valid_cindex:
            best_valid_cindex = valid_ci
            best_train_cindex = train_ci
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= DS_EARLY_STOPPING:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    sparsity = model.get_sparsity_info()["sparsity_ratio"]

    return {
        "valid_cindex": best_valid_cindex,
        "train_cindex": best_train_cindex,
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "sparsity": sparsity,
    }


def tune_deepsurv(X, y_event, y_time, gene_names, cohort, device, out_dir):
    """Tune DeepSurv hyperparameters via Optuna with adapted search spaces."""
    n_features = X.shape[1]
    space = get_deepsurv_search_space(n_features, cohort)

    logger.info(f"    Tuning DeepSurv for {cohort.upper()} "
                f"({n_features} genes, {DS_N_TRIALS} trials)...")
    logger.info(f"      Search space: layers={space['n_layers']}, "
                f"hidden=[{space['hidden_size_min']},{space['hidden_size_max']}], "
                f"alpha=[{space['alpha_min']:.1e},{space['alpha_max']:.1e}]")

    strata = create_strata(y_event, y_time)
    skf = StratifiedKFold(n_splits=DS_N_FOLDS, shuffle=True, random_state=SEED)
    fold_indices = list(skf.split(X, strata))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    def objective(trial):
        n_layers = trial.suggest_categorical("n_layers", space["n_layers"])
        hidden_sizes = []
        for i in range(n_layers):
            if i == 0:
                size = trial.suggest_int(
                    "hidden_size_0",
                    space["hidden_size_min"], space["hidden_size_max"],
                    step=32,
                )
            else:
                prev = hidden_sizes[-1]
                size = trial.suggest_int(f"hidden_size_{i}", 32, prev, step=32)
            hidden_sizes.append(size)

        alpha = trial.suggest_float("alpha", space["alpha_min"], space["alpha_max"], log=True)
        l1_ratio = trial.suggest_categorical("l1_ratio", DS_L1_RATIO_CHOICES)
        dropout = trial.suggest_float("dropout", space["dropout_min"], space["dropout_max"])
        lr = trial.suggest_float("learning_rate", DS_LR_MIN, DS_LR_MAX, log=True)
        batch_size = trial.suggest_categorical("batch_size", space["batch_size_choices"])
        activation = trial.suggest_categorical("activation", DS_ACTIVATION_CHOICES)
        batch_norm = trial.suggest_categorical("batch_norm", [True, False])

        params = {
            "hidden_sizes": hidden_sizes, "alpha": alpha, "l1_ratio": l1_ratio,
            "dropout": dropout, "learning_rate": lr, "batch_size": batch_size,
            "activation": activation, "batch_norm": batch_norm,
        }

        fold_results = []
        for fold_idx, (tr_idx, va_idx) in enumerate(fold_indices):
            try:
                result = train_single_fold_ds(
                    X[tr_idx], y_time[tr_idx], y_event[tr_idx],
                    X[va_idx], y_time[va_idx], y_event[va_idx],
                    gene_names, params, device,
                )
                fold_results.append(result)
            except Exception:
                fold_results.append({
                    "valid_cindex": 0.5, "train_cindex": 0.5,
                    "best_epoch": 0, "total_epochs": 0, "sparsity": 0.0,
                })

        mean_valid = np.mean([f["valid_cindex"] for f in fold_results])
        trial.set_user_attr("std_cindex", np.std([f["valid_cindex"] for f in fold_results]))
        trial.set_user_attr("mean_train_cindex", np.mean([f["train_cindex"] for f in fold_results]))
        trial.set_user_attr("mean_best_epoch", np.mean([f["best_epoch"] for f in fold_results]))
        trial.set_user_attr("mean_sparsity", np.mean([f["sparsity"] for f in fold_results]))
        trial.set_user_attr("hidden_sizes", hidden_sizes)
        trial.set_user_attr("fold_results", fold_results)

        if (trial.number + 1) % 20 == 0:
            logger.info(f"        Trial {trial.number+1}/{DS_N_TRIALS}: "
                        f"arch={hidden_sizes}, CV={mean_valid:.4f}")

        return mean_valid

    start = time.time()
    study.optimize(objective, n_trials=DS_N_TRIALS, show_progress_bar=False)
    elapsed = time.time() - start

    best = study.best_trial
    results = {
        "cohort": cohort,
        "best_params": {
            "hidden_sizes": best.user_attrs["hidden_sizes"],
            "alpha": best.params["alpha"],
            "l1_ratio": best.params["l1_ratio"],
            "learning_rate": best.params["learning_rate"],
            "dropout": best.params["dropout"],
            "batch_size": best.params["batch_size"],
            "activation": best.params["activation"],
            "batch_norm": best.params["batch_norm"],
        },
        "best_mean_cindex": best.value,
        "best_std_cindex": best.user_attrs["std_cindex"],
        "best_mean_train_cindex": best.user_attrs["mean_train_cindex"],
        "best_mean_epoch": best.user_attrs["mean_best_epoch"],
        "n_features": n_features,
        "n_samples": len(y_event),
        "n_events": int(y_event.sum()),
        "elapsed_seconds": elapsed,
    }

    out_file = out_dir / f"deepsurv_{cohort}_best.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else o)

    bp = results["best_params"]
    logger.info(f"      Best: arch={bp['hidden_sizes']}, α={bp['alpha']:.2e}, "
                f"l1={bp['l1_ratio']}, CV={best.value:.4f} ({elapsed:.0f}s)")

    return results


# =============================================================================
# STAGE 2: Phase 7 — Bidirectional Validation
# =============================================================================

def phase7_cox(
    X_train, y_event_train, y_time_train, train_ids,
    X_test, y_event_test, y_time_test, test_ids,
    l1_ratio, gene_names, direction_label, out_dir,
):
    """Train Cox and evaluate bidirectionally (from phase7 script)."""
    y_train = np.array(
        [(bool(e), t) for e, t in zip(y_event_train, y_time_train)],
        dtype=[("event", bool), ("time", float)],
    )
    y_test = np.array(
        [(bool(e), t) for e, t in zip(y_event_test, y_time_test)],
        dtype=[("event", bool), ("time", float)],
    )

    model = CoxnetSurvivalAnalysis(
        l1_ratio=l1_ratio, n_alphas=COX_N_ALPHAS,
        normalize=False, tol=1e-7, max_iter=100000,
    )
    model.fit(X_train, y_train)
    outer_alphas = model.alphas_

    # Inner CV to select best alpha
    strata = create_strata(y_event_train, y_time_train)
    skf = StratifiedKFold(n_splits=P7_COX_INNER_CV_FOLDS, shuffle=True, random_state=SEED)
    alpha_scores = {i: [] for i in range(len(outer_alphas))}

    for tr_idx, va_idx in skf.split(X_train, strata):
        try:
            inner = CoxnetSurvivalAnalysis(
                l1_ratio=l1_ratio, alphas=outer_alphas,
                normalize=False, tol=1e-7, max_iter=100000,
            )
            inner.fit(X_train[tr_idx], y_train[tr_idx])
            for i, a in enumerate(inner.alphas_):
                try:
                    pred = inner.predict(X_train[va_idx], alpha=a)
                    ci, _, _, _, _ = concordance_index_censored(
                        y_train[va_idx]["event"], y_train[va_idx]["time"], pred
                    )
                    alpha_scores[i].append(ci)
                except Exception:
                    continue
        except Exception:
            continue

    best_alpha, best_mean_ci = None, 0.0
    for i, scores in alpha_scores.items():
        if len(scores) >= 3:
            m = np.mean(scores)
            if m > best_mean_ci:
                best_mean_ci = m
                best_alpha = outer_alphas[i]
    if best_alpha is None:
        best_alpha = outer_alphas[-1]

    alpha_idx = np.argmin(np.abs(model.alphas_ - best_alpha))
    coefs = model.coef_[:, alpha_idx]
    n_nonzero = int(np.sum(coefs != 0))

    train_pred = model.predict(X_train, alpha=best_alpha)
    train_ci, _, _, _, _ = concordance_index_censored(y_train["event"], y_train["time"], train_pred)

    test_pred = model.predict(X_test, alpha=best_alpha)
    test_ci, _, _, _, _ = concordance_index_censored(y_test["event"], y_test["time"], test_pred)

    # Save model and rankings
    ranking = pd.DataFrame({
        "gene_id": gene_names,
        "abs_coef": np.abs(coefs),
        "coef": coefs,
    }).sort_values("abs_coef", ascending=False)
    ranking["rank"] = range(1, len(ranking) + 1)
    ranking.to_csv(out_dir / f"cox_gene_ranking_{direction_label}.csv", index=False)

    with open(out_dir / f"cox_model_{direction_label}.pkl", "wb") as f:
        pickle.dump({"model": model, "best_alpha": best_alpha, "gene_names": gene_names}, f)

    return train_ci, test_ci, n_nonzero, best_alpha


def phase7_deepsurv(
    X_train, y_event_train, y_time_train, train_ids,
    X_test, y_event_test, y_time_test, test_ids,
    gene_names, params, device, direction_label, out_dir,
):
    """Train DeepSurv and compute IG (from phase7 script)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    n_features = X_train.shape[1]

    # 10% internal split for early stopping
    strata_es = create_strata(y_event_train, y_time_train)
    tr_idx, es_idx = train_test_split(
        np.arange(len(X_train)), test_size=0.1,
        stratify=strata_es, random_state=SEED,
    )

    train_ds = FoldDataset(X_train[tr_idx], y_time_train[tr_idx], y_event_train[tr_idx],
                           [train_ids[i] for i in tr_idx], gene_names)
    es_ds = FoldDataset(X_train[es_idx], y_time_train[es_idx], y_event_train[es_idx],
                        [train_ids[i] for i in es_idx], gene_names)

    train_loader = make_loader(train_ds, params["batch_size"], shuffle=True)
    es_loader = make_loader(es_ds, params["batch_size"], shuffle=False)

    model = ElasticDeepSurv(
        n_features=n_features,
        hidden_sizes=params["hidden_sizes"],
        dropout=params["dropout"],
        activation=params["activation"],
        batch_norm=params["batch_norm"],
        l1_ratio=params["l1_ratio"],
        alpha=params["alpha"],
    )

    trainer = ElasticDeepSurvTrainer(
        model=model, learning_rate=params["learning_rate"],
        weight_decay=DS_WEIGHT_DECAY, scheduler_patience=P7_SCHEDULER_PATIENCE,
        device=device,
    )

    best_cindex, best_epoch, patience_counter, best_state = 0.0, 0, 0, None
    for epoch in range(P7_MAX_EPOCHS):
        trainer.train_epoch(train_loader)
        _, _, _, es_ci = trainer.evaluate(es_loader)
        trainer.scheduler.step(es_ci)
        if es_ci > best_cindex:
            best_cindex = es_ci
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= P7_EARLY_STOPPING:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save model
    torch.save(model.state_dict(), out_dir / f"deepsurv_model_{direction_label}.pt")

    # Evaluate
    model.eval()
    model.to(device)

    test_ds = FoldDataset(X_test, y_time_test, y_event_test, test_ids, gene_names)
    test_loader = make_loader(test_ds, params["batch_size"], shuffle=False)
    _, _, _, test_ci = trainer.evaluate(test_loader)

    full_train_ds = FoldDataset(X_train, y_time_train, y_event_train, train_ids, gene_names)
    full_train_loader = make_loader(full_train_ds, params["batch_size"], shuffle=False)
    _, _, _, train_ci = trainer.evaluate(full_train_loader)

    # Compute IG on both cohorts
    ig = IntegratedGradients(model)
    baseline = torch.zeros(1, n_features).to(device)

    def compute_ig(X_data, sample_ids_data):
        X_tensor = torch.from_numpy(X_data.astype(np.float32)).to(device)
        all_attr = []
        for start in range(0, len(X_data), P7_IG_BATCH_SIZE):
            end = min(start + P7_IG_BATCH_SIZE, len(X_data))
            batch = X_tensor[start:end]
            attr = ig.attribute(
                batch, baselines=baseline.expand(batch.shape[0], -1),
                n_steps=P7_IG_N_STEPS, method="gausslegendre",
            )
            all_attr.append(attr.cpu().detach().numpy())
        attributions = np.concatenate(all_attr, axis=0)
        ig_df = pd.DataFrame(attributions, index=sample_ids_data, columns=gene_names)
        ig_df.index.name = "sampleID"
        return ig_df

    ig_train_df = compute_ig(X_train, train_ids)
    ig_test_df = compute_ig(X_test, test_ids)

    ig_train_df.to_csv(out_dir / f"deepsurv_ig_{direction_label}_train.csv")
    ig_test_df.to_csv(out_dir / f"deepsurv_ig_{direction_label}_test.csv")

    model.cpu()
    return train_ci, test_ci


# =============================================================================
# STAGE 3: Phase 8 — External Validation
# =============================================================================

def make_survival_array(surv_df):
    return np.array(
        [(bool(e), t) for e, t in zip(surv_df["event"], surv_df["time"])],
        dtype=[("event", bool), ("time", float)],
    )


def cv_ridge_cox_single(X, y_struct, events, n_folds, alphas, seed):
    """Single-seed ridge Cox CV."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []

    for train_idx, test_idx in skf.split(X, events):
        test_events = y_struct[test_idx]["event"].sum()
        if test_events < 2:
            continue
        try:
            model = CoxnetSurvivalAnalysis(
                l1_ratio=0.01, alphas=alphas,
                normalize=False, tol=1e-7, max_iter=100000,
            )
            model.fit(X[train_idx], y_struct[train_idx])
            fold_cindices = [np.nan] * len(alphas)
            for ai in range(len(model.alphas_)):
                try:
                    match_idx = np.argmin(np.abs(alphas - model.alphas_[ai]))
                    scores = X[test_idx] @ model.coef_[:, ai]
                    c = concordance_index_censored(
                        y_struct[test_idx]["event"], y_struct[test_idx]["time"], scores
                    )[0]
                    fold_cindices[match_idx] = c
                except Exception:
                    pass
            fold_results.append(fold_cindices)
        except Exception:
            continue

    if not fold_results:
        return None, None, None
    arr = np.array(fold_results)
    mean_c = np.nanmean(arr, axis=0)
    best_idx = np.nanargmax(mean_c)
    return mean_c[best_idx], np.nanstd(arr[:, best_idx]), alphas[best_idx]


def run_multi_seed_cv(X, y_struct, events, alphas=None):
    """100-seed repeated CV for stable C-index estimation."""
    if alphas is None:
        alphas = P8_ALPHAS_GENES
    results = []
    for seed in P8_SEEDS:
        c, s, a = cv_ridge_cox_single(X, y_struct, events, P8_N_FOLDS, alphas, seed)
        if c is not None:
            results.append(c)
    if not results:
        return None
    return {
        "median_cindex": float(np.median(results)),
        "mean_cindex": float(np.mean(results)),
        "sd": float(np.std(results)),
        "ci_lower": float(np.percentile(results, 2.5)),
        "ci_upper": float(np.percentile(results, 97.5)),
        "n_seeds": len(results),
    }


def encode_clinical_41613(clin, surv):
    samples = surv["sampleID"].values
    clin = clin.set_index("sampleID").loc[samples].reset_index()
    features = pd.DataFrame(index=samples)
    features["gender"] = clin["gender"].values
    features["tumor_stage"] = (clin["tumor_stage"] == "III/IV").astype(int).values
    age_order = {"19-39": 0, "40-49": 1, "50-59": 2, "60-88": 3}
    features["age_ordinal"] = clin["age_bin"].map(age_order).values
    return features


def encode_clinical_42743(clin, surv):
    samples = surv["sampleID"].values
    clin = clin.set_index("sampleID").loc[samples].reset_index()
    features = pd.DataFrame(index=samples)
    features["gender"] = clin["gender"].values
    features["age"] = clin["age"].values
    features["smoking"] = (clin["smoking"] == "Ever").astype(int).values
    features["T_stage"] = (clin["T_stage"] == "T3-T4").astype(int).values
    features["N_stage"] = (clin["N_stage"] == "N+").astype(int).values
    return features


def load_geo_dataset(name):
    expr = pd.read_csv(EXTERNAL_DATA_DIR / f"{name}_expression.csv", index_col=0)
    surv = pd.read_csv(EXTERNAL_DATA_DIR / f"{name}_survival.csv")
    clin = pd.read_csv(EXTERNAL_DATA_DIR / f"{name}_clinical.csv")
    if "sampleID" not in clin.columns:
        if clin.columns[0].startswith("Unnamed") or clin.columns[0] == "":
            clin = pd.read_csv(EXTERNAL_DATA_DIR / f"{name}_clinical.csv", index_col=0)
            clin["sampleID"] = clin.index
        else:
            clin["sampleID"] = clin.iloc[:, 0]
    return expr, surv, clin


def run_phase8(consensus_genes, out_dir):
    """Run Phase 8 external validation for one gene set."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for name, encode_fn in [("GSE41613", encode_clinical_41613),
                             ("GSE42743", encode_clinical_42743)]:
        expr, surv, clin = load_geo_dataset(name)
        samples = surv["sampleID"].values

        # Filter to consensus genes present in this dataset
        available = [g for g in consensus_genes if g in expr.index]
        expr_aligned = expr.loc[available][samples].values.T
        y_struct = make_survival_array(surv)
        events = surv["event"].values

        # Gene signature
        res_genes = run_multi_seed_cv(expr_aligned, y_struct, events)

        # Clinical-only baseline
        clin_features = encode_fn(clin, surv)
        X_clin = clin_features.values.astype(float)
        missing = np.isnan(X_clin).any(axis=1)
        if missing.any():
            X_clin = X_clin[~missing]
            y_clin = y_struct[~missing]
            ev_clin = events[~missing]
        else:
            y_clin = y_struct
            ev_clin = events
        res_clin = run_multi_seed_cv(X_clin, y_clin, ev_clin, alphas=P8_ALPHAS_CLINICAL)

        results[name] = {
            "n_genes_available": len(available),
            "n_genes_total": len(consensus_genes),
            "n_samples": len(samples),
            "n_events": int(events.sum()),
            "gene_signature": res_genes,
            "clinical_only": res_clin,
        }

        gs_str = f"{res_genes['median_cindex']:.3f}" if res_genes else "N/A"
        cl_str = f"{res_clin['median_cindex']:.3f}" if res_clin else "N/A"
        logger.info(f"      {name}: gene={gs_str}, clinical={cl_str}")

    # Save
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_dir / "phase8_summary.json", "w") as f:
        json.dump(make_serializable(results), f, indent=2)

    return results


# =============================================================================
# MAIN SWEEP
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sensitivity Sweep")
    parser.add_argument("--phase4-dir", type=str, default=PHASE4_DIR)
    parser.add_argument("--skip-tuning", action="store_true",
                        help="Skip Stage 1 (use existing tuning results)")
    parser.add_argument("--skip-phase7", action="store_true",
                        help="Skip Stage 2 (use existing Phase 7 results)")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated subset, e.g. '0.6_0.60,0.7_0.60'")
    args = parser.parse_args()

    total_start = time.time()

    # Parse config subset if provided
    if args.configs:
        configs = []
        for c in args.configs.split(","):
            mult, pi = c.strip().split("_")
            configs.append((float(mult), float(pi)))
    else:
        configs = SWEEP_CONFIGS

    # Setup
    SWEEP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Log to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(SWEEP_OUTPUT_DIR / f"sweep_{timestamp}.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("=" * 70)
    logger.info("SENSITIVITY SWEEP: Retune + Validate All Combinations")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Configurations to evaluate: {len(configs)}")
    logger.info(f"DeepSurv trials per cohort: {DS_N_TRIALS}")
    logger.info(f"Phase 8 seeds: {len(P8_SEEDS)}")

    # =========================================================================
    # Load development cohort data (shared across all configs)
    # =========================================================================
    logger.info("\n--- Loading development cohort data ---")
    tcga_expr, tcga_surv = load_cohort("tcga", DATA_DIR)
    orien_expr, orien_surv = load_cohort("orien", DATA_DIR)

    all_gene_names = tcga_expr.index.tolist()
    assert list(orien_expr.index) == all_gene_names

    tcga_X_full = tcga_expr.values.T.astype(np.float32)
    tcga_event = tcga_surv["event"].values.astype(np.float32)
    tcga_time = tcga_surv["time"].values.astype(np.float32)
    tcga_ids = tcga_surv.index.tolist()

    orien_X_full = orien_expr.values.T.astype(np.float32)
    orien_event = orien_surv["event"].values.astype(np.float32)
    orien_time = orien_surv["time"].values.astype(np.float32)
    orien_ids = orien_surv.index.tolist()

    logger.info(f"TCGA: {tcga_X_full.shape[0]} samples, ORIEN: {orien_X_full.shape[0]} samples")
    logger.info(f"Total genes: {len(all_gene_names)}")

    # Annotation
    annot = pd.read_csv(ANNOTATION_PATH)
    ensg_to_symbol = dict(zip(annot["gene_id"], annot["gene_symbol"]))

    # =========================================================================
    # Sweep loop
    # =========================================================================
    comparison_rows = []

    for config_idx, (mult, pi) in enumerate(configs):
        config_label = f"mult{mult}_pi{int(pi*100):02d}"
        config_start = time.time()

        logger.info(f"\n{'='*70}")
        logger.info(f"  CONFIG {config_idx+1}/{len(configs)}: {config_label}")
        logger.info(f"{'='*70}")

        config_dir = SWEEP_OUTPUT_DIR / config_label
        config_dir.mkdir(parents=True, exist_ok=True)

        # Load gene set
        try:
            stable_genes = load_phase4_genes(args.phase4_dir, mult, pi)
        except FileNotFoundError as e:
            logger.warning(f"  Skipping {config_label}: {e}")
            continue

        union_genes = stable_genes["union"]
        n_genes = len(union_genes)

        logger.info(f"  Gene sets: TCGA={len(stable_genes['tcga'])}, "
                     f"ORIEN={len(stable_genes['orien'])}, "
                     f"union={n_genes}, overlap={len(stable_genes['overlap'])}")

        if n_genes == 0:
            logger.warning(f"  Empty gene set — skipping")
            continue

        # Save union gene list
        union_df = pd.DataFrame([{
            "gene": g,
            "symbol": ensg_to_symbol.get(g, ""),
            "stable_in": "both" if g in stable_genes["overlap"]
                         else ("tcga" if g in stable_genes["tcga"] else "orien"),
        } for g in union_genes])
        union_df.to_csv(config_dir / "union_gene_set.csv", index=False)
        with open(config_dir / "union_genes.txt", "w") as f:
            f.write("\n".join(union_genes))

        # Subset expression
        tcga_X_union = subset_expression(tcga_X_full, all_gene_names, union_genes)
        orien_X_union = subset_expression(orien_X_full, all_gene_names, union_genes)

        # Also subset for per-cohort Cox
        tcga_X_tcga_genes = subset_expression(tcga_X_full, all_gene_names, stable_genes["tcga"])
        orien_X_tcga_genes = subset_expression(orien_X_full, all_gene_names, stable_genes["tcga"])
        orien_X_orien_genes = subset_expression(orien_X_full, all_gene_names, stable_genes["orien"])
        tcga_X_orien_genes = subset_expression(tcga_X_full, all_gene_names, stable_genes["orien"])

        # =====================================================================
        # STAGE 1: Tuning
        # =====================================================================
        tuning_dir = config_dir / "tuning"
        tuning_dir.mkdir(parents=True, exist_ok=True)

        if not args.skip_tuning:
            logger.info(f"\n  --- Stage 1: Tuning ({n_genes} genes) ---")

            # Cox tuning (on union genes for simplicity — l1_ratio is the key param)
            cox_params = {}
            for cohort, X, y_ev, y_t in [
                ("tcga", tcga_X_union, tcga_event, tcga_time),
                ("orien", orien_X_union, orien_event, orien_time),
            ]:
                cox_res = tune_cox(X, y_ev, y_t, cohort, tuning_dir)
                cox_params[cohort] = {"l1_ratio": cox_res["best_l1_ratio"]}

            # DeepSurv tuning
            ds_params = {}
            for cohort, X, y_ev, y_t in [
                ("tcga", tcga_X_union, tcga_event, tcga_time),
                ("orien", orien_X_union, orien_event, orien_time),
            ]:
                gene_names_list = union_genes
                ds_res = tune_deepsurv(
                    X, y_ev, y_t, gene_names_list, cohort, device, tuning_dir
                )
                ds_params[cohort] = ds_res["best_params"]
        else:
            logger.info(f"\n  --- Stage 1: Skipped (loading existing) ---")
            cox_params = {}
            ds_params = {}
            for cohort in ["tcga", "orien"]:
                cox_file = tuning_dir / f"cox_{cohort}_best.json"
                with open(cox_file) as f:
                    cox_params[cohort] = {"l1_ratio": json.load(f)["best_l1_ratio"]}
                ds_file = tuning_dir / f"deepsurv_{cohort}_best.json"
                with open(ds_file) as f:
                    ds_params[cohort] = json.load(f)["best_params"]

        # =====================================================================
        # STAGE 2: Phase 7 Bidirectional Validation
        # =====================================================================
        val_dir = config_dir / "validation"

        if not args.skip_phase7:
            logger.info(f"\n  --- Stage 2: Phase 7 Validation ---")

            cox_dir = val_dir / "cox"
            cox_dir.mkdir(parents=True, exist_ok=True)
            ds_dir = val_dir / "deepsurv"
            ds_dir.mkdir(parents=True, exist_ok=True)

            # Cox per-cohort bidirectional
            d1_train, d1_test, d1_nz, d1_alpha = phase7_cox(
                tcga_X_tcga_genes, tcga_event, tcga_time, tcga_ids,
                orien_X_tcga_genes, orien_event, orien_time, orien_ids,
                cox_params["tcga"]["l1_ratio"], stable_genes["tcga"],
                "tcga_to_orien", cox_dir,
            )
            d2_train, d2_test, d2_nz, d2_alpha = phase7_cox(
                orien_X_orien_genes, orien_event, orien_time, orien_ids,
                tcga_X_orien_genes, tcga_event, tcga_time, tcga_ids,
                cox_params["orien"]["l1_ratio"], stable_genes["orien"],
                "orien_to_tcga", cox_dir,
            )
            cox_mean = (d1_test + d2_test) / 2

            # Cox union bidirectional
            u1_train, u1_test, u1_nz, u1_alpha = phase7_cox(
                tcga_X_union, tcga_event, tcga_time, tcga_ids,
                orien_X_union, orien_event, orien_time, orien_ids,
                cox_params["tcga"]["l1_ratio"], union_genes,
                "tcga_to_orien_union", cox_dir,
            )
            u2_train, u2_test, u2_nz, u2_alpha = phase7_cox(
                orien_X_union, orien_event, orien_time, orien_ids,
                tcga_X_union, tcga_event, tcga_time, tcga_ids,
                cox_params["orien"]["l1_ratio"], union_genes,
                "orien_to_tcga_union", cox_dir,
            )
            cox_union_mean = (u1_test + u2_test) / 2

            logger.info(f"    Cox per-cohort: T→O={d1_test:.4f}, O→T={d2_test:.4f}, "
                        f"mean={cox_mean:.4f}")
            logger.info(f"    Cox union:      T→O={u1_test:.4f}, O→T={u2_test:.4f}, "
                        f"mean={cox_union_mean:.4f}")

            # DeepSurv union bidirectional
            ds_d1_train, ds_d1_test = phase7_deepsurv(
                tcga_X_union, tcga_event, tcga_time, tcga_ids,
                orien_X_union, orien_event, orien_time, orien_ids,
                union_genes, ds_params["tcga"], device,
                "tcga_to_orien", ds_dir,
            )
            ds_d2_train, ds_d2_test = phase7_deepsurv(
                orien_X_union, orien_event, orien_time, orien_ids,
                tcga_X_union, tcga_event, tcga_time, tcga_ids,
                union_genes, ds_params["orien"], device,
                "orien_to_tcga", ds_dir,
            )
            ds_mean = (ds_d1_test + ds_d2_test) / 2

            logger.info(f"    DeepSurv union: T→O={ds_d1_test:.4f}, O→T={ds_d2_test:.4f}, "
                        f"mean={ds_mean:.4f}")

            # Save Phase 7 summary
            p7_summary = {
                "config": {"alpha_multiplier": mult, "pi_threshold": pi},
                "gene_sets": {
                    "tcga_stable": len(stable_genes["tcga"]),
                    "orien_stable": len(stable_genes["orien"]),
                    "union": n_genes,
                    "overlap": len(stable_genes["overlap"]),
                },
                "cox_percohort": {
                    "tcga_to_orien": {"train_ci": float(d1_train), "test_ci": float(d1_test)},
                    "orien_to_tcga": {"train_ci": float(d2_train), "test_ci": float(d2_test)},
                    "mean_test_ci": float(cox_mean),
                },
                "cox_union": {
                    "tcga_to_orien": {"train_ci": float(u1_train), "test_ci": float(u1_test)},
                    "orien_to_tcga": {"train_ci": float(u2_train), "test_ci": float(u2_test)},
                    "mean_test_ci": float(cox_union_mean),
                },
                "deepsurv_union": {
                    "tcga_to_orien": {"train_ci": float(ds_d1_train), "test_ci": float(ds_d1_test)},
                    "orien_to_tcga": {"train_ci": float(ds_d2_train), "test_ci": float(ds_d2_test)},
                    "mean_test_ci": float(ds_mean),
                },
                "tuned_params": {
                    "cox": cox_params,
                    "deepsurv_tcga": ds_params["tcga"],
                    "deepsurv_orien": ds_params["orien"],
                },
            }
            with open(val_dir / "phase7_summary.json", "w") as f:
                json.dump(p7_summary, f, indent=2, default=str)

        else:
            logger.info(f"\n  --- Stage 2: Skipped (loading existing) ---")
            with open(val_dir / "phase7_summary.json") as f:
                p7_summary = json.load(f)
            cox_mean = p7_summary["cox_percohort"]["mean_test_ci"]
            cox_union_mean = p7_summary["cox_union"]["mean_test_ci"]
            ds_mean = p7_summary["deepsurv_union"]["mean_test_ci"]
            d1_test = p7_summary["cox_percohort"]["tcga_to_orien"]["test_ci"]
            d2_test = p7_summary["cox_percohort"]["orien_to_tcga"]["test_ci"]
            ds_d1_test = p7_summary["deepsurv_union"]["tcga_to_orien"]["test_ci"]
            ds_d2_test = p7_summary["deepsurv_union"]["orien_to_tcga"]["test_ci"]

        # =====================================================================
        # STAGE 3: Phase 8 External Validation
        # =====================================================================
        logger.info(f"\n  --- Stage 3: Phase 8 External Validation ---")
        ext_dir = config_dir / "external"
        p8_results = run_phase8(union_genes, ext_dir)

        # Extract external C-indices
        gse41613_ci = (p8_results.get("GSE41613", {}).get("gene_signature", {})
                       .get("median_cindex", np.nan))
        gse42743_ci = (p8_results.get("GSE42743", {}).get("gene_signature", {})
                       .get("median_cindex", np.nan))

        # =====================================================================
        # Record for comparison
        # =====================================================================
        config_elapsed = time.time() - config_start

        row = {
            "mult": mult,
            "pi": pi,
            "n_genes_union": n_genes,
            "n_genes_tcga": len(stable_genes["tcga"]),
            "n_genes_orien": len(stable_genes["orien"]),
            "n_overlap": len(stable_genes["overlap"]),
            "cox_percohort_t2o": float(d1_test),
            "cox_percohort_o2t": float(d2_test),
            "cox_percohort_mean": float(cox_mean),
            "cox_union_mean": float(cox_union_mean),
            "deepsurv_t2o": float(ds_d1_test),
            "deepsurv_o2t": float(ds_d2_test),
            "deepsurv_mean": float(ds_mean),
            "gse41613_median_ci": float(gse41613_ci) if not np.isnan(gse41613_ci) else None,
            "gse42743_median_ci": float(gse42743_ci) if not np.isnan(gse42743_ci) else None,
            "runtime_seconds": config_elapsed,
        }
        comparison_rows.append(row)

        logger.info(f"\n  Config {config_label} complete ({config_elapsed:.0f}s)")
        logger.info(f"    Cox per-cohort mean: {cox_mean:.4f}")
        logger.info(f"    Cox union mean:      {cox_union_mean:.4f}")
        logger.info(f"    DeepSurv mean:       {ds_mean:.4f}")
        logger.info(f"    GSE41613:            {gse41613_ci:.3f}")
        logger.info(f"    GSE42743:            {gse42743_ci:.3f}")

        # Save incremental comparison (in case of crash)
        pd.DataFrame(comparison_rows).to_csv(
            SWEEP_OUTPUT_DIR / "sweep_comparison_partial.csv", index=False
        )

    # =========================================================================
    # Final comparison table
    # =========================================================================
    total_elapsed = time.time() - total_start

    logger.info(f"\n{'='*70}")
    logger.info("SWEEP COMPLETE")
    logger.info(f"{'='*70}")

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df.sort_values("deepsurv_mean", ascending=False)
    comparison_df.to_csv(SWEEP_OUTPUT_DIR / "sweep_comparison.csv", index=False)

    logger.info(f"\n{comparison_df.to_string(index=False)}")

    # Highlight best
    if len(comparison_df) > 0:
        best_row = comparison_df.iloc[0]
        logger.info(f"\n  BEST CONFIG (by DeepSurv mean):")
        logger.info(f"    mult={best_row['mult']}, pi={best_row['pi']}")
        logger.info(f"    {int(best_row['n_genes_union'])} genes")
        logger.info(f"    Cox per-cohort mean: {best_row['cox_percohort_mean']:.4f}")
        logger.info(f"    Cox union mean:      {best_row['cox_union_mean']:.4f}")
        logger.info(f"    DeepSurv mean:       {best_row['deepsurv_mean']:.4f}")
        logger.info(f"    GSE41613:            {best_row['gse41613_median_ci']}")
        logger.info(f"    GSE42743:            {best_row['gse42743_median_ci']}")

    logger.info(f"\n  Total runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    logger.info(f"  Results: {SWEEP_OUTPUT_DIR / 'sweep_comparison.csv'}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
