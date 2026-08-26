#!/usr/bin/env python3
"""
Phase 7: Bidirectional Validation with Per-Cohort Stable Genes
===============================================================

Train on 100% of each cohort using that cohort's stable genes
(from Phase 4 randomized lasso), evaluate on the opposite cohort.

Design:
    - Per-cohort gene sets from Phase 4 randomized lasso stability selection
    - TCGA→ORIEN: train Cox on TCGA with TCGA-stable genes, test on ORIEN
    - ORIEN→TCGA: train Cox on ORIEN with ORIEN-stable genes, test on TCGA
    - Union gene set: train DeepSurv for IG heterogeneity analysis
    - Phase 2 hyperparameters used for both Cox and DeepSurv

    Cox: train on full regularization path, select lambda by inner 5-fold CV.
    DeepSurv: 10% internal validation split for early stopping,
              IG attributions computed on ALL patients in BOTH cohorts.

Inputs (linked from other phases):
    - Phase 2: Hyperparameters (results/phase2_tuning/)
    - Phase 4: Per-cohort stable gene lists (results/phase4_stability_randomized_lasso/)
    - IQR filter: Gene list (data/processed/iqr_filtered/)
    - Annotation: Gene symbols (data/raw/protein_coding_genes_annotation.csv)

References:
    - Katzman et al. (2018) BMC Med Res Methodol — DeepSurv
    - Meinshausen & Bühlmann (2010) JRSS-B — Randomized lasso stability selection
    - Sundararajan et al. (2017) ICML — Integrated Gradients
    - Bernau et al. (2014) Bioinformatics — Cross-study validation

Usage:
    cd ~/hpchome/oscc-biomarker-grouplasso
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/phase7_bidirectional_percohort.py

    # Custom Phase 4 results directory:
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/phase7_bidirectional_percohort.py \
        --phase4-dir results/phase4_stability_randomized_lasso \
        --alpha-mult 0.3 --pi-thresh 0.60
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
from typing import Dict, List, Tuple

from sklearn.model_selection import StratifiedKFold, train_test_split
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from captum.attr import IntegratedGradients

from src.data.data_loader import load_cohort
from src.models.elastic_deepsurv import ElasticDeepSurv, ElasticDeepSurvTrainer
from src.utils.batch_samplers import StratifiedBatchSampler
from torch.utils.data import DataLoader

# =============================================================================
# Setup
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

logging.getLogger("src.models.deepsurv").setLevel(logging.ERROR)
logging.getLogger("src.models.elastic_deepsurv").setLevel(logging.ERROR)
logging.getLogger("src.utils.batch_samplers").setLevel(logging.ERROR)
logging.getLogger("src.data.data_loader").setLevel(logging.WARNING)

# =============================================================================
# Configuration
# =============================================================================

SEED = 42
DATA_DIR = "data/processed"
ANNOTATION_PATH = Path("data/raw/protein_coding_genes_annotation.csv")

# Training config
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 20
SCHEDULER_PATIENCE = 10
WEIGHT_DECAY = 0.0
COX_N_ALPHAS = 100
COX_INNER_CV_FOLDS = 5

# IG config
IG_N_STEPS = 300
IG_BATCH_SIZE = 32


# =============================================================================
# Load Phase 2 parameters
# =============================================================================

def load_phase2_deepsurv_params(phase2_dir="results/phase2_tuning"):
    """Load best DeepSurv hyperparameters from Phase 2."""
    phase2_path = Path(phase2_dir)
    tuning_dirs = sorted(phase2_path.glob("deepsurv_tuning_*"))
    if not tuning_dirs:
        raise FileNotFoundError(f"No deepsurv tuning results found in {phase2_path}")
    latest = tuning_dirs[-1]
    results_file = latest / "all_results.json"
    with open(results_file) as f:
        all_results = json.load(f)
    logger.info(f"Loaded DeepSurv Phase 2 params from: {results_file}")
    return {
        cohort: all_results[cohort]['best_params']
        for cohort in ['tcga', 'orien']
    }


def load_phase2_cox_params(phase2_dir="results/phase2_tuning"):
    """Load best Cox hyperparameters from Phase 2."""
    phase2_path = Path(phase2_dir)
    tuning_dirs = sorted(phase2_path.glob("cox_tuning_*"))
    if not tuning_dirs:
        raise FileNotFoundError(f"No cox tuning results found in {phase2_path}")
    latest = tuning_dirs[-1]
    results_file = latest / "all_results.json"
    with open(results_file) as f:
        all_results = json.load(f)
    logger.info(f"Loaded Cox Phase 2 params from: {results_file}")
    return {
        cohort: {
            'l1_ratio': all_results[cohort]['best_l1_ratio'],
            'alpha': all_results[cohort].get('actual_path_alpha',
                     all_results[cohort].get('recommended_alpha')),
        }
        for cohort in ['tcga', 'orien']
    }


# =============================================================================
# Load Phase 4 stable gene lists
# =============================================================================

def load_phase4_stable_genes(phase4_dir, alpha_mult, pi_thresh):
    """
    Load per-cohort stable gene lists from Phase 4 randomized lasso.

    Args:
        phase4_dir: Path to Phase 4 results
        alpha_mult: Alpha multiplier used (e.g., 0.3)
        pi_thresh: Pi threshold (e.g., 0.60)

    Returns:
        dict with 'tcga' and 'orien' gene lists and the union set
    """
    phase4_path = Path(phase4_dir)
    consensus_dir = phase4_path / "consensus"

    pi_label = int(pi_thresh * 100)

    stable_genes = {}
    for cohort in ['tcga', 'orien']:
        fname = consensus_dir / f"{cohort}_stable_mult{alpha_mult}_pi{pi_label:02d}.csv"
        if not fname.exists():
            raise FileNotFoundError(
                f"Phase 4 stable gene file not found: {fname}\n"
                f"Run Phase 4 with --save-all or ensure the file exists."
            )
        df = pd.read_csv(fname)
        stable_genes[cohort] = df['gene'].tolist()
        logger.info(f"  {cohort.upper()} stable genes (mult={alpha_mult}, pi>={pi_thresh}): "
                     f"{len(stable_genes[cohort])}")

    # Union set
    union_genes = sorted(set(stable_genes['tcga']) | set(stable_genes['orien']))
    overlap_genes = sorted(set(stable_genes['tcga']) & set(stable_genes['orien']))

    logger.info(f"  Union: {len(union_genes)} genes")
    logger.info(f"  Overlap: {len(overlap_genes)} genes")

    return {
        'tcga': stable_genes['tcga'],
        'orien': stable_genes['orien'],
        'union': union_genes,
        'overlap': overlap_genes,
    }


# =============================================================================
# Helpers
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


# =============================================================================
# Cox: train and evaluate bidirectionally
# =============================================================================

def train_and_evaluate_cox(
    X_train, y_event_train, y_time_train, train_ids,
    X_test, y_event_test, y_time_test, test_ids,
    l1_ratio, gene_names, direction_label, out_dir,
):
    """
    Train Cox on training cohort, evaluate on test cohort.

    Returns:
        train_ci, test_ci, n_nonzero, best_alpha, coefs
    """
    logger.info(f"    Training Cox ({direction_label})...")
    logger.info(f"      Train: {X_train.shape[0]} samples, {X_train.shape[1]} genes")
    logger.info(f"      Test: {X_test.shape[0]} samples")

    # Structured arrays
    y_train = np.array(
        [(bool(e), t) for e, t in zip(y_event_train, y_time_train)],
        dtype=[("event", bool), ("time", float)],
    )
    y_test = np.array(
        [(bool(e), t) for e, t in zip(y_event_test, y_time_test)],
        dtype=[("event", bool), ("time", float)],
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

    outer_alphas = model.alphas_

    # Inner CV to select best lambda
    strata = create_strata(y_event_train, y_time_train)
    skf = StratifiedKFold(
        n_splits=COX_INNER_CV_FOLDS, shuffle=True, random_state=SEED
    )

    alpha_scores = {i: [] for i in range(len(outer_alphas))}

    for train_idx, valid_idx in skf.split(X_train, strata):
        inner_model = CoxnetSurvivalAnalysis(
            l1_ratio=l1_ratio,
            alphas=outer_alphas,
            normalize=False,
            tol=1e-7,
            max_iter=100000,
        )
        try:
            inner_model.fit(X_train[train_idx], y_train[train_idx])
        except Exception as e:
            logger.warning(f"      Inner CV fold failed: {e}")
            continue

        for i, alpha_val in enumerate(inner_model.alphas_):
            try:
                pred = inner_model.predict(X_train[valid_idx], alpha=alpha_val)
                ci, _, _, _, _ = concordance_index_censored(
                    y_train[valid_idx]["event"], y_train[valid_idx]["time"], pred
                )
                alpha_scores[i].append(ci)
            except Exception:
                continue

    # Select best alpha
    best_alpha = None
    best_mean_ci = 0.0
    for i, scores in alpha_scores.items():
        if len(scores) >= 3:
            mean_ci = np.mean(scores)
            if mean_ci > best_mean_ci:
                best_mean_ci = mean_ci
                best_alpha = outer_alphas[i]

    if best_alpha is None:
        best_alpha = outer_alphas[-1]
        logger.warning("      Inner CV could not select alpha, using smallest lambda")

    logger.info(f"      Best alpha: {best_alpha:.6e}, inner CV C-index: {best_mean_ci:.4f}")

    # Extract coefficients at best alpha
    alpha_idx = np.argmin(np.abs(model.alphas_ - best_alpha))
    coefs = model.coef_[:, alpha_idx]
    n_nonzero = int(np.sum(coefs != 0))
    logger.info(f"      Non-zero genes: {n_nonzero}/{len(gene_names)}")

    # Training C-index
    train_pred = model.predict(X_train, alpha=best_alpha)
    train_ci, _, _, _, _ = concordance_index_censored(
        y_train["event"], y_train["time"], train_pred
    )
    logger.info(f"      Training C-index: {train_ci:.4f}")

    # Test C-index (external validation)
    test_pred = model.predict(X_test, alpha=best_alpha)
    test_ci, _, _, _, _ = concordance_index_censored(
        y_test["event"], y_test["time"], test_pred
    )
    logger.info(f"      Test C-index: {test_ci:.4f}")

    # Save gene ranking
    abs_coefs = np.abs(coefs)
    rank_order = np.argsort(-abs_coefs)
    gene_ranking = [(gene_names[i], float(abs_coefs[i]), float(coefs[i])) for i in rank_order]
    ranking_df = pd.DataFrame(gene_ranking, columns=["gene_id", "abs_coef", "coef"])
    ranking_df["rank"] = range(1, len(ranking_df) + 1)
    ranking_df.to_csv(out_dir / f"cox_gene_ranking_{direction_label}.csv", index=False)

    # Save model
    with open(out_dir / f"cox_model_{direction_label}.pkl", "wb") as f:
        pickle.dump({"model": model, "best_alpha": best_alpha, "gene_names": gene_names}, f)

    return train_ci, test_ci, n_nonzero, best_alpha, coefs


# =============================================================================
# DeepSurv: train and compute IG on both cohorts
# =============================================================================

def train_and_evaluate_deepsurv(
    X_train, y_event_train, y_time_train, train_ids,
    X_test, y_event_test, y_time_test, test_ids,
    gene_names, params, device, direction_label, out_dir,
):
    """
    Train DeepSurv on training cohort, compute IG on both cohorts.

    Returns:
        train_ci, test_ci, ig_train_df, ig_test_df
    """
    logger.info(f"    Training DeepSurv ({direction_label})...")
    logger.info(f"      Train: {X_train.shape[0]} samples, {X_train.shape[1]} genes")
    logger.info(f"      Test: {X_test.shape[0]} samples")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    n_features = X_train.shape[1]

    # 10% internal validation split for early stopping
    strata_es = create_strata(y_event_train, y_time_train)
    tr_idx, es_idx = train_test_split(
        np.arange(len(X_train)),
        test_size=0.1,
        stratify=strata_es,
        random_state=SEED,
    )

    logger.info(f"      Internal split: train={len(tr_idx)}, ES holdout={len(es_idx)}")

    train_ds = FoldDataset(
        X_train[tr_idx], y_time_train[tr_idx], y_event_train[tr_idx],
        [train_ids[i] for i in tr_idx], gene_names,
    )
    es_ds = FoldDataset(
        X_train[es_idx], y_time_train[es_idx], y_event_train[es_idx],
        [train_ids[i] for i in es_idx], gene_names,
    )

    train_loader = make_loader(train_ds, params["batch_size"], shuffle=True)
    es_loader = make_loader(es_ds, params["batch_size"], shuffle=False)

    # Create model
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
        model=model,
        learning_rate=params["learning_rate"],
        weight_decay=WEIGHT_DECAY,
        scheduler_patience=SCHEDULER_PATIENCE,
        device=device,
    )

    # Training loop with early stopping
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

    if best_state is not None:
        model.load_state_dict(best_state)

    logger.info(f"      Best epoch: {best_epoch}, ES C-index: {best_cindex:.4f}")

    # Save model
    torch.save(model.state_dict(), out_dir / f"deepsurv_model_{direction_label}.pt")

    # Evaluate on test cohort
    model.eval()
    model.to(device)

    # Test C-index
    test_ds = FoldDataset(X_test, y_time_test, y_event_test, test_ids, gene_names)
    test_loader = make_loader(test_ds, params["batch_size"], shuffle=False)
    _, _, _, test_ci = trainer.evaluate(test_loader)

    # Training C-index (full training set)
    full_train_ds = FoldDataset(
        X_train, y_time_train, y_event_train, train_ids, gene_names
    )
    full_train_loader = make_loader(full_train_ds, params["batch_size"], shuffle=False)
    _, _, _, train_ci = trainer.evaluate(full_train_loader)

    logger.info(f"      Training C-index: {train_ci:.4f}")
    logger.info(f"      Test C-index: {test_ci:.4f}")

    # --- Compute IG on BOTH cohorts ---
    ig = IntegratedGradients(model)
    baseline = torch.zeros(1, n_features).to(device)

    def compute_ig(X_data, sample_ids_data, label):
        logger.info(f"      Computing IG on {label} ({len(X_data)} patients)...")
        X_tensor = torch.from_numpy(X_data.astype(np.float32)).to(device)
        all_attr = []
        for start in range(0, len(X_data), IG_BATCH_SIZE):
            end = min(start + IG_BATCH_SIZE, len(X_data))
            batch_input = X_tensor[start:end]
            batch_baseline = baseline.expand(batch_input.shape[0], -1)
            attr = ig.attribute(
                batch_input, baselines=batch_baseline,
                n_steps=IG_N_STEPS, method="gausslegendre",
            )
            all_attr.append(attr.cpu().detach().numpy())
        attributions = np.concatenate(all_attr, axis=0)
        ig_df = pd.DataFrame(attributions, index=sample_ids_data, columns=gene_names)
        ig_df.index.name = "sampleID"
        return ig_df

    ig_train_df = compute_ig(X_train, train_ids, "training cohort")
    ig_test_df = compute_ig(X_test, test_ids, "test cohort")

    # Save IG attributions
    ig_train_df.to_csv(out_dir / f"deepsurv_ig_{direction_label}_train.csv")
    ig_test_df.to_csv(out_dir / f"deepsurv_ig_{direction_label}_test.csv")

    model.cpu()

    return train_ci, test_ci, ig_train_df, ig_test_df


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 7: Bidirectional Validation with Per-Cohort Stable Genes"
    )
    parser.add_argument(
        '--phase4-dir', type=str,
        default='results/phase4_stability_randomized_lasso',
        help="Phase 4 results directory"
    )
    parser.add_argument(
        '--alpha-mult', type=float, default=0.3,
        help="Alpha multiplier from Phase 4 (default: 0.3)"
    )
    parser.add_argument(
        '--pi-thresh', type=float, default=0.60,
        help="Pi threshold from Phase 4 (default: 0.60)"
    )
    parser.add_argument(
        '--results-dir', type=str, default=None,
        help="Override results directory"
    )
    parser.add_argument(
        '--cox-only', action='store_true',
        help="Run Cox validation only (skip DeepSurv)"
    )
    args = parser.parse_args()

    start_time = time.time()

    # Results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path(f"results/phase7_percohort_mult{args.alpha_mult}_pi{int(args.pi_thresh*100):02d}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Log to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(results_dir / f"phase7_{timestamp}.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 70)
    logger.info("Phase 7: Bidirectional Validation with Per-Cohort Stable Genes")
    logger.info("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Seed: {SEED}")
    logger.info(f"Phase 4 dir: {args.phase4_dir}")
    logger.info(f"Alpha multiplier: {args.alpha_mult}")
    logger.info(f"Pi threshold: {args.pi_thresh}")

    # =========================================================================
    # Load data
    # =========================================================================
    logger.info("\n--- Loading data ---")
    tcga_expr, tcga_surv = load_cohort("tcga", DATA_DIR)
    orien_expr, orien_surv = load_cohort("orien", DATA_DIR)

    all_gene_names = tcga_expr.index.tolist()
    assert list(orien_expr.index) == all_gene_names, "Gene order mismatch"

    # Full expression matrices (samples × genes)
    tcga_X_full = tcga_expr.values.T.astype(np.float32)
    tcga_event = tcga_surv["event"].values.astype(np.float32)
    tcga_time = tcga_surv["time"].values.astype(np.float32)
    tcga_ids = tcga_surv.index.tolist()

    orien_X_full = orien_expr.values.T.astype(np.float32)
    orien_event = orien_surv["event"].values.astype(np.float32)
    orien_time = orien_surv["time"].values.astype(np.float32)
    orien_ids = orien_surv.index.tolist()

    logger.info(f"TCGA: {tcga_X_full.shape[0]} samples × {tcga_X_full.shape[1]} genes, "
                f"{int(tcga_event.sum())} events")
    logger.info(f"ORIEN: {orien_X_full.shape[0]} samples × {orien_X_full.shape[1]} genes, "
                f"{int(orien_event.sum())} events")

    # =========================================================================
    # Load Phase 4 stable gene lists
    # =========================================================================
    logger.info("\n--- Loading Phase 4 stable gene lists ---")
    stable_genes = load_phase4_stable_genes(
        args.phase4_dir, args.alpha_mult, args.pi_thresh
    )

    # Load annotation for gene symbols
    annot = pd.read_csv(ANNOTATION_PATH)
    ensg_to_symbol = dict(zip(annot["gene_id"], annot["gene_symbol"]))

    # Log gene lists with symbols
    for cohort in ['tcga', 'orien']:
        logger.info(f"\n  {cohort.upper()} stable genes ({len(stable_genes[cohort])}):")
        for gene in stable_genes[cohort]:
            sym = ensg_to_symbol.get(gene, "?")
            logger.info(f"    {gene} ({sym})")

    logger.info(f"\n  Union gene set ({len(stable_genes['union'])} genes)")

    # Save union gene list
    union_df = pd.DataFrame([
        {
            'gene': g,
            'symbol': ensg_to_symbol.get(g, ''),
            'stable_in': 'both' if g in stable_genes['overlap']
                         else ('tcga' if g in stable_genes['tcga'] else 'orien'),
        }
        for g in stable_genes['union']
    ])
    union_df.to_csv(results_dir / "union_gene_set.csv", index=False)

    # Load Phase 2 params
    COX_PARAMS = load_phase2_cox_params()
    DEEPSURV_PARAMS = load_phase2_deepsurv_params()

    # =========================================================================
    # Cox Bidirectional Validation (per-cohort gene sets)
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("COX BIDIRECTIONAL VALIDATION (per-cohort stable genes)")
    logger.info("=" * 70)

    cox_dir = results_dir / "cox"
    cox_dir.mkdir(parents=True, exist_ok=True)

    # --- Direction 1: TCGA → ORIEN ---
    logger.info("\n  Direction 1: TCGA → ORIEN")
    logger.info(f"  Using {len(stable_genes['tcga'])} TCGA-stable genes")

    # GUARD (2026-08): at the SS-primary CPSS cell TCGA has 0 stable genes.
    # An empty design matrix cannot be fit; record the direction as not
    # evaluable rather than crashing. This is a reportable finding.
    if len(stable_genes['tcga']) == 0:
        logger.warning("  Direction 1 NOT EVALUABLE: 0 TCGA-stable genes at this cell.")
        d1_train_ci = d1_test_ci = d1_alpha = None
        d1_nz = 0
    else:
        tcga_X_tcga_genes = subset_expression(tcga_X_full, all_gene_names, stable_genes['tcga'])
        orien_X_tcga_genes = subset_expression(orien_X_full, all_gene_names, stable_genes['tcga'])

        d1_train_ci, d1_test_ci, d1_nz, d1_alpha, d1_coefs = train_and_evaluate_cox(
            X_train=tcga_X_tcga_genes,
            y_event_train=tcga_event, y_time_train=tcga_time, train_ids=tcga_ids,
            X_test=orien_X_tcga_genes,
            y_event_test=orien_event, y_time_test=orien_time, test_ids=orien_ids,
            l1_ratio=COX_PARAMS['tcga']['l1_ratio'],
            gene_names=stable_genes['tcga'],
            direction_label="tcga_to_orien",
            out_dir=cox_dir,
        )

    # --- Direction 2: ORIEN → TCGA ---
    logger.info("\n  Direction 2: ORIEN → TCGA")
    logger.info(f"  Using {len(stable_genes['orien'])} ORIEN-stable genes")

    if len(stable_genes['orien']) == 0:
        logger.warning("  Direction 2 NOT EVALUABLE: 0 ORIEN-stable genes at this cell.")
        d2_train_ci = d2_test_ci = d2_alpha = None
        d2_nz = 0
    else:
        orien_X_orien_genes = subset_expression(orien_X_full, all_gene_names, stable_genes['orien'])
        tcga_X_orien_genes = subset_expression(tcga_X_full, all_gene_names, stable_genes['orien'])

        d2_train_ci, d2_test_ci, d2_nz, d2_alpha, d2_coefs = train_and_evaluate_cox(
            X_train=orien_X_orien_genes,
            y_event_train=orien_event, y_time_train=orien_time, train_ids=orien_ids,
            X_test=tcga_X_orien_genes,
            y_event_test=tcga_event, y_time_test=tcga_time, test_ids=tcga_ids,
            l1_ratio=COX_PARAMS['orien']['l1_ratio'],
            gene_names=stable_genes['orien'],
            direction_label="orien_to_tcga",
            out_dir=cox_dir,
        )

    def _fmt(v):
        return "n/a" if v is None else f"{v:.4f}"

    _eval_cis = [ci for ci in (d1_test_ci, d2_test_ci) if ci is not None]
    cox_mean_test = float(np.mean(_eval_cis)) if _eval_cis else None

    logger.info(f"\n  Cox Summary:")
    logger.info(f"    TCGA->ORIEN: train={_fmt(d1_train_ci)}, test={_fmt(d1_test_ci)} "
                f"({len(stable_genes['tcga'])} genes)")
    logger.info(f"    ORIEN->TCGA: train={_fmt(d2_train_ci)}, test={_fmt(d2_test_ci)} "
                f"({len(stable_genes['orien'])} genes)")
    logger.info(f"    Mean test C-index over evaluable directions ({len(_eval_cis)}): "
                f"{_fmt(cox_mean_test)}")

    # =========================================================================
    # DeepSurv Bidirectional Validation (union gene set + IG)
    # =========================================================================
    ds_results = {}

    if not args.cox_only:
        logger.info("\n" + "=" * 70)
        logger.info("DEEPSURV BIDIRECTIONAL VALIDATION (union gene set + IG)")
        logger.info("=" * 70)

        ds_dir = results_dir / "deepsurv"
        ds_dir.mkdir(parents=True, exist_ok=True)

        union_genes = stable_genes['union']

        # Subset to union genes
        tcga_X_union = subset_expression(tcga_X_full, all_gene_names, union_genes)
        orien_X_union = subset_expression(orien_X_full, all_gene_names, union_genes)

        # --- Direction 1: TCGA → ORIEN ---
        logger.info(f"\n  Direction 1: TCGA → ORIEN (union {len(union_genes)} genes)")

        ds_d1_train_ci, ds_d1_test_ci, d1_ig_train, d1_ig_test = train_and_evaluate_deepsurv(
            X_train=tcga_X_union,
            y_event_train=tcga_event, y_time_train=tcga_time, train_ids=tcga_ids,
            X_test=orien_X_union,
            y_event_test=orien_event, y_time_test=orien_time, test_ids=orien_ids,
            gene_names=union_genes,
            params=DEEPSURV_PARAMS['tcga'],
            device=device,
            direction_label="tcga_to_orien",
            out_dir=ds_dir,
        )

        # --- Direction 2: ORIEN → TCGA ---
        logger.info(f"\n  Direction 2: ORIEN → TCGA (union {len(union_genes)} genes)")

        ds_d2_train_ci, ds_d2_test_ci, d2_ig_train, d2_ig_test = train_and_evaluate_deepsurv(
            X_train=orien_X_union,
            y_event_train=orien_event, y_time_train=orien_time, train_ids=orien_ids,
            X_test=tcga_X_union,
            y_event_test=tcga_event, y_time_test=tcga_time, test_ids=tcga_ids,
            gene_names=union_genes,
            params=DEEPSURV_PARAMS['orien'],
            device=device,
            direction_label="orien_to_tcga",
            out_dir=ds_dir,
        )

        ds_mean_test = (ds_d1_test_ci + ds_d2_test_ci) / 2

        logger.info(f"\n  DeepSurv Summary:")
        logger.info(f"    TCGA→ORIEN: train={ds_d1_train_ci:.4f}, test={ds_d1_test_ci:.4f}")
        logger.info(f"    ORIEN→TCGA: train={ds_d2_train_ci:.4f}, test={ds_d2_test_ci:.4f}")
        logger.info(f"    Mean test C-index: {ds_mean_test:.4f}")

        ds_results = {
            "tcga_to_orien": {
                "train_ci": float(ds_d1_train_ci),
                "test_ci": float(ds_d1_test_ci),
            },
            "orien_to_tcga": {
                "train_ci": float(ds_d2_train_ci),
                "test_ci": float(ds_d2_test_ci),
            },
            "mean_test_ci": float(ds_mean_test),
            "n_union_genes": len(union_genes),
        }

    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - start_time

    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)

    summary = {
        "phase4_source": {
            "directory": str(args.phase4_dir),
            "alpha_multiplier": args.alpha_mult,
            "pi_threshold": args.pi_thresh,
        },
        "gene_sets": {
            "tcga_stable": len(stable_genes['tcga']),
            "orien_stable": len(stable_genes['orien']),
            "union": len(stable_genes['union']),
            "overlap": len(stable_genes['overlap']),
            "overlap_genes": stable_genes['overlap'],
        },
        "cox_validation": {
            "tcga_to_orien": {
                "n_genes": len(stable_genes['tcga']),
                "evaluable": d1_test_ci is not None,
                "train_ci": None if d1_train_ci is None else float(d1_train_ci),
                "test_ci": None if d1_test_ci is None else float(d1_test_ci),
                "n_nonzero": d1_nz,
                "best_alpha": None if d1_alpha is None else float(d1_alpha),
            },
            "orien_to_tcga": {
                "n_genes": len(stable_genes['orien']),
                "evaluable": d2_test_ci is not None,
                "train_ci": None if d2_train_ci is None else float(d2_train_ci),
                "test_ci": None if d2_test_ci is None else float(d2_test_ci),
                "n_nonzero": d2_nz,
                "best_alpha": None if d2_alpha is None else float(d2_alpha),
            },
            "mean_test_ci": None if cox_mean_test is None else float(cox_mean_test),
            "n_evaluable_directions": len(_eval_cis),
        },
        "deepsurv_validation": ds_results,
        "config": {
            "seed": SEED,
            "cox_inner_cv_folds": COX_INNER_CV_FOLDS,
            "cox_n_alphas": COX_N_ALPHAS,
            "deepsurv_max_epochs": MAX_EPOCHS,
            "deepsurv_early_stopping": EARLY_STOPPING_PATIENCE,
            "ig_n_steps": IG_N_STEPS,
        },
        "runtime_seconds": elapsed,
    }

    with open(results_dir / "phase7_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print summary
    logger.info(f"\n  Phase 4 source: {args.phase4_dir}")
    logger.info(f"  Alpha mult: {args.alpha_mult}, Pi thresh: {args.pi_thresh}")
    logger.info(f"  Gene sets: TCGA={len(stable_genes['tcga'])}, "
                f"ORIEN={len(stable_genes['orien'])}, "
                f"union={len(stable_genes['union'])}, "
                f"overlap={len(stable_genes['overlap'])}")
    logger.info(f"\n  Cox bidirectional validation:")
    logger.info(f"    TCGA→ORIEN ({len(stable_genes['tcga'])} genes): "
                f"test C-index = {_fmt(d1_test_ci)}")
    logger.info(f"    ORIEN→TCGA ({len(stable_genes['orien'])} genes): "
                f"test C-index = {_fmt(d2_test_ci)}")
    logger.info(f"    Mean test C-index: {_fmt(cox_mean_test)}")

    if ds_results:
        logger.info(f"\n  DeepSurv bidirectional validation (union {len(stable_genes['union'])} genes):")
        logger.info(f"    TCGA→ORIEN: test C-index = {ds_results['tcga_to_orien']['test_ci']:.4f}")
        logger.info(f"    ORIEN→TCGA: test C-index = {ds_results['orien_to_tcga']['test_ci']:.4f}")
        logger.info(f"    Mean test C-index: {ds_results['mean_test_ci']:.4f}")

    logger.info(f"\n  Runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"  Results saved to: {results_dir}")
    logger.info("\nPhase 7 complete.")


if __name__ == "__main__":
    main()
