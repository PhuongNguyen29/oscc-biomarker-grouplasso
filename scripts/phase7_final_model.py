#!/usr/bin/env python3
"""
Phase 7: Component 4 — Final Full-Data Model (Stage B)
=======================================================

Train on 100% of each cohort to produce the final consensus genes
for external validation (Phase 8).

Design:
    - 2 methods (elastic net Cox + ElasticDeepSurv)
    - 2 directions (TCGA→ORIEN, ORIEN→TCGA)
    - 1 seed (42, pre-specified)
    - Fixed hyperparameters from Phase 2
    = 4 model training runs total

    For ElasticDeepSurv: 10% internal validation split for early stopping
    (same strategy as Phase 3), then compute IG attributions on ALL
    patients in the opposite cohort.

    For Cox: train on full regularization path, select lambda by
    inner 5-fold CV (same as Phase 3).

Step 7.2: Compare full-data consensus with Phase 4 stability-selected
    consensus via Jaccard overlap.

References:
    - Katzman et al. (2018) BMC Med Res Methodol — DeepSurv
    - Simon et al. (2011) JSS — Regularization paths for Cox
    - Sundararajan et al. (2017) ICML — Integrated Gradients
    - Meinshausen & Buhlmann (2010) JRSS-B — Stability selection

Usage:
    cd ~/hpchome/oscc-biomarker-elasticdeepsurv
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/phase7_final_model.py

Output:
    results/phase7_final_model/
        tcga_to_orien/
            cox_model.pkl
            cox_gene_ranking.csv
            deepsurv_model.pt
            deepsurv_ig_attributions.csv
            deepsurv_gene_ranking.csv
        orien_to_tcga/
            ...
        consensus/
            cox_fulldata_consensus_k1000.txt
            deepsurv_fulldata_consensus_k1000.txt
        phase7_summary.json
        phase7_summary.txt

Author: Phuong Nguyen
Date: February 2026
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
from typing import Dict, Tuple

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

# Suppress verbose sub-module logging
logging.getLogger("src.models.deepsurv").setLevel(logging.ERROR)
logging.getLogger("src.models.elastic_deepsurv").setLevel(logging.ERROR)
logging.getLogger("src.utils.batch_samplers").setLevel(logging.ERROR)
logging.getLogger("src.data.data_loader").setLevel(logging.WARNING)

# =============================================================================
# Configuration
# =============================================================================

SEED = 42  # Pre-specified primary seed
DATA_DIR = "data/processed"
RESULTS_DIR = Path("results/phase7_final_model")
PHASE4_CONSENSUS_DIR = Path("results/phase4_stability_mad3000/consensus")
ANNOTATION_PATH = Path("data/raw/protein_coding_genes_annotation.csv")

# Same k as Phase 4 (no plateau observed, used largest k)
K_SELECTION = 1000

# --- Phase 2 best hyperparameters (identical to Phase 3) ---

DEEPSURV_PARAMS = {
    "tcga": {
        "hidden_sizes": [96, 64],
        "alpha": 1.92e-03,
        "l1_ratio": 0.3,
        "learning_rate": 4.6e-04,
        "dropout": 0.29,
        "batch_size": 24,
        "activation": "relu",
        "batch_norm": False,
    },
    "orien": {
        "hidden_sizes": [128, 96],
        "alpha": 5.60e-03,
        "l1_ratio": 0.9,
        "learning_rate": 4.6e-04,
        "dropout": 0.35,
        "batch_size": 64,
        "activation": "relu",
        "batch_norm": True,
    },
}

COX_PARAMS = {
    "tcga": {"l1_ratio": 0.7},
    "orien": {"l1_ratio": 1.0},
}

# Training config (identical to Phase 3)
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 20
SCHEDULER_PATIENCE = 10
WEIGHT_DECAY = 0.0
COX_N_ALPHAS = 100
COX_INNER_CV_FOLDS = 5

# IG config (identical to Phase 3b)
IG_N_STEPS = 300
IG_BATCH_SIZE = 32


# =============================================================================
# Helper: stratification for CV and train/val splits
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
    """Wraps StratifiedBatchSampler, filters batches with only 1 sample."""

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
        return len(self.inner)


def make_loader(dataset, batch_size, shuffle=True):
    """Create DataLoader with StratifiedBatchSampler."""
    sampler = _SafeBatchSampler(
        events=dataset.y_event,
        batch_size=batch_size,
        min_events_per_batch=2,
        shuffle=shuffle,
    )
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=True)


# =============================================================================
# Cox: train on full data, rank genes
# =============================================================================

def train_cox_full(X, y_event, y_time, sample_ids, l1_ratio, gene_names, out_dir):
    """
    Train elastic net Cox on 100% of cohort data.
    Select lambda by inner 5-fold CV (same as Phase 3).

    Returns:
        gene_ranking: list of (gene_id, abs_coef) sorted descending
        model, best_alpha
    """
    logger.info("    Fitting Cox regularization path...")

    # Structured array for sksurv
    y = np.array(
        [(bool(e), t) for e, t in zip(y_event, y_time)],
        dtype=[("event", bool), ("time", float)],
    )

    # Fit full regularization path on ALL data
    model = CoxnetSurvivalAnalysis(
        l1_ratio=l1_ratio,
        n_alphas=COX_N_ALPHAS,
        normalize=False,
        tol=1e-7,
        max_iter=100000,
    )
    model.fit(X, y)

    outer_alphas = model.alphas_

    # Inner CV to select best lambda
    strata = create_strata(y_event, y_time)
    skf = StratifiedKFold(
        n_splits=COX_INNER_CV_FOLDS, shuffle=True, random_state=SEED
    )

    alpha_scores = {i: [] for i in range(len(outer_alphas))}

    for train_idx, valid_idx in skf.split(X, strata):
        inner_model = CoxnetSurvivalAnalysis(
            l1_ratio=l1_ratio,
            alphas=outer_alphas,
            normalize=False,
            tol=1e-7,
            max_iter=100000,
        )
        try:
            inner_model.fit(X[train_idx], y[train_idx])
        except Exception as e:
            logger.warning(f"    Inner CV fold failed: {e}")
            continue

        for i, alpha_val in enumerate(inner_model.alphas_):
            try:
                pred = inner_model.predict(X[valid_idx], alpha=alpha_val)
                ci, _, _, _, _ = concordance_index_censored(
                    y[valid_idx]["event"], y[valid_idx]["time"], pred
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
        logger.warning("    Inner CV could not select alpha, using smallest lambda")

    logger.info(f"    Best alpha: {best_alpha:.6e}, inner CV C-index: {best_mean_ci:.4f}")

    # Extract coefficients at best alpha
    alpha_idx = np.argmin(np.abs(model.alphas_ - best_alpha))
    coefs = model.coef_[:, alpha_idx]
    n_nonzero = int(np.sum(coefs != 0))
    logger.info(f"    Non-zero genes: {n_nonzero}/{len(gene_names)}")

    # Gene ranking by |coefficient|
    abs_coefs = np.abs(coefs)
    rank_order = np.argsort(-abs_coefs)
    gene_ranking = [(gene_names[i], float(abs_coefs[i]), float(coefs[i])) for i in rank_order]

    # Save
    ranking_df = pd.DataFrame(gene_ranking, columns=["gene_id", "abs_coef", "coef"])
    ranking_df["rank"] = range(1, len(ranking_df) + 1)
    ranking_df.to_csv(out_dir / "cox_gene_ranking.csv", index=False)

    with open(out_dir / "cox_model.pkl", "wb") as f:
        pickle.dump({"model": model, "best_alpha": best_alpha}, f)

    # Training C-index (in-sample, for reporting)
    train_pred = model.predict(X, alpha=best_alpha)
    train_ci, _, _, _, _ = concordance_index_censored(y["event"], y["time"], train_pred)
    logger.info(f"    Training C-index (in-sample): {train_ci:.4f}")

    return gene_ranking, model, best_alpha, train_ci


# =============================================================================
# DeepSurv: train on full data, compute IG, rank genes
# =============================================================================

def train_deepsurv_full(
    X, y_event, y_time, sample_ids, gene_names,
    params, device, out_dir,
    X_target, target_ids, target_gene_names,
):
    """
    Train ElasticDeepSurv on 100% of cohort data.
    Uses 10% internal validation split for early stopping (same as Phase 3).
    Compute IG attributions on ALL patients in the TARGET cohort.

    Returns:
        gene_ranking: list of (gene_id, mean_abs_ig)
        model, ig_df
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    n_features = X.shape[1]

    # --- 10% internal validation split for early stopping ---
    strata_es = create_strata(y_event, y_time)
    tr_idx, es_idx = train_test_split(
        np.arange(len(X)),
        test_size=0.1,
        stratify=strata_es,
        random_state=SEED,
    )

    logger.info(f"    Train: {len(tr_idx)} samples, ES holdout: {len(es_idx)} samples")

    train_ds = FoldDataset(
        X[tr_idx], y_time[tr_idx], y_event[tr_idx],
        [sample_ids[i] for i in tr_idx], gene_names,
    )
    es_ds = FoldDataset(
        X[es_idx], y_time[es_idx], y_event[es_idx],
        [sample_ids[i] for i in es_idx], gene_names,
    )

    train_loader = make_loader(train_ds, params["batch_size"], shuffle=True)
    es_loader = make_loader(es_ds, params["batch_size"], shuffle=False)

    # --- Create and train model ---
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

    # Manual training loop with early stopping (same as Phase 3)
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

    logger.info(f"    Best epoch: {best_epoch}, ES C-index: {best_cindex:.4f}")

    # Save model
    torch.save(model.state_dict(), out_dir / "deepsurv_model.pt")

    # --- Compute IG on TARGET cohort (all patients) ---
    logger.info(f"    Computing IG attributions ({X_target.shape[0]} target patients)...")

    model.eval()
    model.to(device)

    ig = IntegratedGradients(model)
    baseline = torch.zeros(1, n_features).to(device)
    X_target_tensor = torch.from_numpy(X_target.astype(np.float32)).to(device)

    all_attr = []
    for start in range(0, len(X_target), IG_BATCH_SIZE):
        end = min(start + IG_BATCH_SIZE, len(X_target))
        batch_input = X_target_tensor[start:end]
        batch_baseline = baseline.expand(batch_input.shape[0], -1)

        attr = ig.attribute(
            batch_input,
            baselines=batch_baseline,
            n_steps=IG_N_STEPS,
            method="gausslegendre",
        )
        all_attr.append(attr.cpu().detach().numpy())

    attributions = np.concatenate(all_attr, axis=0)

    ig_df = pd.DataFrame(
        attributions,
        index=target_ids,
        columns=target_gene_names,
    )
    ig_df.index.name = "sampleID"
    ig_df.to_csv(out_dir / "deepsurv_ig_attributions.csv")

    logger.info(
        f"    IG computed: {ig_df.shape[0]} patients × {ig_df.shape[1]} genes"
    )

    # --- Gene ranking by mean |IG| ---
    mean_abs_ig = ig_df.abs().mean(axis=0).sort_values(ascending=False)
    gene_ranking = [
        (gene, float(mean_abs_ig[gene])) for gene in mean_abs_ig.index
    ]

    ranking_df = pd.DataFrame(gene_ranking, columns=["gene_id", "mean_abs_ig"])
    ranking_df["rank"] = range(1, len(ranking_df) + 1)
    ranking_df.to_csv(out_dir / "deepsurv_gene_ranking.csv", index=False)

    # Move model to CPU for return
    model.cpu()

    return gene_ranking, model, ig_df


# =============================================================================
# Consensus comparison (Step 7.2)
# =============================================================================

def compute_consensus_and_compare(
    ranking_dir1, ranking_dir2, method, k, phase4_consensus_path, out_dir
):
    """
    Take top-k genes from each direction, intersect, compare with Phase 4.

    Args:
        ranking_dir1: Path containing {method}_gene_ranking.csv for direction 1
        ranking_dir2: Path containing {method}_gene_ranking.csv for direction 2
        method: 'cox' or 'deepsurv'
        k: number of top genes to select
        phase4_consensus_path: Path to Phase 4 consensus gene list
        out_dir: Path to save results

    Returns:
        dict with consensus genes and Jaccard comparison
    """
    col_name = "abs_coef" if method == "cox" else "mean_abs_ig"

    # Load rankings
    r1 = pd.read_csv(ranking_dir1 / f"{method}_gene_ranking.csv")
    r2 = pd.read_csv(ranking_dir2 / f"{method}_gene_ranking.csv")

    # Top-k from each direction
    top_k_d1 = set(r1.nlargest(k, col_name)["gene_id"].tolist())
    top_k_d2 = set(r2.nlargest(k, col_name)["gene_id"].tolist())

    # Intersection = full-data consensus
    fulldata_consensus = sorted(top_k_d1 & top_k_d2)

    logger.info(f"    {method.upper()} full-data consensus:")
    logger.info(f"      Direction 1 top-{k}: {len(top_k_d1)} genes")
    logger.info(f"      Direction 2 top-{k}: {len(top_k_d2)} genes")
    logger.info(f"      Intersection: {len(fulldata_consensus)} genes")

    # Save full-data consensus
    consensus_file = out_dir / f"{method}_fulldata_consensus_k{k}.txt"
    with open(consensus_file, "w") as f:
        for gene in fulldata_consensus:
            f.write(gene + "\n")

    # Compare with Phase 4 stability-selected consensus
    if phase4_consensus_path.exists():
        with open(phase4_consensus_path) as f:
            phase4_genes = set(line.strip() for line in f if line.strip())

        overlap = set(fulldata_consensus) & phase4_genes
        union = set(fulldata_consensus) | phase4_genes

        jaccard = len(overlap) / len(union) if len(union) > 0 else 0.0

        logger.info(f"      Phase 4 stability consensus: {len(phase4_genes)} genes")
        logger.info(f"      Overlap with Phase 4: {len(overlap)} genes")
        logger.info(f"      Jaccard: {jaccard:.4f}")

        if jaccard < 0.50:
            logger.info(
                f"      → Jaccard < 0.50: USE Phase 4 stability-selected set (more defensible)"
            )
            recommendation = "use_phase4"
        else:
            logger.info(
                f"      → Jaccard >= 0.50: USE full-data consensus set"
            )
            recommendation = "use_fulldata"
    else:
        phase4_genes = set()
        overlap = set()
        jaccard = None
        recommendation = "phase4_not_found"
        logger.warning(f"      Phase 4 consensus file not found: {phase4_consensus_path}")

    return {
        "method": method,
        "k": k,
        "n_fulldata_consensus": len(fulldata_consensus),
        "fulldata_genes": fulldata_consensus,
        "n_phase4_consensus": len(phase4_genes),
        "n_overlap": len(overlap),
        "jaccard": jaccard,
        "recommendation": recommendation,
        "overlap_genes": sorted(overlap),
        "fulldata_only": sorted(set(fulldata_consensus) - phase4_genes),
        "phase4_only": sorted(phase4_genes - set(fulldata_consensus)),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    import time

    start_time = time.time()

    logger.info("=" * 70)
    logger.info("Phase 7: Final Full-Data Model (Stage B)")
    logger.info("=" * 70)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Seed: {SEED}")
    logger.info(f"k for gene selection: {K_SELECTION}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Log to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(RESULTS_DIR / f"phase7_{timestamp}.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    # --- Load data ---
    logger.info("\n--- Loading data ---")
    tcga_expr, tcga_surv = load_cohort("tcga", DATA_DIR)
    orien_expr, orien_surv = load_cohort("orien", DATA_DIR)

    gene_names = tcga_expr.index.tolist()
    n_features = len(gene_names)

    # Verify gene order consistency
    assert list(orien_expr.index) == gene_names, "Gene order mismatch"

    # Prepare arrays (samples × genes)
    tcga_X = tcga_expr.values.T.astype(np.float32)
    tcga_event = tcga_surv["event"].values.astype(np.float32)
    tcga_time = tcga_surv["time"].values.astype(np.float32)
    tcga_ids = tcga_surv.index.tolist()

    orien_X = orien_expr.values.T.astype(np.float32)
    orien_event = orien_surv["event"].values.astype(np.float32)
    orien_time = orien_surv["time"].values.astype(np.float32)
    orien_ids = orien_surv.index.tolist()

    logger.info(
        f"TCGA: {tcga_X.shape[0]} samples × {tcga_X.shape[1]} genes, "
        f"{int(tcga_event.sum())} events"
    )
    logger.info(
        f"ORIEN: {orien_X.shape[0]} samples × {orien_X.shape[1]} genes, "
        f"{int(orien_event.sum())} events"
    )

    # --- Direction 1: TCGA → ORIEN ---
    logger.info("\n" + "=" * 70)
    logger.info("Direction 1: TCGA → ORIEN (train on TCGA, IG on ORIEN)")
    logger.info("=" * 70)

    d1_dir = RESULTS_DIR / "tcga_to_orien"
    d1_dir.mkdir(parents=True, exist_ok=True)

    # Cox
    logger.info("\n  --- Cox ---")
    d1_cox_ranking, d1_cox_model, d1_cox_alpha, d1_cox_ci = train_cox_full(
        X=tcga_X,
        y_event=tcga_event,
        y_time=tcga_time,
        sample_ids=tcga_ids,
        l1_ratio=COX_PARAMS["tcga"]["l1_ratio"],
        gene_names=gene_names,
        out_dir=d1_dir,
    )

    # DeepSurv
    logger.info("\n  --- DeepSurv ---")
    d1_ds_ranking, d1_ds_model, d1_ig_df = train_deepsurv_full(
        X=tcga_X,
        y_event=tcga_event,
        y_time=tcga_time,
        sample_ids=tcga_ids,
        gene_names=gene_names,
        params=DEEPSURV_PARAMS["tcga"],
        device=device,
        out_dir=d1_dir,
        X_target=orien_X,
        target_ids=orien_ids,
        target_gene_names=gene_names,
    )

    # --- Direction 2: ORIEN → TCGA ---
    logger.info("\n" + "=" * 70)
    logger.info("Direction 2: ORIEN → TCGA (train on ORIEN, IG on TCGA)")
    logger.info("=" * 70)

    d2_dir = RESULTS_DIR / "orien_to_tcga"
    d2_dir.mkdir(parents=True, exist_ok=True)

    # Cox
    logger.info("\n  --- Cox ---")
    d2_cox_ranking, d2_cox_model, d2_cox_alpha, d2_cox_ci = train_cox_full(
        X=orien_X,
        y_event=orien_event,
        y_time=orien_time,
        sample_ids=orien_ids,
        l1_ratio=COX_PARAMS["orien"]["l1_ratio"],
        gene_names=gene_names,
        out_dir=d2_dir,
    )

    # DeepSurv
    logger.info("\n  --- DeepSurv ---")
    d2_ds_ranking, d2_ds_model, d2_ig_df = train_deepsurv_full(
        X=orien_X,
        y_event=orien_event,
        y_time=orien_time,
        sample_ids=orien_ids,
        gene_names=gene_names,
        params=DEEPSURV_PARAMS["orien"],
        device=device,
        out_dir=d2_dir,
        X_target=tcga_X,
        target_ids=tcga_ids,
        target_gene_names=gene_names,
    )

    # --- Step 7.2: Consensus comparison ---
    logger.info("\n" + "=" * 70)
    logger.info("Step 7.2: Consensus Gene Set Comparison")
    logger.info("=" * 70)

    consensus_dir = RESULTS_DIR / "consensus"
    consensus_dir.mkdir(parents=True, exist_ok=True)

    # Cox consensus
    logger.info("\n  --- Cox consensus ---")
    cox_comparison = compute_consensus_and_compare(
        ranking_dir1=d1_dir,
        ranking_dir2=d2_dir,
        method="cox",
        k=K_SELECTION,
        phase4_consensus_path=PHASE4_CONSENSUS_DIR / "cox_consensus_pi080.txt",
        out_dir=consensus_dir,
    )

    # DeepSurv consensus
    logger.info("\n  --- DeepSurv consensus ---")
    ds_comparison = compute_consensus_and_compare(
        ranking_dir1=d1_dir,
        ranking_dir2=d2_dir,
        method="deepsurv",
        k=K_SELECTION,
        phase4_consensus_path=PHASE4_CONSENSUS_DIR / "deepsurv_consensus_pi080.txt",
        out_dir=consensus_dir,
    )

    # --- Summary ---
    elapsed = time.time() - start_time

    summary = {
        "seed": SEED,
        "k_selection": K_SELECTION,
        "n_features": n_features,
        "tcga_n": len(tcga_ids),
        "tcga_events": int(tcga_event.sum()),
        "orien_n": len(orien_ids),
        "orien_events": int(orien_event.sum()),
        "direction_1": {
            "cox_train_cindex": d1_cox_ci,
            "cox_best_alpha": d1_cox_alpha,
            "deepsurv_best_epoch": "see_log",
        },
        "direction_2": {
            "cox_train_cindex": d2_cox_ci,
            "cox_best_alpha": d2_cox_alpha,
            "deepsurv_best_epoch": "see_log",
        },
        "cox_consensus": {
            "n_fulldata": cox_comparison["n_fulldata_consensus"],
            "n_phase4": cox_comparison["n_phase4_consensus"],
            "n_overlap": cox_comparison["n_overlap"],
            "jaccard": cox_comparison["jaccard"],
            "recommendation": cox_comparison["recommendation"],
        },
        "deepsurv_consensus": {
            "n_fulldata": ds_comparison["n_fulldata_consensus"],
            "n_phase4": ds_comparison["n_phase4_consensus"],
            "n_overlap": ds_comparison["n_overlap"],
            "jaccard": ds_comparison["jaccard"],
            "recommendation": ds_comparison["recommendation"],
        },
        "runtime_seconds": elapsed,
    }

    with open(RESULTS_DIR / "phase7_summary.json", "w") as f:
        json.dump(
            summary, f, indent=2,
            default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o,
        )

    # Text summary
    summary_lines = [
        "=" * 70,
        "Phase 7: Final Full-Data Model — Summary",
        "=" * 70,
        "",
        f"Seed: {SEED}",
        f"k for gene selection: {K_SELECTION}",
        f"Features: {n_features} genes",
        "",
        "--- Direction 1: TCGA → ORIEN ---",
        f"  Cox training C-index: {d1_cox_ci:.4f}",
        f"  Cox best alpha: {d1_cox_alpha:.6e}",
        "",
        "--- Direction 2: ORIEN → TCGA ---",
        f"  Cox training C-index: {d2_cox_ci:.4f}",
        f"  Cox best alpha: {d2_cox_alpha:.6e}",
        "",
        "--- Cox Consensus (k={}) ---".format(K_SELECTION),
        f"  Full-data consensus: {cox_comparison['n_fulldata_consensus']} genes",
        f"  Phase 4 stability consensus: {cox_comparison['n_phase4_consensus']} genes",
        f"  Overlap: {cox_comparison['n_overlap']} genes",
        f"  Jaccard: {cox_comparison['jaccard']:.4f}",
        f"  Recommendation: {cox_comparison['recommendation']}",
        "",
        "--- DeepSurv Consensus (k={}) ---".format(K_SELECTION),
        f"  Full-data consensus: {ds_comparison['n_fulldata_consensus']} genes",
        f"  Phase 4 stability consensus: {ds_comparison['n_phase4_consensus']} genes",
        f"  Overlap: {ds_comparison['n_overlap']} genes",
        f"  Jaccard: {ds_comparison['jaccard']:.4f}" if ds_comparison['jaccard'] is not None else "  Jaccard: N/A",
        f"  Recommendation: {ds_comparison['recommendation']}",
        "",
        f"Runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)",
    ]

    summary_text = "\n".join(summary_lines)
    logger.info("\n" + summary_text)

    with open(RESULTS_DIR / "phase7_summary.txt", "w") as f:
        f.write(summary_text)

    logger.info(f"\nResults saved to: {RESULTS_DIR}")
    logger.info("Phase 7 complete.")


if __name__ == "__main__":
    main()
