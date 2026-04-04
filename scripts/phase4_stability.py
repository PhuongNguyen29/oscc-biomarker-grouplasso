"""
Phase 4: Stability Selection with IQR Pre-Filtering (Component 3)

Motivation:
    At p=14,675, bidirectional stability selection yields near-empty consensus
    because TCGA (n=339, ~270 per subsample) produces noisy rankings when
    p/n ≈ 54. IQR filtering reduces p to ~4,448 genes using expression
    variability only (no survival outcome), preserving the decoupled design.

IQR filtering rationale:
    Genes with low expression variability across samples cannot contribute
    meaningfully to survival prediction — they carry minimal discriminative
    information. IQR is robust to outliers and is the default variability
    metric in the Bioconductor genefilter package. Per-cohort median IQR
    filtering followed by intersection ensures retained genes are variable
    in BOTH cohorts, providing an implicit cross-cohort filter.

    The IQR-filtered gene list is pre-computed by scripts/filter_genes_iqr.py
    and loaded from data/processed/iqr_filtered/retained_genes.txt.

All other Phase 4 logic (subsampling, Ridge Cox, DeepSurv+IG, selection
probabilities, consensus, plateau rule) is identical to the original
phase4_stability.py.

References:
    - Bourgon et al. (2010) PNAS — Independent filtering principle
    - Bioconductor genefilter — IQR with 0.5 quantile default
    - Meinshausen & Buhlmann (2010) JRSS-B — Stability selection
    - Sundararajan et al. (2017) ICML — Integrated Gradients
    - Hoerl & Kennard (1970) Technometrics — Ridge regression

Usage:
    cd ~/hpchome/oscc-biomarker-grouplasso
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/phase4_stability.py

    # Single cohort:
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/phase4_stability.py --cohort tcga

Output:
    results/phase4_stability_iqr/
        ridge_lambda_selection.json
        subsample_rankings/  (same structure as original)
        selection_probabilities/  (same structure as original)
        consensus/  (same structure as original)
        summary.json
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
from typing import Dict, List, Tuple

from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

from src.data.data_loader import load_cohort
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

logging.getLogger('src.models.deepsurv').setLevel(logging.ERROR)
logging.getLogger('src.models.elastic_deepsurv').setLevel(logging.ERROR)
logging.getLogger('src.utils.batch_samplers').setLevel(logging.ERROR)
logging.getLogger('src.data.dataset').setLevel(logging.ERROR)
logging.getLogger('src.data.data_loader').setLevel(logging.WARNING)

# =============================================================================
# Configuration
# =============================================================================

B_SUBSAMPLES = 50
SUBSAMPLE_FRACTION = 0.80
RANDOM_SEED = 42
DATA_DIR = "data/processed"
RESULTS_BASE_DEFAULT = "results/phase4_stability_iqr"

# k scan values
K_SCAN = [50, 100, 150, 200, 300, 500, 750, 1000]

# Stability thresholds
PI_THRESHOLDS = [0.70, 0.80, 0.90]
PI_PRIMARY = 0.80

# Ridge Cox: near-pure L2 penalty via CoxnetSurvivalAnalysis(l1_ratio=1e-10)
RIDGE_L1_RATIO = 1e-10
RIDGE_N_ALPHAS = 100
RIDGE_CV_FOLDS = 3

# DeepSurv hyperparameters (from Phase 2)
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

DEEPSURV_PARAMS = load_phase2_deepsurv_params()

# DeepSurv training config
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 20
SCHEDULER_PATIENCE = 10
WEIGHT_DECAY = 0.0

# IG config
IG_N_STEPS = 300


# =============================================================================
# Helpers
# =============================================================================

def create_strata(y_event, y_time):
    """Event × time quartile stratification."""
    time_quartiles = pd.qcut(y_time, q=4, labels=False, duplicates='drop')
    return y_event.astype(int) * 4 + time_quartiles


class FoldDataset:
    """Lightweight dataset for training/OOB subsets."""
    def __init__(self, X, y_time, y_event, gene_names):
        self.X = X.astype(np.float32)
        self.y_time = y_time.astype(np.float32)
        self.y_event = y_event.astype(np.float32)
        self.gene_names = gene_names
        self.n_features = X.shape[1]
        self.sample_ids = [f"s_{i}" for i in range(len(X))]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'features': torch.from_numpy(self.X[idx]),
            'time': torch.tensor(self.y_time[idx], dtype=torch.float32),
            'event': torch.tensor(self.y_event[idx], dtype=torch.float32),
            'sample_id': self.sample_ids[idx],
        }


class _SafeBatchSampler:
    """Wraps StratifiedBatchSampler, drops batches with only 1 sample."""
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


# =============================================================================
# Step 1: Ridge lambda selection (once per cohort)
# =============================================================================

def select_ridge_lambda(X, y_event, y_time, cohort_name):
    """
    Select ridge alpha via warm-started path fitting with 3-fold CV.

    Fits the entire alpha path in a single call (warm-started coordinate
    descent — much faster than individual fits). If small alphas cause
    numerical overflow, retries with a higher alpha floor.
    """
    logger.info(f"  Selecting ridge alpha for {cohort_name}...")

    y_struct = np.array(
        [(bool(e), t) for e, t in zip(y_event, y_time)],
        dtype=[('event', bool), ('time', float)]
    )

    # 3-fold CV
    strata = create_strata(y_event, y_time)
    skf = StratifiedKFold(n_splits=RIDGE_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    folds = list(skf.split(X, strata))

    # Try progressively higher alpha floors until path fits succeed
    floors = [-1, 0, 0.5, 1.0, 1.5, 2.0]
    alpha_scores = None
    used_alphas = None

    for floor in floors:
        alphas = np.logspace(floor, 3, 20)
        all_fold_scores = {i: [] for i in range(len(alphas))}
        n_successful_folds = 0

        for fold_idx, (train_idx, valid_idx) in enumerate(folds):
            try:
                model = CoxnetSurvivalAnalysis(
                    l1_ratio=RIDGE_L1_RATIO,
                    alphas=alphas,
                    normalize=False,
                    tol=1e-7,
                    max_iter=100000,
                )
                model.fit(X[train_idx], y_struct[train_idx])
                n_successful_folds += 1

                for i, alpha_val in enumerate(model.alphas_):
                    try:
                        pred = model.predict(X[valid_idx], alpha=alpha_val)
                        ci, _, _, _, _ = concordance_index_censored(
                            y_struct[valid_idx]['event'],
                            y_struct[valid_idx]['time'], pred
                        )
                        all_fold_scores[i].append(ci)
                    except Exception:
                        continue

            except ArithmeticError:
                logger.info(f"    Floor={floor}: fold {fold_idx+1} overflow, skipping")
                continue

        if n_successful_folds >= 1:
            alpha_scores = all_fold_scores
            used_alphas = alphas
            logger.info(f"    Floor={floor}: {n_successful_folds}/{RIDGE_CV_FOLDS} folds OK")
            break
        else:
            logger.info(f"    Floor={floor}: all folds failed, raising floor")

    # Best alpha by mean CV C-index
    best_alpha = None
    best_ci = 0.0
    if alpha_scores is not None:
        for i, scores in alpha_scores.items():
            if len(scores) >= 1:
                mean_ci = np.mean(scores)
                if mean_ci > best_ci:
                    best_ci = mean_ci
                    best_alpha = used_alphas[i]

    if best_alpha is None:
        best_alpha = 10.0
        logger.warning(f"  Ridge CV failed for {cohort_name}, using alpha=10.0")

    # Log top-5 alphas
    if alpha_scores is not None:
        scored = [(used_alphas[i], np.mean(s), len(s))
                  for i, s in alpha_scores.items() if len(s) >= 1]
        scored.sort(key=lambda x: x[1], reverse=True)
        for alpha_val, ci, nf in scored[:5]:
            marker = " ← best" if alpha_val == best_alpha else ""
            logger.info(f"    alpha={alpha_val:>10.4f}: CV C-index={ci:.4f} "
                        f"({nf} folds){marker}")

    # Verify nonzero count on full data
    try:
        full_model = CoxnetSurvivalAnalysis(
            l1_ratio=RIDGE_L1_RATIO,
            alphas=[best_alpha],
            normalize=False,
            tol=1e-7,
            max_iter=100000,
        )
        full_model.fit(X, y_struct)
        coefs = full_model.coef_[:, 0]
        n_nonzero = int(np.sum(coefs != 0))
        n_total = len(coefs)
    except ArithmeticError:
        best_alpha = best_alpha * 2
        logger.warning(f"  Numerical error, doubling to alpha={best_alpha:.6f}")
        full_model = CoxnetSurvivalAnalysis(
            l1_ratio=RIDGE_L1_RATIO,
            alphas=[best_alpha],
            normalize=False,
            tol=1e-7,
            max_iter=100000,
        )
        full_model.fit(X, y_struct)
        coefs = full_model.coef_[:, 0]
        n_nonzero = int(np.sum(coefs != 0))
        n_total = len(coefs)

    logger.info(f"  {cohort_name}: best alpha={best_alpha:.6f}, "
                f"CV C-index={best_ci:.4f}, "
                f"nonzero={n_nonzero}/{n_total} genes")

    return best_alpha, best_ci


# =============================================================================
# Step 2: Generate subsamples
# =============================================================================

def generate_subsamples(n_samples, y_event, n_subsamples, fraction, seed):
    """
    Generate stratified subsamples.

    Each subsample contains ~fraction of the cohort, stratified by event status
    to maintain event rates.
    """
    rng = np.random.RandomState(seed)
    subsamples = []

    for b in range(n_subsamples):
        # Stratified subsampling
        event_idx = np.where(y_event == 1)[0]
        censor_idx = np.where(y_event == 0)[0]

        n_event = int(len(event_idx) * fraction)
        n_censor = int(len(censor_idx) * fraction)

        sub_event = rng.choice(event_idx, size=n_event, replace=False)
        sub_censor = rng.choice(censor_idx, size=n_censor, replace=False)

        train_idx = np.sort(np.concatenate([sub_event, sub_censor]))
        oob_idx = np.sort(np.setdiff1d(np.arange(n_samples), train_idx))

        subsamples.append({
            'train_idx': train_idx,
            'oob_idx': oob_idx,
            'n_train': len(train_idx),
            'n_oob': len(oob_idx),
            'n_train_events': len(sub_event),
            'n_oob_events': int(y_event[oob_idx].sum()),
        })

    return subsamples


# =============================================================================
# Step 3a: Ridge Cox ranking per subsample
# =============================================================================

def rank_genes_cox(X_train, y_event_train, y_time_train, gene_names,
                   fixed_alpha):
    """
    Train ridge Cox on subsample, rank genes by |coefficient|.

    Uses CoxnetSurvivalAnalysis(l1_ratio~0) with a single fixed alpha.
    If numerical overflow occurs, doubles alpha until fit succeeds.

    Returns:
        ranking: list of (gene_name, importance_score) sorted by |coef| descending.
        signed_coefs: dict mapping gene_name → signed coefficient value.
    """
    y_struct = np.array(
        [(bool(e), t) for e, t in zip(y_event_train, y_time_train)],
        dtype=[('event', bool), ('time', float)]
    )

    alpha = fixed_alpha
    for attempt in range(5):
        try:
            model = CoxnetSurvivalAnalysis(
                l1_ratio=RIDGE_L1_RATIO,
                alphas=[alpha],
                normalize=False,
                tol=1e-7,
                max_iter=100000,
            )
            model.fit(X_train, y_struct)
            break
        except ArithmeticError:
            alpha = alpha * 2
            if attempt == 4:
                raise RuntimeError(f"Cox fit failed after 5 attempts, alpha={alpha}")

    # Get signed coefficients (single alpha → coef_ is (n_features, 1))
    raw_coefs = model.coef_[:, 0]
    abs_coefs = np.abs(raw_coefs)

    # Signed coefficient dict for sign concordance analysis
    signed_coefs = dict(zip(gene_names, raw_coefs))

    # Rank by |coefficient| descending
    ranking = sorted(
        zip(gene_names, abs_coefs),
        key=lambda x: x[1],
        reverse=True,
    )

    return ranking, signed_coefs


# =============================================================================
# Step 3b: DeepSurv + OOB IG ranking per subsample
# =============================================================================

def rank_genes_deepsurv_ig(
    X_train, y_event_train, y_time_train,
    X_oob,
    gene_names, params, device, train_seed,
):
    """
    Train ElasticDeepSurv on subsample, compute IG on OOB patients,
    rank genes by mean |IG|.

    Returns: list of (gene_name, importance_score) sorted descending.
    """
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(train_seed)

    n_features = X_train.shape[1]

    # Internal ES split (10% of training for early stopping)
    from sklearn.model_selection import train_test_split
    strata = create_strata(y_event_train, y_time_train)
    tr_idx, es_idx = train_test_split(
        np.arange(len(X_train)), test_size=0.1,
        stratify=strata, random_state=train_seed,
    )

    train_ds = FoldDataset(
        X_train[tr_idx], y_time_train[tr_idx], y_event_train[tr_idx], gene_names
    )
    es_ds = FoldDataset(
        X_train[es_idx], y_time_train[es_idx], y_event_train[es_idx], gene_names
    )

    train_loader = make_loader(train_ds, params['batch_size'], shuffle=True)
    es_loader = make_loader(es_ds, params['batch_size'], shuffle=False)

    # Create and train model
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

    best_cindex = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        trainer.train_epoch(train_loader)
        _, _, _, es_cindex = trainer.evaluate(es_loader)
        trainer.scheduler.step(es_cindex)

        if es_cindex > best_cindex:
            best_cindex = es_cindex
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Compute IG on OOB patients
    model.eval()
    model.to(device)

    from captum.attr import IntegratedGradients
    ig = IntegratedGradients(model)

    # Zero baseline (population mean after z-scoring)
    baseline = torch.zeros(1, n_features).to(device)

    # Compute IG for each OOB patient
    all_attributions = []

    X_oob_tensor = torch.from_numpy(X_oob.astype(np.float32)).to(device)

    # Process in batches to manage GPU memory
    ig_batch_size = 32
    for start in range(0, len(X_oob), ig_batch_size):
        end = min(start + ig_batch_size, len(X_oob))
        batch_input = X_oob_tensor[start:end]
        batch_baseline = baseline.expand(batch_input.shape[0], -1)

        attr = ig.attribute(
            batch_input,
            baselines=batch_baseline,
            n_steps=IG_N_STEPS,
            method='gausslegendre',
        )
        all_attributions.append(attr.cpu().detach().numpy())

    attributions = np.concatenate(all_attributions, axis=0)  # (n_oob, n_features)

    # Rank by mean |IG| across OOB patients
    mean_abs_ig = np.mean(np.abs(attributions), axis=0)  # (n_features,)

    ranking = sorted(
        zip(gene_names, mean_abs_ig),
        key=lambda x: x[1],
        reverse=True,
    )

    model.cpu()

    return ranking


# =============================================================================
# Step 4: Compute selection probabilities
# =============================================================================

def compute_selection_probs(all_rankings, gene_names, k_values):
    """
    For each gene, compute the proportion of subsamples where it appears
    in the top-k, for each k.

    Args:
        all_rankings: list of B rankings, each a list of (gene, score) pairs
        gene_names: list of all gene names
        k_values: list of k values to evaluate

    Returns:
        DataFrame: genes × k_values, values are selection probabilities
    """
    B = len(all_rankings)
    probs = pd.DataFrame(0.0, index=gene_names, columns=[f"k_{k}" for k in k_values])

    for ranking in all_rankings:
        ranked_genes = [g for g, _ in ranking]
        for k in k_values:
            top_k = set(ranked_genes[:k])
            for gene in top_k:
                if gene in probs.index:
                    probs.loc[gene, f"k_{k}"] += 1

    probs = probs / B
    return probs


# =============================================================================
# Step 5: Consensus and plateau rule
# =============================================================================

def find_consensus(probs_dir1, probs_dir2, k_values, pi):
    """
    For each k, find genes with selection probability >= pi in BOTH directions.
    """
    consensus_sizes = {}
    consensus_genes = {}

    for k in k_values:
        col = f"k_{k}"
        stable_d1 = set(probs_dir1.index[probs_dir1[col] >= pi])
        stable_d2 = set(probs_dir2.index[probs_dir2[col] >= pi])
        consensus = stable_d1 & stable_d2
        consensus_sizes[k] = len(consensus)
        consensus_genes[k] = sorted(consensus)

    return consensus_sizes, consensus_genes


def apply_plateau_rule(consensus_sizes, k_values):
    """
    Hybrid plateau rule: select the smallest k where the plateau condition
    holds for TWO consecutive steps (k_i→k_{i+1} and k_{i+1}→k_{i+2}).

    Plateau condition for a step:
      - consensus size does not decrease (C_next >= C_curr)
      - relative change < 5%
      - absolute change <= 2 genes

    Returns:
        selected_k: chosen k (or k_max if never triggered)
        plateau_triggered: bool
    """
    def step_plateau(c_curr, c_next):
        if c_curr <= 0:
            return False
        if c_next < c_curr:
            return False
        abs_change = c_next - c_curr
        rel_change = abs_change / c_curr
        return (rel_change < 0.05) and (abs_change <= 2)

    for i in range(len(k_values) - 2):
        k0, k1, k2 = k_values[i], k_values[i + 1], k_values[i + 2]
        c0, c1, c2 = consensus_sizes[k0], consensus_sizes[k1], consensus_sizes[k2]

        if step_plateau(c0, c1) and step_plateau(c1, c2):
            return k0, True

    # Plateau never triggered
    return k_values[-1], False


# =============================================================================
# Main per-cohort runner
# =============================================================================

def run_stability_for_cohort(
    cohort: str,
    X, y_event, y_time, gene_names,
    ridge_lambda: float,
    results_dir: Path,
    device: str,
):
    """Run stability selection for one cohort (as training cohort)."""

    logger.info(f"")
    logger.info(f"{'='*70}")
    logger.info(f"STABILITY SELECTION: {cohort.upper()} (as training cohort)")
    logger.info(f"{'='*70}")
    logger.info(f"  Samples: {len(y_event)}, Events: {int(y_event.sum())}")
    logger.info(f"  Features (genes): {X.shape[1]}")
    logger.info(f"  Ridge lambda: {ridge_lambda:.6f}")
    logger.info(f"  Subsamples: {B_SUBSAMPLES}, Fraction: {SUBSAMPLE_FRACTION}")

    # Generate subsamples
    subsamples = generate_subsamples(
        len(y_event), y_event, B_SUBSAMPLES, SUBSAMPLE_FRACTION, RANDOM_SEED
    )

    logger.info(f"  Subsample sizes: train~{subsamples[0]['n_train']}, "
                f"OOB~{subsamples[0]['n_oob']}")

    # Create output directories
    cox_rank_dir = results_dir / "subsample_rankings" / cohort / "cox"
    ds_rank_dir = results_dir / "subsample_rankings" / cohort / "deepsurv"
    cox_rank_dir.mkdir(parents=True, exist_ok=True)
    ds_rank_dir.mkdir(parents=True, exist_ok=True)

    # --- Cox rankings ---
    logger.info(f"")
    logger.info(f"  Computing Cox rankings ({B_SUBSAMPLES} subsamples)...")

    cox_rankings = []
    cox_signed_coefs = []  # Store signed coefficients per subsample
    for b in range(B_SUBSAMPLES):
        ss = subsamples[b]
        ranking, signed_coefs = rank_genes_cox(
            X[ss['train_idx']], y_event[ss['train_idx']], y_time[ss['train_idx']],
            gene_names, fixed_alpha=ridge_lambda,
        )
        cox_rankings.append(ranking)
        cox_signed_coefs.append(signed_coefs)

        # Save ranking (absolute values, same as before)
        rank_df = pd.DataFrame(ranking, columns=['gene', 'importance'])
        rank_df.to_csv(cox_rank_dir / f"subsample_{b+1:02d}.csv", index=False)

        if (b + 1) % 10 == 0:
            logger.info(f"    Cox: {b+1}/{B_SUBSAMPLES} complete")

    # Save signed coefficient matrix (genes × subsamples)
    signed_coef_dir = results_dir / "signed_coefficients" / cohort
    signed_coef_dir.mkdir(parents=True, exist_ok=True)

    signed_df = pd.DataFrame(cox_signed_coefs, columns=gene_names).T
    signed_df.columns = [f"subsample_{b+1:02d}" for b in range(B_SUBSAMPLES)]
    signed_df.to_csv(signed_coef_dir / "cox_signed_coefficients.csv")
    logger.info(f"  Saved signed coefficient matrix: {signed_df.shape[0]} genes × {signed_df.shape[1]} subsamples")

    # --- DeepSurv + IG rankings ---
    logger.info(f"")
    logger.info(f"  Computing DeepSurv + OOB IG rankings ({B_SUBSAMPLES} subsamples)...")

    ds_params = DEEPSURV_PARAMS[cohort]
    ds_rankings = []

    for b in range(B_SUBSAMPLES):
        ss = subsamples[b]

        ranking = rank_genes_deepsurv_ig(
            X_train=X[ss['train_idx']],
            y_event_train=y_event[ss['train_idx']],
            y_time_train=y_time[ss['train_idx']],
            X_oob=X[ss['oob_idx']],
            gene_names=gene_names,
            params=ds_params,
            device=device,
            train_seed=RANDOM_SEED + b,
        )
        ds_rankings.append(ranking)

        # Save ranking
        rank_df = pd.DataFrame(ranking, columns=['gene', 'importance'])
        rank_df.to_csv(ds_rank_dir / f"subsample_{b+1:02d}.csv", index=False)

        if (b + 1) % 5 == 0:
            logger.info(f"    DeepSurv: {b+1}/{B_SUBSAMPLES} complete")

    # --- Selection probabilities ---
    logger.info(f"")
    logger.info(f"  Computing selection probabilities...")

    cox_probs = compute_selection_probs(cox_rankings, gene_names, K_SCAN)
    ds_probs = compute_selection_probs(ds_rankings, gene_names, K_SCAN)

    # Save
    prob_dir = results_dir / "selection_probabilities"
    prob_dir.mkdir(parents=True, exist_ok=True)
    cox_probs.to_csv(prob_dir / f"{cohort}_cox_selection_probs.csv")
    ds_probs.to_csv(prob_dir / f"{cohort}_deepsurv_selection_probs.csv")

    return {
        'cox_probs': cox_probs,
        'deepsurv_probs': ds_probs,
        'cox_rankings': cox_rankings,
        'deepsurv_rankings': ds_rankings,
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 4: Stability Selection with IQR Pre-Filtering"
    )
    parser.add_argument(
    '--filter-dir', type=str, default='iqr_filtered',
    help="Subdirectory under data/processed/ with retained_genes.txt (default: iqr_filtered)"
    )
    parser.add_argument(
        '--cohort', type=str, default='both',
        choices=['tcga', 'orien', 'both'],
        help="Which cohort to run as training cohort (default: both)"
    )
    parser.add_argument(
        '--results-dir', type=str, default=None,
        help=f"Override results directory (default: {RESULTS_BASE_DEFAULT})"
    )
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path(RESULTS_BASE_DEFAULT)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Log to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(results_dir / f"phase4_iqr_{timestamp}.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(file_handler)

    logger.info(f"Phase 4: Stability Selection with IQR Pre-Filtering")
    logger.info(f"Device: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Results: {results_dir}")

    # =========================================================================
    # Load data
    # =========================================================================
    logger.info(f"Loading data...")
    tcga_expr, tcga_surv = load_cohort('tcga', DATA_DIR)
    orien_expr, orien_surv = load_cohort('orien', DATA_DIR)

    # Sanity check: gene order must match
    gene_names_full = tcga_expr.index.tolist()
    if list(orien_expr.index) != gene_names_full:
        raise ValueError(
            "Gene sets/order differ between TCGA and ORIEN. "
            "Ensure preprocessing produced an identical shared gene matrix "
            "with the same gene ordering in both cohorts before Phase 4."
        )

    logger.info(f"Full gene set: {len(gene_names_full)} genes")

    # =========================================================================
    # IQR pre-filtering (load pre-computed gene list)
    # =========================================================================
    logger.info(f"")
    logger.info(f"{'='*70}")
    logger.info(f"IQR PRE-FILTERING")
    logger.info(f"{'='*70}")

    iqr_gene_file = Path(DATA_DIR) / args.filter_dir / "retained_genes.txt"
    iqr_metadata_file = Path(DATA_DIR) / args.filter_dir / "filter_metadata.json"

    if not iqr_gene_file.exists():
        raise FileNotFoundError(
            f"IQR filtered gene list not found: {iqr_gene_file}. "
            f"Run scripts/filter_genes_iqr.py first."
        )

    with open(iqr_gene_file) as f:
        retained_genes = [line.strip() for line in f if line.strip()]

    iqr_metadata = {}
    if iqr_metadata_file.exists():
        with open(iqr_metadata_file) as f:
            iqr_metadata = json.load(f)

    logger.info(f"  IQR filter: {len(gene_names_full)} → {len(retained_genes)} genes")
    if iqr_metadata:
        logger.info(f"  TCGA median IQR threshold: {iqr_metadata.get('tcga_median_iqr', 'N/A')}")
        logger.info(f"  ORIEN median IQR threshold: {iqr_metadata.get('orien_median_iqr', 'N/A')}")

    # Subset expression matrices to retained genes
    tcga_expr_filtered = tcga_expr.loc[retained_genes]
    orien_expr_filtered = orien_expr.loc[retained_genes]

    # Transpose to samples × genes for modeling
    tcga_X = tcga_expr_filtered.values.T.astype(np.float32)
    tcga_event = tcga_surv['event'].values.astype(np.float32)
    tcga_time = tcga_surv['time'].values.astype(np.float32)

    orien_X = orien_expr_filtered.values.T.astype(np.float32)
    orien_event = orien_surv['event'].values.astype(np.float32)
    orien_time = orien_surv['time'].values.astype(np.float32)

    gene_names = retained_genes
    n_features = len(gene_names)

    logger.info(f"")
    logger.info(f"Post-filter dimensions:")
    logger.info(f"  TCGA: {tcga_X.shape[0]} samples × {tcga_X.shape[1]} genes, "
                f"{int(tcga_event.sum())} events")
    logger.info(f"  ORIEN: {orien_X.shape[0]} samples × {orien_X.shape[1]} genes, "
                f"{int(orien_event.sum())} events")
    logger.info(f"  p/n ratio (TCGA full): {n_features / tcga_X.shape[0]:.1f}")
    logger.info(f"  p/n ratio (TCGA subsample): "
                f"{n_features / int(tcga_X.shape[0] * SUBSAMPLE_FRACTION):.1f}")

    # =========================================================================
    # Step 1: Ridge lambda selection (on IQR-filtered gene set)
    # =========================================================================
    logger.info(f"")
    logger.info(f"{'='*70}")
    logger.info(f"STEP 1: Ridge Lambda Selection (on {n_features} IQR-filtered genes)")
    logger.info(f"{'='*70}")

    tcga_lambda, tcga_ridge_ci = select_ridge_lambda(
        tcga_X, tcga_event, tcga_time, 'tcga'
    )
    orien_lambda, orien_ridge_ci = select_ridge_lambda(
        orien_X, orien_event, orien_time, 'orien'
    )

    lambda_info = {
        'tcga': {'alpha': tcga_lambda, 'cv_cindex': tcga_ridge_ci},
        'orien': {'alpha': orien_lambda, 'cv_cindex': orien_ridge_ci},
        'method': 'CoxnetSurvivalAnalysis(l1_ratio=1e-10) — near-pure L2 ridge',
        'n_genes': n_features,
        'filter': f'IQR above cohort-specific median, intersected ({n_features} genes)',
        'note': 'Alpha selected once per cohort via within-cohort CV on IQR-filtered genes. '
                'Held fixed across subsamples to isolate sampling perturbation.',
    }
    with open(results_dir / "ridge_lambda_selection.json", 'w') as f:
        json.dump(lambda_info, f, indent=2)

    # =========================================================================
    # Steps 2-3: Run stability per cohort
    # =========================================================================
    cohort_data = {
        'tcga': (tcga_X, tcga_event, tcga_time, tcga_lambda),
        'orien': (orien_X, orien_event, orien_time, orien_lambda),
    }

    cohorts_to_run = ['tcga', 'orien'] if args.cohort == 'both' else [args.cohort]
    all_results = {}

    for cohort in cohorts_to_run:
        X, y_event, y_time, ridge_lam = cohort_data[cohort]
        result = run_stability_for_cohort(
            cohort, X, y_event, y_time, gene_names,
            ridge_lam, results_dir, device,
        )
        all_results[cohort] = result

    # =========================================================================
    # Steps 4-5: Consensus (requires both cohorts)
    # =========================================================================
    if len(all_results) == 2:
        logger.info(f"")
        logger.info(f"{'='*70}")
        logger.info(f"CONSENSUS ANALYSIS")
        logger.info(f"{'='*70}")

        consensus_dir = results_dir / "consensus"
        consensus_dir.mkdir(parents=True, exist_ok=True)

        methods = ['cox', 'deepsurv']
        method_selected_ks = {}

        for method in methods:
            logger.info(f"")
            logger.info(f"  --- {method.upper()} ---")

            probs_key = f'{method}_probs'
            tcga_probs = all_results['tcga'][probs_key]
            orien_probs = all_results['orien'][probs_key]

            for pi in PI_THRESHOLDS:
                consensus_sizes, consensus_genes = find_consensus(
                    tcga_probs, orien_probs, K_SCAN, pi
                )

                logger.info(f"  π={pi:.2f}:")
                for k in K_SCAN:
                    logger.info(f"    k={k:>4}: {consensus_sizes[k]:>4} consensus genes")

                # Save gene lists at all k and pi
                for k in K_SCAN:
                    if consensus_genes[k]:
                        gene_file = consensus_dir / f"{method}_consensus_k{k}_pi{int(pi*100):02d}.txt"
                        with open(gene_file, 'w') as f:
                            for gene in consensus_genes[k]:
                                f.write(gene + '\n')

                # Apply plateau rule at primary threshold
                if pi == PI_PRIMARY:
                    selected_k, plateau_triggered = apply_plateau_rule(
                        consensus_sizes, K_SCAN
                    )
                    method_selected_ks[method] = {
                        'selected_k': selected_k,
                        'plateau_triggered': plateau_triggered,
                        'consensus_sizes': consensus_sizes,
                        'consensus_genes_by_k': consensus_genes,
                    }

                    if plateau_triggered:
                        logger.info(f"  ✓ Method-specific plateau at k={selected_k}")
                    else:
                        logger.info(f"  ⚠ No plateau. Method-specific k_max={selected_k}")

        # Shared k = max of method-specific k values
        shared_k = max(res['selected_k'] for res in method_selected_ks.values())
        shared_plateau = all(res['plateau_triggered'] for res in method_selected_ks.values())

        logger.info(f"")
        logger.info(f"  Shared k = {shared_k} "
                    f"(Cox k={method_selected_ks['cox']['selected_k']}, "
                    f"DeepSurv k={method_selected_ks['deepsurv']['selected_k']})")

        # Extract consensus genes at shared k for each method
        plateau_results = {}
        for method in methods:
            info = method_selected_ks[method]
            probs_key = f'{method}_probs'
            tcga_probs = all_results['tcga'][probs_key]
            orien_probs = all_results['orien'][probs_key]
            consensus_sizes, consensus_genes = find_consensus(
                tcga_probs, orien_probs, K_SCAN, PI_PRIMARY
            )

            plateau_results[method] = {
                'method_specific_k': info['selected_k'],
                'method_specific_plateau': info['plateau_triggered'],
                'shared_k': shared_k,
                'shared_plateau': shared_plateau,
                'consensus_sizes': consensus_sizes,
                'n_consensus_at_shared_k': consensus_sizes[shared_k],
                'consensus_genes': consensus_genes[shared_k],
            }

            # Save primary consensus gene list at shared k
            primary_file = consensus_dir / f"{method}_consensus_pi080.txt"
            with open(primary_file, 'w') as f:
                for gene in consensus_genes[shared_k]:
                    f.write(gene + '\n')

        # Save plateau analysis
        plateau_file = consensus_dir / "plateau_analysis.json"
        serializable = {}
        for method, res in plateau_results.items():
            serializable[method] = {
                'method_specific_k': res['method_specific_k'],
                'method_specific_plateau': res['method_specific_plateau'],
                'shared_k': res['shared_k'],
                'shared_plateau': res['shared_plateau'],
                'consensus_sizes': {str(k): v for k, v in res['consensus_sizes'].items()},
                'n_consensus': res['n_consensus_at_shared_k'],
                'consensus_genes': res['consensus_genes'],
            }
        with open(plateau_file, 'w') as f:
            json.dump(serializable, f, indent=2)

        # --- Contingency checks ---
        for method, res in plateau_results.items():
            n_genes = res['n_consensus_at_shared_k']
            if n_genes == 0:
                logger.warning(
                    f"  ⚠ {method}: consensus set empty at pi=0.80. "
                    f"This is a negative result. Check secondary thresholds (pi=0.70)."
                )
            elif n_genes < 5:
                logger.warning(
                    f"  ⚠ {method}: consensus set has only {n_genes} genes at pi=0.80. "
                    f"Consider reporting pi=0.70 as secondary threshold."
                )

        # --- Final summary ---
        logger.info(f"")
        logger.info(f"{'='*70}")
        logger.info(f"FINAL SUMMARY — PHASE 4 (IQR-filtered)")
        logger.info(f"{'='*70}")

        logger.info(f"  IQR filter: {len(gene_names_full)} → {n_features} genes "
                    f"(above-median IQR in both cohorts, intersected)")

        for method in methods:
            res = plateau_results[method]
            logger.info(
                f"  {method.upper():>8}: shared_k={res['shared_k']}, "
                f"method_k={res['method_specific_k']}, "
                f"plateau={'YES' if res['method_specific_plateau'] else 'NO'}, "
                f"consensus genes={res['n_consensus_at_shared_k']}"
            )

        # Gene overlap between methods
        cox_genes = set(plateau_results['cox']['consensus_genes'])
        ds_genes = set(plateau_results['deepsurv']['consensus_genes'])
        overlap = cox_genes & ds_genes
        union = cox_genes | ds_genes
        jaccard = len(overlap) / len(union) if union else 0

        logger.info(f"")
        logger.info(f"  Cox consensus: {len(cox_genes)} genes")
        logger.info(f"  DeepSurv consensus: {len(ds_genes)} genes")
        logger.info(f"  Overlap: {len(overlap)} genes")
        logger.info(f"  Jaccard: {jaccard:.3f}")
        logger.info(f"{'='*70}")

        # Save overall summary
        summary = {
            'iqr_filter': iqr_metadata,
            'ridge_lambdas': lambda_info,
            'plateau_results': serializable,
            'gene_overlap': {
                'cox_n': len(cox_genes),
                'deepsurv_n': len(ds_genes),
                'overlap_n': len(overlap),
                'jaccard': jaccard,
                'overlap_genes': sorted(overlap),
            },
            'config': {
                'B_subsamples': B_SUBSAMPLES,
                'subsample_fraction': SUBSAMPLE_FRACTION,
                'k_scan': K_SCAN,
                'pi_thresholds': PI_THRESHOLDS,
                'pi_primary': PI_PRIMARY,
                'ridge_method': 'CoxnetSurvivalAnalysis(l1_ratio=1e-10) — near-pure L2 ridge',
                'ig_n_steps': IG_N_STEPS,
                'ig_computed_on': 'OOB (out-of-subsample 20%)',
                'filter_method': 'IQR above cohort-specific median, intersected',
                'filter_reference': 'Bourgon et al. (2010) PNAS; Bioconductor genefilter',
            },
        }
        with open(results_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

    else:
        logger.info("Only one cohort run. Consensus analysis requires both cohorts.")
        logger.info("Run with --cohort both to compute consensus.")

    logger.info(f"\nPhase 4 (IQR-filtered) complete.")
