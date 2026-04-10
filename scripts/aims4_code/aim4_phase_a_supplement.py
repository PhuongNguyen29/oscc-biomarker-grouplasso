#!/usr/bin/env python3
"""
Aim 4 Phase A Supplement: Complete Comparison Table
====================================================

Adds the missing comparisons from Phase A:
  1. Single-gene LOCKED validation on GEO (no refitting)
  2. Bootstrap confidence intervals for all C-indices
  3. Full 2×2 comparison table: {single-gene, pair} × {locked, refit}

This resolves the question: is the GSE41613 pair advantage from
the pair representation or from the locked validation design?

Usage:
    cd ~/hpchome/oscc-biomarker-grouplasso
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/aim4_phase_a_supplement.py

Author: Phuong Nguyen
Date: April 2026
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from itertools import combinations
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import StratifiedKFold
import warnings
import logging
import json

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================
SEED = 42
DATA_DIR = "data/processed"
EXTERNAL_DIR = Path("data/external")
SWEEP_DIR = Path("results/sensitivity_sweep/mult0.6_pi60")
COX_DIR = SWEEP_DIR / "validation" / "cox"
UNION_GENES_FILE = SWEEP_DIR / "union_genes.txt"
OUTPUT_DIR = Path("results/aim4_phase_a")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ridge Cox config
ALPHAS_GENES = np.logspace(-1, 4, 50)
ALPHAS_PAIRS = np.logspace(0, 5, 50)
L1_RATIO = 0.01

# GEO repeated CV
GEO_SEEDS = list(range(42, 142))
GEO_N_FOLDS = 3

# Bootstrap config
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42


# =============================================================================
# Data Loading
# =============================================================================
def load_union_genes(filepath):
    with open(filepath) as f:
        return [line.strip() for line in f if line.strip()]


def subset_expression(X_full, all_gene_names, selected_genes):
    gene_indices = [all_gene_names.index(g) for g in selected_genes]
    return X_full[:, gene_indices]


def load_development_cohorts(data_dir, union_genes):
    from src.data.data_loader import load_cohort
    tcga_expr, tcga_surv = load_cohort("tcga", data_dir)
    orien_expr, orien_surv = load_cohort("orien", data_dir)
    all_gene_names = tcga_expr.index.tolist()
    tcga_X = subset_expression(tcga_expr.values.T.astype(np.float32), all_gene_names, union_genes)
    orien_X = subset_expression(orien_expr.values.T.astype(np.float32), all_gene_names, union_genes)
    return {
        "TCGA": {
            "X": tcga_X,
            "time": tcga_surv["time"].values,
            "event": tcga_surv["event"].values,
        },
        "ORIEN": {
            "X": orien_X,
            "time": orien_surv["time"].values,
            "event": orien_surv["event"].values,
        },
    }


def load_geo_dataset(name, union_genes):
    expr = pd.read_csv(EXTERNAL_DIR / f"{name}_expression.csv", index_col=0)
    surv = pd.read_csv(EXTERNAL_DIR / f"{name}_survival.csv")
    samples = surv["sampleID"].values
    expr_filtered = expr.loc[expr.index.isin(union_genes)]
    X = expr_filtered[samples].values.T.astype(np.float32)
    return {
        "X": X,
        "time": surv["time"].values.astype(float),
        "event": surv["event"].values.astype(float),
    }


def make_survival_array(events, times):
    return np.array(
        [(bool(e), t) for e, t in zip(events, times)],
        dtype=[("event", bool), ("time", float)],
    )


def encode_pairs(X, gene_names):
    n_samples, n_genes = X.shape
    pairs = list(combinations(range(n_genes), 2))
    X_pairs = np.zeros((n_samples, len(pairs)), dtype=np.float32)
    pair_names = []
    for idx, (i, j) in enumerate(pairs):
        X_pairs[:, idx] = (X[:, i] > X[:, j]).astype(np.float32)
        pair_names.append(f"{gene_names[i]} > {gene_names[j]}")
    return X_pairs, pair_names


# =============================================================================
# Fitting
# =============================================================================
def fit_ridge_cox(X_train, y_train, alphas):
    for attempt in range(5):
        try:
            model = CoxnetSurvivalAnalysis(
                l1_ratio=L1_RATIO, alphas=alphas,
                normalize=False, tol=1e-7, max_iter=100000,
            )
            model.fit(X_train, y_train)
            return model
        except ArithmeticError:
            alphas = alphas * 2
    raise RuntimeError("Ridge Cox fitting failed after 5 attempts")


def select_best_alpha_cv(X, y_struct, events, alphas, n_folds=5, seed=SEED):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    alpha_scores = {}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, events)):
        try:
            model = fit_ridge_cox(X[train_idx], y_struct[train_idx], alphas)
            for ai, alpha_val in enumerate(model.alphas_):
                try:
                    pred = X[val_idx] @ model.coef_[:, ai]
                    ci = concordance_index_censored(
                        y_struct[val_idx]["event"], y_struct[val_idx]["time"], pred
                    )[0]
                    if ai not in alpha_scores:
                        alpha_scores[ai] = []
                    alpha_scores[ai].append(ci)
                except Exception:
                    continue
        except Exception:
            continue

    best_alpha_idx, best_ci = 0, 0.5
    for ai, scores in alpha_scores.items():
        if len(scores) >= 2:
            mean_ci = np.mean(scores)
            if mean_ci > best_ci:
                best_ci = mean_ci
                best_alpha_idx = ai

    return best_alpha_idx, best_ci


# =============================================================================
# Bootstrap CI
# =============================================================================
def bootstrap_cindex(events, times, risk_scores, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Compute bootstrap 95% CI for C-index."""
    rng = np.random.RandomState(seed)
    n = len(events)
    y_struct = make_survival_array(events, times)

    # Observed C-index
    observed_ci = concordance_index_censored(
        y_struct["event"], y_struct["time"], risk_scores
    )[0]

    # Bootstrap
    boot_cis = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            ci = concordance_index_censored(
                y_struct[idx]["event"], y_struct[idx]["time"], risk_scores[idx]
            )[0]
            boot_cis.append(ci)
        except Exception:
            continue

    boot_cis = np.array(boot_cis)
    ci_lower = np.percentile(boot_cis, 2.5)
    ci_upper = np.percentile(boot_cis, 97.5)

    return {
        "cindex": float(observed_ci),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_boot": len(boot_cis),
    }


# =============================================================================
# GEO Refit C-index (100 seeds × 3-fold)
# =============================================================================
def geo_refit_cindex(X, y_struct, events, alphas):
    seed_cindices = []
    for seed in GEO_SEEDS:
        skf = StratifiedKFold(n_splits=GEO_N_FOLDS, shuffle=True, random_state=seed)
        fold_cindices = []
        for train_idx, test_idx in skf.split(X, events):
            test_events = y_struct[test_idx]["event"].sum()
            if test_events < 2:
                continue
            try:
                model = fit_ridge_cox(X[train_idx], y_struct[train_idx], alphas)
                best_ci, best_ai = 0, 0
                for ai in range(len(model.alphas_)):
                    try:
                        pred = X[test_idx] @ model.coef_[:, ai]
                        ci = concordance_index_censored(
                            y_struct[test_idx]["event"], y_struct[test_idx]["time"], pred
                        )[0]
                        if ci > best_ci:
                            best_ci = ci
                            best_ai = ai
                    except Exception:
                        continue
                fold_cindices.append(best_ci)
            except Exception:
                continue
        if fold_cindices:
            seed_cindices.append(np.mean(fold_cindices))
    return float(np.median(seed_cindices)) if seed_cindices else np.nan


# =============================================================================
# Main
# =============================================================================
def main():
    logger.info("=" * 70)
    logger.info("Aim 4 Phase A Supplement: Complete Comparison Table")
    logger.info("=" * 70)

    # Load genes
    union_genes = load_union_genes(UNION_GENES_FILE)
    n_genes = len(union_genes)

    # Load annotation for symbols
    annot_path = Path("data/raw/protein_coding_genes_annotation.csv")
    if annot_path.exists():
        annot = pd.read_csv(annot_path)
        ensg_to_symbol = dict(zip(annot["gene_id"], annot["gene_symbol"]))
    else:
        ensg_to_symbol = {}
    gene_symbols = [ensg_to_symbol.get(g, g) for g in union_genes]

    # Load data
    logger.info("\n--- Loading data ---")
    dev_data = load_development_cohorts(DATA_DIR, union_genes)
    geo_data = {}
    for name in ["GSE41613", "GSE42743"]:
        geo_data[name] = load_geo_dataset(name, union_genes)

    # Encode pairs
    logger.info("--- Encoding pairs ---")
    pairs_dev = {}
    for name in ["TCGA", "ORIEN"]:
        X_p, p_names = encode_pairs(dev_data[name]["X"], gene_symbols)
        pairs_dev[name] = X_p

    pairs_geo = {}
    for name in ["GSE41613", "GSE42743"]:
        X_p, p_names = encode_pairs(geo_data[name]["X"], gene_symbols)
        pairs_geo[name] = X_p

    pair_names = p_names

    # =====================================================================
    # STEP 1: Train models on dev cohorts (both gene and pair)
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Training models on development cohorts")
    logger.info("=" * 70)

    models = {}

    for train_name in ["TCGA", "ORIEN"]:
        logger.info(f"\n  --- Training on {train_name} ---")

        y_train = make_survival_array(dev_data[train_name]["event"],
                                       dev_data[train_name]["time"])
        events_train = dev_data[train_name]["event"]

        # Single-gene model
        X_gene = dev_data[train_name]["X"]
        best_ai_gene, cv_ci_gene = select_best_alpha_cv(
            X_gene, y_train, events_train, ALPHAS_GENES
        )
        model_gene = fit_ridge_cox(X_gene, y_train, ALPHAS_GENES)
        best_alpha_gene = model_gene.alphas_[best_ai_gene]
        alpha_idx_gene = np.argmin(np.abs(model_gene.alphas_ - best_alpha_gene))
        coefs_gene = model_gene.coef_[:, alpha_idx_gene]
        logger.info(f"    Gene: CV C-index={cv_ci_gene:.4f}, "
                    f"alpha={best_alpha_gene:.2e}, "
                    f"nonzero={np.sum(np.abs(coefs_gene)>1e-10)}/{len(coefs_gene)}")

        # Pair model
        X_pair = pairs_dev[train_name]
        best_ai_pair, cv_ci_pair = select_best_alpha_cv(
            X_pair, y_train, events_train, ALPHAS_PAIRS
        )
        model_pair = fit_ridge_cox(X_pair, y_train, ALPHAS_PAIRS)
        best_alpha_pair = model_pair.alphas_[best_ai_pair]
        alpha_idx_pair = np.argmin(np.abs(model_pair.alphas_ - best_alpha_pair))
        coefs_pair = model_pair.coef_[:, alpha_idx_pair]
        logger.info(f"    Pair: CV C-index={cv_ci_pair:.4f}, "
                    f"alpha={best_alpha_pair:.2e}, "
                    f"nonzero={np.sum(np.abs(coefs_pair)>1e-10)}/{len(coefs_pair)}")

        models[train_name] = {
            "gene_coefs": coefs_gene,
            "pair_coefs": coefs_pair,
            "gene_model": model_gene,
            "pair_model": model_pair,
            "gene_alpha": best_alpha_gene,
            "pair_alpha": best_alpha_pair,
        }

    # =====================================================================
    # STEP 2: Compute all C-indices with bootstrap CIs
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Full comparison table with bootstrap 95% CIs")
    logger.info("=" * 70)

    results = []

    for geo_name in ["GSE41613", "GSE42743"]:
        logger.info(f"\n  --- {geo_name} ---")

        X_gene_geo = geo_data[geo_name]["X"]
        X_pair_geo = pairs_geo[geo_name]
        events_geo = geo_data[geo_name]["event"]
        times_geo = geo_data[geo_name]["time"]
        y_geo = make_survival_array(events_geo, times_geo)

        # --- LOCKED validation (use each dev model) ---
        for train_name in ["TCGA", "ORIEN"]:
            # Single-gene locked
            scores_gene = X_gene_geo @ models[train_name]["gene_coefs"]
            boot_gene = bootstrap_cindex(events_geo, times_geo, scores_gene)
            logger.info(f"    Gene locked ({train_name}-trained): "
                        f"{boot_gene['cindex']:.3f} "
                        f"[{boot_gene['ci_lower']:.3f}, {boot_gene['ci_upper']:.3f}]")

            results.append({
                "dataset": geo_name,
                "feature": "single-gene",
                "validation": "locked",
                "trained_on": train_name,
                **boot_gene,
            })

            # Pair locked
            scores_pair = X_pair_geo @ models[train_name]["pair_coefs"]
            boot_pair = bootstrap_cindex(events_geo, times_geo, scores_pair)
            logger.info(f"    Pair locked ({train_name}-trained): "
                        f"{boot_pair['cindex']:.3f} "
                        f"[{boot_pair['ci_lower']:.3f}, {boot_pair['ci_upper']:.3f}]")

            results.append({
                "dataset": geo_name,
                "feature": "pair",
                "validation": "locked",
                "trained_on": train_name,
                **boot_pair,
            })

        # --- REFIT validation (100 seeds × 3-fold) ---
        # Single-gene refit
        gene_refit_ci = geo_refit_cindex(X_gene_geo, y_geo, events_geo, ALPHAS_GENES)
        logger.info(f"    Gene refit (100 seeds): {gene_refit_ci:.3f}")
        results.append({
            "dataset": geo_name,
            "feature": "single-gene",
            "validation": "refit",
            "trained_on": "within-GEO",
            "cindex": gene_refit_ci,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_boot": 0,
        })

        # Pair refit
        pair_refit_ci = geo_refit_cindex(X_pair_geo, y_geo, events_geo, ALPHAS_PAIRS)
        logger.info(f"    Pair refit (100 seeds): {pair_refit_ci:.3f}")
        results.append({
            "dataset": geo_name,
            "feature": "pair",
            "validation": "refit",
            "trained_on": "within-GEO",
            "cindex": pair_refit_ci,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_boot": 0,
        })

    # =====================================================================
    # STEP 3: Cross-cohort bidirectional (locked)
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Cross-cohort bidirectional with bootstrap CIs")
    logger.info("=" * 70)

    crosscohort_results = []

    for train_name, test_name in [("TCGA", "ORIEN"), ("ORIEN", "TCGA")]:
        direction = f"{train_name}→{test_name}"
        events_test = dev_data[test_name]["event"]
        times_test = dev_data[test_name]["time"]

        # Single-gene locked
        scores_gene = dev_data[test_name]["X"] @ models[train_name]["gene_coefs"]
        boot_gene = bootstrap_cindex(events_test, times_test, scores_gene)
        logger.info(f"  {direction} gene locked: {boot_gene['cindex']:.3f} "
                    f"[{boot_gene['ci_lower']:.3f}, {boot_gene['ci_upper']:.3f}]")
        crosscohort_results.append({
            "direction": direction,
            "feature": "single-gene",
            **boot_gene,
        })

        # Pair locked
        scores_pair = pairs_dev[test_name] @ models[train_name]["pair_coefs"]
        boot_pair = bootstrap_cindex(events_test, times_test, scores_pair)
        logger.info(f"  {direction} pair locked: {boot_pair['cindex']:.3f} "
                    f"[{boot_pair['ci_lower']:.3f}, {boot_pair['ci_upper']:.3f}]")
        crosscohort_results.append({
            "direction": direction,
            "feature": "pair",
            **boot_pair,
        })

    # =====================================================================
    # STEP 4: Print formatted comparison table
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE 2×2 COMPARISON TABLE")
    logger.info("=" * 70)

    # Format GEO results
    logger.info(f"\n  {'Dataset':<12s} {'Feature':<14s} {'Validation':<10s} "
                f"{'Trained on':<12s} {'C-index':>8s} {'95% CI':>20s}")
    logger.info(f"  {'-'*78}")

    for r in sorted(results, key=lambda x: (x['dataset'], x['feature'], x['validation'])):
        ci_str = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]" if not np.isnan(r['ci_lower']) else "N/A (100-seed)"
        logger.info(f"  {r['dataset']:<12s} {r['feature']:<14s} {r['validation']:<10s} "
                    f"{r['trained_on']:<12s} {r['cindex']:>8.3f} {ci_str:>20s}")

    # Condensed table for presentation
    logger.info(f"\n  --- Condensed (best locked per dataset) ---")
    logger.info(f"\n  {'Dataset':<12s} {'Gene locked':>13s} {'Gene refit':>12s} "
                f"{'Pair locked':>13s} {'Pair refit':>12s}")
    logger.info(f"  {'-'*64}")

    for geo_name in ["GSE41613", "GSE42743"]:
        geo_results = [r for r in results if r['dataset'] == geo_name]

        gene_locked = [r for r in geo_results
                       if r['feature'] == 'single-gene' and r['validation'] == 'locked']
        best_gene_locked = max(gene_locked, key=lambda x: x['cindex'])

        pair_locked = [r for r in geo_results
                       if r['feature'] == 'pair' and r['validation'] == 'locked']
        best_pair_locked = max(pair_locked, key=lambda x: x['cindex'])

        gene_refit = [r for r in geo_results
                      if r['feature'] == 'single-gene' and r['validation'] == 'refit'][0]
        pair_refit = [r for r in geo_results
                      if r['feature'] == 'pair' and r['validation'] == 'refit'][0]

        logger.info(f"  {geo_name:<12s} "
                    f"{best_gene_locked['cindex']:>6.3f} ({best_gene_locked['trained_on'][:3]:>3s}) "
                    f"{gene_refit['cindex']:>12.3f} "
                    f"{best_pair_locked['cindex']:>6.3f} ({best_pair_locked['trained_on'][:3]:>3s}) "
                    f"{pair_refit['cindex']:>12.3f}")

    # Cross-cohort summary
    logger.info(f"\n  --- Cross-cohort bidirectional ---")
    logger.info(f"\n  {'Direction':<16s} {'Gene locked':>13s} {'Pair locked':>13s}")
    logger.info(f"  {'-'*44}")

    gene_cc = [r for r in crosscohort_results if r['feature'] == 'single-gene']
    pair_cc = [r for r in crosscohort_results if r['feature'] == 'pair']

    for g, p in zip(gene_cc, pair_cc):
        logger.info(f"  {g['direction']:<16s} "
                    f"{g['cindex']:>6.3f} [{g['ci_lower']:.3f},{g['ci_upper']:.3f}] "
                    f"{p['cindex']:>6.3f} [{p['ci_lower']:.3f},{p['ci_upper']:.3f}]")

    gene_mean = np.mean([r['cindex'] for r in gene_cc])
    pair_mean = np.mean([r['cindex'] for r in pair_cc])
    logger.info(f"  {'Mean':<16s} {gene_mean:>6.3f}               {pair_mean:>6.3f}")

    # =====================================================================
    # STEP 5: Key diagnostic — is the gain from pairs or from locking?
    # =====================================================================
    logger.info(f"\n" + "=" * 70)
    logger.info("DIAGNOSTIC: Source of GSE41613 improvement")
    logger.info("=" * 70)

    gse41_results = [r for r in results if r['dataset'] == 'GSE41613']

    gene_locked_orien = [r for r in gse41_results
                         if r['feature'] == 'single-gene'
                         and r['validation'] == 'locked'
                         and r['trained_on'] == 'ORIEN'][0]
    pair_locked_orien = [r for r in gse41_results
                         if r['feature'] == 'pair'
                         and r['validation'] == 'locked'
                         and r['trained_on'] == 'ORIEN'][0]
    gene_refit = [r for r in gse41_results
                  if r['feature'] == 'single-gene'
                  and r['validation'] == 'refit'][0]

    logger.info(f"\n  GSE41613 (ORIEN-trained models):")
    logger.info(f"    Gene refit:  {gene_refit['cindex']:.3f}")
    logger.info(f"    Gene locked: {gene_locked_orien['cindex']:.3f} "
                f"[{gene_locked_orien['ci_lower']:.3f}, {gene_locked_orien['ci_upper']:.3f}]")
    logger.info(f"    Pair locked: {pair_locked_orien['cindex']:.3f} "
                f"[{pair_locked_orien['ci_lower']:.3f}, {pair_locked_orien['ci_upper']:.3f}]")

    gene_locked_ci = gene_locked_orien['cindex']
    pair_locked_ci = pair_locked_orien['cindex']
    gene_refit_ci = gene_refit['cindex']

    logger.info(f"\n  Decomposition:")
    logger.info(f"    Gain from locking (gene locked - gene refit): "
                f"{gene_locked_ci - gene_refit_ci:+.3f}")
    logger.info(f"    Gain from pairs (pair locked - gene locked):  "
                f"{pair_locked_ci - gene_locked_ci:+.3f}")
    logger.info(f"    Total gain (pair locked - gene refit):         "
                f"{pair_locked_ci - gene_refit_ci:+.3f}")

    if pair_locked_ci > gene_locked_ci + 0.02:
        logger.info(f"\n  CONCLUSION: Pair representation contributes beyond locking.")
    elif gene_locked_ci > gene_refit_ci + 0.02:
        logger.info(f"\n  CONCLUSION: Gain is primarily from locked design, not pairs.")
    else:
        logger.info(f"\n  CONCLUSION: Differences are small; no clear advantage.")

    # Save all results
    all_output = {
        "geo_results": results,
        "crosscohort_results": crosscohort_results,
    }
    with open(OUTPUT_DIR / "phase_a_supplement.json", "w") as f:
        json.dump(all_output, f, indent=2, default=str)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "phase_a_full_comparison.csv", index=False)

    cc_df = pd.DataFrame(crosscohort_results)
    cc_df.to_csv(OUTPUT_DIR / "phase_a_crosscohort_comparison.csv", index=False)

    logger.info(f"\n  Results saved to {OUTPUT_DIR}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
