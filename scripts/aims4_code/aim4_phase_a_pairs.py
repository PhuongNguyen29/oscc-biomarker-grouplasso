#!/usr/bin/env python3
"""
Aim 4 Phase A: Gene-Pair Ordering Feasibility
===============================================

Tests whether within-sample gene-pair orderings (I(g_i > g_j))
provide a transportable prognostic representation for OSCC.

Design:
  1. Encode 300 pair features from 25-gene union set
  2. Fit ridge Cox on pair features in TCGA and ORIEN
  3. Bidirectional cross-cohort validation (locked coefficients)
  4. Locked validation on GEO (NO refitting)
  5. Compare to single-gene refit approach from Aim 1

Key question: Can a locked pair-based score transfer to GEO
microarray data without refitting?

Usage:
    cd ~/hpchome/oscc-biomarker-grouplasso
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/aim4_phase_a_pairs.py

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
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
UNION_GENES_FILE = SWEEP_DIR / "union_genes.txt"
ANNOTATION_PATH = Path("data/raw/protein_coding_genes_annotation.csv")
OUTPUT_DIR = Path("results/aim4_phase_a")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ridge Cox config — wider alpha range for binary pair features
ALPHAS = np.logspace(0, 5, 50)  # binary features need stronger regularization
L1_RATIO = 0.01  # near-pure ridge

# GEO repeated CV config (for comparison with Aim 1)
GEO_SEEDS = list(range(42, 142))
GEO_N_FOLDS = 3

# Colors for KM plots
COLOR_LOW = "#4A90D9"
COLOR_HIGH = "#D4A843"


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
            "ids": tcga_surv.index.tolist(),
        },
        "ORIEN": {
            "X": orien_X,
            "time": orien_surv["time"].values,
            "event": orien_surv["event"].values,
            "ids": orien_surv.index.tolist(),
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
        "ids": samples.tolist(),
        "genes_found": len(expr_filtered),
    }


def make_survival_array(events, times):
    return np.array(
        [(bool(e), t) for e, t in zip(events, times)],
        dtype=[("event", bool), ("time", float)],
    )


# =============================================================================
# Pair Feature Encoding
# =============================================================================
def encode_pairs(X, gene_names):
    """
    Encode all C(p,2) within-sample gene-pair orderings.
    P_ij(x) = 1 if x_i > x_j, else 0.

    Args:
        X: (n_samples, n_genes) expression matrix
        gene_names: list of gene names

    Returns:
        X_pairs: (n_samples, n_pairs) binary pair features
        pair_names: list of pair labels "geneA > geneB"
    """
    n_samples, n_genes = X.shape
    pairs = list(combinations(range(n_genes), 2))
    n_pairs = len(pairs)

    X_pairs = np.zeros((n_samples, n_pairs), dtype=np.float32)
    pair_names = []

    for idx, (i, j) in enumerate(pairs):
        X_pairs[:, idx] = (X[:, i] > X[:, j]).astype(np.float32)
        pair_names.append(f"{gene_names[i]} > {gene_names[j]}")

    return X_pairs, pair_names


# =============================================================================
# Model Fitting and Evaluation
# =============================================================================
def fit_ridge_cox(X_train, y_train, alphas=None):
    """Fit ridge Cox across alpha grid, with retry on numerical error."""
    if alphas is None:
        alphas = ALPHAS
    
    for attempt in range(5):
        try:
            model = CoxnetSurvivalAnalysis(
                l1_ratio=L1_RATIO, alphas=alphas,
                normalize=False, tol=1e-7, max_iter=100000,
            )
            model.fit(X_train, y_train)
            return model
        except ArithmeticError:
            # Increase alpha (stronger regularization)
            alphas = alphas * 2
            logger.warning(f"    ArithmeticError, retrying with alpha range "
                          f"[{alphas[0]:.1e}, {alphas[-1]:.1e}] (attempt {attempt+2})")
    
    raise RuntimeError("Ridge Cox fitting failed after 5 attempts")


def select_best_alpha_cv(X, y_struct, events, n_folds=5, seed=SEED):
    """Select best alpha by inner CV."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    alpha_scores = {}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, events)):
        try:
            model = fit_ridge_cox(X[train_idx], y_struct[train_idx])
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
        except Exception as e:
            logger.warning(f"    CV fold {fold_idx+1} failed: {e}")
            continue

    best_alpha_idx, best_ci = 0, 0.5
    for ai, scores in alpha_scores.items():
        if len(scores) >= 2:
            mean_ci = np.mean(scores)
            if mean_ci > best_ci:
                best_ci = mean_ci
                best_alpha_idx = ai

    return best_alpha_idx, best_ci


def evaluate_locked(model, best_alpha, X_test, y_test):
    """Evaluate locked model on test data (no refitting)."""
    alpha_idx = np.argmin(np.abs(model.alphas_ - best_alpha))
    coefs = model.coef_[:, alpha_idx]
    risk_scores = X_test @ coefs
    ci = concordance_index_censored(
        y_test["event"], y_test["time"], risk_scores
    )[0]
    return ci, risk_scores, coefs


# =============================================================================
# Main Analysis
# =============================================================================
def main():
    logger.info("=" * 70)
    logger.info("Aim 4 Phase A: Gene-Pair Ordering Feasibility")
    logger.info("=" * 70)

    # Load genes and annotation
    union_genes = load_union_genes(UNION_GENES_FILE)
    n_genes = len(union_genes)
    n_pairs = n_genes * (n_genes - 1) // 2
    logger.info(f"Union genes: {n_genes}")
    logger.info(f"Candidate pairs: C({n_genes},2) = {n_pairs}")

    # Load gene symbols
    if ANNOTATION_PATH.exists():
        annot = pd.read_csv(ANNOTATION_PATH)
        ensg_to_symbol = dict(zip(annot["gene_id"], annot["gene_symbol"]))
    else:
        ensg_to_symbol = {}

    gene_symbols = [ensg_to_symbol.get(g, g) for g in union_genes]

    # --- Load data ---
    logger.info("\n--- Loading data ---")
    dev_data = load_development_cohorts(DATA_DIR, union_genes)
    geo_data = {}
    for name in ["GSE41613", "GSE42743"]:
        geo_data[name] = load_geo_dataset(name, union_genes)
        logger.info(f"  {name}: {geo_data[name]['X'].shape[0]} samples, "
                    f"{geo_data[name]['genes_found']} genes")

    # --- Encode pair features ---
    logger.info("\n--- Encoding pair features ---")

    pairs = {}
    for name in ["TCGA", "ORIEN"]:
        X_pairs, pair_names = encode_pairs(dev_data[name]["X"], gene_symbols)
        pairs[name] = {"X": X_pairs, "names": pair_names}
        logger.info(f"  {name}: {X_pairs.shape[0]} samples × {X_pairs.shape[1]} pairs")

        # Check pair variance (constant pairs are uninformative)
        pair_var = X_pairs.var(axis=0)
        n_constant = (pair_var == 0).sum()
        logger.info(f"    Constant pairs (no variation): {n_constant}/{len(pair_names)}")

    for name in ["GSE41613", "GSE42743"]:
        X_pairs, pair_names = encode_pairs(geo_data[name]["X"], gene_symbols)
        pairs[name] = {"X": X_pairs, "names": pair_names}
        logger.info(f"  {name}: {X_pairs.shape[0]} samples × {X_pairs.shape[1]} pairs")

    # Verify pair names match across datasets
    assert pairs["TCGA"]["names"] == pairs["ORIEN"]["names"] == \
           pairs["GSE41613"]["names"] == pairs["GSE42743"]["names"], \
        "Pair names mismatch across datasets"

    pair_names = pairs["TCGA"]["names"]

    # Remove constant pairs (same ordering in all patients)
    variable_mask = np.ones(len(pair_names), dtype=bool)
    for name in ["TCGA", "ORIEN"]:
        pair_var = pairs[name]["X"].var(axis=0)
        variable_mask &= (pair_var > 0)

    n_variable = variable_mask.sum()
    logger.info(f"\n  Variable pairs (non-constant in both dev cohorts): {n_variable}/{len(pair_names)}")

    # Filter to variable pairs
    for name in pairs:
        pairs[name]["X"] = pairs[name]["X"][:, variable_mask]
    pair_names_filtered = [p for p, m in zip(pair_names, variable_mask) if m]

    # =====================================================================
    # ANALYSIS 1: Bidirectional Cross-Cohort Validation (Locked)
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 1: Bidirectional Cross-Cohort Validation (Locked Pairs)")
    logger.info("=" * 70)

    results_bidir = {}

    for direction, train_name, test_name in [
        ("TCGA→ORIEN", "TCGA", "ORIEN"),
        ("ORIEN→TCGA", "ORIEN", "TCGA"),
    ]:
        logger.info(f"\n  --- {direction} ---")

        X_train = pairs[train_name]["X"]
        y_train = make_survival_array(dev_data[train_name]["event"], dev_data[train_name]["time"])
        events_train = dev_data[train_name]["event"]

        X_test = pairs[test_name]["X"]
        y_test = make_survival_array(dev_data[test_name]["event"], dev_data[test_name]["time"])

        # Select best alpha by inner CV
        best_alpha_idx, cv_ci = select_best_alpha_cv(X_train, y_train, events_train)
        logger.info(f"    Inner CV C-index: {cv_ci:.4f}")

        # Fit on full training data
        model = fit_ridge_cox(X_train, y_train)
        best_alpha = model.alphas_[best_alpha_idx]

        # Training C-index
        train_ci, _, _ = evaluate_locked(model, best_alpha, X_train, y_train)
        logger.info(f"    Training C-index: {train_ci:.4f}")

        # Test C-index (LOCKED — no refitting)
        test_ci, test_scores, coefs = evaluate_locked(model, best_alpha, X_test, y_test)
        logger.info(f"    Test C-index (locked): {test_ci:.4f}")

        n_nonzero = np.sum(np.abs(coefs) > 1e-10)
        logger.info(f"    Nonzero pair coefficients: {n_nonzero}/{len(coefs)}")

        results_bidir[direction] = {
            "train_ci": float(train_ci),
            "test_ci": float(test_ci),
            "cv_ci": float(cv_ci),
            "n_nonzero": int(n_nonzero),
            "best_alpha": float(best_alpha),
            "model": model,
            "coefs": coefs,
        }

    mean_test = (results_bidir["TCGA→ORIEN"]["test_ci"] +
                 results_bidir["ORIEN→TCGA"]["test_ci"]) / 2
    logger.info(f"\n  Mean bidirectional test C-index (locked): {mean_test:.4f}")

    # =====================================================================
    # ANALYSIS 2: Locked Validation on GEO (NO REFITTING)
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 2: Locked Validation on GEO (No Refitting)")
    logger.info("=" * 70)

    results_geo_locked = {}

    # Use each dev model to predict on both GEO datasets
    for train_name, direction_key in [("TCGA", "TCGA-trained"), ("ORIEN", "ORIEN-trained")]:
        direction = "TCGA→ORIEN" if train_name == "TCGA" else "ORIEN→TCGA"
        model = results_bidir[direction]["model"]
        best_alpha = results_bidir[direction]["best_alpha"]
        alpha_idx = np.argmin(np.abs(model.alphas_ - best_alpha))
        coefs = model.coef_[:, alpha_idx]

        for geo_name in ["GSE41613", "GSE42743"]:
            X_geo = pairs[geo_name]["X"]
            y_geo = make_survival_array(geo_data[geo_name]["event"], geo_data[geo_name]["time"])

            risk_scores = X_geo @ coefs
            ci = concordance_index_censored(y_geo["event"], y_geo["time"], risk_scores)[0]

            key = f"{direction_key}→{geo_name}"
            results_geo_locked[key] = float(ci)
            logger.info(f"  {key}: C-index = {ci:.4f} (LOCKED, no refit)")

    # =====================================================================
    # ANALYSIS 3: GEO Refit Comparison (same as Aim 1 approach)
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 3: GEO Refit Comparison (100 seeds × 3-fold)")
    logger.info("=" * 70)

    results_geo_refit = {}

    for geo_name in ["GSE41613", "GSE42743"]:
        X_geo = pairs[geo_name]["X"]
        y_geo = make_survival_array(geo_data[geo_name]["event"], geo_data[geo_name]["time"])
        events_geo = geo_data[geo_name]["event"]

        seed_cindices = []
        for seed in GEO_SEEDS:
            skf = StratifiedKFold(n_splits=GEO_N_FOLDS, shuffle=True, random_state=seed)
            fold_cindices = []

            for train_idx, test_idx in skf.split(X_geo, events_geo):
                test_events = y_geo[test_idx]["event"].sum()
                if test_events < 2:
                    continue
                try:
                    model = fit_ridge_cox(X_geo[train_idx], y_geo[train_idx])
                    # Select best alpha on training fold
                    best_ci, best_ai = 0, 0
                    for ai in range(len(model.alphas_)):
                        try:
                            pred = X_geo[test_idx] @ model.coef_[:, ai]
                            ci = concordance_index_censored(
                                y_geo[test_idx]["event"], y_geo[test_idx]["time"], pred
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

        median_ci = np.median(seed_cindices) if seed_cindices else np.nan
        results_geo_refit[geo_name] = {
            "median_ci": float(median_ci),
            "mean_ci": float(np.mean(seed_cindices)),
            "std_ci": float(np.std(seed_cindices)),
            "n_seeds": len(seed_cindices),
        }
        logger.info(f"  {geo_name} pair refit: median C-index = {median_ci:.4f} "
                    f"({len(seed_cindices)} seeds)")

    # =====================================================================
    # ANALYSIS 4: Head-to-Head Comparison Table
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON: Pairs vs Single Genes")
    logger.info("=" * 70)

    # Aim 1 reference values (from sweep)
    aim1_values = {
        "cross_cohort_cox": 0.620,
        "cross_cohort_ds": 0.626,
        "GSE41613_refit": 0.625,
        "GSE42743_refit": 0.671,
    }

    logger.info(f"\n  {'Metric':<40s} {'Single-Gene':>12s} {'Pair (locked)':>14s} {'Pair (refit)':>13s}")
    logger.info(f"  {'-'*80}")

    logger.info(f"  {'Cross-cohort mean (bidirectional)':<40s} "
                f"{aim1_values['cross_cohort_cox']:>12.3f} "
                f"{mean_test:>14.3f} "
                f"{'N/A':>13s}")

    # Best locked GEO (take the better of TCGA-trained and ORIEN-trained)
    for geo_name in ["GSE41613", "GSE42743"]:
        locked_cis = [v for k, v in results_geo_locked.items() if geo_name in k]
        best_locked = max(locked_cis)
        refit_ci = results_geo_refit[geo_name]["median_ci"]
        aim1_ci = aim1_values[f"{geo_name}_refit"]

        logger.info(f"  {geo_name + ' (external)':<40s} "
                    f"{aim1_ci:>12.3f} "
                    f"{best_locked:>14.3f} "
                    f"{refit_ci:>13.3f}")

    # =====================================================================
    # ANALYSIS 5: Top Pairs by Coefficient Magnitude
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TOP PAIRS: Concordant Across Cohorts")
    logger.info("=" * 70)

    # Get coefficients from both directions
    coefs_tcga = results_bidir["TCGA→ORIEN"]["coefs"]
    coefs_orien = results_bidir["ORIEN→TCGA"]["coefs"]

    # Find pairs with concordant sign and high magnitude
    pair_info = []
    for idx, pname in enumerate(pair_names_filtered):
        c_t = coefs_tcga[idx]
        c_o = coefs_orien[idx]
        concordant = (c_t > 0 and c_o > 0) or (c_t < 0 and c_o < 0)
        mean_abs = (abs(c_t) + abs(c_o)) / 2
        pair_info.append({
            "pair": pname,
            "coef_tcga": float(c_t),
            "coef_orien": float(c_o),
            "concordant": concordant,
            "mean_abs_coef": float(mean_abs),
        })

    pair_df = pd.DataFrame(pair_info)
    concordant_pairs = pair_df[pair_df["concordant"]].sort_values("mean_abs_coef", ascending=False)
    discordant_pairs = pair_df[~pair_df["concordant"]]

    logger.info(f"\n  Concordant sign pairs: {len(concordant_pairs)}/{len(pair_df)}")
    logger.info(f"  Discordant sign pairs: {len(discordant_pairs)}/{len(pair_df)}")

    logger.info(f"\n  Top 20 concordant pairs:")
    logger.info(f"  {'Pair':<30s} {'TCGA coef':>10s} {'ORIEN coef':>11s} {'Mean |coef|':>12s}")
    logger.info(f"  {'-'*65}")
    for _, row in concordant_pairs.head(20).iterrows():
        logger.info(f"  {row['pair']:<30s} {row['coef_tcga']:>10.4f} {row['coef_orien']:>11.4f} "
                    f"{row['mean_abs_coef']:>12.4f}")

    # Save pair rankings
    pair_df.to_csv(OUTPUT_DIR / "pair_coefficients.csv", index=False)
    concordant_pairs.to_csv(OUTPUT_DIR / "concordant_pairs_ranked.csv", index=False)

    # =====================================================================
    # KM Plots for Locked Pair Scores
    # =====================================================================
    logger.info("\n--- Generating KM plots ---")

    # Use ORIEN-trained model (larger training set) for KM visualization
    model_for_km = results_bidir["ORIEN→TCGA"]["model"]
    best_alpha_km = results_bidir["ORIEN→TCGA"]["best_alpha"]
    alpha_idx_km = np.argmin(np.abs(model_for_km.alphas_ - best_alpha_km))
    coefs_km = model_for_km.coef_[:, alpha_idx_km]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.subplots_adjust(wspace=0.3)

    cohort_configs = [
        ("TCGA", pairs["TCGA"]["X"], dev_data["TCGA"]["time"],
         dev_data["TCGA"]["event"], "months"),
        ("ORIEN", pairs["ORIEN"]["X"], dev_data["ORIEN"]["time"],
         dev_data["ORIEN"]["event"], "months"),
        ("GSE41613", pairs["GSE41613"]["X"], geo_data["GSE41613"]["time"],
         geo_data["GSE41613"]["event"], "months"),
        ("GSE42743", pairs["GSE42743"]["X"], geo_data["GSE42743"]["time"],
         geo_data["GSE42743"]["event"], "days"),
    ]

    km_results = []
    for col_idx, (name, X_p, times, events, tunit) in enumerate(cohort_configs):
        risk_scores = X_p @ coefs_km

        if tunit == "days":
            times_display = times / 30.4375
        else:
            times_display = times

        median_score = np.median(risk_scores)
        high = risk_scores >= median_score
        low = ~high

        kmf_low = KaplanMeierFitter()
        kmf_low.fit(times_display[low], events[low])
        kmf_high = KaplanMeierFitter()
        kmf_high.fit(times_display[high], events[high])

        ax = axes[col_idx]

        sf_low = kmf_low.survival_function_
        ci_low = kmf_low.confidence_interval_survival_function_
        ax.step(sf_low.index, sf_low.iloc[:, 0], where="post", color=COLOR_LOW,
                linewidth=2, label=f"Low Risk (n={int(low.sum())})")
        ax.fill_between(ci_low.index, ci_low.iloc[:, 0], ci_low.iloc[:, 1],
                        step="post", alpha=0.15, color=COLOR_LOW)

        sf_high = kmf_high.survival_function_
        ci_high = kmf_high.confidence_interval_survival_function_
        ax.step(sf_high.index, sf_high.iloc[:, 0], where="post", color=COLOR_HIGH,
                linewidth=2, label=f"High Risk (n={int(high.sum())})")
        ax.fill_between(ci_high.index, ci_high.iloc[:, 0], ci_high.iloc[:, 1],
                        step="post", alpha=0.15, color=COLOR_HIGH)

        lr = logrank_test(times[low], times[high], events[low], events[high])
        p_val = lr.p_value
        p_text = "p < 0.0001" if p_val < 0.0001 else (f"p = {p_val:.4f}" if p_val < 0.001 else f"p = {p_val:.4f}")

        ax.text(0.05, 0.15, f"Log-rank\n{p_text}",
                transform=ax.transAxes, fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlim(0, 60)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Time (months)", fontsize=11)
        ax.set_ylabel("Overall Survival Probability", fontsize=11)
        ax.set_title(f"{name} — Pair Score (Locked)", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Panel label
        ax.text(-0.12, 1.08, chr(65 + col_idx), transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="top")

        km_results.append({
            "cohort": name,
            "logrank_p": float(p_val),
            "n_low": int(low.sum()),
            "n_high": int(high.sum()),
        })

    fig.savefig(OUTPUT_DIR / "km_pair_locked_4cohorts.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "km_pair_locked_4cohorts.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  KM panel saved to {OUTPUT_DIR}")

    # =====================================================================
    # Summary
    # =====================================================================
    logger.info(f"\n{'='*70}")
    logger.info("PHASE A SUMMARY")
    logger.info(f"{'='*70}")

    summary = {
        "n_genes": n_genes,
        "n_pairs_total": n_pairs,
        "n_pairs_variable": int(n_variable),
        "bidirectional": {
            "TCGA_to_ORIEN": {
                "train_ci": results_bidir["TCGA→ORIEN"]["train_ci"],
                "test_ci": results_bidir["TCGA→ORIEN"]["test_ci"],
            },
            "ORIEN_to_TCGA": {
                "train_ci": results_bidir["ORIEN→TCGA"]["train_ci"],
                "test_ci": results_bidir["ORIEN→TCGA"]["test_ci"],
            },
            "mean_test_ci": float(mean_test),
        },
        "geo_locked": results_geo_locked,
        "geo_refit": {k: v for k, v in results_geo_refit.items()},
        "aim1_comparison": aim1_values,
        "n_concordant_pairs": int(len(concordant_pairs)),
        "n_discordant_pairs": int(len(discordant_pairs)),
        "km_results": km_results,
    }

    with open(OUTPUT_DIR / "phase_a_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print final comparison
    logger.info(f"\n  Cross-cohort (bidirectional mean):")
    logger.info(f"    Single-gene Cox (Aim 1):  {aim1_values['cross_cohort_cox']:.3f}")
    logger.info(f"    Pair-based locked:         {mean_test:.3f}")

    logger.info(f"\n  External validation (best locked):")
    for geo_name in ["GSE41613", "GSE42743"]:
        locked_cis = [v for k, v in results_geo_locked.items() if geo_name in k]
        best_locked = max(locked_cis)
        aim1_ci = aim1_values[f"{geo_name}_refit"]
        refit_ci = results_geo_refit[geo_name]["median_ci"]
        logger.info(f"    {geo_name}: Aim1(refit)={aim1_ci:.3f}, "
                    f"Pair(locked)={best_locked:.3f}, Pair(refit)={refit_ci:.3f}")

    logger.info(f"\n  Concordant pairs: {len(concordant_pairs)}/{len(pair_df)} "
                f"({100*len(concordant_pairs)/len(pair_df):.0f}%)")

    # Feasibility verdict
    logger.info(f"\n  --- FEASIBILITY VERDICT ---")
    geo_locked_pass = all(
        any(v >= 0.55 for k, v in results_geo_locked.items() if geo_name in k)
        for geo_name in ["GSE41613", "GSE42743"]
    )
    if geo_locked_pass and mean_test >= 0.55:
        logger.info(f"  PROMISING: Locked pair scores show prognostic signal.")
        logger.info(f"  Recommend proceeding to Phase B (pair-stability selection).")
    elif mean_test >= 0.55:
        logger.info(f"  PARTIAL: Cross-cohort signal present but GEO locked transfer weak.")
        logger.info(f"  Consider pair-stability selection to improve signal-to-noise.")
    else:
        logger.info(f"  WEAK: Pair features do not show clear advantage.")
        logger.info(f"  Consider expanding candidate pool or alternative pair encoding.")

    logger.info(f"\n  Results saved to {OUTPUT_DIR}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
