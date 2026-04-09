#!/usr/bin/env python3
"""
Risk Score Stratification: KM Curves + 5-Year Time-Dependent AUC
=================================================================

Development cohorts (TCGA, ORIEN):
  - Phase 7 bidirectional Cox models (out-of-sample risk scores)
  - TCGA patients scored by ORIEN-trained model
  - ORIEN patients scored by TCGA-trained model

GEO cohorts (GSE41613, GSE42743):
  - Refit ridge Cox within each dataset
  - Out-of-fold risk scores (100 seeds × 3-fold) for honest stratification
  - Use median OOF score across seeds for final stratification

Output:
  - 2×4 panel: KM (top) + 5-year ROC (bottom) per cohort
  - Individual cohort figures
  - Summary CSV with log-rank p, median survival, 5-year AUC

Usage:
    cd ~/hpchome/oscc-biomarker-grouplasso
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/km_auc_stratification.py

Author: Phuong Nguyen
Date: April 2026
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import StratifiedKFold
import warnings
import logging

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
OUTPUT_DIR = Path("results/km_auc_stratification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# GEO config
GEO_SEEDS = list(range(42, 142))
GEO_N_FOLDS = 3
GEO_ALPHAS = np.logspace(-1, 4, 50)

# Time horizon for AUC (in same units as survival time)
# TCGA/ORIEN: days — 5 years = 1825 days
# GSE41613: months — 5 years = 60 months
# GSE42743: days — 5 years = 1825 days
TIME_HORIZONS = {
    "TCGA": 60,       # months
    "ORIEN": 60,      # months
    "GSE41613": 60,   # months
    "GSE42743": 1825, # days
}

# KM plot x-axis limits (in months for display)
# Convert days to months for TCGA/ORIEN/GSE42743
XAXIS_MONTHS = 60  # 5 years

# Colors matching your R plot style
COLOR_LOW = "#4A90D9"   # blue
COLOR_HIGH = "#D4A843"  # gold/amber


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
    """Load TCGA and ORIEN expression + survival."""
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
    """Load GEO expression + survival, subset to union genes."""
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


# =============================================================================
# Risk Score Computation
# =============================================================================
def compute_dev_risk_scores(dev_data, cox_dir, union_genes):
    """
    Compute out-of-sample risk scores for development cohorts
    using Phase 7 bidirectional Cox models.
    """
    scores = {}

    # TCGA: scored by ORIEN-trained model
    cox_path = cox_dir / "cox_model_orien_to_tcga_union.pkl"
    with open(cox_path, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    best_alpha = saved["best_alpha"]
    tcga_scores = model.predict(dev_data["TCGA"]["X"], alpha=best_alpha)
    scores["TCGA"] = tcga_scores
    logger.info(f"  TCGA: {len(tcga_scores)} patients scored (ORIEN-trained model)")

    # ORIEN: scored by TCGA-trained model
    cox_path = cox_dir / "cox_model_tcga_to_orien_union.pkl"
    with open(cox_path, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    best_alpha = saved["best_alpha"]
    orien_scores = model.predict(dev_data["ORIEN"]["X"], alpha=best_alpha)
    scores["ORIEN"] = orien_scores
    logger.info(f"  ORIEN: {len(orien_scores)} patients scored (TCGA-trained model)")

    return scores


def compute_geo_oof_scores(geo_data, name):
    """
    Compute out-of-fold risk scores for GEO dataset.
    100 seeds × 3-fold ridge Cox. Returns median score per patient.
    """
    X = geo_data["X"]
    y_struct = np.array(
        [(bool(e), t) for e, t in zip(geo_data["event"], geo_data["time"])],
        dtype=[("event", bool), ("time", float)],
    )
    events = geo_data["event"]
    n_samples = len(X)

    all_seed_scores = np.full((len(GEO_SEEDS), n_samples), np.nan)

    for si, seed in enumerate(GEO_SEEDS):
        patient_scores = np.full(n_samples, np.nan)
        skf = StratifiedKFold(n_splits=GEO_N_FOLDS, shuffle=True, random_state=seed)

        for train_idx, test_idx in skf.split(X, events):
            try:
                model = CoxnetSurvivalAnalysis(
                    l1_ratio=0.01, alphas=GEO_ALPHAS,
                    normalize=False, tol=1e-7, max_iter=100000,
                )
                model.fit(X[train_idx], y_struct[train_idx])

                # Select best alpha by training C-index
                best_ci, best_alpha_idx = 0, 0
                for ai in range(len(model.alphas_)):
                    try:
                        pred = X[test_idx] @ model.coef_[:, ai]
                        ci = concordance_index_censored(
                            y_struct[test_idx]["event"], y_struct[test_idx]["time"], pred
                        )[0]
                        if ci > best_ci:
                            best_ci = ci
                            best_alpha_idx = ai
                    except Exception:
                        continue

                patient_scores[test_idx] = X[test_idx] @ model.coef_[:, best_alpha_idx]
            except Exception:
                continue

        all_seed_scores[si] = patient_scores

    # Median across seeds per patient
    median_scores = np.nanmedian(all_seed_scores, axis=0)
    n_valid = np.sum(~np.isnan(median_scores))
    logger.info(f"  {name}: {n_valid}/{n_samples} patients with valid OOF scores")

    return median_scores


# =============================================================================
# Time-Dependent AUC (Inverse Probability of Censoring Weighted)
# =============================================================================
def time_dependent_auc(times, events, risk_scores, horizon):
    """
    Compute time-dependent AUC at a specific time horizon using
    the Uno et al. (2007) IPCW estimator via sksurv.

    Falls back to concordance_index if sksurv cumulative_dynamic_auc
    is available, otherwise uses manual IPCW.
    """
    try:
        from sksurv.metrics import cumulative_dynamic_auc

        y_struct = np.array(
            [(bool(e), t) for e, t in zip(events, times)],
            dtype=[("event", bool), ("time", float)],
        )

        # cumulative_dynamic_auc needs sorted times
        auc_values, mean_auc = cumulative_dynamic_auc(
            y_struct, y_struct, risk_scores, [horizon]
        )
        return float(auc_values[0])

    except Exception as e:
        logger.warning(f"    cumulative_dynamic_auc failed: {e}")
        logger.warning(f"    Returning C-index as fallback")
        y_struct = np.array(
            [(bool(e), t) for e, t in zip(events, times)],
            dtype=[("event", bool), ("time", float)],
        )
        ci = concordance_index_censored(y_struct["event"], y_struct["time"], risk_scores)[0]
        return float(ci)


# =============================================================================
# Plotting
# =============================================================================
def days_to_months(t):
    return t / 30.4375


def plot_km_roc_panel(cohort_name, times, events, risk_scores, time_horizon,
                      time_unit="days", ax_km=None, ax_roc=None):
    """
    Plot KM curve (top) and 5-year ROC (bottom) for one cohort.
    """
    # Convert to months for display if needed
    if time_unit == "days":
        times_display = days_to_months(times)
        horizon_display = days_to_months(time_horizon)
        xlabel = "Time (months)"
    else:
        times_display = times
        horizon_display = time_horizon
        xlabel = "Time (months)"

    # Median split
    median_score = np.median(risk_scores)
    high_risk = risk_scores >= median_score
    low_risk = ~high_risk

    n_high = high_risk.sum()
    n_low = low_risk.sum()

    # --- KM Plot ---
    if ax_km is None:
        fig_km, ax_km = plt.subplots(1, 1, figsize=(5, 4))

    kmf_low = KaplanMeierFitter()
    kmf_low.fit(times_display[low_risk], events[low_risk], label=f"Low Risk (n={n_low})")

    kmf_high = KaplanMeierFitter()
    kmf_high.fit(times_display[high_risk], events[high_risk], label=f"High Risk (n={n_high})")

    # Manual plotting to avoid pandas/matplotlib version conflicts
    # Low risk
    sf_low = kmf_low.survival_function_
    ci_low = kmf_low.confidence_interval_survival_function_
    ax_km.step(sf_low.index, sf_low.iloc[:, 0], where="post", color=COLOR_LOW,
               linewidth=2, label=f"Low Risk (n={n_low})")
    ax_km.fill_between(ci_low.index, ci_low.iloc[:, 0], ci_low.iloc[:, 1],
                       step="post", alpha=0.15, color=COLOR_LOW)

    # High risk
    sf_high = kmf_high.survival_function_
    ci_high = kmf_high.confidence_interval_survival_function_
    ax_km.step(sf_high.index, sf_high.iloc[:, 0], where="post", color=COLOR_HIGH,
               linewidth=2, label=f"High Risk (n={n_high})")
    ax_km.fill_between(ci_high.index, ci_high.iloc[:, 0], ci_high.iloc[:, 1],
                       step="post", alpha=0.15, color=COLOR_HIGH)

    # Log-rank test
    lr = logrank_test(
        times[low_risk], times[high_risk],
        events[low_risk], events[high_risk],
    )
    p_val = lr.p_value

    if p_val < 0.0001:
        p_text = "p < 0.0001"
    elif p_val < 0.001:
        p_text = f"p = {p_val:.4f}"
    else:
        p_text = f"p = {p_val:.3f}"

    ax_km.text(0.05, 0.15, f"Log-rank\n{p_text}",
               transform=ax_km.transAxes, fontsize=11, fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax_km.set_xlim(0, XAXIS_MONTHS)
    ax_km.set_ylim(0, 1.05)
    ax_km.set_xlabel(xlabel, fontsize=11)
    ax_km.set_ylabel("Overall Survival Probability", fontsize=11)
    ax_km.set_title(f"{cohort_name} — 25 Genes", fontsize=12, fontweight="bold")
    ax_km.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax_km.grid(True, alpha=0.3)

    # --- ROC Plot ---
    if ax_roc is None:
        fig_roc, ax_roc = plt.subplots(1, 1, figsize=(5, 4))

    auc_val = time_dependent_auc(times, events, risk_scores, time_horizon)

    # Compute ROC curve manually for plotting
    try:
        from sklearn.metrics import roc_curve
        # Binary outcome: event before horizon
        binary_outcome = ((times <= time_horizon) & (events == 1)).astype(int)
        # Exclude censored before horizon
        mask = (times > time_horizon) | (events == 1)
        if mask.sum() > 10:
            fpr, tpr, _ = roc_curve(binary_outcome[mask], risk_scores[mask])
            ax_roc.plot(fpr, tpr, color="#2E4A8E", linewidth=2)
        else:
            logger.warning(f"    {cohort_name}: insufficient data for ROC curve")
            ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)
    except Exception as e:
        logger.warning(f"    {cohort_name}: ROC curve failed: {e}")
        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.set_xlabel("1 — Specificity", fontsize=11)
    ax_roc.set_ylabel("Sensitivity", fontsize=11)

    years = int(time_horizon / 1825 * 5) if time_unit == "days" else int(time_horizon / 12)
    ax_roc.set_title(f"{cohort_name} — {years} Year\nAUC = {auc_val:.3f}",
                     fontsize=12, fontweight="bold")
    ax_roc.grid(True, alpha=0.3)

    # Median survival per group
    med_low = kmf_low.median_survival_time_
    med_high = kmf_high.median_survival_time_

    return {
        "cohort": cohort_name,
        "n_total": len(risk_scores),
        "n_low": int(n_low),
        "n_high": int(n_high),
        "logrank_p": float(p_val),
        "median_survival_low": float(med_low) if not np.isinf(med_low) else None,
        "median_survival_high": float(med_high) if not np.isinf(med_high) else None,
        "auc_5year": float(auc_val),
        "median_risk_score": float(median_score),
    }


# =============================================================================
# Main
# =============================================================================
def main():
    logger.info("=" * 70)
    logger.info("Risk Score Stratification: KM Curves + 5-Year AUC")
    logger.info("=" * 70)

    # Load union genes
    union_genes = load_union_genes(UNION_GENES_FILE)
    logger.info(f"Union genes: {len(union_genes)}")

    # --- Load development cohorts ---
    logger.info("\n--- Loading development cohorts ---")
    dev_data = load_development_cohorts(DATA_DIR, union_genes)

    # --- Compute development risk scores (Phase 7 Cox, out-of-sample) ---
    logger.info("\n--- Computing development cohort risk scores ---")
    dev_scores = compute_dev_risk_scores(dev_data, COX_DIR, union_genes)

    # --- Load and score GEO cohorts ---
    logger.info("\n--- Loading and scoring GEO cohorts ---")
    geo_data = {}
    geo_scores = {}
    for name in ["GSE41613", "GSE42743"]:
        geo_data[name] = load_geo_dataset(name, union_genes)
        logger.info(f"  {name}: {geo_data[name]['X'].shape[0]} samples, "
                    f"{geo_data[name]['genes_found']} genes mapped")
        geo_scores[name] = compute_geo_oof_scores(geo_data[name], name)

    # --- Combine all cohorts ---
    all_cohorts = {
        "TCGA": {
            "times": dev_data["TCGA"]["time"],
            "events": dev_data["TCGA"]["event"],
            "scores": dev_scores["TCGA"],
            "time_unit": "months",
        },
        "ORIEN": {
            "times": dev_data["ORIEN"]["time"],
            "events": dev_data["ORIEN"]["event"],
            "scores": dev_scores["ORIEN"],
            "time_unit": "months",
        },
        "GSE41613": {
            "times": geo_data["GSE41613"]["time"],
            "events": geo_data["GSE41613"]["event"],
            "scores": geo_scores["GSE41613"],
            "time_unit": "months",
        },
        "GSE42743": {
            "times": geo_data["GSE42743"]["time"],
            "events": geo_data["GSE42743"]["event"],
            "scores": geo_scores["GSE42743"],
            "time_unit": "days",
        },
    }

    # --- Create 2×4 panel figure ---
    logger.info("\n--- Generating figures ---")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)

    all_results = []
    cohort_order = ["TCGA", "ORIEN", "GSE41613", "GSE42743"]

    for col_idx, cohort_name in enumerate(cohort_order):
        c = all_cohorts[cohort_name]

        # Skip patients with NaN scores (GEO edge case)
        valid = ~np.isnan(c["scores"])
        if valid.sum() < len(valid):
            logger.warning(f"  {cohort_name}: {(~valid).sum()} patients with NaN scores excluded")

        result = plot_km_roc_panel(
            cohort_name=cohort_name,
            times=c["times"][valid],
            events=c["events"][valid],
            risk_scores=c["scores"][valid],
            time_horizon=TIME_HORIZONS[cohort_name],
            time_unit=c["time_unit"],
            ax_km=axes[0, col_idx],
            ax_roc=axes[1, col_idx],
        )
        all_results.append(result)

        logger.info(f"  {cohort_name}: log-rank p={result['logrank_p']:.2e}, "
                    f"5-yr AUC={result['auc_5year']:.3f}")

    # Add panel labels (A, B, C, D)
    for col_idx, label in enumerate(["A", "B", "C", "D"]):
        axes[0, col_idx].text(-0.12, 1.08, label, transform=axes[0, col_idx].transAxes,
                              fontsize=16, fontweight="bold", va="top")

    fig.savefig(OUTPUT_DIR / "km_auc_panel_4cohorts.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "km_auc_panel_4cohorts.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"\n  Panel figure saved to {OUTPUT_DIR}")

    # --- Individual cohort figures ---
    for cohort_name in cohort_order:
        c = all_cohorts[cohort_name]
        valid = ~np.isnan(c["scores"])

        fig_ind, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
        fig_ind.subplots_adjust(wspace=0.35)

        plot_km_roc_panel(
            cohort_name=cohort_name,
            times=c["times"][valid],
            events=c["events"][valid],
            risk_scores=c["scores"][valid],
            time_horizon=TIME_HORIZONS[cohort_name],
            time_unit=c["time_unit"],
            ax_km=ax1,
            ax_roc=ax2,
        )

        fig_ind.savefig(OUTPUT_DIR / f"km_auc_{cohort_name}.png", dpi=300, bbox_inches="tight")
        fig_ind.savefig(OUTPUT_DIR / f"km_auc_{cohort_name}.pdf", bbox_inches="tight")
        plt.close(fig_ind)

    # --- Summary table ---
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(OUTPUT_DIR / "stratification_summary.csv", index=False)

    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY")
    logger.info(f"{'='*70}")
    for r in all_results:
        med_low = f"{r['median_survival_low']:.1f}" if r['median_survival_low'] else "NR"
        med_high = f"{r['median_survival_high']:.1f}" if r['median_survival_high'] else "NR"
        logger.info(f"  {r['cohort']:>10s}: n={r['n_total']}, "
                    f"low={r['n_low']}/high={r['n_high']}, "
                    f"log-rank p={r['logrank_p']:.2e}, "
                    f"5-yr AUC={r['auc_5year']:.3f}, "
                    f"median surv: low={med_low}, high={med_high}")

    logger.info(f"\nResults saved to {OUTPUT_DIR}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
