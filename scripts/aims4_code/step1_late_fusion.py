"""
Step 1: Late Fusion Second-Stage Cox (GEO — Primary Analysis)
==============================================================
Chapter 3, Aim 1: Clinical Integration via Late Fusion

Tests whether combining gene expression and clinical risk scores
improves prognostic prediction beyond either alone.

Design:
  - Input: per-patient out-of-fold risk scores from Step 0B
    (gene-only ridge Cox + clinical-only ridge Cox)
  - Second-stage model: UNPENALIZED Cox with 2 predictors
    h(t) = h0(t) × exp(β_gene × gene_score + β_clin × clin_score)
  - Unpenalized because only 2 predictors with n=71/97 (EPV > 20)
  - Enables interpretable β coefficients, SEs, and likelihood ratio tests

Analysis per dataset:
  1. Apparent fit: Cox on all patients → β coefficients, 95% CI, p-values
  2. Likelihood ratio tests:
     - Combined vs clinical-only (does gene score add value?)
     - Combined vs gene-only (does clinical score add value?)
  3. CV-corrected C-index: 5-fold stratified CV, 3 seeds
  4. Comparison: combined vs gene-only vs clinical-only C-indices

Library usage:
  - lifelines.CoxPHFitter: model fitting (β, SE, log-likelihood, LRT)
  - sksurv.metrics.concordance_index_censored: ALL C-index computation
    (consistent with Steps 0B and Phase 8 pipeline)

Seed strategy:
  - Primary first-stage seed: 42 (main analysis)
  - Sensitivity: first-stage seeds 123, 456
  - Second-stage CV seeds: 42, 123, 456

Inferential framework:
  - GEO datasets are EXTERNAL (not used in gene selection)
  - Results are inferential for the late fusion question
  - Post-selection does NOT apply (genes selected on TCGA/ORIEN)

References:
  - Late fusion: Dai et al. (2021) Brief Bioinform 22(3):bbab048
  - Stacking: Van der Laan et al. (2007) Stat Appl Genet Mol Biol
  - Cox regression: Cox (1972) JRSS-B 34(2):187-220
  - Likelihood ratio test: Wilks (1938)
  - C-index: Harrell et al. (1982) JAMA 247(18):2543-2546

Author: Phuong Nguyen
Date: March 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import CoxPHFitter
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import warnings
import logging
import json

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================
FIRST_STAGE_SEEDS = [42, 123, 456]
PRIMARY_SEED = 42
SECOND_STAGE_SEEDS = [42, 123, 456]
N_FOLDS = 5

INPUT_DIR = Path("results/chapter3/step0b")
OUTPUT_DIR = Path("results/chapter3/step1_late_fusion")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["GSE42743", "GSE41613"]


# =============================================================================
# Utility: Consistent C-Index Computation
# =============================================================================
def compute_cindex(events, times, risk_scores):
    """
    Compute C-index using sksurv for consistency across entire pipeline.

    All C-index values in Steps 0B, 1, and Phase 8 use this function.
    """
    return concordance_index_censored(
        events.astype(bool), times, risk_scores
    )[0]


# =============================================================================
# Data Loading
# =============================================================================
def load_oof_scores(dataset, model_type):
    """Load per-patient out-of-fold scores from Step 0B."""
    path = INPUT_DIR / f"{dataset}_{model_type}_oof_scores.csv"
    df = pd.read_csv(path)
    logger.info(f"  Loaded {path.name}: {len(df)} patients")
    return df


def prepare_fusion_data(dataset, first_stage_seed):
    """
    Merge gene and clinical OOF scores for a given first-stage seed.

    Returns DataFrame with columns:
        sampleID, gene_score, clinical_score, event, time
    """
    gene_df = load_oof_scores(dataset, "gene")
    clin_df = load_oof_scores(dataset, "clinical")

    score_col = f"risk_score_seed{first_stage_seed}"

    # Verify both have the same patients in same order
    assert (gene_df["sampleID"] == clin_df["sampleID"]).all(), \
        "Sample ID mismatch between gene and clinical scores"
    assert (gene_df["event"] == clin_df["event"]).all(), \
        "Event mismatch between gene and clinical scores"

    merged = pd.DataFrame({
        "sampleID": gene_df["sampleID"],
        "gene_score": gene_df[score_col].values,
        "clinical_score": clin_df[score_col].values,
        "event": gene_df["event"].values,
        "time": gene_df["time"].values,
    })

    # Check for NaN scores
    n_nan = merged[["gene_score", "clinical_score"]].isna().sum().sum()
    if n_nan > 0:
        logger.warning(f"  WARNING: {n_nan} NaN values in scores")

    return merged


# =============================================================================
# Apparent Fit: Full-Data Cox Models
# =============================================================================
def fit_cox_and_summarize(df, covariates, label=""):
    """
    Fit unpenalized Cox model using lifelines (for β, SE, log-likelihood).
    Compute C-index using sksurv (for consistency).

    Returns:
        cph: fitted CoxPHFitter object
        summary: dict with coefficients, log-likelihood, C-index
    """
    fit_df = df[covariates + ["event", "time"]].copy()

    cph = CoxPHFitter()
    try:
        cph.fit(fit_df, duration_col="time", event_col="event")
    except Exception as e:
        logger.error(f"  {label}: Cox fitting failed — {e}")
        return None, None

    # C-index via sksurv (consistent with rest of pipeline)
    risk_scores = cph.predict_partial_hazard(df[covariates]).values.flatten()
    c_index = compute_cindex(
        df["event"].values, df["time"].values, risk_scores
    )

    # Extract results
    summary = {}
    summary["n"] = len(fit_df)
    summary["n_events"] = int(fit_df["event"].sum())
    summary["concordance"] = c_index
    summary["log_likelihood"] = cph.log_likelihood_
    summary["AIC"] = cph.AIC_partial_
    summary["covariates"] = covariates

    # Per-covariate results
    coef_results = []
    for var in covariates:
        coef_results.append({
            "variable": var,
            "coef": float(cph.params_[var]),
            "exp_coef": float(np.exp(cph.params_[var])),
            "se": float(cph.summary.loc[var, "se(coef)"]),
            "z": float(cph.summary.loc[var, "z"]),
            "p": float(cph.summary.loc[var, "p"]),
            "ci_lower_coef": float(cph.confidence_intervals_.loc[var].iloc[0]),
            "ci_upper_coef": float(cph.confidence_intervals_.loc[var].iloc[1]),
            "hr_lower": float(np.exp(cph.confidence_intervals_.loc[var].iloc[0])),
            "hr_upper": float(np.exp(cph.confidence_intervals_.loc[var].iloc[1])),
        })
    summary["coefficients"] = coef_results

    return cph, summary


def likelihood_ratio_test(ll_full, ll_reduced, df_diff):
    """
    Likelihood ratio test comparing nested Cox models.

    LRT statistic = -2 × (ll_reduced - ll_full)
    Under H0: LRT ~ chi-squared(df_diff)
    """
    lrt_stat = -2 * (ll_reduced - ll_full)
    p_value = stats.chi2.sf(lrt_stat, df=df_diff)
    return {
        "lrt_statistic": float(lrt_stat),
        "df": int(df_diff),
        "p_value": float(p_value)
    }


def run_apparent_analysis(data, dataset_name):
    """
    Fit three Cox models on full data and compare:
      1. Combined: gene_score + clinical_score
      2. Gene-only: gene_score
      3. Clinical-only: clinical_score

    Perform LRT for nested model comparisons.
    """
    logger.info(f"\n  --- Apparent Fit (full data, n={len(data)}, "
                f"events={data['event'].sum()}) ---")

    results = {}

    # 1. Combined model
    cph_combined, res_combined = fit_cox_and_summarize(
        data, ["gene_score", "clinical_score"], label="Combined"
    )
    if res_combined:
        results["combined"] = res_combined
        logger.info(f"    Combined C-index (sksurv): {res_combined['concordance']:.4f}")
        for c in res_combined["coefficients"]:
            logger.info(f"      {c['variable']}: β={c['coef']:.4f}, "
                        f"HR={c['exp_coef']:.3f} "
                        f"(95% CI: {c['hr_lower']:.3f}-{c['hr_upper']:.3f}), "
                        f"p={c['p']:.4f}")

    # 2. Gene-only model
    cph_gene, res_gene = fit_cox_and_summarize(
        data, ["gene_score"], label="Gene-only"
    )
    if res_gene:
        results["gene_only"] = res_gene
        logger.info(f"    Gene-only C-index (sksurv): {res_gene['concordance']:.4f}")

    # 3. Clinical-only model
    cph_clin, res_clin = fit_cox_and_summarize(
        data, ["clinical_score"], label="Clinical-only"
    )
    if res_clin:
        results["clinical_only"] = res_clin
        logger.info(f"    Clinical-only C-index (sksurv): {res_clin['concordance']:.4f}")

    # --- Likelihood Ratio Tests ---
    logger.info(f"\n  --- Likelihood Ratio Tests ---")

    # Test 1: Combined vs Clinical-only (does gene score add value?)
    if res_combined and res_clin:
        lrt_gene = likelihood_ratio_test(
            res_combined["log_likelihood"],
            res_clin["log_likelihood"],
            df_diff=1
        )
        results["lrt_gene_adds"] = lrt_gene
        sig = "***" if lrt_gene["p_value"] < 0.001 else \
              "**" if lrt_gene["p_value"] < 0.01 else \
              "*" if lrt_gene["p_value"] < 0.05 else "ns"
        logger.info(f"    Combined vs Clinical-only (does gene add value?):")
        logger.info(f"      LRT χ² = {lrt_gene['lrt_statistic']:.4f}, "
                    f"df = {lrt_gene['df']}, "
                    f"p = {lrt_gene['p_value']:.4f} {sig}")

    # Test 2: Combined vs Gene-only (does clinical score add value?)
    if res_combined and res_gene:
        lrt_clin = likelihood_ratio_test(
            res_combined["log_likelihood"],
            res_gene["log_likelihood"],
            df_diff=1
        )
        results["lrt_clinical_adds"] = lrt_clin
        sig = "***" if lrt_clin["p_value"] < 0.001 else \
              "**" if lrt_clin["p_value"] < 0.01 else \
              "*" if lrt_clin["p_value"] < 0.05 else "ns"
        logger.info(f"    Combined vs Gene-only (does clinical add value?):")
        logger.info(f"      LRT χ² = {lrt_clin['lrt_statistic']:.4f}, "
                    f"df = {lrt_clin['df']}, "
                    f"p = {lrt_clin['p_value']:.4f} {sig}")

    return results


# =============================================================================
# CV-Corrected C-Index
# =============================================================================
def cv_model_cindex(data, covariates, second_stage_seed):
    """
    5-fold stratified CV for C-index estimation.

    In each fold:
      - Train unpenalized Cox on training set
      - Predict risk on held-out set using lifelines
      - Compute held-out C-index using sksurv

    Returns: (mean_cindex, std_cindex, fold_cindices)
    """
    events = data["event"].values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=second_stage_seed)

    fold_cindices = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(data, events)):
        train_df = data.iloc[train_idx].copy()
        test_df = data.iloc[test_idx].copy()

        test_events = test_df["event"].sum()
        if test_events < 2:
            logger.warning(f"      Fold {fold_idx+1}: <2 events, skipping")
            continue

        # Fit on training data (lifelines)
        cph = CoxPHFitter()
        try:
            fit_cols = covariates + ["event", "time"]
            cph.fit(train_df[fit_cols], duration_col="time", event_col="event")
        except Exception as e:
            logger.warning(f"      Fold {fold_idx+1}: fitting failed — {e}")
            continue

        # Predict on test data (lifelines partial hazard)
        risk_scores = cph.predict_partial_hazard(
            test_df[covariates]
        ).values.flatten()

        # C-index on held-out fold (sksurv — consistent with pipeline)
        try:
            c_idx = compute_cindex(
                test_df["event"].values,
                test_df["time"].values,
                risk_scores
            )
            fold_cindices.append(c_idx)
        except Exception as e:
            logger.warning(f"      Fold {fold_idx+1}: C-index failed — {e}")

    if len(fold_cindices) == 0:
        return np.nan, np.nan, []

    return np.mean(fold_cindices), np.std(fold_cindices), fold_cindices


def run_cv_analysis(data, dataset_name):
    """
    Run CV-corrected C-index estimation for combined, gene-only,
    and clinical-only models, across 3 second-stage seeds.

    All three models use the SAME fold splits per seed for fair comparison.
    """
    logger.info(f"\n  --- CV-Corrected C-Index (5-fold, 3 second-stage seeds) ---")

    cv_results = {"combined": [], "gene_only": [], "clinical_only": []}

    for seed in SECOND_STAGE_SEEDS:
        logger.info(f"    Second-stage seed {seed}:")

        # Combined (same folds for all three)
        c_comb, s_comb, _ = cv_model_cindex(
            data, ["gene_score", "clinical_score"], seed
        )
        cv_results["combined"].append(c_comb)
        logger.info(f"      Combined:      {c_comb:.4f} ± {s_comb:.4f}")

        # Gene-only (same folds)
        c_gene, s_gene, _ = cv_model_cindex(
            data, ["gene_score"], seed
        )
        cv_results["gene_only"].append(c_gene)
        logger.info(f"      Gene-only:     {c_gene:.4f} ± {s_gene:.4f}")

        # Clinical-only (same folds)
        c_clin, s_clin, _ = cv_model_cindex(
            data, ["clinical_score"], seed
        )
        cv_results["clinical_only"].append(c_clin)
        logger.info(f"      Clinical-only: {c_clin:.4f} ± {s_clin:.4f}")

    # Summary across second-stage seeds
    logger.info(f"\n  CV Summary (mean across second-stage seeds):")
    for model in ["combined", "gene_only", "clinical_only"]:
        vals = cv_results[model]
        mean_c = np.mean(vals)
        range_c = np.max(vals) - np.min(vals)
        logger.info(f"    {model:15s}: {mean_c:.4f} "
                    f"(seeds: {[f'{v:.4f}' for v in vals]}, "
                    f"range: {range_c:.4f})")

    return cv_results


# =============================================================================
# Sensitivity: Naive Score Addition
# =============================================================================
def run_naive_addition(data, dataset_name):
    """
    Sensitivity analysis: simple addition of gene + clinical scores
    without refitting. Tests if even naive combination helps.

    Per Chapter 3 plan Step 1.4c.
    """
    logger.info(f"\n  --- Sensitivity: Naive Score Addition ---")

    data_copy = data.copy()
    data_copy["naive_combined"] = data_copy["gene_score"] + data_copy["clinical_score"]

    c_naive = compute_cindex(
        data_copy["event"].values,
        data_copy["time"].values,
        data_copy["naive_combined"].values
    )

    c_gene = compute_cindex(
        data_copy["event"].values,
        data_copy["time"].values,
        data_copy["gene_score"].values
    )

    c_clin = compute_cindex(
        data_copy["event"].values,
        data_copy["time"].values,
        data_copy["clinical_score"].values
    )

    logger.info(f"    Naive (gene + clinical):  {c_naive:.4f}")
    logger.info(f"    Gene-only (raw OOF):      {c_gene:.4f}")
    logger.info(f"    Clinical-only (raw OOF):  {c_clin:.4f}")

    return {
        "naive_combined": float(c_naive),
        "gene_only_raw": float(c_gene),
        "clinical_only_raw": float(c_clin)
    }


# =============================================================================
# Main Analysis for One Dataset + One First-Stage Seed
# =============================================================================
def analyze_dataset_seed(dataset, first_stage_seed):
    """Run complete late fusion analysis for one dataset and first-stage seed."""

    logger.info(f"\n{'='*60}")
    logger.info(f"  {dataset} | First-stage seed: {first_stage_seed}")
    logger.info(f"{'='*60}")

    # Load and merge scores
    data = prepare_fusion_data(dataset, first_stage_seed)
    logger.info(f"  Merged data: {len(data)} patients, "
                f"{data['event'].sum()} events")
    logger.info(f"  Gene score:     mean={data['gene_score'].mean():.4f}, "
                f"std={data['gene_score'].std():.4f}")
    logger.info(f"  Clinical score: mean={data['clinical_score'].mean():.4f}, "
                f"std={data['clinical_score'].std():.4f}")

    # Correlation between scores (measures complementarity)
    corr = data["gene_score"].corr(data["clinical_score"])
    logger.info(f"  Pearson correlation (gene vs clinical): {corr:.4f}")
    logger.info(f"    (Low correlation → more complementary → fusion more likely to help)")

    results = {}
    results["dataset"] = dataset
    results["first_stage_seed"] = first_stage_seed
    results["n_patients"] = len(data)
    results["n_events"] = int(data["event"].sum())
    results["score_correlation"] = float(corr)

    # 1. Apparent fit + LRT
    apparent = run_apparent_analysis(data, dataset)
    results["apparent"] = apparent

    # 2. CV-corrected C-index
    cv = run_cv_analysis(data, dataset)
    results["cv"] = cv

    # 3. Naive addition sensitivity
    naive = run_naive_addition(data, dataset)
    results["naive_addition"] = naive

    return results


# =============================================================================
# Results Formatting
# =============================================================================
def format_summary_table(all_results):
    """Create a summary table across all datasets and seeds."""
    rows = []

    for result in all_results:
        dataset = result["dataset"]
        fs_seed = result["first_stage_seed"]
        is_primary = fs_seed == PRIMARY_SEED

        # Apparent C-indices
        for model in ["combined", "gene_only", "clinical_only"]:
            if model in result["apparent"]:
                rows.append({
                    "dataset": dataset,
                    "first_stage_seed": fs_seed,
                    "primary": is_primary,
                    "model": model,
                    "apparent_cindex": result["apparent"][model]["concordance"],
                    "cv_cindex_mean": np.mean(result["cv"][model]),
                    "cv_cindex_seeds": result["cv"][model],
                })

        # LRT results
        if "lrt_gene_adds" in result["apparent"]:
            lrt = result["apparent"]["lrt_gene_adds"]
            rows[-3]["lrt_gene_adds_p"] = lrt["p_value"]
            rows[-3]["lrt_gene_adds_chi2"] = lrt["lrt_statistic"]

        if "lrt_clinical_adds" in result["apparent"]:
            lrt = result["apparent"]["lrt_clinical_adds"]
            rows[-3]["lrt_clinical_adds_p"] = lrt["p_value"]
            rows[-3]["lrt_clinical_adds_chi2"] = lrt["lrt_statistic"]

    return pd.DataFrame(rows)


def format_coefficient_table(all_results):
    """Create a table of combined model coefficients."""
    rows = []

    for result in all_results:
        dataset = result["dataset"]
        fs_seed = result["first_stage_seed"]

        if "combined" in result["apparent"]:
            for coef in result["apparent"]["combined"]["coefficients"]:
                rows.append({
                    "dataset": dataset,
                    "first_stage_seed": fs_seed,
                    "variable": coef["variable"],
                    "beta": coef["coef"],
                    "HR": coef["exp_coef"],
                    "HR_lower": coef["hr_lower"],
                    "HR_upper": coef["hr_upper"],
                    "se": coef["se"],
                    "z": coef["z"],
                    "p": coef["p"],
                })

    return pd.DataFrame(rows)


# =============================================================================
# JSON Serialization Helper
# =============================================================================
def make_serializable(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return obj


# =============================================================================
# Main
# =============================================================================
def main():
    logger.info("Step 1: Late Fusion Second-Stage Cox (GEO)")
    logger.info("=" * 60)
    logger.info(f"Primary first-stage seed: {PRIMARY_SEED}")
    logger.info(f"Sensitivity seeds: {[s for s in FIRST_STAGE_SEEDS if s != PRIMARY_SEED]}")
    logger.info(f"Second-stage CV seeds: {SECOND_STAGE_SEEDS}")

    all_results = []

    # --- Primary analysis: seed 42 ---
    for dataset in DATASETS:
        result = analyze_dataset_seed(dataset, PRIMARY_SEED)
        all_results.append(result)

    # --- Sensitivity: seeds 123, 456 ---
    for dataset in DATASETS:
        for seed in FIRST_STAGE_SEEDS:
            if seed == PRIMARY_SEED:
                continue
            result = analyze_dataset_seed(dataset, seed)
            all_results.append(result)

    # =================================================================
    # Summary Output
    # =================================================================
    logger.info(f"\n{'='*60}")
    logger.info("  STEP 1 FINAL SUMMARY")
    logger.info(f"{'='*60}")

    # --- Primary results (seed 42) ---
    logger.info(f"\n  PRIMARY RESULTS (first-stage seed {PRIMARY_SEED}):")
    logger.info(f"  {'─'*55}")

    for dataset in DATASETS:
        primary = [r for r in all_results
                   if r["dataset"] == dataset and r["first_stage_seed"] == PRIMARY_SEED][0]

        logger.info(f"\n  {dataset} (n={primary['n_patients']}, "
                    f"events={primary['n_events']}):")
        logger.info(f"    Score correlation: {primary['score_correlation']:.4f}")

        # Apparent C-indices
        app = primary["apparent"]
        logger.info(f"    Apparent C-index:")
        for model in ["combined", "gene_only", "clinical_only"]:
            if model in app:
                logger.info(f"      {model:15s}: {app[model]['concordance']:.4f}")

        # CV C-indices
        cv = primary["cv"]
        logger.info(f"    CV C-index (mean across second-stage seeds):")
        for model in ["combined", "gene_only", "clinical_only"]:
            logger.info(f"      {model:15s}: {np.mean(cv[model]):.4f}")

        # LRT
        if "lrt_gene_adds" in app:
            lrt = app["lrt_gene_adds"]
            logger.info(f"    LRT gene adds to clinical: "
                        f"χ²={lrt['lrt_statistic']:.2f}, p={lrt['p_value']:.4f}")
        if "lrt_clinical_adds" in app:
            lrt = app["lrt_clinical_adds"]
            logger.info(f"    LRT clinical adds to gene: "
                        f"χ²={lrt['lrt_statistic']:.2f}, p={lrt['p_value']:.4f}")

    # --- Save results ---
    # 1. Summary table
    summary_df = format_summary_table(all_results)
    summary_df.to_csv(OUTPUT_DIR / "step1_summary.csv", index=False)
    logger.info(f"\n  Saved: {OUTPUT_DIR / 'step1_summary.csv'}")

    # 2. Coefficient table
    coef_df = format_coefficient_table(all_results)
    coef_df.to_csv(OUTPUT_DIR / "step1_coefficients.csv", index=False)
    logger.info(f"  Saved: {OUTPUT_DIR / 'step1_coefficients.csv'}")

    # 3. Full detailed results (JSON)
    with open(OUTPUT_DIR / "step1_detailed.json", "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    logger.info(f"  Saved: {OUTPUT_DIR / 'step1_detailed.json'}")

    logger.info(f"\nStep 1 complete.")


if __name__ == "__main__":
    main()
