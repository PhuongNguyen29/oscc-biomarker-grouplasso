#!/usr/bin/env python3
"""
Analysis 5.4: Cox Interaction Patterns (Exploratory)
=====================================================

Chapter 3 — ElasticDeepSurv with Interpretable Biomarker Discovery

Purpose:
    For each consensus gene G and clinical variable C, fit:
        h(t) = h0(t) * exp(β1*G + β2*C + β3*G*C)
    and examine β3 (interaction term).

    Report effect sizes and directions. FDR-adjusted q-values computed
    per cohort and reported in supplement as descriptive summaries,
    not used for significance claims.

Reference:
    - Cox (1972). Regression models and life-tables. JRSS-B.
    - Benjamini & Hochberg (1995). Controlling the false discovery rate. JRSS-B.
    - Royston et al. (2006). Dichotomizing continuous predictors in multiple
      regression: a bad idea. Statistics in Medicine.

Design:
    - 64 consensus genes × 6 clinical variables × 2 cohorts = 768 models
    - Age: continuous (per Royston et al., 2006)
    - Other 5 clinical variables: binary (0/1)
    - FDR correction: BH per cohort (384 tests each)
    - Unknown values: excluded per-variable, not globally

Input files:
    - data/processed/{tcga,orien}_preprocessed_v4.csv
    - data/processed/surv_{tcga,orien}_aligned_v4.csv
    - data/processed/clinical_{tcga,orien}_aligned_v4.csv
    - results/phase4_stability_mad3000/consensus/cox_consensus_pi080.txt
    - data/raw/protein_coding_genes_annotation.csv

Output files:
    - results/phase5/analysis_5_4/analysis_5_4_interaction_results.csv
    - results/phase5/analysis_5_4/analysis_5_4_summary.txt

Author: Phuong Nguyen
Date: February 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from lifelines import CoxPHFitter
from statsmodels.stats.multitest import multipletests
import warnings
import logging
import time

# ==============================================================================
# Configuration
# ==============================================================================

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"
CONSENSUS_PATH = (
    BASE_DIR / "results" / "phase4_stability_mad3000" / "consensus"
    / "cox_consensus_pi080.txt"
)
ANNOTATION_PATH = RAW_DIR / "protein_coding_genes_annotation.csv"
OUTPUT_DIR = BASE_DIR / "results" / "phase5" / "analysis_5_4"

COHORTS = ["tcga", "orien"]

# Clinical variable definitions
# Key: column name in clinical CSV
# Value: dict with encoding instructions
CLINICAL_VARS = {
    "age": {
        "type": "continuous",
        "exclude_values": [],  # no unknowns for age
    },
    "gender": {
        "type": "binary",
        "exclude_values": [],  # already 0/1, no unknowns
        # 0 = Female, 1 = Male (already encoded)
    },
    "smoking": {
        "type": "binary",
        "exclude_values": ["Unknown"],
        "encoding": {"Never": 0, "Ever": 1},
    },
    "alcohol": {
        "type": "binary",
        "exclude_values": ["Unknown"],
        "encoding": {"Never": 0, "Ever": 1},
    },
    "N_stage": {
        "type": "binary",
        "exclude_values": ["Unknown"],
        "encoding": {"N0": 0, "N1": 1, "N2-3": 1},  # collapse N1, N2-3 → N+
    },
    "T_stage": {
        "type": "binary",
        "exclude_values": ["Unknown"],
        "encoding": {"T1-T2": 0, "T3-T4": 1},
    },
}

MIN_EVENTS_PER_GROUP = 5  # minimum events for model fitting safety

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Data Loading Functions
# ==============================================================================


def load_expression(cohort_name):
    """Load preprocessed expression data (genes × samples), transpose to samples × genes."""
    path = DATA_DIR / f"{cohort_name}_preprocessed_v4.csv"
    expr = pd.read_csv(path, index_col=0)
    # Expression files: genes as rows, samples as columns → transpose
    expr = expr.T
    logger.info(f"  {cohort_name.upper()} expression: {expr.shape[0]} samples × {expr.shape[1]} genes")
    return expr


def load_survival(cohort_name):
    """Load survival data, handling R row-number column."""
    path = DATA_DIR / f"surv_{cohort_name}_aligned_v4.csv"
    surv = pd.read_csv(path)
    # Handle R row-number column
    if surv.columns[0] == "" or "Unnamed" in str(surv.columns[0]):
        surv = surv.drop(columns=[surv.columns[0]])
    surv = surv.set_index("sampleID")
    logger.info(
        f"  {cohort_name.upper()} survival: {len(surv)} samples, "
        f"{int(surv['event'].sum())} events ({100*surv['event'].mean():.1f}%)"
    )
    return surv


def load_clinical(cohort_name):
    """Load clinical data."""
    path = DATA_DIR / f"clinical_{cohort_name}_aligned_v4.csv"
    clin = pd.read_csv(path)
    clin = clin.set_index("sampleID")
    logger.info(f"  {cohort_name.upper()} clinical: {len(clin)} samples, columns: {list(clin.columns)}")
    return clin


def load_consensus_genes():
    """Load consensus gene list."""
    genes = pd.read_csv(CONSENSUS_PATH, header=None)[0].tolist()
    logger.info(f"  Consensus genes: {len(genes)}")
    return genes


def load_gene_annotation():
    """Load gene annotation for ENSG → symbol mapping."""
    annot = pd.read_csv(ANNOTATION_PATH)
    # Create mapping dict: gene_id → gene_symbol
    gene_map = dict(zip(annot["gene_id"], annot["gene_symbol"]))
    logger.info(f"  Gene annotation: {len(gene_map)} entries")
    return gene_map


# ==============================================================================
# Clinical Encoding
# ==============================================================================


def encode_clinical_variable(clinical_df, var_name, var_config):
    """
    Encode a clinical variable and return the encoded series + valid sample mask.

    Returns:
        encoded: pd.Series with encoded values (NaN for excluded)
        valid_mask: boolean Series (True = sample is usable)
    """
    raw = clinical_df[var_name].copy()

    if var_config["type"] == "continuous":
        # Age: already numeric, just check for NaN
        encoded = pd.to_numeric(raw, errors="coerce")
        valid_mask = encoded.notna()
    else:
        # Binary variables
        exclude_vals = var_config["exclude_values"]

        if "encoding" in var_config:
            # Map string values to 0/1
            encoded = raw.map(var_config["encoding"])
            valid_mask = raw.isin(var_config["encoding"].keys())
        else:
            # Already numeric (e.g., gender)
            encoded = pd.to_numeric(raw, errors="coerce")
            valid_mask = encoded.notna()

        # Exclude specified values
        for val in exclude_vals:
            valid_mask = valid_mask & (raw != val)

    return encoded, valid_mask


# ==============================================================================
# Cox Interaction Model
# ==============================================================================


def fit_cox_interaction(expr_values, clinical_values, surv_df, gene_id, var_name):
    """
    Fit Cox model: h(t) = h0(t) * exp(β1*G + β2*C + β3*G*C)

    Parameters:
        expr_values: pd.Series, gene expression for valid samples
        clinical_values: pd.Series, encoded clinical variable for valid samples
        surv_df: pd.DataFrame with 'time' and 'event' for valid samples
        gene_id: str, ENSG ID (for logging)
        var_name: str, clinical variable name (for logging)

    Returns:
        dict with model results, or None if model failed
    """
    # Build model dataframe
    df = pd.DataFrame(
        {
            "gene": expr_values.values,
            "clinical": clinical_values.values,
            "gene_x_clinical": expr_values.values * clinical_values.values,
            "time": surv_df["time"].values,
            "event": surv_df["event"].values,
        }
    )

    # Safety checks
    if df["time"].min() <= 0:
        df = df[df["time"] > 0]
    if len(df) < 10:
        return None
    if df["event"].sum() < MIN_EVENTS_PER_GROUP:
        return None

    # Fit Cox model
    try:
        cph = CoxPHFitter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(
                df,
                duration_col="time",
                event_col="event",
                show_progress=False,
            )

        # Extract interaction term results
        summary = cph.summary
        interaction_row = summary.loc["gene_x_clinical"]

        result = {
            "beta_interaction": interaction_row["coef"],
            "se_interaction": interaction_row["se(coef)"],
            "pvalue": interaction_row["p"],
            "hr_interaction": interaction_row["exp(coef)"],
            "hr_lower95": interaction_row["exp(coef) lower 95%"],
            "hr_upper95": interaction_row["exp(coef) upper 95%"],
            "n_samples": len(df),
            "n_events": int(df["event"].sum()),
            "converged": True,
        }

        # Also extract main effects for context
        result["beta_gene"] = summary.loc["gene", "coef"]
        result["beta_clinical"] = summary.loc["clinical", "coef"]

        return result

    except Exception as e:
        logger.warning(f"  Model failed for {gene_id} × {var_name}: {e}")
        return {
            "beta_interaction": np.nan,
            "se_interaction": np.nan,
            "pvalue": np.nan,
            "hr_interaction": np.nan,
            "hr_lower95": np.nan,
            "hr_upper95": np.nan,
            "n_samples": len(df),
            "n_events": int(df["event"].sum()),
            "converged": False,
            "beta_gene": np.nan,
            "beta_clinical": np.nan,
        }


# ==============================================================================
# Main Analysis
# ==============================================================================


def run_analysis():
    """Run Analysis 5.4: Cox Interaction Patterns."""
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("Analysis 5.4: Cox Interaction Patterns (Exploratory)")
    logger.info("=" * 70)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    logger.info("\n--- Loading data ---")
    consensus_genes = load_consensus_genes()
    gene_map = load_gene_annotation()

    data = {}
    for cohort in COHORTS:
        logger.info(f"\nLoading {cohort.upper()}:")
        data[cohort] = {
            "expr": load_expression(cohort),
            "surv": load_survival(cohort),
            "clin": load_clinical(cohort),
        }

    # ---- Verify gene availability ----
    available_genes = []
    for gene in consensus_genes:
        in_tcga = gene in data["tcga"]["expr"].columns
        in_orien = gene in data["orien"]["expr"].columns
        if in_tcga and in_orien:
            available_genes.append(gene)
        else:
            logger.warning(f"  Gene {gene} missing: TCGA={in_tcga}, ORIEN={in_orien}")

    logger.info(f"\nGenes available in both cohorts: {len(available_genes)}/{len(consensus_genes)}")

    # ---- Run interaction models ----
    logger.info("\n--- Fitting Cox interaction models ---")

    results = []
    total_models = len(available_genes) * len(CLINICAL_VARS) * len(COHORTS)
    model_count = 0
    failed_count = 0

    for cohort in COHORTS:
        expr_df = data[cohort]["expr"]
        surv_df = data[cohort]["surv"]
        clin_df = data[cohort]["clin"]

        # Align samples across expression, survival, clinical
        common_samples = sorted(
            set(expr_df.index) & set(surv_df.index) & set(clin_df.index)
        )
        expr_df = expr_df.loc[common_samples]
        surv_df = surv_df.loc[common_samples]
        clin_df = clin_df.loc[common_samples]

        logger.info(f"\n{cohort.upper()}: {len(common_samples)} aligned samples")

        for var_name, var_config in CLINICAL_VARS.items():
            # Encode clinical variable
            encoded_clin, valid_mask = encode_clinical_variable(
                clin_df, var_name, var_config
            )

            # Apply valid mask
            valid_samples = valid_mask[valid_mask].index
            n_excluded = len(common_samples) - len(valid_samples)

            if n_excluded > 0:
                logger.info(
                    f"  {var_name}: {n_excluded} samples excluded "
                    f"(Unknown/missing), {len(valid_samples)} remaining"
                )

            for gene in available_genes:
                model_count += 1

                # Subset to valid samples
                gene_expr = expr_df.loc[valid_samples, gene]
                clin_vals = encoded_clin.loc[valid_samples]
                surv_valid = surv_df.loc[valid_samples]

                # Fit model
                result = fit_cox_interaction(
                    gene_expr, clin_vals, surv_valid, gene, var_name
                )

                if result is None:
                    failed_count += 1
                    result = {
                        "beta_interaction": np.nan,
                        "se_interaction": np.nan,
                        "pvalue": np.nan,
                        "hr_interaction": np.nan,
                        "hr_lower95": np.nan,
                        "hr_upper95": np.nan,
                        "n_samples": len(valid_samples),
                        "n_events": np.nan,
                        "converged": False,
                        "beta_gene": np.nan,
                        "beta_clinical": np.nan,
                    }
                elif not result["converged"]:
                    failed_count += 1

                # Add identifiers
                result["gene_id"] = gene
                result["gene_symbol"] = gene_map.get(gene, "Unknown")
                result["clinical_var"] = var_name
                result["cohort"] = cohort

                results.append(result)

                # Progress logging
                if model_count % 100 == 0:
                    elapsed = time.time() - start_time
                    pct = 100 * model_count / total_models
                    logger.info(
                        f"  Progress: {model_count}/{total_models} ({pct:.0f}%) "
                        f"- {elapsed:.0f}s elapsed"
                    )

    # ---- Compile results ----
    logger.info("\n--- Compiling results ---")
    results_df = pd.DataFrame(results)

    # Reorder columns
    col_order = [
        "gene_id", "gene_symbol", "clinical_var", "cohort",
        "n_samples", "n_events", "converged",
        "beta_gene", "beta_clinical", "beta_interaction",
        "se_interaction", "pvalue",
        "hr_interaction", "hr_lower95", "hr_upper95",
    ]
    results_df = results_df[col_order]

    # ---- FDR correction per cohort ----
    logger.info("\n--- Computing FDR (BH, per cohort) ---")
    results_df["qvalue"] = np.nan

    for cohort in COHORTS:
        mask = (results_df["cohort"] == cohort) & results_df["pvalue"].notna()
        pvals = results_df.loc[mask, "pvalue"].values

        if len(pvals) > 0:
            _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
            results_df.loc[mask, "qvalue"] = qvals
            n_nominal = (pvals < 0.05).sum()
            n_fdr = (qvals < 0.05).sum()
            logger.info(
                f"  {cohort.upper()}: {len(pvals)} tests, "
                f"{n_nominal} nominal p<0.05, {n_fdr} FDR q<0.05"
            )

    # ---- Save results ----
    output_path = OUTPUT_DIR / "analysis_5_4_interaction_results.csv"
    results_df.to_csv(output_path, index=False, float_format="%.6f")
    logger.info(f"\nResults saved to: {output_path}")

    # ---- Summary statistics ----
    elapsed_total = time.time() - start_time

    summary_lines = [
        "=" * 70,
        "Analysis 5.4: Cox Interaction Patterns — Summary",
        "=" * 70,
        "",
        f"Total models fitted: {model_count}",
        f"Failed/non-converged: {failed_count}",
        f"Consensus genes: {len(available_genes)}",
        f"Clinical variables: {len(CLINICAL_VARS)}",
        f"Cohorts: {', '.join(COHORTS)}",
        "",
        "--- Per-cohort summary ---",
    ]

    for cohort in COHORTS:
        cohort_df = results_df[results_df["cohort"] == cohort]
        converged = cohort_df["converged"].sum()
        n_nom = (cohort_df["pvalue"] < 0.05).sum()
        n_fdr = (cohort_df["qvalue"] < 0.05).sum()
        summary_lines.extend([
            f"",
            f"{cohort.upper()}:",
            f"  Models converged: {converged}/{len(cohort_df)}",
            f"  Interaction p < 0.05 (nominal): {n_nom}",
            f"  Interaction q < 0.05 (FDR): {n_fdr}",
        ])

    summary_lines.extend([
        "",
        "--- Per-variable summary (interaction β3 direction counts) ---",
    ])

    for var_name in CLINICAL_VARS:
        var_df = results_df[
            (results_df["clinical_var"] == var_name) & results_df["beta_interaction"].notna()
        ]
        for cohort in COHORTS:
            sub = var_df[var_df["cohort"] == cohort]
            n_pos = (sub["beta_interaction"] > 0).sum()
            n_neg = (sub["beta_interaction"] < 0).sum()
            n_nom = (sub["pvalue"] < 0.05).sum()
            summary_lines.append(
                f"  {var_name} ({cohort.upper()}): "
                f"{n_pos} positive β3, {n_neg} negative β3, "
                f"{n_nom} nominal p<0.05"
            )

    summary_lines.extend([
        "",
        f"Total runtime: {elapsed_total:.1f} seconds ({elapsed_total/60:.1f} minutes)",
        "",
        "NOTE: All results are descriptive/exploratory.",
        "FDR q-values are reported as descriptive summaries, not for significance claims.",
        "Gene set was selected using these cohorts' outcomes (post-selection inference).",
    ])

    summary_text = "\n".join(summary_lines)
    logger.info("\n" + summary_text)

    summary_path = OUTPUT_DIR / "analysis_5_4_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    logger.info(f"Summary saved to: {summary_path}")

    return results_df


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    results = run_analysis()
