#!/usr/bin/env python3
"""
Analysis 5.6: Conditional Permutation Null (Exploratory)
=========================================================

Chapter 3 — ElasticDeepSurv with Interpretable Biomarker Discovery

Purpose:
    Test whether the observed number of concordant gene × stratum-level
    HR directions (across TCGA and ORIEN) exceeds what would be expected
    by chance, GIVEN the selected consensus gene set.

    This is a conditional null — conditioned on the gene set.
    It does NOT test whether the full pipeline (including gene selection)
    beats random noise.

Reference:
    - Phipson & Smyth (2010). Permutation p-values should never be zero.
      Statistical Applications in Genetics and Molecular Biology.
    - Cox (1972). Regression models and life-tables. JRSS-B.

Design:
    - Observed statistic: count of (gene × stratum_level) pairs where
      expression-based HR is on the same side of 1.0 in both cohorts,
      out of 25 genes × 6 variables × 2 levels = 300 total units.
    - Expression-based stratification: split at z-score = 0 within each
      clinical stratum, fit univariate Cox, extract HR direction.
    - Permutation: shuffle (time, event) within each stratum level,
      independently per cohort, per clinical variable. 1,000 iterations.
    - Testable set: fixed from observed data (≥5 events per expression
      group within each stratum in both cohorts).
    - Permutation p-value: (#{null ≥ observed} + 1) / (1001)
      per Phipson & Smyth (2010).

Input files:
    - data/processed/{tcga,orien}_preprocessed_v4.csv
    - data/processed/surv_{tcga,orien}_aligned_v4.csv
    - data/processed/clinical_{tcga,orien}_aligned_v4.csv
    - results/sensitivity_sweep/mult0.6_pi60/union_genes.txt

Output files:
    - results/phase5_corrected/analysis_5_6/analysis_5_6_permutation_results.csv
    - results/phase5_corrected/analysis_5_6/analysis_5_6_null_distribution.csv
    - results/phase5_corrected/analysis_5_6/analysis_5_6_per_variable_breakdown.csv
    - results/phase5_corrected/analysis_5_6/analysis_5_6_summary.txt
    - results/phase5_corrected/analysis_5_6/checkpoint_null_dist.npy (periodic)

Author: Phuong Nguyen
Date: February 2026 (updated April 2026 for corrected 25-gene union set)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from lifelines import CoxPHFitter
import warnings
import logging
import time

# ==============================================================================
# Configuration
# ==============================================================================

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data" / "processed"
CONSENSUS_PATH = (
    BASE_DIR / "results" / "sensitivity_sweep" / "mult0.6_pi60" / "union_genes.txt"
)
OUTPUT_DIR = BASE_DIR / "results" / "phase5_corrected" / "analysis_5_6"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint_null_dist.npy"

COHORTS = ["tcga", "orien"]
N_PERMUTATIONS = 1000
RANDOM_SEED = 42
MIN_EVENTS_PER_GROUP = 5
CHECKPOINT_INTERVAL = 100  # save every N permutations

# Clinical variable definitions
# For stratification: all variables are binary (including age thresholded at 65)
CLINICAL_STRATA = {
    "age": {
        "threshold": 65,
        "levels": ["le65", "gt65"],  # ≤65 vs >65
        "exclude_values": [],
    },
    "gender": {
        "levels": [0, 1],  # 0=Female, 1=Male
        "exclude_values": [],
    },
    "smoking": {
        "levels": ["Ever", "Never"],
        "exclude_values": ["Unknown"],
    },
    "alcohol": {
        "levels": ["Ever", "Never"],
        "exclude_values": ["Unknown"],
    },
    "N_stage": {
        "levels": ["N0", "N+"],
        "remap": {"N0": "N0", "N1": "N+", "N2-3": "N+"},
        "exclude_values": ["Unknown"],
    },
    "T_stage": {
        "levels": ["T1-T2", "T3-T4"],
        "exclude_values": ["Unknown"],
    },
}

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
    """Load preprocessed expression data, transpose to samples × genes."""
    path = DATA_DIR / f"{cohort_name}_preprocessed_v4.csv"
    expr = pd.read_csv(path, index_col=0).T
    logger.info(f"  {cohort_name.upper()} expression: {expr.shape[0]} samples × {expr.shape[1]} genes")
    return expr


def load_survival(cohort_name):
    """Load survival data, handling R row-number column."""
    path = DATA_DIR / f"surv_{cohort_name}_aligned_v4.csv"
    surv = pd.read_csv(path)
    if surv.columns[0] == "" or "Unnamed" in str(surv.columns[0]):
        surv = surv.drop(columns=[surv.columns[0]])
    surv = surv.set_index("sampleID")
    logger.info(
        f"  {cohort_name.upper()} survival: {len(surv)} samples, "
        f"{int(surv['event'].sum())} events"
    )
    return surv


def load_clinical(cohort_name):
    """Load clinical data."""
    path = DATA_DIR / f"clinical_{cohort_name}_aligned_v4.csv"
    clin = pd.read_csv(path).set_index("sampleID")
    logger.info(f"  {cohort_name.upper()} clinical: {len(clin)} samples")
    return clin


def load_consensus_genes():
    """Load consensus gene list."""
    genes = pd.read_csv(CONSENSUS_PATH, header=None)[0].tolist()
    logger.info(f"  Consensus genes: {len(genes)}")
    return genes


# ==============================================================================
# Stratum Assignment
# ==============================================================================


def get_stratum_assignments(clinical_df, var_name, var_config):
    """
    Assign each sample to a stratum level for a given clinical variable.

    Returns:
        strata: pd.Series with stratum labels, NaN for excluded samples
    """
    raw = clinical_df[var_name].copy()

    if var_name == "age":
        # Threshold at 65
        age_vals = pd.to_numeric(raw, errors="coerce")
        strata = pd.Series(index=raw.index, dtype=object)
        strata[age_vals <= 65] = "le65"
        strata[age_vals > 65] = "gt65"
        strata[age_vals.isna()] = np.nan
        return strata

    # Exclude unknown values
    exclude = var_config["exclude_values"]
    valid_mask = ~raw.isin(exclude)

    # Apply remapping if needed (e.g., N_stage)
    if "remap" in var_config:
        strata = raw.map(var_config["remap"])
    else:
        strata = raw.copy()

    # Set excluded to NaN
    strata[~valid_mask] = np.nan

    return strata


# ==============================================================================
# Expression-Based HR Computation
# ==============================================================================


def compute_expression_hr(expr_values, surv_df):
    """
    Split expression at 0 (z-score population mean), fit univariate Cox,
    return HR direction and metadata.

    Parameters:
        expr_values: pd.Series, gene expression values for samples in this stratum
        surv_df: pd.DataFrame with 'time' and 'event' columns for same samples

    Returns:
        dict with 'hr', 'hr_direction' (+1 or -1), 'n_high', 'n_low',
        'events_high', 'events_low', 'success'
        or None if insufficient data
    """
    # Split at z-score = 0
    high_mask = expr_values > 0

    n_high = high_mask.sum()
    n_low = (~high_mask).sum()

    if n_high == 0 or n_low == 0:
        return None

    events_high = int(surv_df.loc[high_mask[high_mask].index, "event"].sum())
    events_low = int(surv_df.loc[high_mask[~high_mask].index, "event"].sum())

    # Check minimum events
    if events_high < MIN_EVENTS_PER_GROUP or events_low < MIN_EVENTS_PER_GROUP:
        return None

    # Fit univariate Cox on binary expression group
    df = pd.DataFrame({
        "high_expr": high_mask.astype(int).values,
        "time": surv_df["time"].values,
        "event": surv_df["event"].values,
    })

    # Remove zero/negative times
    df = df[df["time"] > 0]

    if len(df) < 10 or df["event"].sum() < MIN_EVENTS_PER_GROUP:
        return None

    try:
        cph = CoxPHFitter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(df, duration_col="time", event_col="event", show_progress=False)

        hr = cph.summary.loc["high_expr", "exp(coef)"]
        hr_direction = 1 if hr > 1.0 else -1

        return {
            "hr": hr,
            "hr_direction": hr_direction,
            "n_high": int(n_high),
            "n_low": int(n_low),
            "events_high": events_high,
            "events_low": events_low,
            "success": True,
        }

    except Exception:
        return None


# ==============================================================================
# Observed Statistic Computation
# ==============================================================================


def compute_observed_statistic(data, consensus_genes):
    """
    Compute the observed concordance count and identify the testable set.

    For each (gene, clinical_var, stratum_level) in each cohort:
        - Within the stratum, split expression at 0
        - Fit univariate Cox, extract HR direction
        - Check if both cohorts' HRs are on the same side of 1.0

    Returns:
        observed_count: int, number of concordant pairs
        testable_set: list of (gene, var_name, level) tuples that are testable
        observed_details: pd.DataFrame with per-unit results
    """
    logger.info("\n--- Computing observed statistic ---")

    # Compute HRs for all (gene, var, level, cohort) combinations
    all_results = []

    for var_name, var_config in CLINICAL_STRATA.items():
        for cohort in COHORTS:
            expr_df = data[cohort]["expr"]
            surv_df = data[cohort]["surv"]
            clin_df = data[cohort]["clin"]

            # Get stratum assignments
            strata = get_stratum_assignments(clin_df, var_name, var_config)

            for level in var_config["levels"]:
                # Get samples in this stratum
                level_mask = strata == level
                level_samples = level_mask[level_mask].index

                # Intersect with available expression and survival
                valid = sorted(
                    set(level_samples) & set(expr_df.index) & set(surv_df.index)
                )

                if len(valid) < 10:
                    continue

                for gene in consensus_genes:
                    if gene not in expr_df.columns:
                        continue

                    gene_expr = expr_df.loc[valid, gene]
                    gene_surv = surv_df.loc[valid]

                    result = compute_expression_hr(gene_expr, gene_surv)

                    row = {
                        "gene": gene,
                        "clinical_var": var_name,
                        "level": level,
                        "cohort": cohort,
                    }

                    if result is not None:
                        row.update(result)
                    else:
                        row["success"] = False

                    all_results.append(row)

    details_df = pd.DataFrame(all_results)

    # Identify testable set: units where BOTH cohorts have successful HR computation
    testable_set = []
    concordance_results = []

    for (gene, var_name, level), group in details_df.groupby(
        ["gene", "clinical_var", "level"]
    ):
        cohort_results = {}
        for _, row in group.iterrows():
            if row.get("success", False):
                cohort_results[row["cohort"]] = row["hr_direction"]

        if len(cohort_results) == 2:  # both cohorts have valid HR
            testable_set.append((gene, var_name, level))
            tcga_dir = cohort_results["tcga"]
            orien_dir = cohort_results["orien"]
            is_concordant = tcga_dir == orien_dir

            concordance_results.append({
                "gene": gene,
                "clinical_var": var_name,
                "level": level,
                "tcga_hr_direction": tcga_dir,
                "orien_hr_direction": orien_dir,
                "concordant": is_concordant,
            })

    concordance_df = pd.DataFrame(concordance_results)
    observed_count = int(concordance_df["concordant"].sum())
    total_testable = len(testable_set)

    logger.info(f"  Testable (gene × stratum_level) units: {total_testable}")
    logger.info(f"  Observed concordant: {observed_count}/{total_testable}")
    logger.info(f"  Concordance rate: {observed_count/total_testable:.3f}")

    # Per-variable breakdown
    logger.info("\n  Per-variable breakdown:")
    for var_name in CLINICAL_STRATA:
        var_sub = concordance_df[concordance_df["clinical_var"] == var_name]
        if len(var_sub) > 0:
            n_conc = int(var_sub["concordant"].sum())
            n_total = len(var_sub)
            logger.info(f"    {var_name}: {n_conc}/{n_total} concordant ({n_conc/n_total:.3f})")

    return observed_count, testable_set, concordance_df, details_df


# ==============================================================================
# Permutation Loop
# ==============================================================================


def run_single_permutation(data, consensus_genes, testable_set, rng):
    """
    Run one permutation: shuffle (time, event) within each stratum level
    independently per cohort per clinical variable, then recompute HRs
    and count concordant pairs.

    Parameters:
        data: dict of cohort data
        consensus_genes: list of gene IDs
        testable_set: list of (gene, var_name, level) tuples
        rng: np.random.RandomState for this permutation

    Returns:
        int: concordant count for this permutation
    """
    # Build a lookup for fast access: (gene, var, level, cohort) → hr_direction
    hr_directions = {}

    for var_name, var_config in CLINICAL_STRATA.items():
        for cohort in COHORTS:
            expr_df = data[cohort]["expr"]
            surv_df = data[cohort]["surv"]
            clin_df = data[cohort]["clin"]

            # Get stratum assignments
            strata = get_stratum_assignments(clin_df, var_name, var_config)

            # Permute survival within each stratum level
            permuted_surv = surv_df.copy()
            for level in var_config["levels"]:
                level_mask = strata == level
                level_samples = sorted(
                    set(level_mask[level_mask].index)
                    & set(surv_df.index)
                )

                if len(level_samples) < 2:
                    continue

                # Shuffle (time, event) together within this stratum
                surv_block = surv_df.loc[level_samples, ["time", "event"]].values
                perm_idx = rng.permutation(len(surv_block))
                permuted_surv.loc[level_samples, "time"] = surv_block[perm_idx, 0]
                permuted_surv.loc[level_samples, "event"] = surv_block[perm_idx, 1]

            # Now compute HRs for all testable units for this (var, cohort)
            for gene, tv, level in testable_set:
                if tv != var_name:
                    continue

                # Get samples in this stratum
                level_mask = strata == level
                level_samples = sorted(
                    set(level_mask[level_mask].index)
                    & set(expr_df.index)
                    & set(permuted_surv.index)
                )

                if len(level_samples) < 10:
                    hr_directions[(gene, var_name, level, cohort)] = None
                    continue

                gene_expr = expr_df.loc[level_samples, gene]
                gene_surv = permuted_surv.loc[level_samples]

                result = compute_expression_hr(gene_expr, gene_surv)

                if result is not None:
                    hr_directions[(gene, var_name, level, cohort)] = result["hr_direction"]
                else:
                    hr_directions[(gene, var_name, level, cohort)] = None

    # Count concordant pairs
    concordant = 0
    for gene, var_name, level in testable_set:
        tcga_dir = hr_directions.get((gene, var_name, level, "tcga"))
        orien_dir = hr_directions.get((gene, var_name, level, "orien"))
        if tcga_dir is not None and orien_dir is not None:
            if tcga_dir == orien_dir:
                concordant += 1

    return concordant


# ==============================================================================
# Main Analysis
# ==============================================================================


def run_analysis():
    """Run Analysis 5.6: Conditional Permutation Null."""
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("Analysis 5.6: Conditional Permutation Null (Exploratory)")
    logger.info("=" * 70)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    logger.info("\n--- Loading data ---")
    consensus_genes = load_consensus_genes()

    data = {}
    for cohort in COHORTS:
        logger.info(f"\nLoading {cohort.upper()}:")
        data[cohort] = {
            "expr": load_expression(cohort),
            "surv": load_survival(cohort),
            "clin": load_clinical(cohort),
        }

    # Align samples per cohort
    for cohort in COHORTS:
        common = sorted(
            set(data[cohort]["expr"].index)
            & set(data[cohort]["surv"].index)
            & set(data[cohort]["clin"].index)
        )
        data[cohort]["expr"] = data[cohort]["expr"].loc[common]
        data[cohort]["surv"] = data[cohort]["surv"].loc[common]
        data[cohort]["clin"] = data[cohort]["clin"].loc[common]
        logger.info(f"  {cohort.upper()} aligned: {len(common)} samples")

    # Verify gene availability
    available_genes = [
        g for g in consensus_genes
        if g in data["tcga"]["expr"].columns and g in data["orien"]["expr"].columns
    ]
    logger.info(f"\nGenes available in both cohorts: {len(available_genes)}/{len(consensus_genes)}")

    # ---- Compute observed statistic ----
    observed_count, testable_set, concordance_df, details_df = (
        compute_observed_statistic(data, available_genes)
    )

    # Save observed concordance details
    concordance_df.to_csv(
        OUTPUT_DIR / "analysis_5_6_observed_concordance.csv",
        index=False,
    )

    # ---- Check for checkpoint ----
    null_distribution = np.full(N_PERMUTATIONS, np.nan)
    start_perm = 0

    if CHECKPOINT_PATH.exists():
        checkpoint = np.load(CHECKPOINT_PATH)
        completed = np.sum(~np.isnan(checkpoint))
        if completed > 0 and completed < N_PERMUTATIONS:
            null_distribution[:len(checkpoint)] = checkpoint
            start_perm = int(completed)
            logger.info(f"\n  Resuming from checkpoint: {start_perm} permutations completed")
        elif completed >= N_PERMUTATIONS:
            logger.info(f"\n  Checkpoint shows all {N_PERMUTATIONS} permutations complete. Re-using.")
            null_distribution = checkpoint[:N_PERMUTATIONS]
            start_perm = N_PERMUTATIONS

    # ---- Permutation loop ----
    if start_perm < N_PERMUTATIONS:
        logger.info(f"\n--- Running permutations ({start_perm+1} to {N_PERMUTATIONS}) ---")

        # Generate seeds for all permutations (deterministic)
        master_rng = np.random.RandomState(RANDOM_SEED)
        perm_seeds = master_rng.randint(0, 2**31, size=N_PERMUTATIONS)

        for i in range(start_perm, N_PERMUTATIONS):
            perm_rng = np.random.RandomState(perm_seeds[i])

            null_count = run_single_permutation(
                data, available_genes, testable_set, perm_rng
            )
            null_distribution[i] = null_count

            # Progress logging
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                perm_done = i + 1 - start_perm
                rate = perm_done / elapsed if elapsed > 0 else 0
                remaining = (N_PERMUTATIONS - i - 1) / rate if rate > 0 else 0

                completed_nulls = null_distribution[:i+1]
                completed_nulls = completed_nulls[~np.isnan(completed_nulls)]

                logger.info(
                    f"  Permutation {i+1}/{N_PERMUTATIONS} | "
                    f"null={null_count} | "
                    f"null mean={completed_nulls.mean():.1f} | "
                    f"rate={rate:.1f}/s | "
                    f"ETA={remaining/60:.0f}min"
                )

            # Checkpoint
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                np.save(CHECKPOINT_PATH, null_distribution)
                logger.info(f"  >> Checkpoint saved at permutation {i+1}")

        # Final checkpoint save
        np.save(CHECKPOINT_PATH, null_distribution)

    # ---- Compute permutation p-value ----
    logger.info("\n--- Computing results ---")

    valid_nulls = null_distribution[~np.isnan(null_distribution)]
    n_ge_observed = np.sum(valid_nulls >= observed_count)
    perm_pvalue = (n_ge_observed + 1) / (len(valid_nulls) + 1)  # Phipson & Smyth

    logger.info(f"  Observed concordant count: {observed_count}")
    logger.info(f"  Null distribution: mean={valid_nulls.mean():.1f}, "
                f"SD={valid_nulls.std():.1f}")
    logger.info(f"  Null 95th percentile: {np.percentile(valid_nulls, 95):.1f}")
    logger.info(f"  Null max: {valid_nulls.max():.0f}")
    logger.info(f"  # null ≥ observed: {int(n_ge_observed)}")
    logger.info(f"  Permutation p-value: {perm_pvalue:.4f}")

    # ---- Per-variable breakdown ----
    logger.info("\n--- Per-variable breakdown ---")
    per_var_results = []

    for var_name in CLINICAL_STRATA:
        var_units = [(g, v, l) for g, v, l in testable_set if v == var_name]
        if len(var_units) == 0:
            continue

        # Observed concordance for this variable
        var_conc = concordance_df[concordance_df["clinical_var"] == var_name]
        obs_var = int(var_conc["concordant"].sum())
        total_var = len(var_conc)

        # Expected under random: 50% concordance (HR direction is binary)
        expected_var = total_var * 0.5

        per_var_results.append({
            "clinical_var": var_name,
            "n_testable_units": total_var,
            "observed_concordant": obs_var,
            "concordance_rate": obs_var / total_var if total_var > 0 else np.nan,
            "expected_under_null": expected_var,
        })

        logger.info(
            f"  {var_name}: {obs_var}/{total_var} concordant "
            f"(rate={obs_var/total_var:.3f}, expected≈{expected_var:.0f})"
        )

    per_var_df = pd.DataFrame(per_var_results)

    # ---- Save outputs ----
    logger.info("\n--- Saving outputs ---")

    # 1. Summary results
    summary_results = pd.DataFrame([{
        "observed_concordant": observed_count,
        "total_testable_units": len(testable_set),
        "observed_concordance_rate": observed_count / len(testable_set),
        "null_mean": valid_nulls.mean(),
        "null_sd": valid_nulls.std(),
        "null_median": np.median(valid_nulls),
        "null_95th_pct": np.percentile(valid_nulls, 95),
        "null_max": valid_nulls.max(),
        "null_min": valid_nulls.min(),
        "n_null_ge_observed": int(n_ge_observed),
        "n_permutations": len(valid_nulls),
        "perm_pvalue": perm_pvalue,
        "random_seed": RANDOM_SEED,
    }])
    summary_results.to_csv(
        OUTPUT_DIR / "analysis_5_6_permutation_results.csv",
        index=False, float_format="%.6f",
    )

    # 2. Full null distribution
    null_df = pd.DataFrame({
        "permutation": np.arange(1, len(valid_nulls) + 1),
        "null_concordant_count": valid_nulls.astype(int),
    })
    null_df.to_csv(
        OUTPUT_DIR / "analysis_5_6_null_distribution.csv",
        index=False,
    )

    # 3. Per-variable breakdown
    per_var_df.to_csv(
        OUTPUT_DIR / "analysis_5_6_per_variable_breakdown.csv",
        index=False, float_format="%.6f",
    )

    # 4. Text summary
    elapsed_total = time.time() - start_time

    n_genes = len(available_genes)
    max_units = n_genes * 6 * 2

    summary_lines = [
        "=" * 70,
        "Analysis 5.6: Conditional Permutation Null — Summary",
        "=" * 70,
        "",
        "Reference: Phipson & Smyth (2010), permutation p-values.",
        "",
        "Design:",
        f"  Consensus genes: {n_genes}",
        f"  Clinical variables: {len(CLINICAL_STRATA)} (6 vars × 2 levels = 12 strata)",
        f"  Total concordance units: {n_genes} × 6 × 2 = {max_units} (theoretical maximum)",
        f"  Testable units (≥{MIN_EVENTS_PER_GROUP} events/group in both cohorts): {len(testable_set)}",
        f"  Permutations: {len(valid_nulls)}",
        f"  Random seed: {RANDOM_SEED}",
        "",
        "Results:",
        f"  Observed concordant count: {observed_count}/{len(testable_set)}",
        f"  Observed concordance rate: {observed_count/len(testable_set):.4f}",
        f"  Null distribution: mean={valid_nulls.mean():.2f}, SD={valid_nulls.std():.2f}",
        f"  Null 95th percentile: {np.percentile(valid_nulls, 95):.1f}",
        f"  Null range: [{valid_nulls.min():.0f}, {valid_nulls.max():.0f}]",
        f"  # permutations with null ≥ observed: {int(n_ge_observed)}",
        f"  Permutation p-value: {perm_pvalue:.4f}",
        "",
        "Per-variable breakdown (observed):",
    ]

    for _, row in per_var_df.iterrows():
        summary_lines.append(
            f"  {row['clinical_var']}: "
            f"{int(row['observed_concordant'])}/{int(row['n_testable_units'])} "
            f"(rate={row['concordance_rate']:.3f})"
        )

    summary_lines.extend([
        "",
        "Interpretation:",
        "  This is a CONDITIONAL null — conditioned on the selected gene set.",
        "  It tests whether concordant HR directions across cohorts exceed",
        "  chance expectations given these specific genes.",
        "  It does NOT test whether the full pipeline (including gene",
        "  selection) beats random noise.",
        "",
        f"Total runtime: {elapsed_total:.1f} seconds ({elapsed_total/60:.1f} minutes)",
        "",
        "NOTE: All results are descriptive/exploratory.",
        "Gene set was selected using these cohorts' outcomes (post-selection inference).",
    ])

    summary_text = "\n".join(summary_lines)
    logger.info("\n" + summary_text)

    summary_path = OUTPUT_DIR / "analysis_5_6_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    logger.info(f"\nSummary saved to: {summary_path}")

    # Clean up checkpoint if complete
    if CHECKPOINT_PATH.exists() and len(valid_nulls) == N_PERMUTATIONS:
        logger.info("  Permutations complete. Checkpoint retained for reproducibility.")

    return summary_results


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    results = run_analysis()
