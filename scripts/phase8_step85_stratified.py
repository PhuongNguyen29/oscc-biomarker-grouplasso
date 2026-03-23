"""
Phase 8, Step 8.5: Stratified Analysis in External Data (Exploratory)
=====================================================================
For each consensus gene, within each clinical stratum:
  - Split patients by expression above/below mean (z-score = 0)
  - KM + log-rank: compare high vs low expression survival
  - Report HR, 95% CI
  - Minimum 5 events per group required

Strata:
  GSE42743 (primary): smoking, T_stage, N_stage, gender, age <=65/>65
  GSE41613 (secondary): gender, tumor_stage (I/II vs III/IV)

Per v4 document:
  - Expression-based only (no IG, no model weights transferred)
  - Exploratory (small sample sizes)
  - Wide CIs = inconclusive, not negative

References:
  - Kaplan & Meier (1958) JASA 53(282):457-481
  - Mantel (1966) Cancer Chemother Rep 50(3):163-170 (log-rank)
  - Cox (1972) JRSS-B 34(2):187-220
"""

import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================
DATA_DIR = Path("data/external")
CONSENSUS_FILE = Path("results/phase4_stability_mad3000/consensus/cox_consensus_pi080.txt")
ANNOTATION_FILE = Path("data/raw/protein_coding_genes_annotation.csv")
OUTPUT_DIR = Path("results/phase8_external_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_EVENTS_PER_GROUP = 5


# =============================================================================
# Data Loading
# =============================================================================
def load_consensus_genes(path):
    genes = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            ensg = parts[-1] if parts else line.strip()
            if ensg.startswith("ENSG"):
                genes.append(ensg)
    return genes


def load_gene_annotation(path):
    ann = pd.read_csv(path)
    if "gene_id" in ann.columns and "gene_symbol" in ann.columns:
        return dict(zip(ann["gene_id"], ann["gene_symbol"]))
    elif "gene_id" in ann.columns and "gene_name" in ann.columns:
        return dict(zip(ann["gene_id"], ann["gene_name"]))
    else:
        return dict(zip(ann.iloc[:, 0], ann.iloc[:, 1]))


def load_geo_dataset(name):
    """Load expression, survival, clinical for a GEO dataset."""
    expr = pd.read_csv(DATA_DIR / f"{name}_expression.csv", index_col=0)
    surv = pd.read_csv(DATA_DIR / f"{name}_survival.csv")
    clin = pd.read_csv(DATA_DIR / f"{name}_clinical.csv")

    # Ensure clinical has a sampleID column
    if "sampleID" not in clin.columns:
        clin = pd.read_csv(DATA_DIR / f"{name}_clinical.csv", index_col=0)
        clin["sampleID"] = clin.index

    logger.info(f"{name}: {expr.shape[1]} samples, {expr.shape[0]} genes, "
                f"{surv['event'].sum()} events")
    return expr, surv, clin


# =============================================================================
# Strata Definitions
# =============================================================================
def define_strata_42743(clin):
    """Define clinical strata for GSE42743.
    
    Strata: smoking, T_stage, N_stage, gender, age <=65/>65
    """
    strata = {}

    # Gender
    strata["gender_male"] = clin["gender"] == 1
    strata["gender_female"] = clin["gender"] == 0

    # Age
    strata["age_leq65"] = clin["age"] <= 65
    strata["age_gt65"] = clin["age"] > 65

    # Smoking
    strata["smoking_ever"] = clin["smoking"] == "Ever"
    strata["smoking_never"] = clin["smoking"] == "Never"

    # T stage
    strata["T_stage_T1T2"] = clin["T_stage"] == "T1-T2"
    strata["T_stage_T3T4"] = clin["T_stage"] == "T3-T4"

    # N stage
    strata["N_stage_N0"] = clin["N_stage"] == "N0"
    strata["N_stage_Npos"] = clin["N_stage"] == "N+"

    return strata


def define_strata_41613(clin):
    """Define clinical strata for GSE41613.
    
    Strata: gender, tumor_stage (I/II vs III/IV)
    """
    strata = {}

    # Gender
    strata["gender_male"] = clin["gender"] == 1
    strata["gender_female"] = clin["gender"] == 0

    # Tumor stage
    strata["stage_I_II"] = clin["tumor_stage"] == "I/II"
    strata["stage_III_IV"] = clin["tumor_stage"] == "III/IV"

    return strata


# =============================================================================
# Stratified Analysis
# =============================================================================
def analyze_gene_stratum(gene_expr, time, event, gene_id, symbol, stratum_name):
    """Analyze one gene within one clinical stratum.
    
    Split by expression above/below mean (z-score = 0 since data is z-scored).
    Returns dict with HR, CI, log-rank p, or None if min events not met.
    """
    # Split by expression above/below mean
    high_mask = gene_expr >= 0  # z-scored, so mean = 0
    low_mask = ~high_mask

    n_high = high_mask.sum()
    n_low = low_mask.sum()
    events_high = event[high_mask].sum()
    events_low = event[low_mask].sum()

    # Check minimum events
    if events_high < MIN_EVENTS_PER_GROUP or events_low < MIN_EVENTS_PER_GROUP:
        return None

    # Log-rank test
    try:
        lr = logrank_test(
            time[high_mask], time[low_mask],
            event_observed_A=event[high_mask],
            event_observed_B=event[low_mask]
        )
        logrank_p = lr.p_value
    except Exception:
        logrank_p = np.nan

    # Cox model: high expression as binary predictor
    df = pd.DataFrame({
        "high_expr": high_mask.astype(int),
        "time": time,
        "event": event
    })

    try:
        cph = CoxPHFitter()
        cph.fit(df, duration_col="time", event_col="event", formula="high_expr")
        summary = cph.summary

        hr = summary.loc["high_expr", "exp(coef)"]
        ci_lower = summary.loc["high_expr", "exp(coef) lower 95%"]
        ci_upper = summary.loc["high_expr", "exp(coef) upper 95%"]
        cox_p = summary.loc["high_expr", "p"]
    except Exception:
        hr, ci_lower, ci_upper, cox_p = np.nan, np.nan, np.nan, np.nan

    return {
        "gene_id": gene_id,
        "symbol": symbol,
        "stratum": stratum_name,
        "n_high": int(n_high),
        "n_low": int(n_low),
        "events_high": int(events_high),
        "events_low": int(events_low),
        "HR": hr,
        "HR_lower95": ci_lower,
        "HR_upper95": ci_upper,
        "logrank_p": logrank_p,
        "cox_p": cox_p
    }


def analyze_dataset(name, expr, surv, clin, consensus_genes, ensg_to_symbol, strata_fn):
    """Run stratified analysis for all genes × strata on one dataset."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  DATASET: {name}")
    logger.info(f"{'='*60}")

    samples = surv["sampleID"].values
    clin_aligned = clin.set_index("sampleID").loc[samples].reset_index()
    strata = strata_fn(clin_aligned)

    # Log strata sizes
    logger.info(f"  Strata:")
    for sname, mask in strata.items():
        n = mask.sum()
        events = surv.loc[mask.values, "event"].sum() if mask.sum() > 0 else 0
        logger.info(f"    {sname}: n={n}, events={events}")

    results = []
    skipped = 0

    for gene_id in consensus_genes:
        if gene_id not in expr.index:
            continue
        symbol = ensg_to_symbol.get(gene_id, gene_id)
        gene_expr = expr.loc[gene_id, samples].values.astype(float)

        for sname, mask in strata.items():
            mask_arr = mask.values
            if mask_arr.sum() < 10:  # Skip tiny strata
                skipped += 1
                continue

            res = analyze_gene_stratum(
                gene_expr=gene_expr[mask_arr],
                time=surv["time"].values[mask_arr],
                event=surv["event"].values[mask_arr],
                gene_id=gene_id,
                symbol=symbol,
                stratum_name=sname
            )

            if res is not None:
                results.append(res)
            else:
                skipped += 1

    results_df = pd.DataFrame(results)
    logger.info(f"\n  Gene-stratum combinations tested: {len(results_df)}")
    logger.info(f"  Skipped (< {MIN_EVENTS_PER_GROUP} events per group): {skipped}")

    if len(results_df) == 0:
        return results_df

    # Summary
    n_nominal = (results_df["logrank_p"] < 0.05).sum()
    logger.info(f"  Nominal log-rank p < 0.05: {n_nominal}/{len(results_df)} "
                f"({100*n_nominal/len(results_df):.1f}%)")

    # Top associations
    top = results_df.nsmallest(10, "logrank_p")
    logger.info(f"\n  Top 10 gene-stratum associations:")
    for _, row in top.iterrows():
        ci_width = row["HR_upper95"] - row["HR_lower95"]
        precision = "narrow" if ci_width < 2.0 else "wide"
        logger.info(f"    {row['symbol']:>10s} | {row['stratum']:<20s} | "
                    f"HR={row['HR']:.2f} ({row['HR_lower95']:.2f}-{row['HR_upper95']:.2f}) | "
                    f"p={row['logrank_p']:.4f} | {precision} CI")

    return results_df


def main():
    logger.info("Phase 8, Step 8.5: Stratified Analysis in External Data")
    logger.info("=" * 60)

    consensus_genes = load_consensus_genes(CONSENSUS_FILE)
    logger.info(f"Loaded {len(consensus_genes)} consensus genes")

    ensg_to_symbol = load_gene_annotation(ANNOTATION_FILE)
    logger.info(f"Loaded {len(ensg_to_symbol)} gene annotations")

    # --- GSE42743 (primary) ---
    expr42743, surv42743, clin42743 = load_geo_dataset("GSE42743")
    results_42743 = analyze_dataset(
        "GSE42743", expr42743, surv42743, clin42743,
        consensus_genes, ensg_to_symbol, define_strata_42743
    )

    # --- GSE41613 (secondary) ---
    expr41613, surv41613, clin41613 = load_geo_dataset("GSE41613")
    results_41613 = analyze_dataset(
        "GSE41613", expr41613, surv41613, clin41613,
        consensus_genes, ensg_to_symbol, define_strata_41613
    )

    # Save results
    if len(results_42743) > 0:
        results_42743.to_csv(OUTPUT_DIR / "step8_5_stratified_GSE42743.csv", index=False)
    if len(results_41613) > 0:
        results_41613.to_csv(OUTPUT_DIR / "step8_5_stratified_GSE41613.csv", index=False)

    # --- Cross-dataset direction concordance (where strata overlap) ---
    logger.info(f"\n{'='*60}")
    logger.info("  CROSS-DATASET DIRECTION CONCORDANCE")
    logger.info(f"{'='*60}")

    # Both datasets share gender strata
    shared_strata = ["gender_male", "gender_female"]
    if len(results_42743) > 0 and len(results_41613) > 0:
        for stratum in shared_strata:
            r42 = results_42743[results_42743["stratum"] == stratum][["gene_id", "symbol", "HR"]].copy()
            r41 = results_41613[results_41613["stratum"] == stratum][["gene_id", "HR"]].copy()

            if len(r42) == 0 or len(r41) == 0:
                continue

            merged = r42.merge(r41, on="gene_id", suffixes=("_42743", "_41613"))
            if len(merged) == 0:
                continue

            same_dir = ((merged["HR_42743"] > 1) == (merged["HR_41613"] > 1)).sum()
            logger.info(f"  {stratum}: {same_dir}/{len(merged)} genes same HR direction "
                        f"({100*same_dir/len(merged):.1f}%)")

    logger.info(f"\nResults saved to {OUTPUT_DIR}")
    logger.info("Step 8.5 complete.")


if __name__ == "__main__":
    main()
