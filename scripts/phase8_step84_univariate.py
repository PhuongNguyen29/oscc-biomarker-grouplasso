"""
Phase 8, Step 8.4: Univariate Gene Signal Checks
=================================================
For each consensus gene individually: univariate Cox on each GEO dataset.
Report HR, 95% CI, and p-value per gene.

These are supplementary signal checks, not individual gene validation.

Reference:
  - Cox (1972) JRSS-B 34(2):187-220
  - lifelines CoxPHFitter: Davidson-Pilon (2019) JOSS 4(40):1317
"""

import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import CoxPHFitter
import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================
DATA_DIR = Path("data/external")
CONSENSUS_FILE = Path("results/sensitivity_sweep/mult0.6_pi60/union_genes.txt")
ANNOTATION_FILE = Path("data/raw/protein_coding_genes_annotation.csv")
OUTPUT_DIR = Path("results/phase8_external_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Data Loading (reuse from step 8.3)
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
    """Load ENSG → gene symbol mapping."""
    ann = pd.read_csv(path)
    # Determine column names — check common patterns
    if "gene_id" in ann.columns and "gene_symbol" in ann.columns:
        return dict(zip(ann["gene_id"], ann["gene_symbol"]))
    elif "gene_id" in ann.columns and "gene_name" in ann.columns:
        return dict(zip(ann["gene_id"], ann["gene_name"]))
    else:
        logger.warning(f"Annotation columns: {ann.columns.tolist()}")
        # Try first two columns
        return dict(zip(ann.iloc[:, 0], ann.iloc[:, 1]))


def load_geo_dataset(name):
    expr = pd.read_csv(DATA_DIR / f"{name}_expression.csv", index_col=0)
    surv = pd.read_csv(DATA_DIR / f"{name}_survival.csv")
    return expr, surv


# =============================================================================
# Univariate Cox
# =============================================================================
def run_univariate_cox(gene_id, expression, surv_df):
    """Run univariate Cox for a single gene.
    
    Returns dict with HR, 95% CI, p-value, concordance, or None if failed.
    """
    df = pd.DataFrame({
        "gene": expression,
        "time": surv_df["time"].values,
        "event": surv_df["event"].values
    })

    # Drop any rows with missing or infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if df["event"].sum() < 2:
        return None

    # Check zero variance
    if df["gene"].std() < 1e-10:
        return None

    try:
        cph = CoxPHFitter()
        cph.fit(df, duration_col="time", event_col="event", formula="gene")
        summary = cph.summary

        hr = summary.loc["gene", "exp(coef)"]
        ci_lower = summary.loc["gene", "exp(coef) lower 95%"]
        ci_upper = summary.loc["gene", "exp(coef) upper 95%"]
        p_value = summary.loc["gene", "p"]
        coef = summary.loc["gene", "coef"]
        concordance = cph.concordance_index_

        return {
            "coef": coef,
            "HR": hr,
            "HR_lower95": ci_lower,
            "HR_upper95": ci_upper,
            "p_value": p_value,
            "concordance": concordance
        }
    except Exception as e:
        logger.warning(f"  {gene_id}: Cox failed — {e}")
        return None


# =============================================================================
# Main
# =============================================================================
def analyze_dataset(name, expr, surv, consensus_genes, ensg_to_symbol):
    """Run univariate Cox for all consensus genes on one dataset."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  DATASET: {name}")
    logger.info(f"{'='*60}")

    samples = surv["sampleID"].values
    results = []

    for gene_id in consensus_genes:
        if gene_id not in expr.index:
            continue

        expression = expr.loc[gene_id, samples].values.astype(float)
        res = run_univariate_cox(gene_id, expression, surv)

        if res is not None:
            symbol = ensg_to_symbol.get(gene_id, gene_id)
            results.append({
                "gene_id": gene_id,
                "symbol": symbol,
                **res
            })

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        logger.info("  No genes produced valid results.")
        return results_df

    # Sort by p-value
    results_df = results_df.sort_values("p_value")

    # Summary statistics
    n_tested = len(results_df)
    n_nominal = (results_df["p_value"] < 0.05).sum()
    n_protective = (results_df["HR"] < 1).sum()
    n_risk = (results_df["HR"] > 1).sum()

    logger.info(f"  Genes tested: {n_tested}")
    logger.info(f"  Nominal p < 0.05: {n_nominal}/{n_tested} "
                f"({100*n_nominal/n_tested:.1f}%)")
    logger.info(f"  HR < 1 (protective): {n_protective}, HR > 1 (risk): {n_risk}")
    logger.info(f"  Top 5 by p-value:")
    for _, row in results_df.head(5).iterrows():
        logger.info(f"    {row['symbol']:>10s}: HR = {row['HR']:.2f} "
                    f"({row['HR_lower95']:.2f}-{row['HR_upper95']:.2f}), "
                    f"p = {row['p_value']:.4f}")

    return results_df


def main():
    logger.info("Phase 8, Step 8.4: Univariate Gene Signal Checks")
    logger.info("=" * 60)

    consensus_genes = load_consensus_genes(CONSENSUS_FILE)
    logger.info(f"Loaded {len(consensus_genes)} consensus genes")

    ensg_to_symbol = load_gene_annotation(ANNOTATION_FILE)
    logger.info(f"Loaded {len(ensg_to_symbol)} gene annotations")

    expr41613, surv41613 = load_geo_dataset("GSE41613")
    expr42743, surv42743 = load_geo_dataset("GSE42743")

    results_41613 = analyze_dataset(
        "GSE41613", expr41613, surv41613, consensus_genes, ensg_to_symbol
    )
    results_42743 = analyze_dataset(
        "GSE42743", expr42743, surv42743, consensus_genes, ensg_to_symbol
    )

    # Save per-dataset results
    results_41613.to_csv(OUTPUT_DIR / "step8_4_univariate_GSE41613.csv", index=False)
    results_42743.to_csv(OUTPUT_DIR / "step8_4_univariate_GSE42743.csv", index=False)

    # --- Cross-dataset comparison ---
    logger.info(f"\n{'='*60}")
    logger.info("  CROSS-DATASET COMPARISON")
    logger.info(f"{'='*60}")

    if len(results_41613) > 0 and len(results_42743) > 0:
        merged = results_41613[["gene_id", "symbol", "coef", "HR", "p_value"]].merge(
            results_42743[["gene_id", "coef", "HR", "p_value"]],
            on="gene_id", suffixes=("_41613", "_42743")
        )

        # Direction concordance
        same_direction = (
            (merged["coef_41613"] > 0) == (merged["coef_42743"] > 0)
        ).sum()
        n_compared = len(merged)
        logger.info(f"  Genes compared: {n_compared}")
        logger.info(f"  Same HR direction in both datasets: {same_direction}/{n_compared} "
                    f"({100*same_direction/n_compared:.1f}%)")

        # Genes nominally significant in both
        both_sig = merged[
            (merged["p_value_41613"] < 0.05) & (merged["p_value_42743"] < 0.05)
        ]
        logger.info(f"  Nominal p < 0.05 in BOTH datasets: {len(both_sig)}")
        if len(both_sig) > 0:
            for _, row in both_sig.iterrows():
                logger.info(f"    {row['symbol']:>10s}: "
                            f"HR_41613={row['HR_41613']:.2f}, "
                            f"HR_42743={row['HR_42743']:.2f}")

        merged.to_csv(OUTPUT_DIR / "step8_4_cross_dataset.csv", index=False)

    logger.info(f"\nResults saved to {OUTPUT_DIR}")
    logger.info("Step 8.4 complete.")


if __name__ == "__main__":
    main()
