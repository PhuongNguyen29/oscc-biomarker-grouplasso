"""
Phase 8, Step 8.4: Univariate Gene Signal Checks
=================================================
Univariate Cox per gene on each external GEO cohort. Reports HR, 95% CI,
p-value, BH-adjusted q-value and concordance.

These are marginal signal checks. A univariate HR is NOT validation of a
multivariate signature, and nothing here should be read as such.

WHY THIS RUNS ON ALL GENES BY DEFAULT
-------------------------------------
Univariate Cox is gene-set independent: a gene's HR is identical whether 2 or
4,835 genes are tested. Running the full filtered universe costs seconds and
answers the question a restricted run cannot -- where does a candidate gene
rank externally among ALL genes, not just among the ones the selection
procedure happened to return. That distinguishes "reproducible signal" from
"top-ranked gene in an unstable procedure".

Use --genes <file> to restrict to a specific list if needed.

FIXES IN THIS REVISION
----------------------
1. CONSENSUS_FILE pointed at results/sensitivity_sweep/mult0.6_pi60/, the OLD
   25-gene union. The corrected Phase 4 run additionally wrote its
   union_genes.txt for the wrong cell (the as-drafted rule ordering selected
   mult 0.5 / pi 0.80, which contains zero genes), so that file is empty.
   Gene selection is now an explicit CLI argument with no silent default.

2. warnings.filterwarnings("ignore") suppressed lifelines convergence warnings
   at module level. A Cox fit that did not converge was recorded as a valid
   result. Convergence is now captured per fit and reported.

3. No multiple-testing correction. With 4,835 tests, nominal p < 0.05 alone is
   uninterpretable. BH q-values (Benjamini & Hochberg 1995) added per cohort.

4. No file logging. Added, with the gene list and cohort counts recorded.

5. No assertion that expression columns align with survival sampleIDs. The
   preprocessing script now guarantees this; the assertion makes the
   dependency explicit rather than assumed.

NOTE ON TIME UNITS
------------------
GSE41613 survival is in MONTHS, GSE42743 in DAYS. Cox HRs and the concordance
index are invariant to monotone rescaling of time, so this analysis is
unaffected. Anything evaluated at a FIXED HORIZON is not.

References:
  Cox (1972) JRSS-B 34:187-220
  Benjamini & Hochberg (1995) JRSS-B 57:289-300
  Davidson-Pilon (2019) JOSS 4:1317 -- lifelines
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from statsmodels.stats.multitest import multipletests

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/external")
ANNOTATION_FILE = Path("data/raw/protein_coding_genes_annotation.csv")
DDIT4 = "ENSG00000168209"


# =============================================================================
# Loading
# =============================================================================

def load_gene_list(path):
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
    ann = ann[~ann["gene_id"].duplicated()]
    if "gene_symbol" in ann.columns:
        return dict(zip(ann["gene_id"], ann["gene_symbol"]))
    if "gene_name" in ann.columns:
        return dict(zip(ann["gene_id"], ann["gene_name"]))
    logger.warning(f"Unexpected annotation columns: {ann.columns.tolist()}")
    return dict(zip(ann.iloc[:, 0], ann.iloc[:, 1]))


def load_geo_dataset(name):
    expr = pd.read_csv(DATA_DIR / f"{name}_expression.csv", index_col=0)
    surv = pd.read_csv(DATA_DIR / f"{name}_survival.csv")
    # The preprocessing script guarantees this; assert rather than assume.
    if list(expr.columns) != list(surv["sampleID"]):
        raise ValueError(
            f"{name}: expression columns do not match survival sampleIDs in "
            f"the same order. Re-run phase8_geo_preprocessing.R."
        )
    return expr, surv


# =============================================================================
# Univariate Cox
# =============================================================================

def run_univariate_cox(gene_id, expression, surv_df):
    """
    Univariate Cox for one gene.

    Convergence is captured explicitly. lifelines emits ConvergenceWarning and
    returns a fitted object rather than raising, so a blanket warning filter
    would record an unconverged fit as a valid result.
    """
    df = pd.DataFrame({
        "gene": expression,
        "time": surv_df["time"].values,
        "event": surv_df["event"].values,
    }).replace([np.inf, -np.inf], np.nan).dropna()

    if df["event"].sum() < 5:          # too few events for a stable estimate
        return None
    if df["gene"].std() < 1e-10:
        return None

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cph = CoxPHFitter()
            cph.fit(df, duration_col="time", event_col="event", formula="gene")
        converged = not any(
            issubclass(w.category, ConvergenceWarning)
            or "converge" in str(w.message).lower()
            for w in caught
        )
        s = cph.summary
        return {
            "coef": s.loc["gene", "coef"],
            "se": s.loc["gene", "se(coef)"],
            "HR": s.loc["gene", "exp(coef)"],
            "HR_lower95": s.loc["gene", "exp(coef) lower 95%"],
            "HR_upper95": s.loc["gene", "exp(coef) upper 95%"],
            "p_value": s.loc["gene", "p"],
            "concordance": cph.concordance_index_,
            "n_used": len(df),
            "n_events": int(df["event"].sum()),
            "converged": converged,
        }
    except Exception as e:
        logger.warning(f"  {gene_id}: Cox failed - {e}")
        return None


def analyze_dataset(name, expr, surv, genes, ensg_to_symbol):
    logger.info(f"\n{'='*64}")
    logger.info(f"  DATASET: {name}  ({expr.shape[1]} samples, "
                f"{int(surv['event'].sum())} events)")
    logger.info(f"{'='*64}")

    present = [g for g in genes if g in expr.index]
    if len(present) < len(genes):
        logger.info(f"  {len(genes) - len(present)} requested genes absent "
                    f"from this platform")

    rows = []
    for gene_id in present:
        res = run_univariate_cox(gene_id, expr.loc[gene_id].values.astype(float), surv)
        if res is not None:
            rows.append({"gene_id": gene_id,
                         "symbol": ensg_to_symbol.get(gene_id, gene_id),
                         **res})

    df = pd.DataFrame(rows)
    if df.empty:
        logger.info("  No genes produced valid results.")
        return df

    # BH across all genes tested in THIS cohort. Essential at 4,835 tests;
    # nominal p alone is uninterpretable at that scale.
    df["q_value"] = multipletests(df["p_value"], method="fdr_bh")[1]
    df = df.sort_values("p_value").reset_index(drop=True)
    df["rank_by_p"] = np.arange(1, len(df) + 1)

    n = len(df)
    n_nonconv = (~df["converged"]).sum()
    logger.info(f"  Genes tested: {n}")
    if n_nonconv:
        logger.info(f"  NON-CONVERGED fits: {n_nonconv} (flagged in output)")
    logger.info(f"  Nominal p < 0.05: {(df.p_value < 0.05).sum()}/{n} "
                f"({100*(df.p_value < 0.05).mean():.1f}%) "
                f"[expected by chance: {0.05*n:.0f}]")
    logger.info(f"  BH q < 0.05:      {(df.q_value < 0.05).sum()}/{n}")
    logger.info(f"  BH q < 0.10:      {(df.q_value < 0.10).sum()}/{n}")
    logger.info(f"  HR < 1: {(df.HR < 1).sum()}, HR > 1: {(df.HR > 1).sum()}")

    logger.info("  Top 10 by p-value:")
    for _, r in df.head(10).iterrows():
        logger.info(f"    {r['symbol']:>12s}: HR={r['HR']:.2f} "
                    f"({r['HR_lower95']:.2f}-{r['HR_upper95']:.2f}), "
                    f"p={r['p_value']:.2e}, q={r['q_value']:.3f}")

    if DDIT4 in set(df["gene_id"]):
        r = df[df.gene_id == DDIT4].iloc[0]
        logger.info(f"  --- DDIT4 ---")
        logger.info(f"    HR={r['HR']:.3f} ({r['HR_lower95']:.2f}-{r['HR_upper95']:.2f}), "
                    f"p={r['p_value']:.4f}, q={r['q_value']:.3f}")
        logger.info(f"    rank {int(r['rank_by_p'])} of {n} by p-value "
                    f"(percentile {100*(1 - r['rank_by_p']/n):.1f})")
        logger.info(f"    concordance={r['concordance']:.3f} "
                    f"(age-alone floor is ~0.55-0.57)")
    else:
        logger.info("  --- DDIT4 not present / not estimable in this cohort ---")

    return df


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Phase 8.4 univariate Cox")
    ap.add_argument("--genes", default="all",
                    help="'all' (every gene in the expression matrix) or a path "
                         "to a gene list. No silent default.")
    ap.add_argument("--output-dir", default="results/phase8_external_validation")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(out / "step8_4_univariate.log", mode="w")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info("Phase 8, Step 8.4: Univariate Gene Signal Checks")
    logger.info("=" * 64)
    logger.info(f"Gene selection: {args.genes}")
    logger.info("Time units: GSE41613 MONTHS, GSE42743 DAYS "
                "(HR and C-index are scale-invariant)")

    ensg_to_symbol = load_gene_annotation(ANNOTATION_FILE)
    expr41613, surv41613 = load_geo_dataset("GSE41613")
    expr42743, surv42743 = load_geo_dataset("GSE42743")

    if args.genes == "all":
        genes = sorted(set(expr41613.index) | set(expr42743.index))
        logger.info(f"Testing all {len(genes)} genes present in either cohort")
    else:
        genes = load_gene_list(args.genes)
        logger.info(f"Loaded {len(genes)} genes from {args.genes}")
        if not genes:
            raise ValueError(f"No ENSG IDs found in {args.genes}")

    r41613 = analyze_dataset("GSE41613", expr41613, surv41613, genes, ensg_to_symbol)
    r42743 = analyze_dataset("GSE42743", expr42743, surv42743, genes, ensg_to_symbol)

    r41613.to_csv(out / "step8_4_univariate_GSE41613.csv", index=False)
    r42743.to_csv(out / "step8_4_univariate_GSE42743.csv", index=False)

    # --- Cross-dataset -------------------------------------------------------
    logger.info(f"\n{'='*64}")
    logger.info("  CROSS-DATASET COMPARISON")
    logger.info(f"{'='*64}")

    if r41613.empty or r42743.empty:
        logger.info("  One cohort produced no results; skipping.")
        return

    m = r41613[["gene_id", "symbol", "coef", "HR", "p_value", "q_value", "rank_by_p"]].merge(
        r42743[["gene_id", "coef", "HR", "p_value", "q_value", "rank_by_p"]],
        on="gene_id", suffixes=("_41613", "_42743"))

    same_dir = ((m["coef_41613"] > 0) == (m["coef_42743"] > 0)).sum()
    logger.info(f"  Genes compared: {len(m)}")
    logger.info(f"  Same HR direction: {same_dir}/{len(m)} "
                f"({100*same_dir/len(m):.1f}%)")
    # A binomial sign test here would assume gene independence, which
    # co-expression violates. Do NOT report this as evidence of concordance
    # without the correlated-gene permutation null (cf. Analysis 5.6).
    logger.info("    (NOT a test: co-expression violates the independence a "
                "sign test assumes)")

    rho = m[["coef_41613", "coef_42743"]].corr(method="spearman").iloc[0, 1]
    logger.info(f"  Spearman correlation of coefficients: {rho:.3f}")

    both = m[(m.p_value_41613 < 0.05) & (m.p_value_42743 < 0.05)]
    logger.info(f"  Nominal p < 0.05 in BOTH: {len(both)}")
    for _, r in both.head(20).iterrows():
        logger.info(f"    {r['symbol']:>12s}: HR_41613={r['HR_41613']:.2f}, "
                    f"HR_42743={r['HR_42743']:.2f}")

    both_q = m[(m.q_value_41613 < 0.10) & (m.q_value_42743 < 0.10)]
    logger.info(f"  BH q < 0.10 in BOTH: {len(both_q)}")

    if DDIT4 in set(m["gene_id"]):
        r = m[m.gene_id == DDIT4].iloc[0]
        logger.info(f"  --- DDIT4 across cohorts ---")
        logger.info(f"    GSE41613: HR={r['HR_41613']:.3f}, p={r['p_value_41613']:.4f}, "
                    f"q={r['q_value_41613']:.3f}, rank {int(r['rank_by_p_41613'])}")
        logger.info(f"    GSE42743: HR={r['HR_42743']:.3f}, p={r['p_value_42743']:.4f}, "
                    f"q={r['q_value_42743']:.3f}, rank {int(r['rank_by_p_42743'])}")
        logger.info("    Underpowered non-replication is expected even for a real "
                    "effect: 51/40 events support HR ~1.48-1.56 per SD "
                    "(Schoenfeld 1983). Report a null as 'not established'.")

    m.to_csv(out / "step8_4_cross_dataset.csv", index=False)
    logger.info(f"\nResults saved to {out}")
    logger.info("Step 8.4 complete.")


if __name__ == "__main__":
    main()
