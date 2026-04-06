"""
GSEA preranked using randomized lasso selection frequencies.

Uses selection probability (pi) from randomized lasso stability selection
as the ranking metric. This is fundamentally different from the ridge
mean rank — pi reflects how consistently the lasso *chose* each gene
across 50 subsamples, not how a dense ranker ordered tiny coefficients.

Genes with high pi are genuinely selected by the sparse model.
Genes with pi=0 were never selected in any subsample.

Usage:
    cd ~/hpchome/oscc-biomarker-grouplasso
    python scripts/gsea_lasso_selection_freq.py
"""

import pandas as pd
import numpy as np
import gseapy as gp
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

SEL_PROB_DIR = Path("results/phase4_stability_randomized_lasso/selection_probabilities")
ANNOT_FILE = Path("data/raw/protein_coding_genes_annotation.csv")
OUTPUT_DIR = Path("results/phase4_stability_randomized_lasso/gsea_pathway")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COHORTS = ["tcga", "orien"]
GSEA_LIBS = ["KEGG_2021_Human", "GO_Biological_Process_2023", "MSigDB_Hallmark_2020"]

# =============================================================================
# Load data
# =============================================================================

print("=" * 60)
print("Loading data")
print("=" * 60)

annot = pd.read_csv(ANNOT_FILE)
ensg_to_symbol = dict(zip(annot["gene_id"], annot["gene_symbol"]))

cohort_rankings = {}

for cohort in COHORTS:
    probs = pd.read_csv(SEL_PROB_DIR / f"{cohort}_cox_selection_probs.csv",
                        index_col=0)

    # The file has one column: selection_prob
    if "selection_prob" in probs.columns:
        sel_freq = probs["selection_prob"]
    else:
        # If column name differs, take first column
        sel_freq = probs.iloc[:, 0]

    print(f"\n  {cohort.upper()}:")
    print(f"    Total genes: {len(sel_freq)}")
    print(f"    Genes with pi > 0: {(sel_freq > 0).sum()}")
    print(f"    Genes with pi = 0: {(sel_freq == 0).sum()}")
    print(f"    Max pi: {sel_freq.max():.2f}")
    print(f"    Mean pi (nonzero only): {sel_freq[sel_freq > 0].mean():.4f}")

    # Build ranked list: gene_symbol -> selection_prob
    # Higher pi = more important
    rnk_records = {}
    for gene, pi_val in sel_freq.items():
        sym = ensg_to_symbol.get(gene)
        if sym and sym not in rnk_records:
            rnk_records[sym] = pi_val

    rnk_series = pd.Series(rnk_records)
    rnk_series = rnk_series.sort_values(ascending=False)

    print(f"    Mapped to symbols: {len(rnk_series)}")
    print(f"    Top 10:")
    for sym, pi_val in rnk_series.head(10).items():
        print(f"      {sym:<15} pi={pi_val:.2f}")

    cohort_rankings[cohort] = rnk_series

# =============================================================================
# GSEA preranked
# =============================================================================

print()
print("=" * 60)
print("GSEA preranked on lasso selection frequencies")
print("=" * 60)

cohort_gsea_results = {}

for cohort in COHORTS:
    rnk_series = cohort_rankings[cohort]
    print(f"\n  {cohort.upper()}:")

    for lib in GSEA_LIBS:
        print(f"\n    Library: {lib}")
        try:
            gsea_res = gp.prerank(
                rnk=rnk_series,
                gene_sets=lib,
                outdir=None,
                no_plot=True,
                verbose=False,
                min_size=10,
                max_size=500,
                permutation_num=1000,
                seed=42,
            )
            res = gsea_res.res2d.copy()

            res["FDR q-val"] = pd.to_numeric(res["FDR q-val"], errors="coerce")
            res["NES"] = pd.to_numeric(res["NES"], errors="coerce")
            res["NOM p-val"] = pd.to_numeric(res["NOM p-val"], errors="coerce")

            # Report at multiple thresholds
            sig_005 = res[res["FDR q-val"] < 0.05].sort_values("FDR q-val")
            sig_025 = res[res["FDR q-val"] < 0.25].sort_values("FDR q-val")

            print(f"      FDR < 0.05: {len(sig_005)} pathways")
            print(f"      FDR < 0.25: {len(sig_025)} pathways")

            # Show top 15 regardless of significance
            top15 = res.sort_values("FDR q-val").head(15)
            print(f"\n      Top 15 pathways:")
            print(f"      {'Term':<55} {'NES':>6} {'NOM p':>8} {'FDR':>8}")
            print(f"      {'-'*80}")
            for _, row in top15.iterrows():
                term = str(row["Term"])
                term = term[:53] + ".." if len(term) > 55 else term
                nes = row["NES"] if pd.notna(row["NES"]) else 0
                nom_p = row["NOM p-val"] if pd.notna(row["NOM p-val"]) else 999
                fdr = row["FDR q-val"] if pd.notna(row["FDR q-val"]) else 999
                print(f"      {term:<55} {nes:>6.2f} {nom_p:>8.4f} {fdr:>8.4f}")

            # Save full results
            res.to_csv(OUTPUT_DIR / f"{cohort}_gsea_{lib.split('_')[0]}.csv",
                      index=False)
            cohort_gsea_results[(cohort, lib)] = res

        except Exception as e:
            print(f"      GSEA failed: {e}")
            import traceback
            traceback.print_exc()

# =============================================================================
# Cross-cohort comparison
# =============================================================================

print()
print("=" * 60)
print("Cross-cohort GSEA pathway overlap")
print("=" * 60)

for lib in GSEA_LIBS:
    tcga_res = cohort_gsea_results.get(("tcga", lib))
    orien_res = cohort_gsea_results.get(("orien", lib))

    if tcga_res is None or orien_res is None:
        print(f"\n  {lib}: missing results, skipping")
        continue

    print(f"\n  {lib}:")

    for fdr_thresh, label in [(0.05, "FDR < 0.05"), (0.10, "FDR < 0.10"),
                               (0.25, "FDR < 0.25")]:
        tcga_sig = set(tcga_res[tcga_res["FDR q-val"] < fdr_thresh]["Term"])
        orien_sig = set(orien_res[orien_res["FDR q-val"] < fdr_thresh]["Term"])

        overlap = tcga_sig & orien_sig
        union = tcga_sig | orien_sig
        jaccard = len(overlap) / len(union) if union else 0

        print(f"    {label}:  TCGA={len(tcga_sig)}, ORIEN={len(orien_sig)}, "
              f"overlap={len(overlap)}, Jaccard={jaccard:.3f}")

        if overlap:
            for term in sorted(overlap):
                tcga_nes = tcga_res[tcga_res["Term"] == term]["NES"].values[0]
                orien_nes = orien_res[orien_res["Term"] == term]["NES"].values[0]
                direction = "SAME" if (tcga_nes > 0) == (orien_nes > 0) else "OPPOSITE"
                print(f"      - {str(term)[:55]:<55} "
                      f"T={tcga_nes:+.2f} O={orien_nes:+.2f} [{direction}]")

# =============================================================================
# NES correlation between cohorts (all pathways)
# =============================================================================

print()
print("=" * 60)
print("NES correlation between cohorts (all pathways)")
print("=" * 60)

from scipy.stats import spearmanr, pearsonr

for lib in GSEA_LIBS:
    tcga_res = cohort_gsea_results.get(("tcga", lib))
    orien_res = cohort_gsea_results.get(("orien", lib))

    if tcga_res is None or orien_res is None:
        continue

    # Merge on Term
    merged = tcga_res[["Term", "NES"]].merge(
        orien_res[["Term", "NES"]],
        on="Term", suffixes=("_tcga", "_orien")
    )
    merged = merged.dropna(subset=["NES_tcga", "NES_orien"])

    if len(merged) < 5:
        print(f"\n  {lib}: too few shared pathways for correlation")
        continue

    rho, p_rho = spearmanr(merged["NES_tcga"], merged["NES_orien"])
    r, p_r = pearsonr(merged["NES_tcga"], merged["NES_orien"])

    print(f"\n  {lib} ({len(merged)} pathways):")
    print(f"    Spearman rho = {rho:.4f} (p = {p_rho:.4e})")
    print(f"    Pearson r    = {r:.4f} (p = {p_r:.4e})")

    # Top concordant pathways (same direction, both near-significant)
    merged["mean_abs_nes"] = (merged["NES_tcga"].abs() + merged["NES_orien"].abs()) / 2
    merged["same_direction"] = (merged["NES_tcga"] > 0) == (merged["NES_orien"] > 0)
    concordant = merged[merged["same_direction"]].sort_values("mean_abs_nes",
                                                               ascending=False)
    if len(concordant) > 0:
        print(f"\n    Top 10 concordant pathways (same NES direction):")
        print(f"    {'Term':<50} {'T_NES':>6} {'O_NES':>6}")
        print(f"    {'-'*65}")
        for _, row in concordant.head(10).iterrows():
            term = str(row["Term"])[:48]
            print(f"    {term:<50} {row['NES_tcga']:>+6.2f} {row['NES_orien']:>+6.2f}")

print(f"\n{'='*60}")
print("Done.")
print(f"{'='*60}")
