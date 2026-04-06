"""
GSEA preranked using mean signed lasso coefficients.

Ranking metric = mean signed coefficient across all 50 subsamples
(including zeros when gene was not selected). This naturally combines:
  - Selection frequency (more selections → larger absolute mean)
  - Effect magnitude (larger coefficients → larger absolute mean)  
  - Direction consistency (sign flipping → cancellation toward zero)

Avoids the 99% ties problem of using selection frequency directly.

Usage:
    cd ~/hpchome/oscc-biomarker-grouplasso
    python scripts/gsea_lasso_mean_coef.py
"""

import pandas as pd
import numpy as np
import gseapy as gp
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

COEF_DIR = Path("results/phase4_stability_randomized_lasso/signed_coefficients")
ANNOT_FILE = Path("data/raw/protein_coding_genes_annotation.csv")
OUTPUT_DIR = Path("results/phase4_stability_randomized_lasso/gsea_mean_coef")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COHORTS = ["tcga", "orien"]
GSEA_LIBS = ["KEGG_2021_Human", "GO_Biological_Process_2023", "MSigDB_Hallmark_2020"]

# =============================================================================
# Load and compute ranking metric
# =============================================================================

print("=" * 60)
print("Loading signed coefficients and computing ranking metric")
print("=" * 60)

annot = pd.read_csv(ANNOT_FILE)
ensg_to_symbol = dict(zip(annot["gene_id"], annot["gene_symbol"]))

cohort_rankings = {}

for cohort in COHORTS:
    coef_df = pd.read_csv(COEF_DIR / cohort / "cox_signed_coefficients.csv",
                          index_col=0)

    print(f"\n  {cohort.upper()}:")
    print(f"    Coefficient matrix: {coef_df.shape[0]} genes x {coef_df.shape[1]} subsamples")

    # Mean signed coefficient across all 50 subsamples (including zeros)
    mean_coef = coef_df.mean(axis=1)

    # Also compute supporting stats
    n_nonzero = (coef_df != 0).sum(axis=1)
    mean_abs_coef = coef_df.abs().mean(axis=1)

    # How many genes have at least 1 nonzero coefficient?
    ever_selected = (n_nonzero > 0).sum()
    never_selected = (n_nonzero == 0).sum()

    print(f"    Ever selected (>=1 subsample): {ever_selected}")
    print(f"    Never selected: {never_selected}")
    print(f"    Unique mean_coef values: {mean_coef.nunique()}")
    print(f"    Genes with mean_coef = 0: {(mean_coef == 0).sum()}")
    print(f"    Ties at zero: {100*(mean_coef == 0).sum()/len(mean_coef):.1f}%")

    # Build ranked list: gene_symbol -> mean_signed_coef
    rnk_records = {}
    for gene in mean_coef.index:
        sym = ensg_to_symbol.get(gene)
        if sym and sym not in rnk_records:
            rnk_records[sym] = mean_coef[gene]

    rnk_series = pd.Series(rnk_records)
    rnk_series = rnk_series.sort_values(ascending=False)

    print(f"    Mapped to symbols: {len(rnk_series)}")
    print(f"    Ties at zero after mapping: {(rnk_series == 0).sum()} "
          f"({100*(rnk_series == 0).sum()/len(rnk_series):.1f}%)")

    print(f"\n    Top 10 (most positive mean coef — hazard-associated):")
    for sym, val in rnk_series.head(10).items():
        gene = [g for g, s in ensg_to_symbol.items() if s == sym]
        n_sel = int(n_nonzero[gene[0]]) if gene else 0
        print(f"      {sym:<15} mean_coef={val:+.6f}  selected={n_sel}/50")

    print(f"\n    Bottom 10 (most negative mean coef — protective):")
    for sym, val in rnk_series.tail(10).items():
        gene = [g for g, s in ensg_to_symbol.items() if s == sym]
        n_sel = int(n_nonzero[gene[0]]) if gene else 0
        print(f"      {sym:<15} mean_coef={val:+.6f}  selected={n_sel}/50")

    cohort_rankings[cohort] = rnk_series

# =============================================================================
# Cross-cohort rank correlation of mean coefficients
# =============================================================================

print()
print("=" * 60)
print("Cross-cohort correlation of mean signed coefficients")
print("=" * 60)

tcga_rnk = cohort_rankings["tcga"]
orien_rnk = cohort_rankings["orien"]

common_syms = sorted(set(tcga_rnk.index) & set(orien_rnk.index))
print(f"  Common gene symbols: {len(common_syms)}")

t_vals = tcga_rnk[common_syms]
o_vals = orien_rnk[common_syms]

rho_all, p_all = spearmanr(t_vals, o_vals)
print(f"  All genes: Spearman rho = {rho_all:.4f} (p = {p_all:.2e})")

# Among ever-selected genes only (nonzero in at least one cohort)
nonzero_mask = (t_vals != 0) | (o_vals != 0)
t_nz = t_vals[nonzero_mask]
o_nz = o_vals[nonzero_mask]
if len(t_nz) > 10:
    rho_nz, p_nz = spearmanr(t_nz, o_nz)
    print(f"  Ever-selected ({nonzero_mask.sum()} genes): "
          f"Spearman rho = {rho_nz:.4f} (p = {p_nz:.2e})")

# =============================================================================
# GSEA preranked
# =============================================================================

print()
print("=" * 60)
print("GSEA preranked on mean signed coefficients")
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

            sig_005 = res[res["FDR q-val"] < 0.05]
            sig_025 = res[res["FDR q-val"] < 0.25]

            print(f"      FDR < 0.05: {len(sig_005)} pathways")
            print(f"      FDR < 0.25: {len(sig_025)} pathways")

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

            res.to_csv(OUTPUT_DIR / f"{cohort}_gsea_{lib.split('_')[0]}.csv",
                      index=False)
            cohort_gsea_results[(cohort, lib)] = res

        except Exception as e:
            print(f"      GSEA failed: {e}")
            import traceback
            traceback.print_exc()

# =============================================================================
# Cross-cohort GSEA pathway overlap
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

    for fdr_thresh, label in [(0.05, "FDR<0.05"), (0.10, "FDR<0.10"),
                               (0.25, "FDR<0.25")]:
        tcga_sig = set(tcga_res[tcga_res["FDR q-val"] < fdr_thresh]["Term"])
        orien_sig = set(orien_res[orien_res["FDR q-val"] < fdr_thresh]["Term"])

        overlap = tcga_sig & orien_sig
        union = tcga_sig | orien_sig
        jaccard = len(overlap) / len(union) if union else 0

        print(f"    {label}:  TCGA={len(tcga_sig)}, ORIEN={len(orien_sig)}, "
              f"overlap={len(overlap)}, Jaccard={jaccard:.3f}")

        if overlap:
            for term in sorted(overlap):
                tcga_nes = float(tcga_res[tcga_res["Term"] == term]["NES"].values[0])
                orien_nes = float(orien_res[orien_res["Term"] == term]["NES"].values[0])
                direction = "SAME" if (tcga_nes > 0) == (orien_nes > 0) else "OPP"
                print(f"      - {str(term)[:50]:<50} "
                      f"T={tcga_nes:+.2f} O={orien_nes:+.2f} [{direction}]")

# =============================================================================
# NES correlation
# =============================================================================

print()
print("=" * 60)
print("NES correlation between cohorts")
print("=" * 60)

for lib in GSEA_LIBS:
    tcga_res = cohort_gsea_results.get(("tcga", lib))
    orien_res = cohort_gsea_results.get(("orien", lib))

    if tcga_res is None or orien_res is None:
        continue

    merged = tcga_res[["Term", "NES"]].merge(
        orien_res[["Term", "NES"]],
        on="Term", suffixes=("_tcga", "_orien")
    ).dropna(subset=["NES_tcga", "NES_orien"])

    if len(merged) < 5:
        continue

    rho, p_rho = spearmanr(merged["NES_tcga"], merged["NES_orien"])

    print(f"\n  {lib} ({len(merged)} pathways):")
    print(f"    Spearman rho = {rho:.4f} (p = {p_rho:.4e})")

    concordant = merged[(merged["NES_tcga"] > 0) == (merged["NES_orien"] > 0)]
    print(f"    Same direction: {len(concordant)}/{len(merged)} "
          f"({100*len(concordant)/len(merged):.0f}%)")

    concordant = concordant.copy()
    concordant["mean_abs"] = (concordant["NES_tcga"].abs() +
                               concordant["NES_orien"].abs()) / 2
    concordant = concordant.sort_values("mean_abs", ascending=False)

    if len(concordant) > 0:
        print(f"\n    Top 10 concordant pathways:")
        print(f"    {'Term':<50} {'T_NES':>6} {'O_NES':>6}")
        print(f"    {'-'*65}")
        for _, row in concordant.head(10).iterrows():
            print(f"    {str(row['Term'])[:48]:<50} "
                  f"{row['NES_tcga']:>+6.2f} {row['NES_orien']:>+6.2f}")

print(f"\n{'='*60}")
print("Done.")
print(f"{'='*60}")
