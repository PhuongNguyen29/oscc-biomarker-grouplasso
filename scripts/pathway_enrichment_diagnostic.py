"""
Pathway enrichment diagnostic: do TCGA-stable and ORIEN-stable genes
converge on shared pathways despite low gene-level overlap?

Runs ORA (overrepresentation analysis) on per-cohort stable gene sets
at multiple k_min thresholds. Compares enriched pathway overlap between
cohorts at each threshold.

Requires:
    - results/phase4_stability_iqr/kmin_analysis/tcga_kmin.csv
    - results/phase4_stability_iqr/kmin_analysis/orien_kmin.csv
    - data/raw/protein_coding_genes_annotation.csv
    - data/processed/iqr_filtered/retained_genes.txt
    - gseapy >= 1.1

References:
    - Subramanian et al. (2005) PNAS — Gene set enrichment
    - Fang et al. (2023) — gseapy
    - Bourgon et al. (2010) PNAS — Independent filtering principle

Usage:
    cd ~/hpchome/oscc-biomarker-grouplasso
    python scripts/pathway_enrichment_diagnostic.py
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

KMIN_DIR = Path("results/phase4_stability_iqr/kmin_analysis")
ANNOT_FILE = Path("data/raw/protein_coding_genes_annotation.csv")
IQR_GENES_FILE = Path("data/processed/iqr_filtered/retained_genes.txt")
OUTPUT_DIR = Path("results/phase4_stability_iqr/pathway_diagnostic")

K_THRESHOLDS = [300, 500, 750, 1000, 1250, 1500]

GENE_SET_LIBS = [
    "KEGG_2021_Human",
    "GO_Biological_Process_2023",
    "MSigDB_Hallmark_2020",
]

FDR_THRESHOLD = 0.05

# =============================================================================
# Functions
# =============================================================================

def get_stable_symbols(kmin_df, k_thresh, ensg_to_sym):
    """Get unique gene symbols for genes with k_min <= threshold."""
    stable_ensg = kmin_df[kmin_df["k_min"] <= k_thresh]["gene"].tolist()
    symbols = []
    seen = set()
    for g in stable_ensg:
        s = ensg_to_sym.get(g)
        if s and s not in seen:
            symbols.append(s)
            seen.add(s)
    return symbols

# =============================================================================
# Cell 1: Load data
# =============================================================================

print("=" * 60)
print("Cell 1: Loading data")
print("=" * 60)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tcga_kmin = pd.read_csv(KMIN_DIR / "tcga_kmin.csv")
orien_kmin = pd.read_csv(KMIN_DIR / "orien_kmin.csv")

annot = pd.read_csv(ANNOT_FILE)
ensg_to_symbol = dict(zip(annot["gene_id"], annot["gene_symbol"]))

with open(IQR_GENES_FILE) as f:
    all_ensg = [line.strip() for line in f if line.strip()]
background = list(set(ensg_to_symbol[g] for g in all_ensg if g in ensg_to_symbol))

print(f"TCGA k_min: {tcga_kmin.shape[0]} genes")
print(f"ORIEN k_min: {orien_kmin.shape[0]} genes")
print(f"Background: {len(background)} unique gene symbols")

# =============================================================================
# Cell 2: Per-cohort stable gene counts at each threshold
# =============================================================================

print()
print("=" * 60)
print("Cell 2: Gene-level overlap at each k_min threshold")
print("=" * 60)

print(f"\n{'k_thresh':>8} | {'TCGA':>6} | {'ORIEN':>6} | {'Overlap':>8} | {'Union':>6} | {'Jaccard':>8}")
print("-" * 60)

for k in K_THRESHOLDS:
    t_syms = set(get_stable_symbols(tcga_kmin, k, ensg_to_symbol))
    o_syms = set(get_stable_symbols(orien_kmin, k, ensg_to_symbol))
    overlap = t_syms & o_syms
    union = t_syms | o_syms
    jaccard = len(overlap) / len(union) if union else 0
    print(f"{k:>8} | {len(t_syms):>6} | {len(o_syms):>6} | {len(overlap):>8} | {len(union):>6} | {jaccard:>8.3f}")

# =============================================================================
# Cell 3: Run ORA at each threshold
# =============================================================================

print()
print("=" * 60)
print("Cell 3: Running ORA at each threshold")
print("=" * 60)

all_results = {}

for k in K_THRESHOLDS:
    tcga_genes = get_stable_symbols(tcga_kmin, k, ensg_to_symbol)
    orien_genes = get_stable_symbols(orien_kmin, k, ensg_to_symbol)

    print(f"\n{'='*60}")
    print(f"k_min <= {k}: TCGA={len(tcga_genes)} genes, ORIEN={len(orien_genes)} genes")
    print(f"{'='*60}")

    for lib in GENE_SET_LIBS:
        print(f"\n  Library: {lib}")

        # TCGA enrichment
        try:
            tcga_enr = gp.enrich(
                gene_list=tcga_genes,
                gene_sets=lib,
                background=background,
                outdir=None,
                no_plot=True,
                verbose=False,
            )
            tcga_res = tcga_enr.results.copy()
        except Exception as e:
            print(f"    TCGA failed: {e}")
            tcga_res = pd.DataFrame()

        # ORIEN enrichment
        try:
            orien_enr = gp.enrich(
                gene_list=orien_genes,
                gene_sets=lib,
                background=background,
                outdir=None,
                no_plot=True,
                verbose=False,
            )
            orien_res = orien_enr.results.copy()
        except Exception as e:
            print(f"    ORIEN failed: {e}")
            orien_res = pd.DataFrame()

        # Compare
        tcga_sig = set()
        orien_sig = set()
        if not tcga_res.empty:
            tcga_sig = set(tcga_res[tcga_res["Adjusted P-value"] < FDR_THRESHOLD]["Term"])
        if not orien_res.empty:
            orien_sig = set(orien_res[orien_res["Adjusted P-value"] < FDR_THRESHOLD]["Term"])

        overlap = tcga_sig & orien_sig
        union = tcga_sig | orien_sig
        jaccard = len(overlap) / len(union) if union else 0.0

        print(f"    TCGA sig pathways:  {len(tcga_sig)}")
        print(f"    ORIEN sig pathways: {len(orien_sig)}")
        print(f"    Overlap:           {len(overlap)}")
        print(f"    Jaccard:           {jaccard:.3f}")

        if overlap:
            print(f"    Shared pathways:")
            for term in sorted(overlap):
                print(f"      - {term}")

        all_results[(k, lib)] = {
            "tcga_res": tcga_res,
            "orien_res": orien_res,
            "tcga_sig": tcga_sig,
            "orien_sig": orien_sig,
            "overlap": overlap,
            "jaccard": jaccard,
        }

# =============================================================================
# Cell 4: Pathway overlap summary across thresholds
# =============================================================================

print()
print("=" * 60)
print("Cell 4: Pathway overlap summary across thresholds")
print("=" * 60)

print(f"\n{'Library':<35} {'k':>5} | {'T_sig':>6} {'O_sig':>6} {'Overlap':>8} {'Jaccard':>8}")
print("-" * 80)

for lib in GENE_SET_LIBS:
    for k in K_THRESHOLDS:
        r = all_results.get((k, lib))
        if r:
            print(f"{lib:<35} {k:>5} | {len(r['tcga_sig']):>6} {len(r['orien_sig']):>6} "
                  f"{len(r['overlap']):>8} {r['jaccard']:>8.3f}")
    print()

# =============================================================================
# Cell 5: Persistent pathways across thresholds
# =============================================================================

print("=" * 60)
print("Cell 5: Persistent pathways (shared at >= 2 thresholds)")
print("=" * 60)

for lib in GENE_SET_LIBS:
    print(f"\n{'='*60}")
    print(f"Library: {lib}")
    print(f"{'='*60}")

    pathway_counts = {}
    for k in K_THRESHOLDS:
        r = all_results.get((k, lib))
        if r and r["overlap"]:
            for term in r["overlap"]:
                if term not in pathway_counts:
                    pathway_counts[term] = []
                pathway_counts[term].append(k)

    if not pathway_counts:
        print("  No shared pathways at any threshold.")
        continue

    sorted_pathways = sorted(pathway_counts.items(), key=lambda x: len(x[1]), reverse=True)

    print(f"\n  Pathways shared at multiple thresholds:")
    print(f"  {'Pathway':<55} {'n_thresh':>8}  k values")
    print(f"  {'-'*90}")
    for term, ks in sorted_pathways:
        if len(ks) >= 2:
            k_str = ", ".join(str(k) for k in ks)
            display_term = term[:53] + ".." if len(term) > 55 else term
            print(f"  {display_term:<55} {len(ks):>8}  {k_str}")

    n_single = sum(1 for _, ks in sorted_pathways if len(ks) == 1)
    if n_single > 0:
        print(f"\n  + {n_single} pathways shared at only 1 threshold (not shown)")

# =============================================================================
# Cell 6: Detailed results at k=1000
# =============================================================================

print()
print("=" * 60)
print("Cell 6: Detailed enrichment at k_min <= 1000")
print("=" * 60)

BEST_K = 1000

for lib in GENE_SET_LIBS:
    r = all_results.get((BEST_K, lib))
    if not r:
        continue

    print(f"\n{'='*60}")
    print(f"{lib} at k_min <= {BEST_K}")
    print(f"{'='*60}")

    tcga_res = r["tcga_res"]
    orien_res = r["orien_res"]

    if not tcga_res.empty:
        tcga_sig = tcga_res[tcga_res["Adjusted P-value"] < FDR_THRESHOLD].sort_values("Adjusted P-value")
        print(f"\n  TCGA significant pathways ({len(tcga_sig)}):")
        if len(tcga_sig) > 0:
            for _, row in tcga_sig.head(20).iterrows():
                in_overlap = "*" if row["Term"] in r["overlap"] else " "
                print(f"    {in_overlap} {row['Term'][:60]:<60} FDR={row['Adjusted P-value']:.4f}")

    if not orien_res.empty:
        orien_sig_df = orien_res[orien_res["Adjusted P-value"] < FDR_THRESHOLD].sort_values("Adjusted P-value")
        print(f"\n  ORIEN significant pathways ({len(orien_sig_df)}):")
        if len(orien_sig_df) > 0:
            for _, row in orien_sig_df.head(20).iterrows():
                in_overlap = "*" if row["Term"] in r["overlap"] else " "
                print(f"    {in_overlap} {row['Term'][:60]:<60} FDR={row['Adjusted P-value']:.4f}")

    print(f"\n  (* = shared between cohorts)")

# =============================================================================
# Cell 7: Save results
# =============================================================================

print()
print("=" * 60)
print("Cell 7: Saving results")
print("=" * 60)

for (k, lib), r in all_results.items():
    lib_short = lib.split("_")[0]

    if not r["tcga_res"].empty:
        fname = OUTPUT_DIR / f"tcga_k{k}_{lib_short}_enrichment.csv"
        r["tcga_res"].to_csv(fname, index=False)

    if not r["orien_res"].empty:
        fname = OUTPUT_DIR / f"orien_k{k}_{lib_short}_enrichment.csv"
        r["orien_res"].to_csv(fname, index=False)

summary_rows = []
for lib in GENE_SET_LIBS:
    for k in K_THRESHOLDS:
        r = all_results.get((k, lib))
        if r:
            summary_rows.append({
                "library": lib,
                "k_threshold": k,
                "tcga_sig": len(r["tcga_sig"]),
                "orien_sig": len(r["orien_sig"]),
                "overlap": len(r["overlap"]),
                "union": len(r["tcga_sig"] | r["orien_sig"]),
                "jaccard": r["jaccard"],
            })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_DIR / "pathway_overlap_summary.csv", index=False)
print(f"Results saved to {OUTPUT_DIR}")
print(summary_df.to_string(index=False))

print(f"\n{'='*60}")
print("Done.")
print(f"{'='*60}")
