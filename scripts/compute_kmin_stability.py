"""
Compute per-gene stability entry point (k_min) from saved subsample rankings.

For each gene in each cohort, finds the smallest k at which the gene's
selection probability (fraction of 50 subsamples where it appears in top-k)
reaches π ≥ 0.80.

Reads from:
    results/phase4_stability_iqr/subsample_rankings/{tcga,orien}/cox/subsample_01.csv
    data/raw/protein_coding_genes_annotation.csv

Outputs:
    results/phase4_stability_iqr/kmin_analysis/tcga_kmin.csv
    results/phase4_stability_iqr/kmin_analysis/orien_kmin.csv
    results/phase4_stability_iqr/kmin_analysis/combined_kmin.csv

Usage:
    cd ~/hpchome/oscc-biomarker-grouplasso
    python scripts/compute_kmin_stability.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

RANKINGS_DIR = Path("results/phase4_stability_iqr/subsample_rankings")
OUTPUT_DIR = Path("results/phase4_stability_iqr/kmin_analysis")
ANNOT_FILE = Path("data/raw/protein_coding_genes_annotation.csv")

B_SUBSAMPLES = 50
PI_THRESHOLD = 0.80

# Dense k grid covering full gene space
K_GRID = [50, 100, 200, 300, 500, 750, 1000,1250, 1500, 1750, 2000, 2500, 3000, 3500, 4000, 4448]

COHORTS = ["tcga", "orien"]

# =============================================================================
# Functions
# =============================================================================

def load_subsample_rankings(cohort: str) -> list:
    """Load all 50 subsample ranking CSVs for one cohort."""
    rank_dir = RANKINGS_DIR / cohort / "cox"
    rankings = []
    for b in range(1, B_SUBSAMPLES + 1):
        fpath = rank_dir / f"subsample_{b:02d}.csv"
        if not fpath.exists():
            raise FileNotFoundError(f"Missing: {fpath}")
        df = pd.read_csv(fpath)
        # Each CSV has columns: gene, importance
        # Genes are already sorted by |importance| descending
        ranked_genes = df["gene"].tolist()
        rankings.append(ranked_genes)
    print(f"  Loaded {len(rankings)} subsample rankings for {cohort.upper()}")
    print(f"  Genes per ranking: {len(rankings[0])}")
    return rankings


def compute_selection_probs(rankings: list, gene_names: list, k_values: list) -> pd.DataFrame:
    """
    Compute selection probability for each gene at each k.

    For each k, selection probability = fraction of B subsamples where
    the gene appears in the top-k.
    """
    B = len(rankings)
    probs = pd.DataFrame(0.0, index=gene_names, columns=[f"k_{k}" for k in k_values])

    for ranking in rankings:
        for k in k_values:
            top_k = set(ranking[:k])
            for gene in top_k:
                if gene in probs.index:
                    probs.loc[gene, f"k_{k}"] += 1

    probs = probs / B
    return probs


def compute_kmin(probs: pd.DataFrame, k_values: list, pi: float) -> pd.DataFrame:
    """
    For each gene, compute:
      - k_min: smallest k where selection prob >= pi (NaN if never)
      - pi_at_k1000: selection prob at k=1000
      - max_pi: highest pi across all k values
      - k_at_max_pi: k where max_pi occurs
    """
    records = []

    for gene in probs.index:
        row = probs.loc[gene]

        # k_min: first k where pi >= threshold
        k_min = np.nan
        for k in k_values:
            if row[f"k_{k}"] >= pi:
                k_min = k
                break

        # pi at k=1000
        pi_at_k1000 = row.get("k_1000", np.nan)

        # max pi and where it occurs
        max_pi = row.max()
        max_col = row.idxmax()
        k_at_max_pi = int(max_col.replace("k_", ""))

        records.append({
            "gene": gene,
            "k_min": k_min,
            "pi_at_k1000": pi_at_k1000,
            "max_pi": max_pi,
            "k_at_max_pi": k_at_max_pi,
        })

    return pd.DataFrame(records)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load gene annotation
    print("Loading gene annotation...")
    annot = pd.read_csv(ANNOT_FILE)
    ensg_to_symbol = dict(zip(annot["gene_id"], annot["gene_symbol"]))
    print(f"  Annotation: {len(ensg_to_symbol)} gene mappings")

    # Process each cohort
    cohort_results = {}

    for cohort in COHORTS:
        print(f"\n{'='*60}")
        print(f"Processing {cohort.upper()}")
        print(f"{'='*60}")

        # Load rankings
        rankings = load_subsample_rankings(cohort)
        gene_names = rankings[0]  # all rankings have same genes

        # Compute selection probs at dense k grid
        print(f"  Computing selection probabilities at {len(K_GRID)} k values...")
        probs = compute_selection_probs(rankings, gene_names, K_GRID)

        # Save full probability matrix (for reference)
        probs.to_csv(OUTPUT_DIR / f"{cohort}_selection_probs_dense.csv")

        # Compute k_min
        print(f"  Computing k_min (pi >= {PI_THRESHOLD})...")
        kmin_df = compute_kmin(probs, K_GRID, PI_THRESHOLD)

        # Add gene symbol
        kmin_df["gene_symbol"] = kmin_df["gene"].map(ensg_to_symbol)

        # Sort by k_min (stable genes first), then by max_pi descending
        kmin_df = kmin_df.sort_values(
            ["k_min", "max_pi"],
            ascending=[True, False],
            na_position="last",
        )

        # Save
        out_path = OUTPUT_DIR / f"{cohort}_kmin.csv"
        kmin_df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

        # Summary
        n_stable = kmin_df["k_min"].notna().sum()
        n_never = kmin_df["k_min"].isna().sum()
        print(f"\n  Summary for {cohort.upper()}:")
        print(f"    Genes reaching pi >= {PI_THRESHOLD} at any k: {n_stable}")
        print(f"    Genes never reaching pi >= {PI_THRESHOLD}:    {n_never}")
        print(f"    Distribution of k_min among stable genes:")
        for k in K_GRID:
            n_at_k = (kmin_df["k_min"] == k).sum()
            n_cumul = (kmin_df["k_min"] <= k).sum()
            if n_at_k > 0:
                print(f"      k={k:>5}: {n_at_k:>4} genes first stable here "
                      f"(cumulative: {n_cumul})")

        cohort_results[cohort] = kmin_df

    # =========================================================================
    # Combined file
    # =========================================================================
    print(f"\n{'='*60}")
    print("Combined cross-cohort analysis")
    print(f"{'='*60}")

    tcga_kmin = cohort_results["tcga"][["gene", "gene_symbol", "k_min", "pi_at_k1000", "max_pi"]].copy()
    orien_kmin = cohort_results["orien"][["gene", "gene_symbol", "k_min", "pi_at_k1000", "max_pi"]].copy()

    tcga_kmin = tcga_kmin.rename(columns={
        "k_min": "k_min_tcga",
        "pi_at_k1000": "pi_k1000_tcga",
        "max_pi": "max_pi_tcga",
    })
    orien_kmin = orien_kmin.rename(columns={
        "k_min": "k_min_orien",
        "pi_at_k1000": "pi_k1000_orien",
        "max_pi": "max_pi_orien",
    })

    # Drop duplicate gene_symbol from orien before merge
    orien_kmin = orien_kmin.drop(columns=["gene_symbol"])

    combined = tcga_kmin.merge(orien_kmin, on="gene", how="outer")

    # k_min_max: the k at which gene is stable in BOTH cohorts
    # (max of the two k_min values; NaN if either is NaN)
    combined["k_min_max"] = combined[["k_min_tcga", "k_min_orien"]].max(axis=1)

    # Stability category
    def categorize(row):
        t = pd.notna(row["k_min_tcga"])
        o = pd.notna(row["k_min_orien"])
        if t and o:
            return "both"
        elif t:
            return "tcga_only"
        elif o:
            return "orien_only"
        else:
            return "neither"

    combined["stable_in"] = combined.apply(categorize, axis=1)

    # Sort: both-cohort stable first (by k_min_max), then single-cohort, then neither
    sort_order = {"both": 0, "tcga_only": 1, "orien_only": 2, "neither": 3}
    combined["_sort"] = combined["stable_in"].map(sort_order)
    combined = combined.sort_values(
        ["_sort", "k_min_max", "k_min_tcga", "k_min_orien"],
        ascending=[True, True, True, True],
        na_position="last",
    )
    combined = combined.drop(columns=["_sort"])

    # Save
    out_path = OUTPUT_DIR / "combined_kmin.csv"
    combined.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Summary statistics
    cats = combined["stable_in"].value_counts()
    print(f"\nStability categories:")
    for cat in ["both", "tcga_only", "orien_only", "neither"]:
        print(f"  {cat:>12}: {cats.get(cat, 0)} genes")

    both_genes = combined[combined["stable_in"] == "both"]
    if len(both_genes) > 0:
        print(f"\nGenes stable in BOTH cohorts ({len(both_genes)}):")
        print(f"  k_min_max range: {both_genes['k_min_max'].min():.0f} - "
              f"{both_genes['k_min_max'].max():.0f}")
        print(f"\n  Top 30 by k_min_max:")
        display_cols = ["gene", "gene_symbol", "k_min_tcga", "k_min_orien",
                        "k_min_max", "pi_k1000_tcga", "pi_k1000_orien"]
        print(both_genes[display_cols].head(30).to_string(index=False))
    else:
        print("\n  No genes stable in both cohorts at any k.")
        print("  Showing genes closest to being stable in both:")
        # Genes stable in one, with highest max_pi in the other
        tcga_only = combined[combined["stable_in"] == "tcga_only"].copy()
        if len(tcga_only) > 0:
            tcga_only = tcga_only.sort_values("max_pi_orien", ascending=False)
            print(f"\n  TCGA-stable genes with highest ORIEN max_pi (top 10):")
            print(tcga_only[["gene", "gene_symbol", "k_min_tcga",
                             "max_pi_orien"]].head(10).to_string(index=False))

        orien_only = combined[combined["stable_in"] == "orien_only"].copy()
        if len(orien_only) > 0:
            orien_only = orien_only.sort_values("max_pi_tcga", ascending=False)
            print(f"\n  ORIEN-stable genes with highest TCGA max_pi (top 10):")
            print(orien_only[["gene", "gene_symbol", "k_min_orien",
                              "max_pi_tcga"]].head(10).to_string(index=False))

    print(f"\n{'='*60}")
    print("Done.")
    print(f"{'='*60}")
