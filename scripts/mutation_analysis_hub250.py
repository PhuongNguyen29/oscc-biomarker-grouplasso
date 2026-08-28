"""
Chapter 5: Somatic-mutation characterization of HUB-SCORE risk groups.

MINIMAL-CHANGE REPLICATION of the legacy analysis (Survival_analysis.ipynb,
cells 6-10, results/mutation_analysis_25gene): identical mutation loading,
identical non-silent/LoF/Missense variant-class definitions, identical binary
matrix construction, identical Fisher + Woolf CI per cohort, identical
Mantel-Haenszel pooling (Robins-Breslow-Greenland variance). The ONLY protocol
change: risk groups come from the 309-gene hub score (K3 canonical
scores_hub_250_{cohort}.csv, median split within cohort) instead of the legacy
25-gene bidirectional Cox score.

PRE-REGISTERED (recorded in plan before running):
  1. Primary: TP53 non-silent mutation frequency, High vs Low hub-score group;
     Fisher per cohort + MH pooled across cohorts. Expectation: replicates in
     direction (OR > 1); magnitude may differ from the legacy 1.54.
  2. Secondary aggregates (new, powered because pooled): TMB (all non-silent
     mutations per patient) and driver-set burden (n mutated genes among
     TP53, CDKN2A, PIK3CA, NOTCH1, FAT1, CASP8 -- TCGA-HNSCC driver list),
     Wilcoxon rank-sum High vs Low per cohort.
  3. TP53 LoF / Missense / Wildtype composition by risk group (chi-square) --
     the risk-group version of the legacy severity gradient.
  Interpretation rule fixed in advance: replication in both scores ->
  "TP53 enrichment is score-robust"; legacy-only -> reported honestly, with
  the note that the hub score captures microenvironment programmes where a
  weaker TP53 link is biologically plausible.

FRAMING NOTE for the chapter: TP53 WAS a member of the legacy 25-gene union
(ORIEN-stable, pi = 0.66) but is NOT in the corrected union or the 309-gene
hub set. The legacy bivariate result (TP53 mutation prognostic, expression
not) explains this: the expression signal was a mutation surrogate. In the
hub-score analysis TP53 mutation is therefore a fully INDEPENDENT molecular
correlate -- no circularity.

LIMITATION carried over from the legacy protocol for comparability (state in
methods): patients with no mutation record are treated as wildtype;
not-sequenced and sequenced-wildtype are not distinguished.

Add-on safety: reads existing artifacts read-only; writes to a NEW directory.

Usage:
    cd ~/hpchome/oscc-biomarker-grouplasso
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/mutation_analysis_hub250.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2

DATA_DIR = Path("data/processed")
K3_DIR = Path("results/phaseK3_bootstrap_variance")
OUT_DIR = Path("results/mutation_analysis_hub250")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Analysis gene sets (pre-registered; NOT the signature genes)
PRIMARY_GENE = "TP53"
DRIVER_GENES = ["TP53", "CDKN2A", "PIK3CA", "NOTCH1", "FAT1", "CASP8"]

# --- Variant-class definitions: byte-identical to the legacy notebook -------
LOF_TCGA = ["Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", "Splice_Site"]
MIS_TCGA = ["Missense_Mutation", "In_Frame_Del", "In_Frame_Ins"]
NONSILENT_TCGA = LOF_TCGA + MIS_TCGA + ["Nonstop_Mutation", "Translation_Start_Site"]

LOF_ORIEN = ["FRAME_SHIFT_DEL", "FRAME_SHIFT_INS", "NONSENSE", "SPLICE_SITE"]
MIS_ORIEN = ["MISSENSE", "IN_FRAME_DEL", "IN_FRAME_INS"]
NONSILENT_ORIEN = LOF_ORIEN + MIS_ORIEN + ["NONSTOP"]


# =============================================================================
# Risk groups from the K3 canonical hub scores (the ONE protocol change)
# =============================================================================

def load_risk_groups(cohort):
    sf = K3_DIR / f"scores_hub_250_{cohort}.csv"
    assert sf.exists(), (f"{sf} not found -- rerun phaseK3 with the "
                         f"per-patient score save block first.")
    sc = pd.read_csv(sf, dtype={"patient_id": str}).set_index("patient_id")
    med = sc["score"].median()
    rg = pd.Series(np.where(sc["score"] >= med, "High Risk", "Low Risk"),
                   index=sc.index, name="risk_group")
    return rg


# =============================================================================
# Mutation loading: identical to legacy cells 6-7 (minus the 25-gene filter --
# the analysis universe here is the pre-registered gene list + TMB over all
# non-silent mutations)
# =============================================================================

def load_mutations():
    mut_tcga = pd.read_csv(DATA_DIR / "TCGA_somatic_mutation.csv", index_col=0)
    mut_tcga["Processed_SampleID"] = (
        mut_tcga["Tumor_Sample_Barcode"].str[:12].str.replace("-", ".", regex=False))
    mut_tcga = mut_tcga[mut_tcga["Variant_Classification"].isin(NONSILENT_TCGA)].copy()
    mut_tcga["mut_category"] = np.where(
        mut_tcga["Variant_Classification"].isin(LOF_TCGA), "LoF",
        np.where(mut_tcga["Variant_Classification"].isin(MIS_TCGA), "Missense", "Other"))

    mut_orien = pd.read_csv(DATA_DIR / "all_mutations_combined.tsv", sep="\t")
    mut_orien = mut_orien[mut_orien["variant_class"].isin(NONSILENT_ORIEN)].copy()
    mut_orien["mut_category"] = np.where(
        mut_orien["variant_class"].isin(LOF_ORIEN), "LoF",
        np.where(mut_orien["variant_class"].isin(MIS_ORIEN), "Missense", "Other"))
    return mut_tcga, mut_orien


def binary_matrix(mut_df, gene_col, sample_col, all_samples, genes):
    """Legacy cell 9 construction: one row per patient-gene, absence = 0."""
    dedup = mut_df.drop_duplicates(subset=[gene_col, sample_col])
    mat = pd.DataFrame(0, index=list(all_samples), columns=genes)
    for _, row in dedup.iterrows():
        g, s = row[gene_col], row[sample_col]
        if g in mat.columns and s in mat.index:
            mat.loc[s, g] = 1
    return mat


# =============================================================================
# Statistics: byte-identical to legacy cells 9-10
# =============================================================================

def fisher_with_woolf(high_mut, high_total, low_mut, low_total):
    table = [[high_mut, high_total - high_mut],
             [low_mut, low_total - low_mut]]
    odds_ratio, p_value = stats.fisher_exact(table)
    a, b = high_mut + 0.5, high_total - high_mut + 0.5
    c, d = low_mut + 0.5, low_total - low_mut + 0.5
    log_or = np.log(a * d / (b * c))
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return odds_ratio, np.exp(log_or - 1.96 * se), np.exp(log_or + 1.96 * se), p_value


def mantel_haenszel(tables):
    """Legacy cell 10, verbatim: MH pooled OR, RBG variance CI, MH chi2 test."""
    numerator = denominator = 0.0
    for tbl in tables:
        a, b = tbl[0]; c, d = tbl[1]
        n = a + b + c + d
        numerator += (a * d) / n
        denominator += (b * c) / n
    if denominator == 0:
        return np.nan, (np.nan, np.nan), np.nan
    or_mh = numerator / denominator
    P = Q = R = 0.0
    S = numerator; T_sum = denominator
    for tbl in tables:
        a, b = tbl[0]; c, d = tbl[1]
        n = a + b + c + d
        P += (a + d) * a * d / n**2
        Q += ((a + d) * b * c + (b + c) * a * d) / n**2
        R += (b + c) * b * c / n**2
    ln_or = np.log(or_mh)
    var_ln = P / (2 * S**2) + Q / (2 * S * T_sum) + R / (2 * T_sum**2)
    se_ln = np.sqrt(var_ln)
    ci = (np.exp(ln_or - 1.96 * se_ln), np.exp(ln_or + 1.96 * se_ln))
    p_value = 1 - chi2.cdf(ln_or**2 / var_ln, df=1)
    return or_mh, ci, p_value


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    mut_tcga, mut_orien = load_mutations()

    cohorts = {}
    for name, mut_df, gcol, scol in [
        ("TCGA", mut_tcga, "Hugo_Symbol", "Processed_SampleID"),
        ("ORIEN", mut_orien, "gene", "sampleID"),
    ]:
        rg = load_risk_groups(name)
        samples = rg.index.tolist()
        n_with_mut = mut_df[mut_df[scol].isin(samples)][scol].nunique()
        print(f"[{name}] cohort n={len(samples)} "
              f"({(rg == 'High Risk').sum()} high / {(rg == 'Low Risk').sum()} low); "
              f"patients with >=1 non-silent mutation record: {n_with_mut}")
        cohorts[name] = dict(rg=rg, mut=mut_df, gcol=gcol, scol=scol,
                             samples=samples,
                             binary=binary_matrix(mut_df, gcol, scol,
                                                  samples, DRIVER_GENES))

    # ---- (1) Per-gene Fisher + MH pooling: driver genes, TP53 primary ------
    rows, mh_rows = [], []
    print("\n=== Driver-gene mutation vs hub-score risk group ===")
    print(f"{'Gene':<8} {'Cohort':<7} {'High':>10} {'Low':>10} "
          f"{'OR':>7} {'95% CI':>15} {'p':>9}")
    for gene in DRIVER_GENES:
        tables = []
        for name, cd in cohorts.items():
            hi = cd["rg"] == "High Risk"
            lo = cd["rg"] == "Low Risk"
            hm = int(cd["binary"].loc[hi[hi].index, gene].sum())
            lm = int(cd["binary"].loc[lo[lo].index, gene].sum())
            ht, lt = int(hi.sum()), int(lo.sum())
            orr, lo_ci, hi_ci, p = fisher_with_woolf(hm, ht, lm, lt)
            print(f"{gene:<8} {name:<7} {hm}/{ht:<7} {lm}/{lt:<7} "
                  f"{orr:>7.2f} ({lo_ci:.2f}-{hi_ci:.2f}) {p:>9.4f}")
            rows.append(dict(gene=gene, cohort=name, high_mut=hm, high_total=ht,
                             low_mut=lm, low_total=lt, odds_ratio=orr,
                             ci_lower=lo_ci, ci_upper=hi_ci, fisher_p=p))
            tables.append([[hm, ht - hm], [lm, lt - lm]])
        or_mh, ci, p_mh = mantel_haenszel(tables)
        print(f"{gene:<8} {'POOLED':<7} {'':>10} {'':>10} "
              f"{or_mh:>7.2f} ({ci[0]:.2f}-{ci[1]:.2f}) {p_mh:>9.4f}  [MH]")
        mh_rows.append(dict(gene=gene, or_mh=or_mh, ci_lower=ci[0],
                            ci_upper=ci[1], mh_p=p_mh))

    pd.DataFrame(rows).to_csv(OUT_DIR / "driver_fisher_by_cohort.csv", index=False)
    pd.DataFrame(mh_rows).to_csv(OUT_DIR / "driver_mantel_haenszel.csv", index=False)

    # ---- (2) Aggregates: TMB and driver-set burden -------------------------
    agg_rows = []
    print("\n=== Aggregate burden by hub-score risk group (Wilcoxon) ===")
    for name, cd in cohorts.items():
        tmb = (cd["mut"][cd["mut"][cd["scol"]].isin(cd["samples"])]
               .groupby(cd["scol"]).size()
               .reindex(cd["samples"]).fillna(0))
        drv = cd["binary"].sum(axis=1)
        for label, series in (("TMB_nonsilent", tmb), ("driver_set_count", drv)):
            hi = series[cd["rg"] == "High Risk"]
            lo = series[cd["rg"] == "Low Risk"]
            u, p = stats.mannwhitneyu(hi, lo, alternative="two-sided")
            print(f"[{name}] {label:<17} median high={hi.median():.1f} "
                  f"low={lo.median():.1f}  Wilcoxon p={p:.4f}")
            agg_rows.append(dict(cohort=name, measure=label,
                                 median_high=float(hi.median()),
                                 median_low=float(lo.median()),
                                 mean_high=float(hi.mean()),
                                 mean_low=float(lo.mean()), wilcoxon_p=float(p)))
    pd.DataFrame(agg_rows).to_csv(OUT_DIR / "aggregate_burden.csv", index=False)

    # ---- (3) TP53 LoF / Missense / Wildtype composition by risk group ------
    comp_rows = []
    print("\n=== TP53 severity composition by hub-score risk group ===")
    for name, cd in cohorts.items():
        tp53 = cd["mut"][(cd["mut"][cd["gcol"]] == PRIMARY_GENE)
                         & (cd["mut"][cd["scol"]].isin(cd["samples"]))]
        lof = set(tp53[tp53["mut_category"] == "LoF"][cd["scol"]])
        mis = set(tp53[tp53["mut_category"] == "Missense"][cd["scol"]]) - lof
        status = pd.Series("Wildtype", index=cd["samples"])
        status[status.index.isin(mis)] = "Missense"
        status[status.index.isin(lof)] = "LoF"
        ct = pd.crosstab(status, cd["rg"])
        ct = ct.reindex(index=["LoF", "Missense", "Wildtype"], fill_value=0)
        chi2_stat, p, _, _ = stats.chi2_contingency(ct)
        print(f"[{name}] chi2 p={p:.4f}\n{ct}\n")
        for grp in ct.index:
            for rgp in ct.columns:
                comp_rows.append(dict(cohort=name, tp53_status=grp,
                                      risk_group=rgp, n=int(ct.loc[grp, rgp]),
                                      chi2_p=float(p)))
    pd.DataFrame(comp_rows).to_csv(OUT_DIR / "tp53_severity_composition.csv",
                                   index=False)

    print(f"\nWritten: {OUT_DIR}/ (driver_fisher_by_cohort.csv, "
          f"driver_mantel_haenszel.csv, aggregate_burden.csv, "
          f"tp53_severity_composition.csv)")
