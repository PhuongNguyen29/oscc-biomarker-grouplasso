"""
Phase 8, Step 8.6: Marginal-effect transportability diagnostics
================================================================

TWO computations, both from files that already exist. These are the LAST
Phase 8 diagnostics -- after this the project moves to Phase K.

(1) DEVELOPMENT <-> EXTERNAL CORRELATION
    Spearman and Pearson correlation of marginal Cox coefficients across all
    genes present in all four cohorts.

    Known so far:
        TCGA <-> ORIEN            r_true = 0.674 (Rung 1, disattenuated)
        GSE41613 <-> GSE42743     rho = 0.480    (step 8.4)
        development <-> external  UNTESTED  <-- this fills it in

    Why it matters: it determines what Phase K can aim at. If development and
    external cohorts share marginal structure, there is something for a
    transfer-learning method to transfer. If not, the honest conclusion is
    that these cohorts do not share estimable structure at gene level, which
    points at module-level work instead.

    Correlations are reported BOTH raw and disattenuated. Raw correlation is
    attenuated by measurement error in both axes: with SEs comparable to the
    effects themselves, the raw value substantially understates the
    correlation of TRUE effects. The disattenuation follows the Rung 1
    approach -- estimate the signal variance as var(beta) - mean(se^2) per
    cohort, then scale.

(2) WINNER'S CURSE CHECK
    Genes were nominated by being extreme in the external cohorts (smallest
    q-value). Selecting on extremeness biases estimates upward: the genes that
    look strongest in a small sample are disproportionately those where noise
    pushed the estimate up. HOPX's external beta is ~8x its pooled development
    beta, which is the signature of this.

    The test: for the top-k genes by external q, compare external beta with
    development beta. If inflation is SYSTEMATIC across the set, the external
    cohorts inflate whatever they rank highly and no single-gene external
    result should be taken at face value. If HOPX is an OUTLIER while the rest
    track the diagonal, HOPX is genuinely peculiar.

    A matched null is included: the same comparison for a random sample of
    genes NOT selected on q. The contrast between the two ratios is the
    quantity of interest -- some shrinkage is expected from platform
    differences alone, and the random set calibrates that baseline.

CAVEAT ON COMPARABILITY
-----------------------
Development betas are per SD of z-scored RNA-seq; external betas are per SD of
z-scored microarray (gcRMA for GSE41613, RMA for GSE42743). Both are on a
per-SD scale so they are comparable in principle, but platform differences in
dynamic range mean a ratio of exactly 1.0 should not be expected even for a
perfectly transported effect. Interpret ratios relative to the random-gene
baseline, not against 1.0.

Usage:
    python scripts/phase8_step86_transportability.py
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DDIT4 = "ENSG00000168209"
HOPX = "ENSG00000171476"


def disattenuate(b1, se1, b2, se2):
    """
    Correlation of TRUE effects, correcting for measurement error in both axes.

    var(observed) = var(true) + mean(se^2), so the attenuation factor is
    sqrt(var_true_1 / var_obs_1) * sqrt(var_true_2 / var_obs_2). Returns
    (r_observed, r_true, attenuation_factor). r_true is capped at 1.0 and
    returns NaN if either estimated signal variance is non-positive, which
    happens when measurement error exceeds observed variance.
    """
    r_obs = np.corrcoef(b1, b2)[0, 1]
    v1 = np.var(b1, ddof=1) - np.mean(se1 ** 2)
    v2 = np.var(b2, ddof=1) - np.mean(se2 ** 2)
    if v1 <= 0 or v2 <= 0:
        return r_obs, np.nan, np.nan
    att = np.sqrt(v1 / np.var(b1, ddof=1)) * np.sqrt(v2 / np.var(b2, ddof=1))
    return r_obs, min(r_obs / att, 1.0), att


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-gene", default="per_gene_results.csv",
                    help="Rung 1 per-gene meta-analysis results")
    ap.add_argument("--external-dir", default="results/phase8_external_validation")
    ap.add_argument("--output-dir", default="results/phase8_transportability")
    ap.add_argument("--top-k", type=int, default=50,
                    help="Number of top external genes for the winner's curse check")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out / "step8_6_transportability.log", mode="w")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info("Phase 8.6: Marginal-effect transportability")
    logger.info("=" * 66)

    # ---- Load ---------------------------------------------------------------
    dev = pd.read_csv(args.per_gene)
    n_before = len(dev)
    dev = dev[~dev.gene_id.duplicated(keep="first")]
    if len(dev) < n_before:
        logger.info(f"Development: dropped {n_before - len(dev)} duplicate gene_id "
                    f"rows (14 pseudoautosomal genes annotated on both chrX and "
                    f"chrY; values are identical -- plan C3)")
    e1 = pd.read_csv(Path(args.external_dir) / "step8_4_univariate_GSE41613.csv")
    e2 = pd.read_csv(Path(args.external_dir) / "step8_4_univariate_GSE42743.csv")

    logger.info(f"Development genes: {len(dev)}")
    logger.info(f"GSE41613: {len(e1)}   GSE42743: {len(e2)}")

    d = (dev[["gene_id", "gene_symbol", "beta_tcga", "se_tcga",
              "beta_orien", "se_orien", "beta_pooled", "se_pooled"]]
         .merge(e1[["gene_id", "coef", "se", "p_value", "q_value"]]
                .rename(columns={"coef": "beta_41613", "se": "se_41613",
                                 "p_value": "p_41613", "q_value": "q_41613"}),
                on="gene_id")
         .merge(e2[["gene_id", "coef", "se", "p_value", "q_value"]]
                .rename(columns={"coef": "beta_42743", "se": "se_42743",
                                 "p_value": "p_42743", "q_value": "q_42743"}),
                on="gene_id"))
    assert d.gene_id.is_unique, "duplicate gene_id survived the merge"
    assert len(d) <= min(len(e1), len(e2)), \
        f"merge produced {len(d)} rows from inputs of {len(e1)} and {len(e2)}"

    # External pooled: inverse-variance fixed-effect, so the two external
    # cohorts are combined on the same footing as beta_pooled combines the
    # two development cohorts.
    w1, w2 = 1 / d.se_41613 ** 2, 1 / d.se_42743 ** 2
    d["beta_ext"] = (w1 * d.beta_41613 + w2 * d.beta_42743) / (w1 + w2)
    d["se_ext"] = np.sqrt(1 / (w1 + w2))

    logger.info(f"Genes in ALL FOUR cohorts: {len(d)}\n")

    # =========================================================================
    # (1) Development <-> external correlation
    # =========================================================================
    logger.info("=" * 66)
    logger.info("(1) MARGINAL-EFFECT CORRELATION ACROSS COHORTS")
    logger.info("=" * 66)
    logger.info(f"{'pair':32s} {'Spearman':>9s} {'Pearson':>9s} "
                f"{'r_true':>8s} {'atten':>7s}")

    pairs = [
        ("TCGA <-> ORIEN",            "beta_tcga",  "se_tcga",  "beta_orien", "se_orien"),
        ("GSE41613 <-> GSE42743",     "beta_41613", "se_41613", "beta_42743", "se_42743"),
        ("dev pooled <-> GSE41613",   "beta_pooled","se_pooled","beta_41613", "se_41613"),
        ("dev pooled <-> GSE42743",   "beta_pooled","se_pooled","beta_42743", "se_42743"),
        ("dev pooled <-> ext pooled", "beta_pooled","se_pooled","beta_ext",   "se_ext"),
        ("TCGA <-> ext pooled",       "beta_tcga",  "se_tcga",  "beta_ext",   "se_ext"),
        ("ORIEN <-> ext pooled",      "beta_orien", "se_orien", "beta_ext",   "se_ext"),
    ]

    corr_rows = []
    for label, b1, s1, b2, s2 in pairs:
        rho = stats.spearmanr(d[b1], d[b2]).statistic
        r_obs, r_true, att = disattenuate(d[b1].values, d[s1].values,
                                          d[b2].values, d[s2].values)
        logger.info(f"{label:32s} {rho:9.3f} {r_obs:9.3f} "
                    f"{r_true:8.3f} {att:7.3f}")
        corr_rows.append({"pair": label, "spearman": rho, "pearson": r_obs,
                          "r_true": r_true, "attenuation": att})

    pd.DataFrame(corr_rows).to_csv(out / "cohort_correlations.csv", index=False)

    sign_dev_ext = ((d.beta_pooled > 0) == (d.beta_ext > 0)).mean()
    logger.info(f"\nSign concordance, dev pooled vs ext pooled: {sign_dev_ext:.3f}")
    logger.info("  (Rung 1: 0.494 without ComBat, 0.567 with. Chance = 0.50.)")
    logger.info("  Not a test -- co-expression violates independence.")

    # =========================================================================
    # (2) Winner's curse
    # =========================================================================
    logger.info("\n" + "=" * 66)
    logger.info(f"(2) WINNER'S CURSE: top {args.top_k} genes by external q")
    logger.info("=" * 66)

    d["q_ext_min"] = d[["q_41613", "q_42743"]].min(axis=1)
    top = d.nsmallest(args.top_k, "q_ext_min").copy()

    # Ratio of |external| to |development| effect. Values >> 1 mean the
    # external estimate is much larger than the development estimate.
    def ratio(df):
        return (df.beta_ext.abs() / df.beta_pooled.abs().clip(lower=1e-6))

    top["infl"] = ratio(top)

    rng = np.random.RandomState(args.seed)
    rand = d.loc[rng.choice(d.index, size=min(500, len(d)), replace=False)].copy()
    rand["infl"] = ratio(rand)

    logger.info(f"  Top-{args.top_k} by external q:")
    logger.info(f"    median |beta_ext| / |beta_dev| = {top.infl.median():.2f}")
    logger.info(f"    IQR = {top.infl.quantile(0.25):.2f} - {top.infl.quantile(0.75):.2f}")
    logger.info(f"    fraction with ratio > 2 = {(top.infl > 2).mean():.2f}")
    logger.info(f"  Random {len(rand)} genes (baseline):")
    logger.info(f"    median ratio = {rand.infl.median():.2f}")
    logger.info(f"    fraction with ratio > 2 = {(rand.infl > 2).mean():.2f}")

    u = stats.mannwhitneyu(top.infl, rand.infl, alternative="greater")
    logger.info(f"  Mann-Whitney (top > random): p = {u.pvalue:.3e}")
    logger.info("  Interpretation: a ratio well above the random baseline means")
    logger.info("  the external cohorts inflate whatever they rank highly, and")
    logger.info("  no single-gene external result should be read at face value.")

    logger.info(f"\n  Sign agreement (dev vs ext) among top {args.top_k}: "
                f"{((top.beta_pooled > 0) == (top.beta_ext > 0)).mean():.2f}")
    logger.info(f"  Development p < 0.05 among top {args.top_k}: "
                f"{(top.beta_pooled.abs() / top.se_pooled > 1.96).sum()}"
                f"/{len(top)}")

    logger.info(f"\n  Top {min(20, args.top_k)} genes:")
    logger.info(f"  {'symbol':>12s} {'b_dev':>8s} {'b_ext':>8s} {'ratio':>7s} "
                f"{'p_dev':>9s} {'q_ext':>8s}")
    for _, r in top.head(20).iterrows():
        p_dev = 2 * (1 - stats.norm.cdf(abs(r.beta_pooled / r.se_pooled)))
        logger.info(f"  {str(r.gene_symbol):>12s} {r.beta_pooled:8.3f} "
                    f"{r.beta_ext:8.3f} {r.infl:7.2f} {p_dev:9.3f} "
                    f"{r.q_ext_min:8.3f}")

    for g, name in ((HOPX, "HOPX"), (DDIT4, "DDIT4")):
        if g in set(d.gene_id):
            r = d[d.gene_id == g].iloc[0]
            rr = abs(r.beta_ext) / max(abs(r.beta_pooled), 1e-6)
            pct = stats.percentileofscore(top.infl, rr) if g in set(top.gene_id) else np.nan
            logger.info(f"\n  --- {name} ---")
            logger.info(f"    beta_dev={r.beta_pooled:.3f}, beta_ext={r.beta_ext:.3f}, "
                        f"ratio={rr:.2f}")
            if not np.isnan(pct):
                logger.info(f"    percentile of ratio within top-{args.top_k}: {pct:.0f}")
                logger.info("    (high percentile => outlier even among selected genes;")
                logger.info("     mid percentile => typical of the selection effect)")

    top.to_csv(out / "winners_curse_top.csv", index=False)
    d.to_csv(out / "four_cohort_betas.csv", index=False)

    # ---- Scatter ------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

        ax = axes[0]
        ax.scatter(d.beta_pooled, d.beta_ext, s=4, alpha=0.2, color="grey",
                   label=f"all genes (n={len(d)})")
        ax.scatter(top.beta_pooled, top.beta_ext, s=22, color="crimson",
                   label=f"top {args.top_k} by external q")
        lim = np.nanpercentile(np.abs(np.r_[d.beta_pooled, d.beta_ext]), 99.5)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="y = x")
        ax.axhline(0, lw=0.5, color="k"); ax.axvline(0, lw=0.5, color="k")
        for g, name in ((HOPX, "HOPX"), (DDIT4, "DDIT4")):
            if g in set(d.gene_id):
                r = d[d.gene_id == g].iloc[0]
                ax.annotate(name, (r.beta_pooled, r.beta_ext),
                            fontsize=9, weight="bold",
                            xytext=(5, 5), textcoords="offset points")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("development pooled beta (TCGA + ORIEN)")
        ax.set_ylabel("external pooled beta (GSE41613 + GSE42743)")
        ax.set_title("Marginal effects: development vs external")
        ax.legend(fontsize=8)

        ax = axes[1]
        ax.hist(np.log2(rand.infl.clip(0.01, 100)), bins=40, alpha=0.6,
                color="grey", density=True, label="random genes")
        ax.hist(np.log2(top.infl.clip(0.01, 100)), bins=25, alpha=0.6,
                color="crimson", density=True, label=f"top {args.top_k}")
        ax.axvline(0, color="k", ls="--", lw=1, label="no inflation")
        ax.set_xlabel("log2( |beta_ext| / |beta_dev| )")
        ax.set_ylabel("density")
        ax.set_title("Effect-size inflation")
        ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(out / "transportability.png", dpi=150)
        logger.info(f"\nFigure: {out / 'transportability.png'}")
    except Exception as e:
        logger.warning(f"Plot skipped: {e}")

    logger.info(f"\nResults: {out}")
    logger.info("Step 8.6 complete. This is the last Phase 8 diagnostic.")


if __name__ == "__main__":
    main()
