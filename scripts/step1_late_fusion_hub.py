"""
Chapter 4 (Section 4.3.3 addendum): incremental value of the hub score over
clinical variables in the external cohorts.

MINIMAL-CHANGE REPLICATION of the legacy late-fusion second stage
(step1_late_fusion.py): identical second-stage design -- unpenalized Cox
(lifelines) on standardized scores, likelihood-ratio test combined vs
clinical-only -- with ONE change: the gene score is the 309-gene hub score
(K3 canonical, coefficients fixed from ORIEN; a deterministic transfer, so no
first-stage OOF machinery or seeds are needed on the gene side). The clinical
score is REUSED from the legacy Step 0B out-of-fold ridge (per-cohort encoded
covariates: GSE42743 gender/age/smoking/T/N; GSE41613 gender/stage/age-binary),
so the clinical side of the comparison is byte-identical to the legacy run.

PRE-REGISTERED (record in plan before running): expectation is that the hub
score adds significant value over the clinical model in both GEO cohorts,
mirroring the legacy 25-gene result (LRT p = 0.0004 / 0.013); either outcome
is reported. Sensitivity: repeat over the legacy clinical first-stage seeds
(42 primary; 123, 456).

Scope note for the chapter: this is the GEO-primary analysis by design
(as in legacy); no development-cohort incremental-value claim is made, and
ORIEN clinical data are not used.

Usage:
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/step1_late_fusion_hub.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import chi2

DATA_DIR = Path("data/external")
OOF_DIR = Path("results/late_fusion/oof_scores")   # EDIT to the legacy Step 0B output dir
K3_DIR = Path("results/phaseK3_bootstrap_variance")
OUT_DIR = Path("results/late_fusion_hub")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["GSE41613", "GSE42743"]
CLINICAL_SEEDS = [42, 123, 456]   # 42 = primary, others = sensitivity
PRIMARY_SEED = 42


def load_clinical_oof(dataset, seed):
    """Legacy Step 0B clinical OOF scores. EDIT the filename pattern if the
    legacy run used a different one (check results/late_fusion/)."""
    f = OOF_DIR / f"{dataset}_clinical_oof_scores.csv"
    if not f.exists():
        raise FileNotFoundError(
            f"{f} not found -- point OOF_DIR/filename at the legacy Step 0B "
            f"outputs (the files step1_late_fusion.py's load_oof_scores read).")
    df = pd.read_csv(f, dtype={"sampleID": str})
    col = f"risk_score_seed{seed}"
    if col not in df.columns:
        raise ValueError(f"{f}: column {col} not found; have {list(df.columns)}")
    return df[["sampleID", col, "event", "time"]].rename(
        columns={col: "clinical_score"})


def load_hub_score(dataset):
    f = K3_DIR / f"scores_hub_250_{dataset}.csv"
    sc = pd.read_csv(f, dtype={"patient_id": str})
    return sc.rename(columns={"patient_id": "sampleID", "score": "hub_score"})


def fit_cox(df, covariates):
    fit_df = df[covariates + ["event", "time"]].copy()
    cph = CoxPHFitter()
    cph.fit(fit_df, duration_col="time", event_col="event")
    return cph


def lrt(ll_full, ll_reduced, df_diff=1):
    stat = 2.0 * (ll_full - ll_reduced)
    return stat, chi2.sf(stat, df_diff)


if __name__ == "__main__":
    rows = []
    for ds in DATASETS:
        hub = load_hub_score(ds)
        for seed in CLINICAL_SEEDS:
            clin = load_clinical_oof(ds, seed)
            d = clin.merge(hub, on="sampleID", how="inner")
            assert len(d) == len(clin) == len(hub), \
                f"{ds}: sample mismatch clinical({len(clin)}) vs hub({len(hub)})"

            # Standardize both scores within cohort (comparable betas; the LRT
            # itself is scale-invariant)
            for c in ("clinical_score", "hub_score"):
                d[c] = (d[c] - d[c].mean()) / d[c].std(ddof=1)

            m_clin = fit_cox(d, ["clinical_score"])
            m_hub = fit_cox(d, ["hub_score"])
            m_both = fit_cox(d, ["clinical_score", "hub_score"])

            lr_add_hub, p_add_hub = lrt(m_both.log_likelihood_,
                                        m_clin.log_likelihood_)
            lr_add_clin, p_add_clin = lrt(m_both.log_likelihood_,
                                          m_hub.log_likelihood_)

            beta = m_both.summary
            rows.append(dict(
                dataset=ds, clinical_seed=seed,
                n=len(d), events=int(d["event"].sum()),
                c_clinical=m_clin.concordance_index_,
                c_hub=m_hub.concordance_index_,
                c_combined=m_both.concordance_index_,
                beta_hub=float(beta.loc["hub_score", "coef"]),
                se_hub=float(beta.loc["hub_score", "se(coef)"]),
                beta_clinical=float(beta.loc["clinical_score", "coef"]),
                lrt_hub_over_clinical=lr_add_hub, p_hub_over_clinical=p_add_hub,
                lrt_clinical_over_hub=lr_add_clin, p_clinical_over_hub=p_add_clin,
            ))

            tag = " (PRIMARY)" if seed == PRIMARY_SEED else ""
            print(f"[{ds} | clinical seed {seed}{tag}] n={len(d)}, "
                  f"ev={int(d['event'].sum())}")
            print(f"  C: clinical={m_clin.concordance_index_:.3f}  "
                  f"hub={m_hub.concordance_index_:.3f}  "
                  f"combined={m_both.concordance_index_:.3f}")
            print(f"  Does the HUB score add to clinical?   "
                  f"LRT={lr_add_hub:6.2f}  p={p_add_hub:.4g}")
            print(f"  Does CLINICAL add to the hub score?   "
                  f"LRT={lr_add_clin:6.2f}  p={p_add_clin:.4g}\n")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "late_fusion_hub_summary.csv", index=False)
    print(f"Written: {OUT_DIR}/late_fusion_hub_summary.csv")
