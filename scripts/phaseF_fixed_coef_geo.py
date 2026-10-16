"""
Phase F: fixed-coefficient transfer of the CPSS gene sets to the GEO cohorts.

Pre-registration: corrected_pipeline_plan.md, section F3 (frozen 14 Oct 2026,
before any Phase F number existed). Design sections B4, B5, B6, F1, F2.

What it does, per CPSS cell (SS primary and M&B sensitivity):
  1. Builds five training-cohort x gene-set configurations (plan F3 table).
     Two reuse the Phase 7 Cox fits (provenance-gated); three are new fits made
     with the Phase 7 fitting function unchanged (alpha by inner CV on the
     training cohort only). Nothing is ever fitted on GEO.
  2. Scores GSE41613 and GSE42743 with the fixed models (linear predictor).
  3. C-index with a paired percentile bootstrap over GEO patients (B = 2,000,
     identical resample indices for every configuration within a cohort).
  4. Paired differences C3 - C2 and C3 - C4.
  5. F2: nested Cox likelihood-ratio test of score_TCGAonly over score_ORIEN,
     plus the mirror test, with rho and the detectable HR per SD
     (Hsieh & Lavori 2000).

Inputs (read only):
  results/phase4_cpss_RULEFIX/consensus/{cohort}_stable_mult{m}_pi{p}.csv
  results/phase7_cpss_mult{m}_pi60_coxonly/{cox/cox_model_*.pkl, phase7_summary.json}
  results/phase2_tuning/.../{tcga,orien}_best_params.json
  data/processed (TCGA, ORIEN via src.data.data_loader.load_cohort)
  data/external/{GSE41613,GSE42743}_{expression,survival}.csv

Outputs: results/phaseF_fixed_coef_geo/
  cindex.csv               one row per cell x configuration x GEO cohort
  paired_differences.csv   C3 - C2 and C3 - C4, per cell and cohort
  f2_lrt.csv               nested LRT and mirror test, per cell and cohort
  coefficients.csv         fixed coefficients used for scoring
  scores/                  per-patient GEO scores, per cell and cohort
  models/                  pickles of the new fits
  run_info.json            git commit, package versions, arguments
  phaseF_<timestamp>.log

Notes:
  - Survival time units differ (GSE42743 days, others months). The C-index and
    the Cox LRT are invariant to monotone rescaling of time, so nothing here
    depends on units.
  - Single-gene configurations: the C-index depends only on the sign of the
    coefficient, so they test direction, not effect size.

Usage (from the repository root):
  PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/phaseF_fixed_coef_geo.py
"""

import argparse
import importlib.util
import json
import logging
import pickle
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import chi2, norm
from sksurv.metrics import concordance_index_censored

from src.data.data_loader import load_cohort

logger = logging.getLogger("phaseF")

# =============================================================================
# Configuration (fixed by the pre-registration, plan F3)
# =============================================================================

SEED = 42
N_BOOT = 2000
GEO_COHORTS = ("GSE41613", "GSE42743")

# Age-alone C-index floors, results/age_floor/age_floor.csv (plan, 27 Aug 2026)
AGE_FLOOR = {"GSE41613": 0.518, "GSE42743": 0.571}

# (cell tag, alpha multiplier, pi threshold, Phase 7 run directory)
DEFAULT_CELLS = [
    ("SS_primary_mult0.3_pi60", 0.3, 0.60, "results/phase7_cpss_mult0.3_pi60_coxonly"),
    ("MB_sensitivity_mult0.4_pi60", 0.4, 0.60, "results/phase7_cpss_mult0.4_pi60_coxonly"),
]

# Plan F3 table. source "phase7:<direction>" reuses that Phase 7 pickle.
CONFIGS = [
    # id    train    gene set        coefficient source
    ("C1", "orien", "tcga_stable",  "fit"),
    ("C2", "orien", "orien_stable", "phase7:orien_to_tcga"),
    ("C3", "orien", "union",        "fit"),
    ("C4", "tcga",  "union",        "fit"),
    ("C5", "tcga",  "tcga_stable",  "phase7:tcga_to_orien"),
]
PAIRED_CONTRASTS = [("C3", "C2"), ("C3", "C4")]
OTHER = {"tcga": "orien", "orien": "tcga"}


# =============================================================================
# Helpers
# =============================================================================

def import_phase7(path):
    """Import the Phase 7 script as a module, so its functions are reused unchanged."""
    spec = importlib.util.spec_from_file_location("phase7", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def package_versions():
    out = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scipy", "sklearn", "sksurv", "lifelines"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = "unavailable"
    return out


def cindex(event, time, score):
    """Censored C-index (higher score = higher risk); NaN if not computable."""
    try:
        return concordance_index_censored(event, time, score)[0]
    except Exception:
        return np.nan


def design(expr, genes):
    """Samples x genes float32 matrix in EXACTLY the model's gene order."""
    missing = [g for g in genes if g not in expr.index]
    if missing:
        raise ValueError(f"model genes absent from expression matrix: {missing}")
    return expr.loc[genes].values.T.astype(np.float32)


def coefs_at_best_alpha(bundle):
    model, alpha = bundle["model"], bundle["best_alpha"]
    idx = int(np.argmin(np.abs(model.alphas_ - alpha)))
    return model.coef_[:, idx]


# =============================================================================
# Data
# =============================================================================

def load_dev(processed_dir):
    """TCGA and ORIEN exactly as Phase 7 loads them (float32 round trip)."""
    dev = {}
    for cohort in ("tcga", "orien"):
        expr, surv = load_cohort(cohort, processed_dir)
        dev[cohort] = {
            "expr": expr,
            "event": surv["event"].values.astype(np.float32),
            "time": surv["time"].values.astype(np.float32),
            "ids": surv.index.tolist(),
        }
        logger.info(f"  {cohort.upper():6s} {expr.shape[1]:5d} samples, "
                    f"{int(dev[cohort]['event'].sum())} events")
    return dev


def load_geo(name, external_dir):
    """GEO expression (genes x samples) and survival, aligned by sample ID."""
    expr = pd.read_csv(Path(external_dir) / f"{name}_expression.csv", index_col=0)
    surv = pd.read_csv(Path(external_dir) / f"{name}_survival.csv")
    if "sampleID" not in surv.columns:
        surv = surv.rename(columns={surv.columns[0]: "sampleID"})
    surv = surv.set_index("sampleID")[["time", "event"]]

    keep = [s for s in expr.columns if s in surv.index]
    surv = surv.loc[keep]
    ok = surv["time"].notna() & surv["event"].notna() & (surv["time"] > 0)
    n_dropped = int((~ok).sum()) + (expr.shape[1] - len(keep))
    surv = surv[ok]
    expr = expr[surv.index]
    assert list(expr.columns) == list(surv.index)

    logger.info(f"  {name:8s} {expr.shape[1]:5d} samples, {int(surv['event'].sum())} "
                f"events, {expr.shape[0]} genes ({n_dropped} samples dropped)")
    return {"expr": expr,
            "event": surv["event"].values.astype(bool),
            "time": surv["time"].values.astype(float),
            "ids": surv.index.tolist()}


# =============================================================================
# Models
# =============================================================================

def load_phase7_model(run_dir, direction, dev, genes_expected):
    """Phase 7 pickle, gated: recomputed test C-index must match the summary to 1e-6."""
    run_dir = Path(run_dir)
    path = run_dir / "cox" / f"cox_model_{direction}.pkl"
    if not path.exists():
        path = run_dir / f"cox_model_{direction}.pkl"
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    if sorted(bundle["gene_names"]) != sorted(genes_expected):
        raise RuntimeError(f"{path}: pickle genes {bundle['gene_names']} differ from "
                           f"the consensus gene set {genes_expected}")

    with open(run_dir / "phase7_summary.json") as f:
        recorded = json.load(f)["cox_validation"][direction]["test_ci"]
    test = dev[direction.split("_to_")[1]]
    pred = bundle["model"].predict(design(test["expr"], bundle["gene_names"]),
                                   alpha=bundle["best_alpha"])
    point = cindex(test["event"].astype(bool), test["time"].astype(float), pred)
    if recorded is None or abs(point - recorded) > 1e-6:
        raise RuntimeError(f"{path}: recomputed C {point:.6f} != recorded {recorded}")
    logger.info(f"    {direction}: provenance gate passed (C = {point:.6f})")
    return bundle, str(path)


def fit_new_model(p7, train, genes, l1_ratio, dev, label, model_dir):
    """New fit with the Phase 7 function unchanged; test set = the other dev cohort."""
    tr, te = dev[train], dev[OTHER[train]]
    p7.train_and_evaluate_cox(
        X_train=design(tr["expr"], genes), y_event_train=tr["event"],
        y_time_train=tr["time"], train_ids=tr["ids"],
        X_test=design(te["expr"], genes), y_event_test=te["event"],
        y_time_test=te["time"], test_ids=te["ids"],
        l1_ratio=l1_ratio, gene_names=genes, direction_label=label, out_dir=model_dir)
    path = model_dir / f"cox_model_{label}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f), str(path)


# =============================================================================
# Statistics
# =============================================================================

def bootstrap_indices(n, event, n_boot, rng):
    """Shared resample indices; resamples with no events are dropped for everyone."""
    idx = rng.integers(0, n, size=(n_boot, n))
    valid = event[idx].sum(axis=1) > 0
    return idx[valid], int((~valid).sum())


def f2_lrt(base, added, event, time):
    """Cox ~ base vs Cox ~ base + added (scores standardised within cohort)."""
    df = pd.DataFrame({"base": (base - base.mean()) / base.std(),
                       "added": (added - added.mean()) / added.std(),
                       "time": time, "event": event.astype(int)})
    m_a = CoxPHFitter().fit(df[["base", "time", "event"]], "time", "event")
    m_b = CoxPHFitter().fit(df, "time", "event")
    lr = 2.0 * (m_b.log_likelihood_ - m_a.log_likelihood_)
    ci = m_b.confidence_intervals_.loc["added"]
    return {"lrt_stat": lr, "p_value": chi2.sf(lr, 1),
            "hr_added_per_sd": float(np.exp(m_b.params_["added"])),
            "hr_added_ci95": [float(np.exp(ci.iloc[0])), float(np.exp(ci.iloc[1]))]}


def detectable_hr(events, rho, alpha=0.05, power=0.80):
    """Hsieh & Lavori (2000): minimum detectable HR per SD, adjusted for rho."""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return float(np.exp(z / np.sqrt(events * (1.0 - rho ** 2))))


# =============================================================================
# One cell
# =============================================================================

def run_cell(tag, mult, pi, phase7_dir, args, p7, cox_params, dev, geo, out_dir, rows):
    logger.info(f"\n=== Cell {tag} (mult {mult}, pi {pi}) ===")
    sets = p7.load_phase4_stable_genes(args.phase4_dir, mult, pi)
    gene_sets = {"tcga_stable": list(sets["tcga"]),
                 "orien_stable": list(sets["orien"]),
                 "union": list(sets["union"])}
    for k, v in gene_sets.items():
        logger.info(f"  {k:12s} {len(v)} genes: {v}")
    if sets["overlap"]:
        logger.warning(f"  TCGA/ORIEN overlap {sets['overlap']}: C5 is not TCGA-only")

    model_dir = out_dir / "models" / tag
    model_dir.mkdir(parents=True, exist_ok=True)

    # --- fixed models -------------------------------------------------------
    models = {}
    for cid, train, gset, source in CONFIGS:
        genes = gene_sets[gset]
        if not genes:
            logger.info(f"  {cid} ({train} x {gset}): NOT EVALUABLE, empty gene set")
            models[cid] = None
            continue
        if source.startswith("phase7:"):
            bundle, path = load_phase7_model(phase7_dir, source.split(":")[1], dev, genes)
        else:
            bundle, path = fit_new_model(p7, train, genes, cox_params[train]["l1_ratio"],
                                         dev, f"{cid}_{train}_{gset}", model_dir)
        coefs = coefs_at_best_alpha(bundle)
        models[cid] = {"bundle": bundle, "coefs": coefs, "path": path}
        for g, b in zip(bundle["gene_names"], coefs):
            rows["coefficients"].append({"cell": tag, "config": cid, "train": train,
                                         "gene_set": gset, "gene": g, "coef": float(b),
                                         "best_alpha": float(bundle["best_alpha"]),
                                         "source": path})
        if np.all(coefs == 0):
            logger.warning(f"  {cid}: all coefficients zero -> constant score, NO MODEL")

    # --- score, bootstrap, paired differences, F2, per GEO cohort -----------
    for name in GEO_COHORTS:
        g = geo[name]
        rng = np.random.default_rng(args.seed)
        idx, n_failed = bootstrap_indices(len(g["ids"]), g["event"], args.n_boot, rng)

        scores, boots = {}, {}
        for cid, train, gset, _ in CONFIGS:
            m = models[cid]
            base = {"cell": tag, "config": cid, "train": train, "gene_set": gset,
                    "geo_cohort": name, "n": len(g["ids"]), "events": int(g["event"].sum()),
                    "age_floor": AGE_FLOOR[name]}
            if m is None:
                rows["cindex"].append({**base, "status": "not evaluable (empty gene set)"})
                continue
            if np.all(m["coefs"] == 0):
                rows["cindex"].append({**base, "status": "no model (all coefficients zero)"})
                continue

            b = m["bundle"]
            s = b["model"].predict(design(g["expr"], b["gene_names"]), alpha=b["best_alpha"])
            scores[cid] = s
            point = cindex(g["event"], g["time"], s)
            boots[cid] = np.array([cindex(g["event"][i], g["time"][i], s[i]) for i in idx])
            lo, hi = np.nanpercentile(boots[cid], [2.5, 97.5])
            rows["cindex"].append({
                **base, "status": "ok", "n_genes": len(b["gene_names"]),
                "n_nonzero": int(np.sum(m["coefs"] != 0)), "c_index": point,
                "ci_lo": lo, "ci_hi": hi, "ci_excludes_0.5": bool(lo > 0.5 or hi < 0.5),
                "ci_lo_above_floor": bool(lo > AGE_FLOOR[name]),
                "n_boot_effective": int(np.sum(~np.isnan(boots[cid]))),
                "n_boot_no_events": n_failed})
            logger.info(f"  {name} {cid} ({train} x {gset}): C = {point:.4f} "
                        f"[{lo:.4f}, {hi:.4f}]")

            pd.DataFrame({"sampleID": g["ids"], "score": s}).to_csv(
                out_dir / "scores" / f"{tag}_{name}_{cid}.csv", index=False)

        for a, b_ in PAIRED_CONTRASTS:
            rec = {"cell": tag, "geo_cohort": name, "contrast": f"{a} - {b_}"}
            if a in boots and b_ in boots:
                d = boots[a] - boots[b_]
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                rec.update({"status": "ok",
                            "difference": cindex(g["event"], g["time"], scores[a])
                                          - cindex(g["event"], g["time"], scores[b_]),
                            "ci_lo": lo, "ci_hi": hi,
                            "ci_excludes_0": bool(lo > 0 or hi < 0)})
            else:
                rec["status"] = "not evaluable"
            rows["paired"].append(rec)

        # F2: score_ORIEN = C2, score_TCGAonly = C5 (valid only with zero overlap)
        rec = {"cell": tag, "geo_cohort": name, "events": int(g["event"].sum())}
        if "C2" in scores and "C5" in scores and not sets["overlap"]:
            rho = float(np.corrcoef(scores["C2"], scores["C5"])[0, 1])
            rec.update({"status": "ok", "rho": rho,
                        "detectable_hr_per_sd": detectable_hr(rec["events"], rho)})
            for label, base_s, add_s in (("TCGAonly_over_ORIEN", scores["C2"], scores["C5"]),
                                         ("ORIEN_over_TCGAonly", scores["C5"], scores["C2"])):
                r = f2_lrt(base_s, add_s, g["event"], g["time"])
                rows["f2"].append({**rec, "test": label, **r})
                logger.info(f"  {name} F2 {label}: LRT {r['lrt_stat']:.3f}, "
                            f"p = {r['p_value']:.4f}, rho = {rho:.3f}")
        else:
            rows["f2"].append({**rec, "test": "TCGAonly_over_ORIEN",
                               "status": "not estimable (score_ORIEN or score_TCGAonly "
                                         "missing, or gene sets overlap)"})
            logger.info(f"  {name} F2: not estimable at this cell")


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Phase F: fixed-coefficient GEO transfer (plan F3)")
    ap.add_argument("--phase4-dir", default="results/phase4_cpss_RULEFIX")
    ap.add_argument("--phase7-script", default="scripts/phase7_bidirectional_percohort.py")
    ap.add_argument("--phase2-cox-tcga",
                    default="results/phase2_tuning/cox_tuning_20260804_231755/tcga_best_params.json")
    ap.add_argument("--phase2-cox-orien",
                    default="results/phase2_tuning/cox_tuning_20260804_225946/orien_best_params.json")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--external-dir", default="data/external")
    ap.add_argument("--out-dir", default="results/phaseF_fixed_coef_geo")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "scores").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Import Phase 7 first: it configures logging at import, so the root logger is
    # reset afterwards. Its fitting messages then go to the same console and log file.
    p7 = import_phase7(args.phase7_script)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(out_dir / f"phaseF_{stamp}.log")):
        h.setFormatter(fmt)
        root.addHandler(h)
    root.setLevel(logging.INFO)
    logging.getLogger("src.data.data_loader").setLevel(logging.WARNING)

    logger.info("Phase F: fixed-coefficient GEO transfer (pre-registered, plan F3)")
    logger.info(f"git commit {git_commit()} | seed {args.seed} | B = {args.n_boot}")

    cox_params = p7.load_phase2_cox_params(args.phase2_cox_tcga, args.phase2_cox_orien)

    logger.info("\n--- Loading data ---")
    dev = load_dev(args.processed_dir)
    geo = {name: load_geo(name, args.external_dir) for name in GEO_COHORTS}

    rows = {"cindex": [], "paired": [], "f2": [], "coefficients": []}
    for tag, mult, pi, phase7_dir in DEFAULT_CELLS:
        run_cell(tag, mult, pi, phase7_dir, args, p7, cox_params, dev, geo, out_dir, rows)

    pd.DataFrame(rows["cindex"]).to_csv(out_dir / "cindex.csv", index=False)
    pd.DataFrame(rows["paired"]).to_csv(out_dir / "paired_differences.csv", index=False)
    pd.DataFrame(rows["f2"]).to_csv(out_dir / "f2_lrt.csv", index=False)
    pd.DataFrame(rows["coefficients"]).to_csv(out_dir / "coefficients.csv", index=False)
    with open(out_dir / "run_info.json", "w") as f:
        json.dump({"timestamp": stamp, "git_commit": git_commit(),
                   "versions": package_versions(), "args": vars(args),
                   "cells": DEFAULT_CELLS, "configs": CONFIGS,
                   "age_floor": AGE_FLOOR}, f, indent=2)
    logger.info(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
