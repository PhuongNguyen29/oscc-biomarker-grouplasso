"""
Phase 7 addendum: percentile bootstrap CIs for the test C-indices.

Why: the Phase 7 point estimates near 0.5 (ORIEN->TCGA 0.547 and 0.543) must not
be described as "above chance" or "near chance" without an interval. This script
attaches a nonparametric percentile bootstrap CI (Efron & Tibshirani 1993) to
each evaluable direction of one or more Phase 7 Cox-only runs.

Design:
  - The fitted model is held FIXED (loaded from cox_model_{direction}.pkl with
    its best_alpha and gene order). Only TEST patients are resampled with
    replacement, so the interval quantifies validation-set uncertainty of the
    C-index for the already-trained score -- the quantity the prose claims are
    about. It does not include training variability (that is a different
    question, answered by the stability analyses).
  - Provenance check: the recomputed point C-index must match the recorded
    phase7_summary.json value to 1e-6, otherwise the script raises. This
    guarantees the CI is attached to the same predictions that produced the
    recorded number (same data, same gene order, same alpha).
  - Degenerate resamples (no events, or no comparable pairs) are skipped and
    counted; with 153+ events they should be essentially absent.

Output per run directory: bootstrap_ci.json with, per direction:
  point estimate, CI bounds, n_boot effective, fraction of resamples <= 0.5
  (descriptive), and whether the CI includes 0.5.

Usage:
    cd ~/hpchome/oscc-biomarker-grouplasso
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/phase7_bootstrap_ci.py \
        --phase7-dir results/phase7_cpss_mult0.3_pi60_coxonly \
        --phase7-dir results/phase7_cpss_mult0.4_pi60_coxonly
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
from sksurv.metrics import concordance_index_censored

from src.data.data_loader import load_cohort

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('src.data.data_loader').setLevel(logging.WARNING)

DATA_DIR = "data/processed"
N_BOOT_DEFAULT = 1000
SEED_DEFAULT = 42

# Direction label -> which cohort is the TEST set
TEST_COHORT = {"tcga_to_orien": "orien", "orien_to_tcga": "tcga"}


def subset_in_order(expr_df, gene_names):
    """Rows of expr_df (genes x samples) for gene_names, in EXACTLY that order.

    Order matters: the pickled model's coefficients are aligned to the gene
    order stored in the pickle, so the design matrix must reproduce it.
    """
    missing = [g for g in gene_names if g not in expr_df.index]
    if missing:
        raise ValueError(f"{len(missing)} model genes absent from expression: {missing}")
    # float32, NOT float64: Phase 7 cast its design matrices to float32
    # (phase7_bidirectional_percohort.py lines 617/622). Rebuilding in float64
    # perturbs predictions at ~1e-7, which flips ~1 tied/near-tied pair and
    # moves the C-index by ~5e-6 -- enough to trip the provenance gate.
    return expr_df.loc[gene_names].values.T.astype(np.float32)  # samples x genes


def bootstrap_cindex(event, time, pred, n_boot, rng):
    """Percentile bootstrap of the censored C-index over test patients."""
    n = len(pred)
    stats = []
    n_failed = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ev = event[idx]
        if ev.sum() == 0:
            n_failed += 1
            continue
        try:
            ci = concordance_index_censored(ev, time[idx], pred[idx])[0]
        except Exception:
            n_failed += 1
            continue
        stats.append(ci)
    stats = np.array(stats)
    return stats, n_failed


def process_direction(run_dir, direction, cohorts, n_boot, seed, recorded):
    model_file = run_dir / "cox" / f"cox_model_{direction}.pkl"
    if not model_file.exists():
        # Phase 7 may write directly under the run dir depending on version
        alt = run_dir / f"cox_model_{direction}.pkl"
        if alt.exists():
            model_file = alt
        else:
            raise FileNotFoundError(
                f"{model_file} not found (also tried {alt}). "
                f"Check the cox output subdirectory name."
            )

    with open(model_file, "rb") as f:
        bundle = pickle.load(f)
    model, best_alpha, gene_names = bundle["model"], bundle["best_alpha"], bundle["gene_names"]

    test_cohort = TEST_COHORT[direction]
    expr, surv = cohorts[test_cohort]
    X_test = subset_in_order(expr, gene_names)
    # Reproduce Phase 7's float32 round-trip for event/time as well, so tie
    # structure in survival times is byte-identical to the recorded run.
    event = surv["event"].values.astype(np.float32).astype(bool)
    time = surv["time"].values.astype(np.float32).astype(float)

    pred = model.predict(X_test, alpha=best_alpha)

    point = concordance_index_censored(event, time, pred)[0]

    # Provenance check: must reproduce the recorded number exactly.
    rec = recorded["test_ci"]
    if rec is None or abs(point - rec) > 1e-6:
        raise RuntimeError(
            f"{run_dir.name}/{direction}: recomputed C-index {point:.6f} does not "
            f"match recorded {rec}. Data, gene order, or alpha differ from the "
            f"run that produced phase7_summary.json -- do not attach a CI."
        )

    rng = np.random.default_rng(seed)
    stats, n_failed = bootstrap_cindex(event, time, pred, n_boot, rng)
    lo, hi = np.percentile(stats, [2.5, 97.5])

    out = {
        "direction": direction,
        "test_cohort": test_cohort,
        "n_test": int(len(pred)),
        "n_test_events": int(event.sum()),
        "n_genes": len(gene_names),
        "point_ci": float(point),
        "ci95": [float(lo), float(hi)],
        "includes_0.5": bool(lo <= 0.5 <= hi),
        "frac_boot_le_0.5": float(np.mean(stats <= 0.5)),
        "n_boot_effective": int(len(stats)),
        "n_boot_failed": int(n_failed),
        "seed": seed,
        "method": ("percentile bootstrap over test patients, model fixed "
                   "(Efron & Tibshirani 1993)"),
    }
    logger.info(f"  {direction} ({test_cohort} test, n={out['n_test']}, "
                f"{out['n_test_events']} ev): C={point:.4f} "
                f"95% CI [{lo:.4f}, {hi:.4f}] "
                f"{'INCLUDES 0.5' if out['includes_0.5'] else 'excludes 0.5'}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Bootstrap CIs for Phase 7 test C-indices")
    ap.add_argument("--phase7-dir", action="append", required=True,
                    help="Phase 7 results directory; repeat for multiple runs")
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = ap.parse_args()

    logger.info("Loading cohorts...")
    cohorts = {
        "tcga": load_cohort("tcga", DATA_DIR),
        "orien": load_cohort("orien", DATA_DIR),
    }

    for d in args.phase7_dir:
        run_dir = Path(d)
        summary_file = run_dir / "phase7_summary.json"
        with open(summary_file) as f:
            summary = json.load(f)

        logger.info(f"Run: {run_dir}")
        results = {}
        for direction, rec in summary["cox_validation"].items():
            if direction not in TEST_COHORT:
                continue  # skip mean_test_ci / n_evaluable_directions entries
            if not rec.get("evaluable", rec.get("test_ci") is not None):
                logger.info(f"  {direction}: not evaluable in this run -- skipped")
                results[direction] = {"evaluable": False}
                continue
            results[direction] = process_direction(
                run_dir, direction, cohorts, args.n_boot, args.seed, rec)

        out_file = run_dir / "bootstrap_ci.json"
        with open(out_file, "w") as f:
            json.dump({"n_boot": args.n_boot, "seed": args.seed,
                       "directions": results}, f, indent=2)
        logger.info(f"  Written: {out_file}")
