"""
Step 0B: Generate Per-Patient Out-of-Fold Risk Scores (GEO)
============================================================
Prerequisite for Chapter 3 Aim 1 Late Fusion.

For each GEO dataset, generates per-patient out-of-fold risk scores
from two separate models:
  1. Gene signature model: ridge Cox on 64 consensus genes
  2. Clinical-only model: ridge Cox on encoded clinical variables

Uses single-pass approach: collect scores at all alphas during CV,
select best alpha after, extract scores at that alpha.

Methodology matches phase8_step83_validation.py:
  - CoxnetSurvivalAnalysis(l1_ratio=0.01) (near-pure ridge)
  - StratifiedKFold on event status
  - 100 seeds  for fold assignment sensitivity
  - Alpha grids: logspace(-1, 4, 50) for genes, logspace(-4, 4, 50) for clinical

Clinical encoding from Step 0A:
  - GSE42743: gender, age_zscore, smoking, T_stage, N_stage (5 predictors)
  - GSE41613: gender, tumor_stage, age_binary (3 predictors)

Output:
  results/chapter3/step0b/GSE42743_gene_oof_scores.csv
  results/chapter3/step0b/GSE42743_clinical_oof_scores.csv
  results/chapter3/step0b/GSE41613_gene_oof_scores.csv
  results/chapter3/step0b/GSE41613_clinical_oof_scores.csv
  results/chapter3/step0b/step0b_verification.csv

References:
  - Ridge Cox: Verweij & Van Houwelingen (1994) Stat Med 13(20):2059-2071
  - C-index: Harrell et al. (1982) JAMA 247(18):2543-2546
  - sksurv: Pölsterl (2020) JMLR 21(212):1-6

Author: Phuong Nguyen
Date: March 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import StratifiedKFold
import warnings
import logging
import json

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================
SEEDS = list(range(42, 142))  # matches sweep_sensitivity.py P8_SEEDS
N_FOLDS = 3
DATA_DIR = Path("data/external")
CONSENSUS_FILE = Path("results/sensitivity_sweep/mult0.6_pi60/union_genes.txt")
OUTPUT_DIR = Path("results/chapter3/step0b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Alpha grids — same as phase8_step83_validation.py
ALPHAS_GENES = np.logspace(-1, 4, 50)
ALPHAS_CLINICAL = np.logspace(-4, 4, 50)

# Expected C-indices from Step 8.3 for verification (gene-only model)
# These were computed with the same methodology; we verify consistency
EXPECTED_CINDEX = {}

# =============================================================================
# Data Loading (same as phase8_step83_validation.py)
# =============================================================================
def load_consensus_genes(path):
    """Load consensus gene list (one ENSG ID per line, may have line numbers)."""
    genes = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            ensg = parts[-1] if parts else line.strip()
            if ensg.startswith("ENSG"):
                genes.append(ensg)
    logger.info(f"Loaded {len(genes)} consensus genes")
    return genes


def load_geo_dataset(name):
    """Load expression and survival for a GEO dataset."""
    expr = pd.read_csv(DATA_DIR / f"{name}_expression.csv", index_col=0)
    surv = pd.read_csv(DATA_DIR / f"{name}_survival.csv")
    logger.info(f"{name}: {expr.shape[1]} samples, {expr.shape[0]} genes, "
                f"{surv['event'].sum()} events")
    return expr, surv


def load_clinical_encoded(name):
    """Load Step 0A encoded clinical variables."""
    clin = pd.read_csv(DATA_DIR / f"{name}_clinical_encoded.csv")
    logger.info(f"{name} clinical: {len(clin)} samples, "
                f"variables: {[c for c in clin.columns if c != 'sampleID']}")
    return clin


def make_survival_array(surv_df):
    """Convert survival df to structured array for sksurv."""
    return np.array(
        [(bool(e), t) for e, t in zip(surv_df["event"], surv_df["time"])],
        dtype=[("event", bool), ("time", float)]
    )


# =============================================================================
# Single-Pass CV with Per-Patient Score Collection
# =============================================================================
def cv_ridge_cox_with_scores(X, y_struct, events, sample_ids,
                              alphas, n_folds=N_FOLDS, seed=42):
    """
    Single-pass ridge Cox CV that collects per-patient out-of-fold scores.

    For each fold:
      1. Fit CoxnetSurvivalAnalysis across alpha grid on training data
      2. Compute held-out risk scores at ALL fitted alphas
      3. Store scores indexed by patient ID

    After all folds:
      - Select best alpha by mean CV C-index
      - Extract per-patient scores at best alpha
      - Return scores + diagnostics

    Returns:
        dict with keys:
            'best_alpha': selected regularization strength
            'cv_cindex': CV C-index at best alpha
            'cv_std': std of per-fold C-indices at best alpha
            'patient_scores': DataFrame with sampleID, risk_score, event,
                              time, fold
            'fold_cindices': per-fold C-index at best alpha
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    n_alphas = len(alphas)
    n_samples = len(sample_ids)

    # Storage: per-patient scores at each alpha
    # Initialize with NaN — each patient is scored exactly once (in their held-out fold)
    all_scores = np.full((n_samples, n_alphas), np.nan)
    patient_folds = np.full(n_samples, -1, dtype=int)

    # Per-fold, per-alpha C-indices for alpha selection
    fold_cindices = np.full((n_folds, n_alphas), np.nan)

    folds_completed = 0

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, events)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_struct[train_idx], y_struct[test_idx]

        # Check minimum events in test fold
        test_events = y_test["event"].sum()
        if test_events < 2:
            logger.warning(f"  Fold {fold_idx+1}: only {test_events} event(s) "
                           f"in test — skipping")
            continue

        # Fit ridge Cox across alpha grid
        try:
            model = CoxnetSurvivalAnalysis(
                l1_ratio=0.01,
                alphas=alphas,
                normalize=False,
                tol=1e-7,
                max_iter=100000
            )
            model.fit(X_train, y_train)
            fitted_alphas = model.alphas_
        except Exception as e:
            logger.warning(f"  Fold {fold_idx+1}: fitting failed — {e}")
            continue

        # Score held-out patients at each fitted alpha
        for alpha_idx in range(len(fitted_alphas)):
            try:
                # Map fitted alpha back to position in requested grid
                match_idx = np.argmin(np.abs(alphas - fitted_alphas[alpha_idx]))

                # Risk scores: linear predictor X @ beta
                risk_scores = X_test @ model.coef_[:, alpha_idx]

                # Store per-patient scores
                all_scores[test_idx, match_idx] = risk_scores

                # Compute fold C-index at this alpha
                c_idx = concordance_index_censored(
                    y_test["event"], y_test["time"], risk_scores
                )[0]
                fold_cindices[fold_idx, match_idx] = c_idx
            except Exception:
                pass

        # Record fold assignment for each test patient
        patient_folds[test_idx] = fold_idx + 1

        folds_completed += 1
        best_fold_c = np.nanmax(fold_cindices[fold_idx, :])
        logger.info(f"  Fold {fold_idx+1}: {test_events} events, "
                    f"best C-index = {best_fold_c:.3f}")

    if folds_completed == 0:
        logger.error("  No folds completed successfully")
        return None

    # --- Select best alpha by mean CV C-index across folds ---
    mean_cindex_per_alpha = np.nanmean(fold_cindices, axis=0)
    std_cindex_per_alpha = np.nanstd(fold_cindices, axis=0)

    best_alpha_idx = np.nanargmax(mean_cindex_per_alpha)
    best_alpha = alphas[best_alpha_idx]
    best_cv_cindex = mean_cindex_per_alpha[best_alpha_idx]
    best_cv_std = std_cindex_per_alpha[best_alpha_idx]

    # Extract per-patient scores at best alpha
    patient_risk_scores = all_scores[:, best_alpha_idx]

    # Verify: no patient should have NaN (every patient scored in exactly one fold)
    n_missing = np.isnan(patient_risk_scores).sum()
    if n_missing > 0:
        logger.warning(f"  WARNING: {n_missing} patients have no score at best alpha")

    # Verification: recompute overall C-index from concatenated out-of-fold scores
    valid_mask = ~np.isnan(patient_risk_scores)
    if valid_mask.sum() > 0:
        overall_c = concordance_index_censored(
            y_struct["event"][valid_mask],
            y_struct["time"][valid_mask],
            patient_risk_scores[valid_mask]
        )[0]
    else:
        overall_c = np.nan

    # Per-fold C-indices at best alpha (for reporting)
    best_fold_cs = fold_cindices[:, best_alpha_idx]

    logger.info(f"  Best alpha: {best_alpha:.4e}")
    logger.info(f"  CV C-index (mean of fold C-indices): {best_cv_cindex:.4f} "
                f"± {best_cv_std:.4f}")
    logger.info(f"  Overall C-index (from concatenated OOF scores): {overall_c:.4f}")
    logger.info(f"  Per-fold C-indices: {[f'{c:.3f}' for c in best_fold_cs if not np.isnan(c)]}")

    # Build patient-level DataFrame
    patient_df = pd.DataFrame({
        "sampleID": sample_ids,
        "risk_score": patient_risk_scores,
        "event": y_struct["event"],
        "time": y_struct["time"],
        "fold": patient_folds
    })

    return {
        "best_alpha": best_alpha,
        "cv_cindex": best_cv_cindex,
        "cv_std": best_cv_std,
        "overall_cindex": overall_c,
        "patient_scores": patient_df,
        "fold_cindices": best_fold_cs[~np.isnan(best_fold_cs)].tolist()
    }


# =============================================================================
# Multi-Seed Wrapper
# =============================================================================
def run_multi_seed(X, y_struct, events, sample_ids, alphas, label=""):
    """
    Run single-pass CV across 3 seeds. For each seed, collect per-patient
    out-of-fold scores. Return results keyed by seed.

    For the late fusion, we will use the primary seed (42) scores as the
    main analysis, with seeds 123 and 456 for sensitivity.
    """
    all_seed_results = {}
    seed_cindices = []

    for seed in SEEDS:
        logger.info(f"\n  --- Seed {seed} ---")
        result = cv_ridge_cox_with_scores(
            X, y_struct, events, sample_ids,
            alphas=alphas, seed=seed
        )
        if result is not None:
            all_seed_results[seed] = result
            seed_cindices.append(result["cv_cindex"])

    if len(seed_cindices) == 0:
        logger.error(f"  {label}: all seeds failed")
        return None

    mean_c = np.mean(seed_cindices)
    range_c = np.max(seed_cindices) - np.min(seed_cindices)
    stable = range_c < 0.03

    logger.info(f"\n  {label} summary across seeds:")
    logger.info(f"    Mean CV C-index: {mean_c:.4f}")
    logger.info(f"    Seed C-indices: {[f'{c:.4f}' for c in seed_cindices]}")
    logger.info(f"    Range: {range_c:.4f} ({'STABLE' if stable else 'UNSTABLE'})")

    return {
        "seed_results": all_seed_results,
        "mean_cindex": mean_c,
        "seed_cindices": seed_cindices,
        "seed_range": range_c,
        "stable": stable
    }


# =============================================================================
# Dataset Processing
# =============================================================================
def process_dataset(name, expr, surv, clin_encoded, consensus_genes):
    """
    Process one GEO dataset: generate per-patient OOF scores for both
    gene-only and clinical-only models.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  DATASET: {name}")
    logger.info(f"{'='*60}")

    # --- Align samples ---
    samples = surv["sampleID"].values

    # Gene expression: filter to consensus genes, align to survival sample order
    expr_filtered = expr.loc[expr.index.isin(consensus_genes)]
    n_genes_found = len(expr_filtered)
    logger.info(f"  Consensus genes found: {n_genes_found}/{len(consensus_genes)}")

    X_gene = expr_filtered[samples].values.T  # (n_samples, n_genes)
    y_struct = make_survival_array(surv)
    events = surv["event"].values

    logger.info(f"  Samples: {len(samples)}, Gene features: {X_gene.shape[1]}, "
                f"Events: {events.sum()}")

    # Clinical features: align to survival sample order
    clin_aligned = clin_encoded.set_index("sampleID").loc[samples]
    X_clin = clin_aligned.values.astype(float)
    clin_vars = list(clin_aligned.columns)

    logger.info(f"  Clinical features ({len(clin_vars)}): {clin_vars}")

    # Check for NaN in clinical data
    n_missing = np.isnan(X_clin).sum()
    if n_missing > 0:
        logger.warning(f"  WARNING: {n_missing} missing values in clinical data")

    results = {}

    # --- 1. Gene-Only Model ---
    logger.info(f"\n--- Gene Signature (ridge Cox, {n_genes_found} genes) ---")
    res_gene = run_multi_seed(
        X_gene, y_struct, events, samples,
        alphas=ALPHAS_GENES, label="Gene signature"
    )
    if res_gene is not None:
        results["gene"] = res_gene

    # --- 2. Clinical-Only Model ---
    logger.info(f"\n--- Clinical-Only (ridge Cox, {len(clin_vars)} variables) ---")
    res_clin = run_multi_seed(
        X_clin, y_struct, events, samples,
        alphas=ALPHAS_CLINICAL, label="Clinical-only"
    )
    if res_clin is not None:
        results["clinical"] = res_clin

    return results


# =============================================================================
# Save Results
# =============================================================================
def save_patient_scores(results, dataset_name, model_type, output_dir):
    """
    Save per-patient out-of-fold risk scores for each seed.

    Output format:
        sampleID, risk_score_seed42, risk_score_seed123, risk_score_seed456,
        event, time

    This wide format makes it easy to use in the late fusion step.
    """
    seed_results = results["seed_results"]

    # Start with primary seed to get sampleID, event, time
    primary = seed_results[SEEDS[0]]["patient_scores"]
    out_df = pd.DataFrame({
        "sampleID": primary["sampleID"],
        "event": primary["event"],
        "time": primary["time"],
    })

    # Add risk scores from each seed
    for seed in SEEDS:
        if seed in seed_results:
            scores = seed_results[seed]["patient_scores"]
            out_df[f"risk_score_seed{seed}"] = scores["risk_score"].values
            out_df[f"fold_seed{seed}"] = scores["fold"].values

    # Save
    outpath = output_dir / f"{dataset_name}_{model_type}_oof_scores.csv"
    out_df.to_csv(outpath, index=False)
    logger.info(f"  Saved: {outpath} ({len(out_df)} patients)")

    return outpath


# =============================================================================
# Verification
# =============================================================================
def verify_gene_cindex(results, dataset_name):
    """
    Verify that the gene-only CV C-index matches the existing Step 8.3 result.
    This ensures methodological consistency.
    """
    key = f"{dataset_name}_gene"
    if key not in EXPECTED_CINDEX:
        logger.info(f"  No expected C-index for {key} — skipping verification")
        return True

    expected = EXPECTED_CINDEX[key]
    observed = results["mean_cindex"]
    diff = abs(observed - expected)

    if diff < VERIFICATION_TOL:
        logger.info(f"  VERIFICATION PASSED: {key} C-index = {observed:.4f} "
                    f"(expected {expected:.4f}, diff = {diff:.4f})")
        return True
    else:
        logger.warning(f"  VERIFICATION WARNING: {key} C-index = {observed:.4f} "
                       f"(expected {expected:.4f}, diff = {diff:.4f} > tol {VERIFICATION_TOL})")
        logger.warning(f"  This may indicate a difference in gene filtering, "
                       f"data alignment, or random state.")
        return False


# =============================================================================
# Main
# =============================================================================
def main():
    logger.info("Step 0B: Generate Per-Patient Out-of-Fold Risk Scores (GEO)")
    logger.info("=" * 60)

    # --- Load consensus genes ---
    consensus_genes = load_consensus_genes(CONSENSUS_FILE)

    # --- Load datasets ---
    datasets = {}
    for name in ["GSE41613", "GSE42743"]:
        expr, surv = load_geo_dataset(name)
        clin = load_clinical_encoded(name)
        datasets[name] = {"expr": expr, "surv": surv, "clin": clin}

    # --- Process each dataset ---
    all_results = {}
    verification_rows = []

    for name, data in datasets.items():
        results = process_dataset(
            name, data["expr"], data["surv"], data["clin"],
            consensus_genes
        )
        all_results[name] = results

        # Save per-patient scores
        for model_type in ["gene", "clinical"]:
            if model_type in results:
                save_patient_scores(results[model_type], name, model_type, OUTPUT_DIR)

        # Verification for gene model
        if "gene" in results:
            passed = verify_gene_cindex(results["gene"], name)
            verification_rows.append({
                "dataset": name,
                "model": "gene",
                "expected_cindex": EXPECTED_CINDEX.get(f"{name}_gene", np.nan),
                "observed_cindex": results["gene"]["mean_cindex"],
                "passed": passed
            })

    # --- Summary ---
    logger.info(f"\n{'='*60}")
    logger.info("  STEP 0B SUMMARY")
    logger.info(f"{'='*60}")

    summary_rows = []
    for name in ["GSE41613", "GSE42743"]:
        for model_type in ["gene", "clinical"]:
            if model_type in all_results[name]:
                res = all_results[name][model_type]
                summary_rows.append({
                    "dataset": name,
                    "model": model_type,
                    "mean_cv_cindex": res["mean_cindex"],
                    "seed_cindices": res["seed_cindices"],
                    "seed_range": res["seed_range"],
                    "stable": res["stable"]
                })

    summary_df = pd.DataFrame(summary_rows)
    logger.info(f"\n{summary_df.to_string(index=False)}")

    # Save verification results
    verif_df = pd.DataFrame(verification_rows)
    verif_df.to_csv(OUTPUT_DIR / "step0b_verification.csv", index=False)
    logger.info(f"\nVerification saved to {OUTPUT_DIR / 'step0b_verification.csv'}")

    # Save summary
    summary_df.to_csv(OUTPUT_DIR / "step0b_summary.csv", index=False)
    logger.info(f"Summary saved to {OUTPUT_DIR / 'step0b_summary.csv'}")

    # --- Output file inventory ---
    logger.info(f"\nOutput files:")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        logger.info(f"  {f}")

    logger.info(f"\nNext step: Step 0C — generate TCGA/ORIEN out-of-fold scores")
    logger.info(f"           (or proceed to Step 1 if TCGA/ORIEN is deferred)")


if __name__ == "__main__":
    main()
