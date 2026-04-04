"""
Phase 8, Step 8.3: Multivariate Signature Validation (External)
================================================================
Validates whether the 64 consensus genes carry prognostic signal in
independent GEO populations using ridge Cox with 5-fold stratified CV.

References:
  - Ridge Cox: Verweij & Van Houwelingen (1994) Stat Med 13(20):2059-2071
  - C-index: Harrell et al. (1982) JAMA 247(18):2543-2546
  - sksurv CoxnetSurvivalAnalysis: Pölsterl (2020) JMLR 21(212):1-6

Per v4 pipeline document:
  - Pure L2 penalty (l1_ratio=0)
  - 5-fold stratified CV within each GEO dataset
  - Report CV-corrected C-index only (not in-sample)
  - Compare to clinical-only ridge Cox baseline
  - Sensitivity: rank-based expression (percentile ranks)
  - Do NOT transfer model weights from development cohorts
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================
SEED =  list(range(42, 142))
N_FOLDS = 3
DATA_DIR = Path("data/external")
CONSENSUS_FILE = Path("results/phase4_stability_iqr/consensus/rra_signfiltered_rank350.txt")
OUTPUT_DIR = Path("results/phase8_external_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ridge Cox alpha grid (L2 penalty strength)
# sksurv uses 'alphas' parameter; larger = stronger regularization
# Gene signature (64 features, n~70-97): needs strong regularization
# Clinical-only (3-5 features): can use weaker regularization
ALPHAS_GENES = np.logspace(-1, 4, 50)    # 0.1 to 10000
ALPHAS_CLINICAL = np.logspace(-4, 4, 50)  # 0.0001 to 10000


# =============================================================================
# Data Loading
# =============================================================================
def load_consensus_genes(path):
    """Load consensus gene list (one ENSG ID per line, may have line numbers)."""
    genes = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            # Handle format with line numbers: "1  ENSG00000004468"
            ensg = parts[-1] if parts else line.strip()
            if ensg.startswith("ENSG"):
                genes.append(ensg)
    logger.info(f"Loaded {len(genes)} consensus genes")
    return genes


def load_geo_dataset(name):
    """Load expression, survival, clinical for a GEO dataset."""
    expr = pd.read_csv(DATA_DIR / f"{name}_expression.csv", index_col=0)
    surv = pd.read_csv(DATA_DIR / f"{name}_survival.csv")
    clin = pd.read_csv(DATA_DIR / f"{name}_clinical.csv")

    # Ensure clinical has a sampleID column
    # GSE42743 has explicit "sampleID" column; GSE41613 has sample IDs as unnamed first column
    if "sampleID" not in clin.columns:
        if clin.columns[0].startswith("Unnamed") or clin.columns[0] == "":
            clin = pd.read_csv(DATA_DIR / f"{name}_clinical.csv", index_col=0)
            clin["sampleID"] = clin.index
        else:
            # First column might be the sample IDs with no header name
            clin["sampleID"] = clin.iloc[:, 0]

    logger.info(f"{name}: {expr.shape[1]} samples, {expr.shape[0]} genes, "
                f"{surv['event'].sum()} events")
    return expr, surv, clin


def make_survival_array(surv_df):
    """Convert survival df to structured array for sksurv."""
    return np.array(
        [(bool(e), t) for e, t in zip(surv_df["event"], surv_df["time"])],
        dtype=[("event", bool), ("time", float)]
    )


# =============================================================================
# Clinical Feature Encoding
# =============================================================================
def encode_clinical_41613(clin, surv):
    """Encode GSE41613 clinical variables for ridge Cox.
    
    Variables: gender (binary), tumor_stage (binary), age_bin (ordinal).
    age_bin has 4 categories with no exact age — use ordinal encoding.
    """
    samples = surv["sampleID"].values
    clin = clin.set_index("sampleID").loc[samples].reset_index()


    features = pd.DataFrame(index=samples)
    features["gender"] = clin["gender"].values

    # tumor_stage: I/II=0, III/IV=1
    features["tumor_stage"] = (clin["tumor_stage"] == "III/IV").astype(int).values

    # age_bin: ordinal encoding (19-39=0, 40-49=1, 50-59=2, 60-88=3)
    age_order = {"19-39": 0, "40-49": 1, "50-59": 2, "60-88": 3}
    features["age_ordinal"] = clin["age_bin"].map(age_order).values

    return features


def encode_clinical_42743(clin, surv):
    """Encode GSE42743 clinical variables for ridge Cox.
    
    Variables: gender (binary), age (continuous), smoking (binary),
    T_stage (binary), N_stage (binary).
    """
    samples = surv["sampleID"].values
    clin = clin.set_index("sampleID").loc[samples].reset_index()

    features = pd.DataFrame(index=samples)
    features["gender"] = clin["gender"].values
    features["age"] = clin["age"].values
    features["smoking"] = (clin["smoking"] == "Ever").astype(int).values
    features["T_stage"] = (clin["T_stage"] == "T3-T4").astype(int).values
    features["N_stage"] = (clin["N_stage"] == "N+").astype(int).values

    return features


# =============================================================================
# Cross-Validated Ridge Cox
# =============================================================================
def cv_ridge_cox(X, y_struct, events, n_folds=N_FOLDS, alphas=None, seed=SEED):
    """Fit ridge Cox with stratified CV, return CV-corrected C-index.
    
    For each fold:
      1. Fit CoxnetSurvivalAnalysis(l1_ratio=0.01) on training data across alpha grid
      2. For each alpha, predict on held-out fold
      3. Compute C-index on held-out fold
    
    Returns per-alpha mean CV C-index and best alpha.
    
    Reference: Verweij & Van Houwelingen (1994) cross-validated
    partial likelihood for Cox regression.
    """
    if alphas is None:
        alphas = ALPHAS_GENES
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Store per-fold, per-alpha C-indices
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, events)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_struct[train_idx], y_struct[test_idx]

        # Check minimum events in test fold
        test_events = y_test["event"].sum()
        if test_events < 2:
            logger.warning(f"  Fold {fold_idx+1}: only {test_events} event(s) in test — skipping")
            continue

        # Fit ridge Cox across alpha grid
        try:
            model = CoxnetSurvivalAnalysis(
                l1_ratio=0.01,  # Near-pure ridge; sksurv 0.23.1 requires l1_ratio > 0
                alphas=alphas,
                normalize=False,  # Data already z-scored
                tol=1e-7,
                max_iter=100000
            )
            model.fit(X_train, y_train)
            fitted_alphas = model.alphas_
        except Exception as e:
            logger.warning(f"  Fold {fold_idx+1}: fitting failed — {e}")
            continue

        # Evaluate each alpha on held-out fold
        # sksurv may fit fewer alphas than requested if numerical issues occur
        fold_cindices = [np.nan] * len(alphas)
        for alpha_idx in range(len(fitted_alphas)):
            try:
                # Find matching position in original alpha grid
                # fitted_alphas may be a subset of the requested alphas
                match_idx = np.argmin(np.abs(alphas - fitted_alphas[alpha_idx]))
                risk_scores = X_test @ model.coef_[:, alpha_idx]
                c_idx = concordance_index_censored(
                    y_test["event"], y_test["time"], risk_scores
                )[0]
                fold_cindices[match_idx] = c_idx
            except Exception:
                pass

        fold_results.append(fold_cindices)
      

    if len(fold_results) == 0:
        return None, None, None

    # Mean C-index across folds for each alpha
    # Note: all folds used same alpha grid, so columns align
    fold_results = np.array(fold_results)
    mean_cindex = np.nanmean(fold_results, axis=0)
    std_cindex = np.nanstd(fold_results, axis=0)

    best_alpha_idx = np.nanargmax(mean_cindex)
    best_alpha = alphas[best_alpha_idx]
    best_cindex = mean_cindex[best_alpha_idx]
    best_std = std_cindex[best_alpha_idx]

    return best_cindex, best_std, best_alpha


# =============================================================================
# Rank-Based Transformation (Sensitivity)
# =============================================================================
def rank_transform(X):
    """Convert expression to within-sample percentile ranks.
    
    Each gene is replaced by its rank (as fraction) across samples.
    Ties are averaged. This removes distributional assumptions.
    """
    from scipy.stats import rankdata
    X_rank = np.zeros_like(X)
    for j in range(X.shape[1]):
        X_rank[:, j] = rankdata(X[:, j]) / X.shape[0]
    return X_rank


SEEDS = list(range(42, 142)) 


# =============================================================================
# Main Analysis
# =============================================================================
def run_multi_seed_cv(X, y_struct, events, alphas=None, label=""):
    """Run CV across multiple seeds, report median and percentile interval.
    
    Uses 100 repeated stratified 3-fold CV for stable estimation.
    """
    seed_results = []
    for seed in SEEDS:
        c, s, a = cv_ridge_cox(X, y_struct, events, alphas=alphas, seed=seed)
        if c is not None:
            seed_results.append({"seed": seed, "cv_cindex": c, "std": s, "alpha": a})

    if len(seed_results) == 0:
        return None

    cindices = [r["cv_cindex"] for r in seed_results]
    median_c = np.median(cindices)
    ci_lower = np.percentile(cindices, 2.5)
    ci_upper = np.percentile(cindices, 97.5)
    mean_c = np.mean(cindices)
    sd_c = np.std(cindices)

    logger.info(f"  {label} across {len(cindices)} seeds:")
    logger.info(f"    Median = {median_c:.3f} (95% CI: {ci_lower:.3f} – {ci_upper:.3f})")
    logger.info(f"    Mean = {mean_c:.3f} ± {sd_c:.3f}")

    return {
        "median_cindex": median_c,
        "mean_cindex": mean_c,
        "sd": sd_c,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "alpha": seed_results[0]["alpha"],
        "seed_cindices": cindices,
        "n_seeds": len(cindices),
    }

def analyze_dataset(name, expr, surv, clin, consensus_genes, encode_fn):
    """Run full Step 8.3 analysis on one GEO dataset."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  DATASET: {name}")
    logger.info(f"{'='*60}")

    # Align samples across expression, survival, clinical
    samples = surv["sampleID"].values
    expr_filtered = expr.loc[expr.index.isin(consensus_genes)]
    expr_aligned = expr_filtered[samples].values.T
    #expr_aligned = expr[samples].values.T  # samples × genes
    y_struct = make_survival_array(surv)
    events = surv["event"].values

    logger.info(f"  Samples: {len(samples)}, Genes: {expr_aligned.shape[1]}, "
                f"Events: {events.sum()}")

    results = {}
    

    # --- 1. Gene Signature Ridge Cox ---
    logger.info(f"\n--- Gene Signature ({len(consensus_genes)} genes, ridge Cox) ---")
    res_genes = run_multi_seed_cv(
        expr_aligned, y_struct, events, label="Gene signature"
    )
    if res_genes is not None:
        results["gene_signature"] = res_genes

    # --- 2. Clinical-Only Ridge Cox Baseline ---
    logger.info(f"\n--- Clinical-Only Baseline ---")
    clin_features = encode_fn(clin, surv)
    X_clin = clin_features.values.astype(float)

    # Check for missing values
    missing_mask = np.isnan(X_clin).any(axis=1)
    if missing_mask.any():
        logger.info(f"  Dropping {missing_mask.sum()} samples with missing clinical data")
        X_clin_clean = X_clin[~missing_mask]
        y_clin = y_struct[~missing_mask]
        events_clin = events[~missing_mask]
    else:
        X_clin_clean = X_clin
        y_clin = y_struct
        events_clin = events

    res_clin = run_multi_seed_cv(
        X_clin_clean, y_clin, events_clin, alphas=ALPHAS_CLINICAL,
        label="Clinical-only"
    )
    if res_clin is not None:
        res_clin["variables"] = list(clin_features.columns)
        res_clin["n_samples"] = len(X_clin_clean)
        results["clinical_only"] = res_clin

    # --- 3. Sensitivity: Rank-Based Expression ---
    logger.info(f"\n--- Sensitivity: Rank-Based Expression ---")
    X_rank = rank_transform(expr_aligned)
    res_rank = run_multi_seed_cv(
        X_rank, y_struct, events, label="Rank-based"
    )
    if res_rank is not None:
        results["rank_sensitivity"] = res_rank

    return results


def main():
    logger.info("Phase 8, Step 8.3: Multivariate Signature Validation")
    logger.info("=" * 60)

    # Load consensus genes
    consensus_genes = load_consensus_genes(CONSENSUS_FILE)

    # Load datasets
    expr41613, surv41613, clin41613 = load_geo_dataset("GSE41613")
    expr42743, surv42743, clin42743 = load_geo_dataset("GSE42743")

    # Verify gene coverage
    for name, expr in [("GSE41613", expr41613), ("GSE42743", expr42743)]:
        found = sum(1 for g in consensus_genes if g in expr.index)
        logger.info(f"  {name}: {found}/{len(consensus_genes)} consensus genes found")

    # Run analyses
    results_41613 = analyze_dataset(
        "GSE41613", expr41613, surv41613, clin41613,
        consensus_genes, encode_clinical_41613
    )
    results_42743 = analyze_dataset(
        "GSE42743", expr42743, surv42743, clin42743,
        consensus_genes, encode_clinical_42743
    )

    # --- Summary ---
    logger.info(f"\n{'='*60}")
    logger.info("  SUMMARY")
    logger.info(f"{'='*60}")

    summary_rows = []
    for name, results in [("GSE41613", results_41613), ("GSE42743", results_42743)]:
        for model, res in results.items():
            summary_rows.append({
                "dataset": name,
                "model": model,
                "median_cindex": res["median_cindex"],
                "mean_cindex": res["mean_cindex"],
                "sd": res["sd"],
                "ci_lower": res["ci_lower"],
                "ci_upper": res["ci_upper"],
                "best_alpha": res["alpha"],
                "n_seeds": res.get("n_seeds", 0),
            })

    summary_df = pd.DataFrame(summary_rows)
    logger.info(f"\n{summary_df.to_string(index=False)}")

    # Save results
    summary_df.to_csv(OUTPUT_DIR / "step8_3_summary.csv", index=False)
    logger.info(f"\nResults saved to {OUTPUT_DIR / 'step8_3_summary.csv'}")

    # Save detailed results
    import json
    detailed = {"GSE41613": results_41613, "GSE42743": results_42743}
    # Convert numpy types for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(OUTPUT_DIR / "step8_3_detailed.json", "w") as f:
        json.dump(make_serializable(detailed), f, indent=2)

    logger.info("Step 8.3 complete.")


if __name__ == "__main__":
    main()
