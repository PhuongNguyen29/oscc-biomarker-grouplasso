"""
Phase 2 — Elastic Net Cox Hyperparameter Tuning

Tunes the l1_ratio parameter for CoxnetSurvivalAnalysis (scikit-survival)
using Optuna with 5-fold stratified CV. Lambda (alpha in glmnet terminology)
is handled internally by the regularization path.

For each l1_ratio, CoxnetSurvivalAnalysis computes a full regularization path
of 100 lambda values. We evaluate each lambda by 5-fold CV C-index and select
the best (l1_ratio, lambda) combination.

References:
    - Simon et al. (2011) JSS — Regularization paths for Cox regression
    - Friedman et al. (2010) JSS — Coordinate descent for GLMs
    - Zou & Hastie (2005) JRSS-B — Elastic net
    - Bergstra et al. (2011) NeurIPS — TPE algorithm (Optuna)

Usage:
    cd ~/hpchome/oscc-biomarker-elasticdeepsurv
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/tune_cox.py

Output:
    results/phase2_tuning/cox_tuning_<timestamp>/
        ├── cox_tuning.log
        ├── tcga_best_params.json
        ├── orien_best_params.json
        └── all_trials.json
"""

import sys
import os
import logging
import json
import numpy as np
import pandas as pd
import optuna
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List

from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

from src.data.data_loader import load_cohort

# =============================================================================
# Setup
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress Optuna's internal logging (too verbose)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# Configuration
# =============================================================================

N_TRIALS = 50
N_FOLDS = 5
RANDOM_SEED = 42
DATA_DIR = "data/processed"

# l1_ratio search space (discrete grid)
# 0.1 = mostly Ridge, 1.0 = pure Lasso
L1_RATIO_CHOICES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

# Number of lambda values in the regularization path
N_ALPHAS = 100


# =============================================================================
# Stratified CV folds (event status + survival time quartiles)
# =============================================================================

def create_stratified_folds(
    y_event: np.ndarray,
    y_time: np.ndarray,
    n_folds: int = 5,
    seed: int = 42
) -> StratifiedKFold:
    """
    Create stratified folds using event status × survival time quartile bins.

    This ensures each fold has similar event rates AND survival time distributions.

    Reference:
        Royston & Altman (2013) BMC Medical Research Methodology —
        stratification for survival data CV.

    Args:
        y_event: Binary event indicator (1=event, 0=censored)
        y_time: Survival times
        n_folds: Number of CV folds
        seed: Random seed

    Returns:
        strata: Combined stratification labels for use with StratifiedKFold
    """
    # Bin survival times into quartiles
    time_quartiles = pd.qcut(y_time, q=4, labels=False, duplicates='drop')

    # Combine: event_status * 4 + time_quartile → 8 possible strata
    strata = y_event.astype(int) * 4 + time_quartiles

    return strata


# =============================================================================
# Cox evaluation for one (l1_ratio, alpha/lambda) combination
# =============================================================================

def evaluate_cox_cv(
    X: np.ndarray,
    y_event: np.ndarray,
    y_time: np.ndarray,
    l1_ratio: float,
    n_folds: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Evaluate elastic net Cox with a given l1_ratio using 5-fold stratified CV.

    CoxnetSurvivalAnalysis fits the full regularization path (100 lambda values).
    For each fold, we find the lambda that gives the best C-index on the
    validation fold. The returned metric is the mean CV C-index at the best
    lambda averaged across folds.

    Args:
        X: Feature matrix (n_samples, n_features)
        y_event: Event indicators
        y_time: Survival times
        l1_ratio: Elastic net mixing parameter
        n_folds: Number of CV folds
        seed: Random seed

    Returns:
        dict with mean_cindex, std_cindex, per_fold results, best_alpha_per_fold
    """
    # Create structured array for sksurv
    y_structured = np.array(
        [(bool(e), t) for e, t in zip(y_event, y_time)],
        dtype=[('event', bool), ('time', float)]
    )

    # Create stratification labels
    strata = create_stratified_folds(y_event, y_time, n_folds, seed)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_results = []

    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, strata)):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train = y_structured[train_idx]
        y_valid = y_structured[valid_idx]

        # Fit Cox model with full regularization path
        try:
            model = CoxnetSurvivalAnalysis(
                l1_ratio=l1_ratio,
                n_alphas=N_ALPHAS,
                normalize=False,       # Data already z-scored
                tol=1e-7,
                max_iter=100000,
            )
            model.fit(X_train, y_train)
        except Exception as e:
            logger.warning(f"  Fold {fold_idx+1}: fitting failed — {e}")
            fold_results.append({
                'fold': fold_idx + 1,
                'best_cindex': 0.5,
                'best_alpha': None,
                'n_nonzero': 0,
                'error': str(e),
            })
            continue

        # Evaluate at each alpha (lambda) in the path
        best_cindex = 0.0
        best_alpha = None
        best_n_nonzero = 0

        for alpha_val in model.alphas_:
            try:
                pred = model.predict(X_valid, alpha=alpha_val)

                # Extract event and time from structured array
                valid_event = y_valid['event']
                valid_time = y_valid['time']

                ci, _, _, _, _ = concordance_index_censored(
                    valid_event, valid_time, pred
                )

                if ci > best_cindex:
                    best_cindex = ci
                    best_alpha = alpha_val
                    # Count nonzero coefficients at this alpha
                    coef_idx = np.where(model.alphas_ == alpha_val)[0][0]
                    best_n_nonzero = int(np.sum(model.coef_[:, coef_idx] != 0))

            except Exception:
                # Some alpha values may produce degenerate predictions
                continue

        fold_results.append({
            'fold': fold_idx + 1,
            'best_cindex': best_cindex,
            'best_alpha': float(best_alpha) if best_alpha is not None else None,
            'n_nonzero': best_n_nonzero,
        })

    # Aggregate
    cindices = [f['best_cindex'] for f in fold_results]
    mean_cindex = np.mean(cindices)
    std_cindex = np.std(cindices)
    mean_nonzero = np.mean([f['n_nonzero'] for f in fold_results])

    return {
        'l1_ratio': l1_ratio,
        'mean_cindex': mean_cindex,
        'std_cindex': std_cindex,
        'mean_nonzero_genes': mean_nonzero,
        'fold_results': fold_results,
    }


# =============================================================================
# Optuna objective
# =============================================================================

def create_objective(X, y_event, y_time):
    """
    Create an Optuna objective function for Cox tuning.

    With only 6 discrete l1_ratio values, Optuna's TPE will efficiently
    explore all of them within 50 trials, spending more trials on
    promising values.
    """
    def objective(trial):
        l1_ratio = trial.suggest_categorical('l1_ratio', L1_RATIO_CHOICES)

        result = evaluate_cox_cv(
            X, y_event, y_time,
            l1_ratio=l1_ratio,
            n_folds=N_FOLDS,
            seed=RANDOM_SEED,
        )

        # Store additional info in trial
        trial.set_user_attr('std_cindex', result['std_cindex'])
        trial.set_user_attr('mean_nonzero_genes', result['mean_nonzero_genes'])
        trial.set_user_attr('fold_results', result['fold_results'])

        logger.info(
            f"  Trial {trial.number+1:>3}: l1_ratio={l1_ratio:.1f} → "
            f"C-index={result['mean_cindex']:.4f} ± {result['std_cindex']:.4f}, "
            f"nonzero genes={result['mean_nonzero_genes']:.0f}"
        )

        return result['mean_cindex']

    return objective


# =============================================================================
# Run tuning for one cohort
# =============================================================================

def tune_cox_for_cohort(
    cohort: str,
    results_dir: Path,
) -> Dict:
    """
    Run full Optuna tuning for elastic net Cox on one cohort.
    """
    logger.info("=" * 70)
    logger.info(f"ELASTIC NET COX TUNING: {cohort.upper()}")
    logger.info("=" * 70)

    # Load data
    expression, survival = load_cohort(cohort, DATA_DIR)

    # Prepare matrices
    # Expression is genes × samples, need samples × genes
    X = expression.values.T.astype(np.float64)
    y_event = survival['event'].values.astype(float)
    y_time = survival['time'].values.astype(float)

    logger.info(f"X shape: {X.shape} (samples × genes)")
    logger.info(f"Events: {int(y_event.sum())}/{len(y_event)} ({y_event.mean():.1%})")

    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        study_name=f"cox_{cohort}",
    )

    # Run optimization
    objective = create_objective(X, y_event, y_time)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    # Best result
    best = study.best_trial
    logger.info("")
    logger.info("-" * 70)
    logger.info(f"BEST RESULT: {cohort.upper()}")
    logger.info(f"  l1_ratio: {best.params['l1_ratio']}")
    logger.info(f"  Mean CV C-index: {best.value:.4f} ± {best.user_attrs['std_cindex']:.4f}")
    logger.info(f"  Mean nonzero genes: {best.user_attrs['mean_nonzero_genes']:.0f}")
    logger.info("-" * 70)

    # Now refit on full data with best l1_ratio to get best alpha
    logger.info("Refitting on full data with best l1_ratio to determine best alpha...")

    y_structured = np.array(
        [(bool(e), t) for e, t in zip(y_event, y_time)],
        dtype=[('event', bool), ('time', float)]
    )

    final_model = CoxnetSurvivalAnalysis(
        l1_ratio=best.params['l1_ratio'],
        n_alphas=N_ALPHAS,
        normalize=False,
        tol=1e-7,
        max_iter=100000,
    )
    final_model.fit(X, y_structured)

    # Find the alpha from CV fold results (median of per-fold best alphas)
    fold_alphas = [
        f['best_alpha'] for f in best.user_attrs['fold_results']
        if f['best_alpha'] is not None
    ]
    recommended_alpha = float(np.median(fold_alphas))

    # Get nonzero genes at recommended alpha
    alpha_idx = np.argmin(np.abs(final_model.alphas_ - recommended_alpha))
    actual_alpha = float(final_model.alphas_[alpha_idx])
    coefs = final_model.coef_[:, alpha_idx]
    n_nonzero = int(np.sum(coefs != 0))
    nonzero_genes = expression.index[coefs != 0].tolist()

    logger.info(f"  Recommended alpha (median of CV best): {recommended_alpha:.6f}")
    logger.info(f"  Closest path alpha: {actual_alpha:.6f}")
    logger.info(f"  Nonzero genes at this alpha: {n_nonzero}")

    # Compile results
    results = {
        'cohort': cohort,
        'best_l1_ratio': best.params['l1_ratio'],
        'best_mean_cindex': best.value,
        'best_std_cindex': best.user_attrs['std_cindex'],
        'best_mean_nonzero_genes': best.user_attrs['mean_nonzero_genes'],
        'recommended_alpha': recommended_alpha,
        'actual_path_alpha': actual_alpha,
        'n_nonzero_at_best': n_nonzero,
        'fold_results': best.user_attrs['fold_results'],
        'all_trial_summary': [
            {
                'trial': t.number,
                'l1_ratio': t.params['l1_ratio'],
                'mean_cindex': t.value,
                'std_cindex': t.user_attrs['std_cindex'],
                'mean_nonzero': t.user_attrs['mean_nonzero_genes'],
            }
            for t in study.trials
        ],
    }

    # Save per-cohort results
    out_file = results_dir / f"{cohort}_best_params.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {out_file}")

    # Also save the nonzero gene list
    gene_file = results_dir / f"{cohort}_cox_selected_genes.txt"
    with open(gene_file, 'w') as f:
        for gene in nonzero_genes:
            f.write(gene + '\n')
    logger.info(f"Selected genes saved to: {gene_file}")

    return results


# =============================================================================
# Summary across l1_ratio values
# =============================================================================

def print_l1_ratio_summary(results: Dict):
    """Print aggregated results per l1_ratio value."""
    trials = results['all_trial_summary']

    # Group by l1_ratio
    from collections import defaultdict
    grouped = defaultdict(list)
    for t in trials:
        grouped[t['l1_ratio']].append(t)

    logger.info("")
    logger.info(f"  {'l1_ratio':>8} | {'n_trials':>8} | {'mean CI':>8} | "
                f"{'best CI':>8} | {'mean genes':>10}")
    logger.info(f"  {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*10}")

    for lr in sorted(grouped.keys()):
        trials_lr = grouped[lr]
        cindices = [t['mean_cindex'] for t in trials_lr]
        genes = [t['mean_nonzero'] for t in trials_lr]
        logger.info(
            f"  {lr:>8.1f} | {len(trials_lr):>8} | "
            f"{np.mean(cindices):>8.4f} | {max(cindices):>8.4f} | "
            f"{np.mean(genes):>10.0f}"
        )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/phase2_tuning/cox_tuning_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Log to file
    file_handler = logging.FileHandler(results_dir / "cox_tuning.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(file_handler)

    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Configuration:")
    logger.info(f"  N_TRIALS: {N_TRIALS}")
    logger.info(f"  N_FOLDS: {N_FOLDS}")
    logger.info(f"  L1_RATIO_CHOICES: {L1_RATIO_CHOICES}")
    logger.info(f"  N_ALPHAS: {N_ALPHAS}")
    logger.info(f"  RANDOM_SEED: {RANDOM_SEED}")

    all_results = {}

    for cohort in ['tcga', 'orien']:
        try:
            results = tune_cox_for_cohort(cohort, results_dir)
            all_results[cohort] = results

            # Print l1_ratio summary
            logger.info(f"\nl1_ratio summary for {cohort.upper()}:")
            print_l1_ratio_summary(results)

        except Exception as e:
            logger.error(f"Cox tuning failed for {cohort}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL SUMMARY — ELASTIC NET COX")
    logger.info("=" * 70)

    for cohort, res in all_results.items():
        logger.info(
            f"  {cohort.upper():>5}: l1_ratio={res['best_l1_ratio']:.1f}, "
            f"C-index={res['best_mean_cindex']:.4f} ± {res['best_std_cindex']:.4f}, "
            f"alpha={res['recommended_alpha']:.6f}, "
            f"nonzero={res['n_nonzero_at_best']} genes"
        )

    logger.info("=" * 70)

    # Save combined results
    combined_file = results_dir / "all_results.json"
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)

    logger.info(f"\nAll results saved to: {combined_file}")
    logger.info("Done.")
