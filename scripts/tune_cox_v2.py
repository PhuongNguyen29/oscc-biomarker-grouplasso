"""
Phase 2 — Elastic Net Cox Hyperparameter Tuning

Tunes the l1_ratio parameter for CoxnetSurvivalAnalysis (scikit-survival)
using Optuna with 5-fold stratified CV. Lambda (alpha in glmnet terminology)
is handled internally by the regularization path.

For each l1_ratio, CoxnetSurvivalAnalysis computes a full regularization path
of 100 lambda values. We evaluate each lambda by 5-fold CV C-index and select
the best (l1_ratio, lambda) combination.

Note: Lambda selection within each CV fold uses the same validation fold for
both alpha selection and performance reporting. This is standard K-fold CV,
not nested CV. Phase 3 re-selects lambda via proper nested inner CV per
training fold, so this tuning script's role is to identify the best l1_ratio.

CONSEQUENCE, RECORDED EXPLICITLY: because alpha is chosen to maximise the
C-index on the same fold it is scored on, `best_mean_cindex` is optimistically
biased and MUST NOT be reported as cross-validated performance. It is used only
to rank l1_ratio values. The exported alpha is an ANCHORING SCALE for Phase 4
(M&B 2010 Sec 2.2), not a performance claim. The output JSON carries
`cindex_is_optimistically_biased: true` so downstream consumers cannot mistake
it. Maximising C-index also tends to favour small alphas (denser models), which
propagates to a larger q in Phase 4 and therefore a weaker E(V) bound.

v2 CHANGES
    - No hard-coded paths: --data-dir, --filter-dir, --results-dir, --cohort,
      --n-trials, --n-folds, --seed, --n-alphas are all CLI arguments.
    - Retained-gene list is intersected with an assertion, not a silent
      isin() mask that cannot distinguish "already filtered" from "wrong file".
    - ConvergenceWarning is captured. Coordinate descent returns a partially
      optimised path at max_iter WITHOUT raising, so an unconverged fit was
      previously recorded as a valid result. Non-converged folds are now
      flagged and counted.
    - Filter provenance (filter_metadata.json) is copied into the output so the
      alpha can always be traced to the gene universe it was tuned on.

References:
    - Simon et al. (2011) JSS — Regularization paths for Cox regression
    - Friedman et al. (2010) JSS — Coordinate descent for GLMs
    - Zou & Hastie (2005) JRSS-B — Elastic net
    - Bergstra et al. (2011) NeurIPS — TPE algorithm (Optuna)

Usage:
    cd ~/hpchome/oscc-biomarker-grouplasso
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/tune_cox.py

Output:
    results/phase2_tuning/cox_tuning_<timestamp>/
        ├── cox_tuning.log
        ├── tcga_best_params.json
        ├── orien_best_params.json
        └── all_results.json
"""

import sys
import os
import argparse
import logging
import json
import warnings
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

# Defaults only -- every one of these is overridable from the command line.
DEFAULT_N_TRIALS = 50
DEFAULT_N_FOLDS = 5
DEFAULT_SEED = 42
DEFAULT_DATA_DIR = "data/processed"
DEFAULT_FILTER_DIR = "iqr_filtered_v4_prez"
DEFAULT_RESULTS_ROOT = "results/phase2_tuning"
DEFAULT_N_ALPHAS = 100

# l1_ratio search space (discrete grid). 0.1 = mostly ridge, 1.0 = pure lasso.
L1_RATIO_CHOICES = [1.0]

MAX_ITER = 100000
TOL = 1e-7

# Populated from CLI in __main__ so helper functions stay parameterised.
CFG = {}


# =============================================================================
# Stratified CV folds (event status + survival time quartiles)
# =============================================================================

def create_stratified_folds(
    y_event: np.ndarray,
    y_time: np.ndarray,
    n_folds: int = 5,
    seed: int = 42
) -> np.ndarray:
    """
    Create stratification labels from event status × survival time quartiles.

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
    n_alphas: int = DEFAULT_N_ALPHAS,
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

        # Fit Cox model with full regularization path.
        # Coordinate descent emits a ConvergenceWarning and returns a partially
        # optimised path at max_iter WITHOUT raising, so the warning must be
        # captured explicitly or an unconverged fit is silently accepted.
        try:
            model = CoxnetSurvivalAnalysis(
                l1_ratio=l1_ratio,
                n_alphas=n_alphas,
                normalize=False,       # Data already z-scored
                tol=TOL,
                max_iter=MAX_ITER,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(X_train, y_train)
            # Match on class name AND message: the warning class has moved
            # between sksurv versions, so do not import a specific class.
            nonconverged = any(
                w.category.__name__ == "ConvergenceWarning"
                or "terminated early" in str(w.message).lower()
                for w in caught
            )
        except Exception as e:
            logger.warning(f"  Fold {fold_idx+1}: fitting failed - {e}")
            fold_results.append({
                'fold': fold_idx + 1,
                'best_cindex': 0.5,
                'best_alpha': None,
                'n_nonzero': 0,
                'converged': False,
                'error': str(e),
            })
            continue

        if nonconverged:
            logger.warning(f"  Fold {fold_idx+1}: path did not converge at "
                           f"max_iter={MAX_ITER}; fold flagged.")

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
            'converged': not nonconverged,
        })

    # Aggregate
    cindices = [f['best_cindex'] for f in fold_results]
    mean_cindex = np.mean(cindices)
    std_cindex = np.std(cindices)
    mean_nonzero = np.mean([f['n_nonzero'] for f in fold_results])
    n_converged = sum(1 for f in fold_results if f.get('converged'))

    return {
        'l1_ratio': l1_ratio,
        'mean_cindex': mean_cindex,
        'std_cindex': std_cindex,
        'mean_nonzero_genes': mean_nonzero,
        'n_folds_converged': n_converged,
        'n_folds': len(fold_results),
        'all_folds_converged': n_converged == len(fold_results),
        'fold_results': fold_results,
    }


# =============================================================================
# Optuna objective
# =============================================================================

def create_objective(X, y_event, y_time, n_folds, seed, n_alphas):
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
            n_folds=n_folds,
            seed=seed,
            n_alphas=n_alphas,
        )

        # Store additional info in trial
        trial.set_user_attr('std_cindex', result['std_cindex'])
        trial.set_user_attr('mean_nonzero_genes', result['mean_nonzero_genes'])
        trial.set_user_attr('fold_results', result['fold_results'])
        trial.set_user_attr('all_folds_converged', result['all_folds_converged'])
        trial.set_user_attr('n_folds_converged', result['n_folds_converged'])

        logger.info(
            f"  Trial {trial.number+1:>3}: l1_ratio={l1_ratio:.1f} → "
            f"C-index={result['mean_cindex']:.4f} ± {result['std_cindex']:.4f}, "
            f"nonzero genes={result['mean_nonzero_genes']:.0f}, "
            f"converged={result['n_folds_converged']}/{result['n_folds']}"
        )

        return result['mean_cindex']

    return objective


# =============================================================================
# Run tuning for one cohort
# =============================================================================

def tune_cox_for_cohort(
    cohort: str,
    results_dir: Path,
    data_dir: str,
    filter_dir: str,
    n_trials: int,
    n_folds: int,
    seed: int,
    n_alphas: int,
) -> Dict:
    """
    Run full Optuna tuning for elastic net Cox on one cohort.
    """
    logger.info("=" * 70)
    logger.info(f"ELASTIC NET COX TUNING: {cohort.upper()}")
    logger.info("=" * 70)

    # Load data
    expression, survival = load_cohort(cohort, data_dir)

    # Prepare matrices
    # Expression is genes × samples, need samples × genes
    # ---- Retained gene set -------------------------------------------------
    # Path comes from --filter-dir; nothing is hard-coded, so the frozen old
    # post-z set and the corrected post-ComBat pre-z set are both reachable and
    # the run log records which was used.
    filter_path = Path(data_dir) / filter_dir
    iqr_gene_file = filter_path / "retained_genes.txt"
    if not iqr_gene_file.exists():
        raise FileNotFoundError(
            f"Retained gene list not found: {iqr_gene_file}. "
            f"Check --filter-dir (given: {filter_dir})."
        )
    with open(iqr_gene_file) as f:
        retained = [ln.strip() for ln in f if ln.strip()]

    # Filter provenance travels with the tuned alpha, so the anchor can always
    # be traced back to the gene universe it was derived on.
    filter_meta_file = filter_path / "filter_metadata.json"
    filter_metadata = (json.load(open(filter_meta_file))
                       if filter_meta_file.exists() else {})

    # Assert rather than mask. The old isin() form could not distinguish
    # "expression is already filtered" from "wrong gene list supplied" -- both
    # silently produce a subset and the run looks fine either way.
    n_before = expression.shape[0]
    missing = set(retained) - set(expression.index)
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(retained)} retained genes are absent from "
            f"the {cohort.upper()} expression matrix. The gene list and the "
            f"preprocessed matrix are out of step."
        )
    expression = expression.loc[retained]
    logger.info(f"  Filter dir: {filter_dir}")
    logger.info(f"  Retained genes: {n_before} -> {expression.shape[0]}")
    if n_before == expression.shape[0]:
        logger.info("  (matrix was already restricted to the retained set)")

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
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=f"cox_{cohort}",
    )

    # Run optimization
    objective = create_objective(X, y_event, y_time, n_folds, seed, n_alphas)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

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

    try:
        final_model = CoxnetSurvivalAnalysis(
            l1_ratio=best.params['l1_ratio'],
            n_alphas=n_alphas,
            normalize=False,
            tol=1e-7,
            max_iter=100000,
        )
        final_model.fit(X, y_structured)
    except (ArithmeticError, Exception) as e:
        logger.error(f"Final full-data refit failed: {e}")
        logger.info("Saving CV results without full-data refit.")

        # Save partial results (CV completed successfully, refit failed)
        results = {
            'cohort': cohort,
            'best_l1_ratio': best.params['l1_ratio'],
            'best_mean_cindex': best.value,
            'best_std_cindex': best.user_attrs['std_cindex'],
            'best_mean_nonzero_genes': best.user_attrs['mean_nonzero_genes'],
            'recommended_alpha': None,
            'actual_path_alpha': None,
            'n_nonzero_at_best': None,
            'refit_error': str(e),
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

        out_file = results_dir / f"{cohort}_best_params.json"
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Partial results saved to: {out_file}")

        return results

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
        # --- provenance and caveats (see module docstring) ---
        'cindex_is_optimistically_biased': True,
        'cindex_bias_reason': (
            'alpha selected on the same validation fold it is scored on; '
            'use for ranking l1_ratio only, never as reported performance'
        ),
        'alpha_role': 'anchoring reference scale for Phase 4 (M&B 2010 Sec 2.2)',
        'alpha_derivation': (
            'median of per-fold best alphas, snapped to nearest alpha on the '
            'full-data path (fold paths and the full-data path have different '
            'alpha grids)'
        ),
        'n_genes': int(X.shape[1]),
        'n_samples': int(X.shape[0]),
        'n_events': int(y_event.sum()),
        'filter_dir': filter_dir,
        'filter_metadata': filter_metadata,
        'all_folds_converged': best.user_attrs.get('all_folds_converged'),
        'n_folds_converged': best.user_attrs.get('n_folds_converged'),
        'config': {'n_trials': n_trials, 'n_folds': n_folds,
                   'seed': seed, 'n_alphas': n_alphas,
                   'max_iter': MAX_ITER, 'tol': TOL},
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
    ap = argparse.ArgumentParser(
        description="Phase 2: elastic net Cox tuning (anchoring alpha for Phase 4)")
    ap.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    ap.add_argument('--filter-dir', default=DEFAULT_FILTER_DIR,
                    help="Subdirectory of --data-dir holding retained_genes.txt")
    ap.add_argument('--results-dir', default=None,
                    help=f"Default: {DEFAULT_RESULTS_ROOT}/cox_tuning_<timestamp>")
    ap.add_argument('--cohort', default='both', choices=['tcga', 'orien', 'both'])
    ap.add_argument('--n-trials', type=int, default=DEFAULT_N_TRIALS)
    ap.add_argument('--n-folds', type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument('--n-alphas', type=int, default=DEFAULT_N_ALPHAS)
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    np.random.seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = (Path(args.results_dir) if args.results_dir
                   else Path(DEFAULT_RESULTS_ROOT) / f"cox_tuning_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(results_dir / "cox_tuning.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    logger.info(f"Results directory: {results_dir}")
    logger.info("Configuration:")
    logger.info(f"  data_dir:    {args.data_dir}")
    logger.info(f"  filter_dir:  {args.filter_dir}")
    logger.info(f"  cohort:      {args.cohort}")
    logger.info(f"  n_trials:    {args.n_trials}")
    logger.info(f"  n_folds:     {args.n_folds}")
    logger.info(f"  n_alphas:    {args.n_alphas}")
    logger.info(f"  seed:        {args.seed}")
    logger.info(f"  l1_ratios:   {L1_RATIO_CHOICES}")
    logger.info(f"  max_iter:    {MAX_ITER}, tol: {TOL}")
    logger.info("NOTE: reported C-index is optimistically biased (alpha chosen "
                "on the scoring fold). It ranks l1_ratio; it is not performance.")

    cohorts = ['tcga', 'orien'] if args.cohort == 'both' else [args.cohort]
    all_results = {}

    for cohort in cohorts:
        try:
            results = tune_cox_for_cohort(
                cohort, results_dir,
                data_dir=args.data_dir,
                filter_dir=args.filter_dir,
                n_trials=args.n_trials,
                n_folds=args.n_folds,
                seed=args.seed,
                n_alphas=args.n_alphas,
            )
            all_results[cohort] = results
            logger.info(f"\nl1_ratio summary for {cohort.upper()}:")
            print_l1_ratio_summary(results)
        except Exception as e:
            logger.error(f"Cox tuning failed for {cohort}: {e}")
            import traceback
            traceback.print_exc()

    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL SUMMARY - ELASTIC NET COX")
    logger.info("=" * 70)
    for cohort, res in all_results.items():
        logger.info(
            f"  {cohort.upper():>5}: l1_ratio={res['best_l1_ratio']:.1f}, "
            f"alpha={res.get('actual_path_alpha')}, "
            f"nonzero={res.get('n_nonzero_at_best')} genes, "
            f"converged={res.get('n_folds_converged')}/{args.n_folds}"
        )
    logger.info("")
    logger.info("  Paste actual_path_alpha and best_l1_ratio into Phase 4's")
    logger.info("  PHASE2_COX_PARAMS, then set PHASE2_PARAMS_ARE_CORRECTED = True.")
    logger.info("=" * 70)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)
    logger.info(f"\nAll results saved to: {results_dir / 'all_results.json'}")
    logger.info("Done.")
