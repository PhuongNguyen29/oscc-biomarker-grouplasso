"""
Phase 4: Stability Selection with Randomized Lasso (Cox Only)

Corrects the original Phase 4 which used ridge Cox (l1_ratio=1e-10) as the
base ranker. Ridge assigns nonzero coefficients to ALL genes, producing
dense rankings where stability selection degenerates into stability ranking
with an arbitrary k threshold.

This version uses randomized lasso (Meinshausen & Bühlmann 2010, Section 3.1):
    - Base procedure: CoxnetSurvivalAnalysis with l1_ratio=1.0 (lasso)
    - Alpha from Phase 2 tuning (TCGA: 0.102, ORIEN: 0.068)
    - Per-subsample random penalty weights W_j ~ Uniform(0.5, 1.0)
    - Selection = nonzero coefficient (no arbitrary k threshold needed)
    - Selection probability = fraction of B subsamples where gene is nonzero

Random penalty weighting is implemented via column scaling:
    X_scaled[:, j] = X[:, j] / W_j
    This makes the effective penalty on gene j equal to lambda * W_j.
    Genes with W_j = 0.5 are "easier" to select; W_j = 1.0 is unchanged.

References:
    - Meinshausen & Bühlmann (2010) JRSS-B — Stability selection
    - Meinshausen & Bühlmann (2010) Section 3.1 — Randomized lasso
    - Bourgon et al. (2010) PNAS — IQR independent filtering

Usage:
    cd ~/hpchome/oscc-biomarker-grouplasso
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/phase4_stability_randomized_lasso.py

Output:
    results/phase4_stability_randomized_lasso/
        subsample_selections/   (per-subsample selected gene lists)
        signed_coefficients/    (per-subsample signed coefficients)
        selection_probabilities/ (per-cohort selection probabilities)
        consensus/              (cross-cohort consensus genes)
        summary.json
"""

import sys
import os
import argparse
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

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

logging.getLogger('src.data.dataset').setLevel(logging.ERROR)
logging.getLogger('src.data.data_loader').setLevel(logging.WARNING)

# =============================================================================
# Configuration
# =============================================================================

B_SUBSAMPLES = 50
SUBSAMPLE_FRACTION = 0.80
RANDOM_SEED = 42
DATA_DIR = "data/processed"
RESULTS_BASE = "results/phase4_stability_randomized_lasso"

# Stability thresholds
PI_THRESHOLDS = [0.60, 0.70, 0.80, 0.90]
PI_PRIMARY = 0.80

# Phase 2 Cox parameters (lasso)
PHASE2_COX_PARAMS = {
    'tcga': {
        'l1_ratio': 1.0,
        'alpha': 0.10217330634724603,   # actual_path_alpha from Phase 2
        'mean_nonzero': 19.2,           # expected genes per fit
    },
    'orien': {
        'l1_ratio': 1.0,
        'alpha': 0.06833270705221954,   # actual_path_alpha from Phase 2
        'mean_nonzero': 24.2,           # expected genes per fit
    },
}

# Randomized lasso config (Meinshausen & Bühlmann 2010, Section 3.1)
WEAKNESS_ALPHA = 0.5  # W_j ~ Uniform(WEAKNESS_ALPHA, 1)
# This means penalty weights are drawn from Uniform(0.5, 1.0)

# Alpha scan: also test alphas that select more/fewer genes
# to find the operating point with best consensus
ALPHA_MULTIPLIERS = [0.5, 0.75, 1.0, 1.5, 2.0]


# =============================================================================
# Helpers
# =============================================================================

def create_strata(y_event, y_time):
    """Event × time quartile stratification."""
    time_quartiles = pd.qcut(y_time, q=4, labels=False, duplicates='drop')
    return y_event.astype(int) * 4 + time_quartiles


def generate_subsamples(n_samples, y_event, n_subsamples, fraction, seed):
    """Generate stratified subsamples (same as original Phase 4)."""
    rng = np.random.RandomState(seed)
    subsamples = []

    for b in range(n_subsamples):
        event_idx = np.where(y_event == 1)[0]
        censor_idx = np.where(y_event == 0)[0]

        n_event = int(len(event_idx) * fraction)
        n_censor = int(len(censor_idx) * fraction)

        sub_event = rng.choice(event_idx, size=n_event, replace=False)
        sub_censor = rng.choice(censor_idx, size=n_censor, replace=False)

        train_idx = np.sort(np.concatenate([sub_event, sub_censor]))
        oob_idx = np.sort(np.setdiff1d(np.arange(n_samples), train_idx))

        subsamples.append({
            'train_idx': train_idx,
            'oob_idx': oob_idx,
            'n_train': len(train_idx),
            'n_oob': len(oob_idx),
            'n_train_events': len(sub_event),
            'n_oob_events': int(y_event[oob_idx].sum()),
        })

    return subsamples


# =============================================================================
# Randomized Lasso Cox per subsample
# =============================================================================

def fit_randomized_lasso_cox(
    X_train, y_event_train, y_time_train, gene_names,
    alpha, l1_ratio, weakness, rng,
):
    """
    Fit lasso Cox with randomized penalty weights on one subsample.

    Randomization (Meinshausen & Bühlmann 2010, Section 3.1):
        For each gene j, draw W_j ~ Uniform(weakness, 1).
        Scale column j by 1/W_j before fitting.
        Effective penalty on gene j becomes alpha * W_j.

    Args:
        X_train: (n_samples, n_features) expression matrix
        y_event_train: event indicators
        y_time_train: survival times
        gene_names: list of gene names
        alpha: lasso regularization strength (from Phase 2)
        l1_ratio: 1.0 for lasso
        weakness: lower bound of Uniform distribution (default 0.5)
        rng: numpy RandomState for reproducibility

    Returns:
        selected_genes: list of gene names with nonzero coefficients
        signed_coefs: dict mapping gene_name -> signed coefficient
        n_nonzero: number of selected genes
        weights: the random weights used (for reproducibility)
    """
    n_features = X_train.shape[1]

    # Draw random weights W_j ~ Uniform(weakness, 1.0)
    weights = rng.uniform(weakness, 1.0, size=n_features)

    # Scale columns: X_scaled[:, j] = X_train[:, j] / W_j
    # This makes effective penalty = alpha * W_j for gene j
    X_scaled = X_train / weights[np.newaxis, :]

    # Build structured survival array
    y_struct = np.array(
        [(bool(e), t) for e, t in zip(y_event_train, y_time_train)],
        dtype=[('event', bool), ('time', float)]
    )

    # Fit lasso Cox
    current_alpha = alpha
    for attempt in range(5):
        try:
            model = CoxnetSurvivalAnalysis(
                l1_ratio=l1_ratio,
                alphas=[current_alpha],
                normalize=False,
                tol=1e-7,
                max_iter=100000,
            )
            model.fit(X_scaled, y_struct)
            break
        except ArithmeticError:
            current_alpha = current_alpha * 2
            if attempt == 4:
                logger.warning(f"    Lasso fit failed after 5 attempts, alpha={current_alpha}")
                return [], {}, 0, weights

    # Get coefficients (on scaled data)
    # Note: coefficients are for X_scaled, but since X_scaled[:, j] = X[:, j] / W_j,
    # the effective coefficient for original gene j is coef_scaled[j] / W_j.
    # For selection purposes (nonzero/zero), this doesn't matter —
    # a gene is selected if and only if its scaled coefficient is nonzero.
    raw_coefs_scaled = model.coef_[:, 0]

    # Transform back to original scale for sign interpretation
    raw_coefs_original = raw_coefs_scaled / weights

    # Identify selected genes (nonzero coefficients)
    nonzero_mask = np.abs(raw_coefs_scaled) > 1e-10
    n_nonzero = int(nonzero_mask.sum())

    selected_genes = [gene_names[j] for j in range(n_features) if nonzero_mask[j]]
    signed_coefs = dict(zip(gene_names, raw_coefs_original))

    return selected_genes, signed_coefs, n_nonzero, weights


# =============================================================================
# Selection probabilities (for randomized lasso — no k needed)
# =============================================================================

def compute_selection_probs_binary(all_selections, gene_names):
    """
    Compute selection probability for each gene.

    Selection probability = fraction of B subsamples where the gene
    has a nonzero coefficient. No k threshold needed.

    Args:
        all_selections: list of B sets, each containing selected gene names
        gene_names: list of all gene names

    Returns:
        Series: gene -> selection probability
    """
    B = len(all_selections)
    counts = pd.Series(0, index=gene_names, dtype=float)

    for selected in all_selections:
        for gene in selected:
            if gene in counts.index:
                counts[gene] += 1

    probs = counts / B
    return probs


# =============================================================================
# Consensus
# =============================================================================

def find_consensus_binary(probs_cohort1, probs_cohort2, pi):
    """
    Find genes with selection probability >= pi in BOTH cohorts.
    No k parameter — selection is defined by nonzero coefficients.
    """
    stable_c1 = set(probs_cohort1.index[probs_cohort1 >= pi])
    stable_c2 = set(probs_cohort2.index[probs_cohort2 >= pi])
    consensus = stable_c1 & stable_c2
    return {
        'n_stable_c1': len(stable_c1),
        'n_stable_c2': len(stable_c2),
        'n_consensus': len(consensus),
        'consensus_genes': sorted(consensus),
        'stable_c1_genes': sorted(stable_c1),
        'stable_c2_genes': sorted(stable_c2),
    }


# =============================================================================
# Per-cohort runner
# =============================================================================

def run_stability_for_cohort(
    cohort, X, y_event, y_time, gene_names,
    alpha, l1_ratio, results_dir,
):
    """Run randomized lasso stability selection for one cohort."""

    logger.info(f"")
    logger.info(f"{'='*70}")
    logger.info(f"RANDOMIZED LASSO STABILITY: {cohort.upper()}")
    logger.info(f"{'='*70}")
    logger.info(f"  Samples: {len(y_event)}, Events: {int(y_event.sum())}")
    logger.info(f"  Features: {X.shape[1]}")
    logger.info(f"  Lasso alpha: {alpha:.6f}")
    logger.info(f"  l1_ratio: {l1_ratio}")
    logger.info(f"  Weakness (W_j lower bound): {WEAKNESS_ALPHA}")
    logger.info(f"  Subsamples: {B_SUBSAMPLES}, Fraction: {SUBSAMPLE_FRACTION}")

    # Generate subsamples (same seed as original for comparability)
    subsamples = generate_subsamples(
        len(y_event), y_event, B_SUBSAMPLES, SUBSAMPLE_FRACTION, RANDOM_SEED
    )

    logger.info(f"  Subsample sizes: train~{subsamples[0]['n_train']}, "
                f"OOB~{subsamples[0]['n_oob']}")

    # Output directories
    sel_dir = results_dir / "subsample_selections" / cohort
    sel_dir.mkdir(parents=True, exist_ok=True)
    coef_dir = results_dir / "signed_coefficients" / cohort
    coef_dir.mkdir(parents=True, exist_ok=True)

    # RNG for randomized weights (separate from subsampling RNG)
    weight_rng = np.random.RandomState(RANDOM_SEED + 1000)

    # --- Run across alpha multipliers ---
    all_alpha_results = {}

    for mult in ALPHA_MULTIPLIERS:
        current_alpha = alpha * mult
        logger.info(f"")
        logger.info(f"  --- Alpha multiplier: {mult}x (alpha={current_alpha:.6f}) ---")

        all_selections = []
        all_signed_coefs = []
        n_nonzero_per_sub = []

        # Reset weight RNG for each alpha so results are comparable
        weight_rng_alpha = np.random.RandomState(RANDOM_SEED + 1000)

        for b in range(B_SUBSAMPLES):
            ss = subsamples[b]

            selected, signed_coefs, n_nonzero, weights = fit_randomized_lasso_cox(
                X[ss['train_idx']],
                y_event[ss['train_idx']],
                y_time[ss['train_idx']],
                gene_names,
                alpha=current_alpha,
                l1_ratio=l1_ratio,
                weakness=WEAKNESS_ALPHA,
                rng=weight_rng_alpha,
            )

            all_selections.append(selected)
            all_signed_coefs.append(signed_coefs)
            n_nonzero_per_sub.append(n_nonzero)

            if (b + 1) % 10 == 0:
                logger.info(f"    Subsample {b+1}/{B_SUBSAMPLES}: "
                            f"{n_nonzero} genes selected")

        # Summary for this alpha
        mean_nonzero = np.mean(n_nonzero_per_sub)
        std_nonzero = np.std(n_nonzero_per_sub)
        logger.info(f"    Mean genes selected per subsample: "
                    f"{mean_nonzero:.1f} ± {std_nonzero:.1f}")
        logger.info(f"    Range: [{min(n_nonzero_per_sub)}, {max(n_nonzero_per_sub)}]")

        # Compute selection probabilities
        sel_probs = compute_selection_probs_binary(all_selections, gene_names)

        # How many genes pass various pi thresholds?
        for pi in PI_THRESHOLDS:
            n_stable = (sel_probs >= pi).sum()
            logger.info(f"    pi >= {pi:.2f}: {n_stable} stable genes")

        all_alpha_results[mult] = {
            'alpha': current_alpha,
            'sel_probs': sel_probs,
            'all_selections': all_selections,
            'all_signed_coefs': all_signed_coefs,
            'mean_nonzero': mean_nonzero,
            'std_nonzero': std_nonzero,
            'n_nonzero_per_sub': n_nonzero_per_sub,
        }

        # Save selections and coefficients for primary multiplier (1.0x)
        if mult == 1.0:
            for b in range(B_SUBSAMPLES):
                # Save selected gene list
                sel_df = pd.DataFrame({
                    'gene': all_selections[b],
                })
                sel_df.to_csv(sel_dir / f"subsample_{b+1:02d}.csv", index=False)

            # Save signed coefficient matrix
            signed_df = pd.DataFrame(all_signed_coefs, columns=gene_names).T
            signed_df.columns = [f"subsample_{b+1:02d}" for b in range(B_SUBSAMPLES)]
            signed_df.to_csv(coef_dir / "cox_signed_coefficients.csv")

            # Save selection probabilities
            prob_dir = results_dir / "selection_probabilities"
            prob_dir.mkdir(parents=True, exist_ok=True)
            sel_probs.to_csv(prob_dir / f"{cohort}_cox_selection_probs.csv",
                            header=["selection_prob"])

    return all_alpha_results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 4: Randomized Lasso Stability Selection (Cox Only)"
    )
    parser.add_argument(
        '--filter-dir', type=str, default='iqr_filtered',
        help="Subdirectory under data/processed/ with retained_genes.txt"
    )
    parser.add_argument(
        '--cohort', type=str, default='both',
        choices=['tcga', 'orien', 'both'],
        help="Which cohort to run (default: both)"
    )
    parser.add_argument(
        '--results-dir', type=str, default=None,
        help=f"Override results directory (default: {RESULTS_BASE})"
    )
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)

    # Results directory
    results_dir = Path(args.results_dir) if args.results_dir else Path(RESULTS_BASE)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Log to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(results_dir / f"phase4_rlasso_{timestamp}.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(file_handler)

    logger.info("Phase 4: Randomized Lasso Stability Selection (Cox Only)")
    logger.info(f"Results: {results_dir}")

    # =========================================================================
    # Load data
    # =========================================================================
    logger.info("Loading data...")
    tcga_expr, tcga_surv = load_cohort('tcga', DATA_DIR)
    orien_expr, orien_surv = load_cohort('orien', DATA_DIR)

    gene_names_full = tcga_expr.index.tolist()
    if list(orien_expr.index) != gene_names_full:
        raise ValueError("Gene sets/order differ between TCGA and ORIEN.")

    logger.info(f"Full gene set: {len(gene_names_full)} genes")

    # =========================================================================
    # IQR pre-filtering
    # =========================================================================
    logger.info(f"")
    logger.info(f"{'='*70}")
    logger.info(f"IQR PRE-FILTERING")
    logger.info(f"{'='*70}")

    iqr_gene_file = Path(DATA_DIR) / args.filter_dir / "retained_genes.txt"
    iqr_metadata_file = Path(DATA_DIR) / args.filter_dir / "filter_metadata.json"

    if not iqr_gene_file.exists():
        raise FileNotFoundError(
            f"IQR filtered gene list not found: {iqr_gene_file}. "
            f"Run scripts/filter_genes_iqr.py first."
        )

    with open(iqr_gene_file) as f:
        retained_genes = [line.strip() for line in f if line.strip()]

    iqr_metadata = {}
    if iqr_metadata_file.exists():
        with open(iqr_metadata_file) as f:
            iqr_metadata = json.load(f)

    logger.info(f"  IQR filter: {len(gene_names_full)} -> {len(retained_genes)} genes")

    # Subset and transpose
    tcga_X = tcga_expr.loc[retained_genes].values.T.astype(np.float32)
    tcga_event = tcga_surv['event'].values.astype(np.float32)
    tcga_time = tcga_surv['time'].values.astype(np.float32)

    orien_X = orien_expr.loc[retained_genes].values.T.astype(np.float32)
    orien_event = orien_surv['event'].values.astype(np.float32)
    orien_time = orien_surv['time'].values.astype(np.float32)

    gene_names = retained_genes
    n_features = len(gene_names)

    logger.info(f"  TCGA: {tcga_X.shape[0]} samples x {tcga_X.shape[1]} genes, "
                f"{int(tcga_event.sum())} events")
    logger.info(f"  ORIEN: {orien_X.shape[0]} samples x {orien_X.shape[1]} genes, "
                f"{int(orien_event.sum())} events")

    # =========================================================================
    # Run stability selection per cohort
    # =========================================================================
    cohorts_to_run = ['tcga', 'orien'] if args.cohort == 'both' else [args.cohort]

    cohort_data = {
        'tcga': (tcga_X, tcga_event, tcga_time),
        'orien': (orien_X, orien_event, orien_time),
    }

    all_cohort_results = {}

    for cohort in cohorts_to_run:
        X, y_event, y_time = cohort_data[cohort]
        params = PHASE2_COX_PARAMS[cohort]

        result = run_stability_for_cohort(
            cohort, X, y_event, y_time, gene_names,
            alpha=params['alpha'],
            l1_ratio=params['l1_ratio'],
            results_dir=results_dir,
        )
        all_cohort_results[cohort] = result

    # =========================================================================
    # Consensus analysis (requires both cohorts)
    # =========================================================================
    if len(all_cohort_results) == 2:
        logger.info(f"")
        logger.info(f"{'='*70}")
        logger.info(f"CONSENSUS ANALYSIS")
        logger.info(f"{'='*70}")

        consensus_dir = results_dir / "consensus"
        consensus_dir.mkdir(parents=True, exist_ok=True)

        # Load gene annotation for symbols
        annot_file = Path("data/raw/protein_coding_genes_annotation.csv")
        if annot_file.exists():
            annot = pd.read_csv(annot_file)
            ensg_to_symbol = dict(zip(annot["gene_id"], annot["gene_symbol"]))
        else:
            ensg_to_symbol = {}

        # For each alpha multiplier, compute consensus
        for mult in ALPHA_MULTIPLIERS:
            tcga_probs = all_cohort_results['tcga'][mult]['sel_probs']
            orien_probs = all_cohort_results['orien'][mult]['sel_probs']

            tcga_mean_nz = all_cohort_results['tcga'][mult]['mean_nonzero']
            orien_mean_nz = all_cohort_results['orien'][mult]['mean_nonzero']

            logger.info(f"")
            logger.info(f"  --- Alpha multiplier: {mult}x ---")
            logger.info(f"  Mean selected: TCGA={tcga_mean_nz:.1f}, ORIEN={orien_mean_nz:.1f}")

            for pi in PI_THRESHOLDS:
                result = find_consensus_binary(tcga_probs, orien_probs, pi)
                logger.info(
                    f"  pi >= {pi:.2f}: TCGA={result['n_stable_c1']:>4}, "
                    f"ORIEN={result['n_stable_c2']:>4}, "
                    f"consensus={result['n_consensus']:>4}"
                )

                # Save consensus gene lists
                if result['n_consensus'] > 0:
                    genes_with_symbols = []
                    for gene in result['consensus_genes']:
                        sym = ensg_to_symbol.get(gene, "")
                        genes_with_symbols.append((gene, sym))

                    gene_file = consensus_dir / f"consensus_mult{mult}_pi{int(pi*100):02d}.csv"
                    pd.DataFrame(genes_with_symbols, columns=['gene', 'symbol']).to_csv(
                        gene_file, index=False
                    )

                    if pi == PI_PRIMARY:
                        logger.info(f"    Consensus genes at pi={pi:.2f}:")
                        for gene, sym in genes_with_symbols:
                            t_pi = tcga_probs[gene]
                            o_pi = orien_probs[gene]
                            logger.info(f"      {gene} ({sym}): "
                                        f"TCGA pi={t_pi:.2f}, ORIEN pi={o_pi:.2f}")

                # Save per-cohort stable gene lists
                if pi == PI_PRIMARY:
                    for cohort_name, stable_genes in [
                        ('tcga', result['stable_c1_genes']),
                        ('orien', result['stable_c2_genes'])
                    ]:
                        stable_file = consensus_dir / f"{cohort_name}_stable_mult{mult}_pi{int(pi*100):02d}.csv"
                        stable_with_sym = [(g, ensg_to_symbol.get(g, "")) for g in stable_genes]
                        pd.DataFrame(stable_with_sym, columns=['gene', 'symbol']).to_csv(
                            stable_file, index=False
                        )

        # =====================================================================
        # Summary
        # =====================================================================
        logger.info(f"")
        logger.info(f"{'='*70}")
        logger.info(f"FINAL SUMMARY")
        logger.info(f"{'='*70}")

        summary = {
            'method': 'Randomized Lasso Stability Selection (Meinshausen & Bühlmann 2010)',
            'base_procedure': 'CoxnetSurvivalAnalysis with l1_ratio=1.0 (lasso)',
            'randomization': f'W_j ~ Uniform({WEAKNESS_ALPHA}, 1.0) per-gene penalty weights',
            'reference': 'Meinshausen & Bühlmann (2010) JRSS-B, Section 3.1',
            'iqr_filter': iqr_metadata,
            'n_genes': n_features,
            'phase2_params': PHASE2_COX_PARAMS,
            'config': {
                'B_subsamples': B_SUBSAMPLES,
                'subsample_fraction': SUBSAMPLE_FRACTION,
                'weakness_alpha': WEAKNESS_ALPHA,
                'alpha_multipliers': ALPHA_MULTIPLIERS,
                'pi_thresholds': PI_THRESHOLDS,
                'pi_primary': PI_PRIMARY,
            },
            'results_by_multiplier': {},
        }

        for mult in ALPHA_MULTIPLIERS:
            tcga_probs = all_cohort_results['tcga'][mult]['sel_probs']
            orien_probs = all_cohort_results['orien'][mult]['sel_probs']

            mult_result = {
                'tcga_alpha': all_cohort_results['tcga'][mult]['alpha'],
                'orien_alpha': all_cohort_results['orien'][mult]['alpha'],
                'tcga_mean_nonzero': all_cohort_results['tcga'][mult]['mean_nonzero'],
                'orien_mean_nonzero': all_cohort_results['orien'][mult]['mean_nonzero'],
            }

            for pi in PI_THRESHOLDS:
                result = find_consensus_binary(tcga_probs, orien_probs, pi)
                mult_result[f'pi_{int(pi*100):02d}'] = {
                    'tcga_stable': result['n_stable_c1'],
                    'orien_stable': result['n_stable_c2'],
                    'consensus': result['n_consensus'],
                    'consensus_genes': result['consensus_genes'],
                }

            summary['results_by_multiplier'][str(mult)] = mult_result

            logger.info(
                f"  mult={mult}x: TCGA q~{mult_result['tcga_mean_nonzero']:.0f}, "
                f"ORIEN q~{mult_result['orien_mean_nonzero']:.0f}, "
                f"consensus(pi=0.80)="
                f"{mult_result[f'pi_{int(PI_PRIMARY*100):02d}']['consensus']}"
            )

        with open(results_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    else:
        logger.info("Only one cohort run. Use --cohort both for consensus.")

    logger.info(f"\nPhase 4 (Randomized Lasso) complete.")
    logger.info(f"Results saved to: {results_dir}")
