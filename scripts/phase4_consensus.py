"""
Phase 4 Consensus Analysis (standalone)

Reads saved selection probability CSVs from Phase 4 subsampling and
computes consensus gene sets, plateau rule, and summary statistics.
Does NOT re-run any model training.

Works with both the original full-dimensionality results and
MAD-filtered results — just point --results-dir to the appropriate
directory.

Usage:
    cd ~/hpchome/oscc-biomarker-elasticdeepsurv

    # Original full-dimensionality results:
    python scripts/phase4_consensus.py

    # MAD-filtered results:
    python scripts/phase4_consensus.py --results-dir results/phase4_stability_mad3000
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration (must match phase4_stability*.py)
RESULTS_BASE_DEFAULT = "results/phase4_stability"
K_SCAN = [50, 100, 150, 200, 300, 500, 750, 1000]
PI_THRESHOLDS = [0.70, 0.80, 0.90]
PI_PRIMARY = 0.80


def find_consensus(probs_dir1, probs_dir2, k_values, pi):
    consensus_sizes = {}
    consensus_genes = {}
    for k in k_values:
        col = f"k_{k}"
        stable_d1 = set(probs_dir1.index[probs_dir1[col] >= pi])
        stable_d2 = set(probs_dir2.index[probs_dir2[col] >= pi])
        consensus = stable_d1 & stable_d2
        consensus_sizes[k] = len(consensus)
        consensus_genes[k] = sorted(consensus)
    return consensus_sizes, consensus_genes


def apply_plateau_rule(consensus_sizes, k_values):
    def step_plateau(c_curr, c_next):
        if c_curr <= 0:
            return False
        if c_next < c_curr:
            return False
        abs_change = c_next - c_curr
        rel_change = abs_change / c_curr
        return (rel_change < 0.05) and (abs_change <= 2)

    for i in range(len(k_values) - 2):
        k0, k1, k2 = k_values[i], k_values[i + 1], k_values[i + 2]
        c0, c1, c2 = consensus_sizes[k0], consensus_sizes[k1], consensus_sizes[k2]
        if step_plateau(c0, c1) and step_plateau(c1, c2):
            return k0, True

    return k_values[-1], False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 4 Consensus Analysis (standalone)"
    )
    parser.add_argument(
        '--results-dir', type=str, default=RESULTS_BASE_DEFAULT,
        help=f"Path to Phase 4 results directory (default: {RESULTS_BASE_DEFAULT})"
    )
    args = parser.parse_args()

    RESULTS_BASE = Path(args.results_dir)

    prob_dir = RESULTS_BASE / "selection_probabilities"
    consensus_dir = RESULTS_BASE / "consensus"
    consensus_dir.mkdir(parents=True, exist_ok=True)

    # Load selection probabilities
    logger.info(f"Loading selection probabilities from {prob_dir}...")
    tcga_cox = pd.read_csv(prob_dir / "tcga_cox_selection_probs.csv", index_col=0)
    tcga_ds = pd.read_csv(prob_dir / "tcga_deepsurv_selection_probs.csv", index_col=0)
    orien_cox = pd.read_csv(prob_dir / "orien_cox_selection_probs.csv", index_col=0)
    orien_ds = pd.read_csv(prob_dir / "orien_deepsurv_selection_probs.csv", index_col=0)

    logger.info(f"  TCGA Cox: {tcga_cox.shape}, TCGA DeepSurv: {tcga_ds.shape}")
    logger.info(f"  ORIEN Cox: {orien_cox.shape}, ORIEN DeepSurv: {orien_ds.shape}")

    probs = {
        'cox': {'tcga': tcga_cox, 'orien': orien_cox},
        'deepsurv': {'tcga': tcga_ds, 'orien': orien_ds},
    }

    logger.info("")
    logger.info("=" * 70)
    logger.info("CONSENSUS ANALYSIS")
    logger.info("=" * 70)

    # --- Shared k selection ---
    methods = ['cox', 'deepsurv']
    method_selected_ks = {}

    for method in methods:
        logger.info(f"")
        logger.info(f"  --- {method.upper()} ---")

        tcga_probs = probs[method]['tcga']
        orien_probs = probs[method]['orien']

        for pi in PI_THRESHOLDS:
            consensus_sizes, consensus_genes = find_consensus(
                tcga_probs, orien_probs, K_SCAN, pi
            )

            logger.info(f"  π={pi:.2f}:")
            for k in K_SCAN:
                logger.info(f"    k={k:>4}: {consensus_sizes[k]:>4} consensus genes")

            # Save gene lists at all k and pi
            for k in K_SCAN:
                if consensus_genes[k]:
                    gene_file = consensus_dir / f"{method}_consensus_k{k}_pi{int(pi*100):02d}.txt"
                    with open(gene_file, 'w') as f:
                        for gene in consensus_genes[k]:
                            f.write(gene + '\n')

            # Apply plateau rule at primary threshold
            if pi == PI_PRIMARY:
                selected_k, plateau_triggered = apply_plateau_rule(
                    consensus_sizes, K_SCAN
                )
                method_selected_ks[method] = {
                    'selected_k': selected_k,
                    'plateau_triggered': plateau_triggered,
                    'consensus_sizes': consensus_sizes,
                    'consensus_genes_by_k': consensus_genes,
                }

                if plateau_triggered:
                    logger.info(f"  ✓ Method-specific plateau at k={selected_k}")
                else:
                    logger.info(f"  ⚠ No plateau. Method-specific k_max={selected_k}")

    # Shared k = max of method-specific k values
    shared_k = max(res['selected_k'] for res in method_selected_ks.values())
    shared_plateau = all(res['plateau_triggered'] for res in method_selected_ks.values())

    logger.info(f"")
    logger.info(f"  Shared k = {shared_k} "
                f"(Cox k={method_selected_ks['cox']['selected_k']}, "
                f"DeepSurv k={method_selected_ks['deepsurv']['selected_k']})")

    # Extract consensus genes at shared k for each method
    plateau_results = {}
    for method in methods:
        info = method_selected_ks[method]
        tcga_probs = probs[method]['tcga']
        orien_probs = probs[method]['orien']
        consensus_sizes, consensus_genes = find_consensus(
            tcga_probs, orien_probs, K_SCAN, PI_PRIMARY
        )

        plateau_results[method] = {
            'method_specific_k': info['selected_k'],
            'method_specific_plateau': info['plateau_triggered'],
            'shared_k': shared_k,
            'shared_plateau': shared_plateau,
            'consensus_sizes': consensus_sizes,
            'n_consensus_at_shared_k': consensus_sizes[shared_k],
            'consensus_genes': consensus_genes[shared_k],
        }

        # Save primary consensus gene list at shared k
        primary_file = consensus_dir / f"{method}_consensus_pi080.txt"
        with open(primary_file, 'w') as f:
            for gene in consensus_genes[shared_k]:
                f.write(gene + '\n')

    # Save plateau analysis
    plateau_file = consensus_dir / "plateau_analysis.json"
    serializable = {}
    for method, res in plateau_results.items():
        serializable[method] = {
            'method_specific_k': res['method_specific_k'],
            'method_specific_plateau': res['method_specific_plateau'],
            'shared_k': res['shared_k'],
            'shared_plateau': res['shared_plateau'],
            'consensus_sizes': {str(k): v for k, v in res['consensus_sizes'].items()},
            'n_consensus': res['n_consensus_at_shared_k'],
            'consensus_genes': res['consensus_genes'],
        }
    with open(plateau_file, 'w') as f:
        json.dump(serializable, f, indent=2)

    # --- Contingency checks ---
    for method, res in plateau_results.items():
        n_genes = res['n_consensus_at_shared_k']
        if n_genes == 0:
            logger.warning(
                f"  ⚠ {method}: consensus set empty at pi=0.80. "
                f"This is a negative result. Check secondary thresholds (pi=0.70)."
            )
        elif n_genes < 5:
            logger.warning(
                f"  ⚠ {method}: consensus set has only {n_genes} genes at pi=0.80. "
                f"Consider reporting pi=0.70 as secondary threshold."
            )

    # --- Final summary ---
    logger.info(f"")
    logger.info(f"{'='*70}")
    logger.info(f"FINAL SUMMARY — PHASE 4")
    logger.info(f"{'='*70}")

    for method in methods:
        res = plateau_results[method]
        logger.info(
            f"  {method.upper():>8}: shared_k={res['shared_k']}, "
            f"method_k={res['method_specific_k']}, "
            f"plateau={'YES' if res['method_specific_plateau'] else 'NO'}, "
            f"consensus genes={res['n_consensus_at_shared_k']}"
        )

    # Gene overlap between methods
    cox_genes = set(plateau_results['cox']['consensus_genes'])
    ds_genes = set(plateau_results['deepsurv']['consensus_genes'])
    overlap = cox_genes & ds_genes
    union = cox_genes | ds_genes
    jaccard = len(overlap) / len(union) if union else 0

    logger.info(f"")
    logger.info(f"  Cox consensus: {len(cox_genes)} genes")
    logger.info(f"  DeepSurv consensus: {len(ds_genes)} genes")
    logger.info(f"  Overlap: {len(overlap)} genes")
    logger.info(f"  Jaccard: {jaccard:.3f}")
    logger.info(f"{'='*70}")

    # Load ridge lambda info if available
    lambda_file = RESULTS_BASE / "ridge_lambda_selection.json"
    lambda_info = {}
    if lambda_file.exists():
        with open(lambda_file) as f:
            lambda_info = json.load(f)

    # Load MAD filter info if available
    mad_file = RESULTS_BASE / "mad_filter" / "mad_filter_metadata.json"
    mad_info = {}
    if mad_file.exists():
        with open(mad_file) as f:
            mad_info = json.load(f)

    # Save overall summary
    summary = {
        'ridge_lambdas': lambda_info,
        'plateau_results': serializable,
        'gene_overlap': {
            'cox_n': len(cox_genes),
            'deepsurv_n': len(ds_genes),
            'overlap_n': len(overlap),
            'jaccard': jaccard,
            'overlap_genes': sorted(overlap),
        },
        'config': {
            'k_scan': K_SCAN,
            'pi_thresholds': PI_THRESHOLDS,
            'pi_primary': PI_PRIMARY,
            'shared_k': shared_k,
        },
    }
    if mad_info:
        summary['mad_filter'] = mad_info

    with open(RESULTS_BASE / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nConsensus analysis complete. Results in {consensus_dir}")
