"""
Phase 4 Extended K-Scan Consensus Analysis

Reads saved selection probability CSVs from Phase 4 MAD-filtered run and
evaluates consensus at extended k values (up to 1500) to check for plateau.

The selection probabilities already contain per-gene probabilities at each
original k. Since probabilities are monotonically non-decreasing with k
(a gene in top-50 is also in top-100), we can evaluate any k up to the
number of genes (2943) without re-running subsamples.

Usage:
    cd ~/hpchome/oscc-biomarker-grouplasso
    python scripts/phase4_consensus_extended_k.py
"""

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

# Configuration
RESULTS_BASE = Path("results/phase4_stability_iqr_p65")
# Extended k scan — original + new values
K_SCAN_EXTENDED = [50, 100, 150, 200, 300, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3500, 4000]
PI_THRESHOLDS = [0.70, 0.80, 0.90]
PI_PRIMARY = 0.80


def recompute_selection_probs_at_k(rankings_dir, gene_names, k):
    """
    Recompute selection probability at a new k value from saved subsample rankings.
    
    For each gene, count the proportion of subsamples where it appears in the top-k.
    """
    ranking_files = sorted(rankings_dir.glob("subsample_*.csv"))
    if not ranking_files:
        return None

    n_subsamples = len(ranking_files)
    gene_counts = {g: 0 for g in gene_names}

    for rf in ranking_files:
        df = pd.read_csv(rf)
        top_k_genes = set(df['gene'].iloc[:k].tolist())
        for g in top_k_genes:
            if g in gene_counts:
                gene_counts[g] += 1

    probs = {g: count / n_subsamples for g, count in gene_counts.items()}
    return probs


def find_consensus(probs_dir1, probs_dir2, k_col, pi):
    """Find genes with selection probability >= pi in BOTH directions."""
    stable_d1 = set(probs_dir1.index[probs_dir1[k_col] >= pi])
    stable_d2 = set(probs_dir2.index[probs_dir2[k_col] >= pi])
    consensus = stable_d1 & stable_d2
    return len(consensus), sorted(consensus)


def apply_plateau_rule(consensus_sizes, k_values):
    """Hybrid plateau rule: two consecutive steps."""
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
    prob_dir = RESULTS_BASE / "selection_probabilities"
    rankings_dir = RESULTS_BASE / "subsample_rankings"
    consensus_dir = RESULTS_BASE / "consensus_extended_k"
    consensus_dir.mkdir(parents=True, exist_ok=True)

    # Load existing selection probabilities
    logger.info("Loading existing selection probabilities...")
    existing_probs = {}
    for cohort in ['tcga', 'orien']:
        for method in ['cox', 'deepsurv']:
            key = f"{cohort}_{method}"
            df = pd.read_csv(prob_dir / f"{key}_selection_probs.csv", index_col=0)
            existing_probs[key] = df

    gene_names = existing_probs['tcga_cox'].index.tolist()
    n_genes = len(gene_names)
    logger.info(f"  Genes: {n_genes}")

    # Check which k values need recomputation
    existing_ks = [int(c.replace('k_', '')) for c in existing_probs['tcga_cox'].columns]
    new_ks = [k for k in K_SCAN_EXTENDED if k not in existing_ks]
    logger.info(f"  Existing k values: {existing_ks}")
    logger.info(f"  New k values to compute: {new_ks}")

    # Recompute selection probs for new k values from saved rankings
    if new_ks:
        logger.info("Recomputing selection probabilities for new k values...")
        for cohort in ['tcga', 'orien']:
            for method in ['cox', 'deepsurv']:
                key = f"{cohort}_{method}"
                method_rankings_dir = rankings_dir / cohort / method

                if not method_rankings_dir.exists():
                    logger.warning(f"  Rankings not found: {method_rankings_dir}")
                    continue

                for k in new_ks:
                    probs = recompute_selection_probs_at_k(
                        method_rankings_dir, gene_names, k
                    )
                    if probs:
                        existing_probs[key][f'k_{k}'] = [
                            probs.get(g, 0.0) for g in gene_names
                        ]
                        logger.info(f"  {key} k={k}: done")

    # Consensus analysis with extended k
    logger.info("")
    logger.info("=" * 70)
    logger.info("CONSENSUS ANALYSIS (Extended K-Scan)")
    logger.info("=" * 70)

    methods = ['cox', 'deepsurv']
    method_selected_ks = {}

    for method in methods:
        logger.info(f"")
        logger.info(f"  --- {method.upper()} ---")

        tcga_probs = existing_probs[f'tcga_{method}']
        orien_probs = existing_probs[f'orien_{method}']

        for pi in PI_THRESHOLDS:
            consensus_sizes = {}
            consensus_genes = {}

            logger.info(f"  π={pi:.2f}:")
            for k in K_SCAN_EXTENDED:
                col = f'k_{k}'
                if col in tcga_probs.columns and col in orien_probs.columns:
                    n_cons, genes = find_consensus(tcga_probs, orien_probs, col, pi)
                    consensus_sizes[k] = n_cons
                    consensus_genes[k] = genes
                    logger.info(f"    k={k:>5}: {n_cons:>4} consensus genes")

                    # Save gene lists
                    if genes:
                        gene_file = consensus_dir / f"{method}_consensus_k{k}_pi{int(pi*100):02d}.txt"
                        with open(gene_file, 'w') as f:
                            for gene in genes:
                                f.write(gene + '\n')

            # Plateau rule at primary threshold
            if pi == PI_PRIMARY:
                available_ks = sorted(consensus_sizes.keys())
                selected_k, plateau_triggered = apply_plateau_rule(
                    consensus_sizes, available_ks
                )
                method_selected_ks[method] = {
                    'selected_k': selected_k,
                    'plateau_triggered': plateau_triggered,
                    'consensus_sizes': consensus_sizes,
                    'consensus_genes_by_k': consensus_genes,
                }

                if plateau_triggered:
                    logger.info(f"  ✓ Plateau at k={selected_k}")
                else:
                    logger.info(f"  ⚠ No plateau. k_max={selected_k}")

    # Shared k
    shared_k = max(res['selected_k'] for res in method_selected_ks.values())

    logger.info(f"")
    logger.info(f"  Shared k = {shared_k} "
                f"(Cox k={method_selected_ks['cox']['selected_k']}, "
                f"DeepSurv k={method_selected_ks['deepsurv']['selected_k']})")

    # Final summary
    plateau_results = {}
    for method in methods:
        info = method_selected_ks[method]
        consensus_sizes = info['consensus_sizes']
        consensus_genes = info['consensus_genes_by_k']

        plateau_results[method] = {
            'method_specific_k': info['selected_k'],
            'method_specific_plateau': info['plateau_triggered'],
            'shared_k': shared_k,
            'consensus_sizes': consensus_sizes,
            'n_consensus_at_shared_k': consensus_sizes.get(shared_k, 0),
            'consensus_genes': consensus_genes.get(shared_k, []),
        }

        # Save primary consensus
        primary_file = consensus_dir / f"{method}_consensus_pi080.txt"
        with open(primary_file, 'w') as f:
            for gene in consensus_genes.get(shared_k, []):
                f.write(gene + '\n')

    # Save plateau analysis
    serializable = {}
    for method, res in plateau_results.items():
        serializable[method] = {
            'method_specific_k': res['method_specific_k'],
            'method_specific_plateau': res['method_specific_plateau'],
            'shared_k': res['shared_k'],
            'consensus_sizes': {str(k): v for k, v in res['consensus_sizes'].items()},
            'n_consensus': res['n_consensus_at_shared_k'],
            'consensus_genes': res['consensus_genes'],
        }
    with open(consensus_dir / "plateau_analysis.json", 'w') as f:
        json.dump(serializable, f, indent=2)

    # Gene overlap
    cox_genes = set(plateau_results['cox']['consensus_genes'])
    ds_genes = set(plateau_results['deepsurv']['consensus_genes'])
    overlap = cox_genes & ds_genes
    union = cox_genes | ds_genes
    jaccard = len(overlap) / len(union) if union else 0

    logger.info(f"")
    logger.info(f"{'='*70}")
    logger.info(f"FINAL SUMMARY — Extended K-Scan")
    logger.info(f"{'='*70}")

    for method in methods:
        res = plateau_results[method]
        logger.info(
            f"  {method.upper():>8}: shared_k={res['shared_k']}, "
            f"method_k={res['method_specific_k']}, "
            f"plateau={'YES' if res['method_specific_plateau'] else 'NO'}, "
            f"consensus={res['n_consensus_at_shared_k']}"
        )

    logger.info(f"")
    logger.info(f"  Cox consensus: {len(cox_genes)} genes")
    logger.info(f"  DeepSurv consensus: {len(ds_genes)} genes")
    logger.info(f"  Overlap: {len(overlap)} genes")
    logger.info(f"  Jaccard: {jaccard:.3f}")
    logger.info(f"{'='*70}")

    # Save summary
    summary = {
        'extended_k_scan': K_SCAN_EXTENDED,
        'plateau_results': serializable,
        'gene_overlap': {
            'cox_n': len(cox_genes),
            'deepsurv_n': len(ds_genes),
            'overlap_n': len(overlap),
            'jaccard': jaccard,
            'overlap_genes': sorted(overlap),
        },
    }
    with open(consensus_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nExtended consensus analysis complete. Results in {consensus_dir}")
