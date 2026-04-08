"""
Phase 5a: Performance Comparison and PH Testing (Corrected Pipeline)

Analyses 5.1 and 5.2 for the 25-gene union set.

Analysis 5.1: Cox vs. DeepSurv Comparison
    - Phase 7 bidirectional C-indices (2 directions × 2 methods)
    - Gene overlap between per-cohort Cox stable sets
    - DeepSurv produced 0 consensus genes (expected)

Analysis 5.2: Proportional Hazards Assumption
    5.2a: Schoenfeld residual test for each consensus gene (univariate Cox)
    5.2b: Schoenfeld test for the overall out-of-sample DeepSurv risk score

All analyses are descriptive/exploratory. No significance claims.
Gene set was selected using TCGA+ORIEN outcomes (post-selection).

References:
    - Grambsch & Therneau (1994) — Schoenfeld residuals
    - Katzman et al. (2018) — DeepSurv

Usage:
    cd ~/hpchome/oscc-biomarker-grouplasso
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/phase5a_performance_ph.py

Output:
    results/phase5_corrected/
        analysis_5_1/
            performance_summary.json
            gene_overlap.json
        analysis_5_2/
            schoenfeld_per_gene_tcga.csv
            schoenfeld_per_gene_orien.csv
            schoenfeld_risk_score.json
            ph_summary.json
"""

import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
import warnings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

SWEEP_DIR = Path("results/sensitivity_sweep/mult0.6_pi60")
PHASE7_DIR = SWEEP_DIR / "validation"
DEEPSURV_DIR = PHASE7_DIR / "deepsurv"

RESULTS_DIR = Path("results/phase5_corrected")
DATA_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")

ANNOTATION_FILE = RAW_DIR / "protein_coding_genes_annotation.csv"
CONSENSUS_PATH = SWEEP_DIR / "union_genes.txt"

# IG attribution paths (out-of-sample test splits from Phase 7)
IG_PATHS = {
    'tcga_to_orien': DEEPSURV_DIR / "deepsurv_ig_tcga_to_orien_test.csv",
    'orien_to_tcga': DEEPSURV_DIR / "deepsurv_ig_orien_to_tcga_test.csv",
}


def load_phase7_summary() -> dict:
    """Load Phase 7 validation summary."""
    summary_file = PHASE7_DIR / "phase7_summary.json"
    with open(summary_file) as f:
        summary = json.load(f)
    logger.info(f"  Loaded Phase 7 summary from {summary_file}")
    return summary


def load_consensus_genes() -> list:
    """Load 25-gene union set."""
    genes = CONSENSUS_PATH.read_text().strip().split('\n')
    logger.info(f"  Consensus genes: {len(genes)}")
    return genes


def load_ig_risk_scores(direction: str) -> pd.Series:
    """
    Derive risk scores from saved IG attributions by summing across genes.

    By the completeness axiom of Integrated Gradients (Sundararajan et al.,
    2017), sum of IG attributions equals f(x) - f(baseline). With a zero
    baseline, this equals the model's log-hazard output. Since the PH test
    only uses ranks, adding a constant (the baseline output) does not affect
    the Schoenfeld residual test.

    Args:
        direction: 'tcga_to_orien' or 'orien_to_tcga'

    Returns:
        pd.Series with sampleID index, IG-sum (≈ log-hazard) as values
    """
    ig_file = IG_PATHS[direction]
    ig_df = pd.read_csv(ig_file, index_col=0)
    risk_scores = ig_df.sum(axis=1)
    risk_scores.name = 'risk_score'
    logger.info(f"  Loaded IG risk scores from {ig_file.name} "
                f"({len(risk_scores)} patients)")
    return risk_scores


# =============================================================================
# Analysis 5.1: Cox vs. DeepSurv Comparison
# =============================================================================

def analysis_5_1(out_dir: Path) -> list:
    """
    5.1a: Performance comparison from Phase 7 bidirectional validation.
    5.1b: Gene overlap between per-cohort Cox stable sets.

    Returns:
        consensus_genes: list of 25 union gene IDs
    """
    logger.info("=" * 70)
    logger.info("ANALYSIS 5.1: Cox vs. DeepSurv Comparison")
    logger.info("=" * 70)

    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # 5.1a: Performance summary from Phase 7
    # -----------------------------------------------------------------
    logger.info("")
    logger.info("  --- 5.1a: Performance Summary ---")

    phase7 = load_phase7_summary()

    # The summary JSON uses 'cox_percohort', 'cox_union', 'deepsurv_union'
    # Cox per-cohort uses each cohort's own stable genes
    # Cox union and DeepSurv union both use the full 25-gene union set

    perf = {
        'cox_percohort': {
            'tcga_to_orien': phase7['cox_percohort']['tcga_to_orien']['test_ci'],
            'orien_to_tcga': phase7['cox_percohort']['orien_to_tcga']['test_ci'],
            'mean_bidirectional': phase7['cox_percohort']['mean_test_ci'],
        },
        'cox_union': {
            'tcga_to_orien': phase7['cox_union']['tcga_to_orien']['test_ci'],
            'orien_to_tcga': phase7['cox_union']['orien_to_tcga']['test_ci'],
            'mean_bidirectional': phase7['cox_union']['mean_test_ci'],
        },
        'deepsurv_union': {
            'tcga_to_orien': phase7['deepsurv_union']['tcga_to_orien']['test_ci'],
            'orien_to_tcga': phase7['deepsurv_union']['orien_to_tcga']['test_ci'],
            'mean_bidirectional': phase7['deepsurv_union']['mean_test_ci'],
        },
        'gene_sets': {
            'tcga_stable': phase7['gene_sets']['tcga_stable'],
            'orien_stable': phase7['gene_sets']['orien_stable'],
            'union': phase7['gene_sets']['union'],
            'overlap': phase7['gene_sets']['overlap'],
        },
        'source': 'Phase 7 bidirectional validation (per-cohort stable genes, union set)',
        'note': (
            'Cox per-cohort models use only that cohort\'s stable genes '
            '(TCGA: 12, ORIEN: 14). Cox union and DeepSurv use all 25 union genes. '
            'C-indices are out-of-sample (train on one cohort, test on the other).'
        ),
    }

    logger.info(f"  Phase 7 C-indices (out-of-sample):")
    logger.info(f"    Cox per-cohort:  T→O={perf['cox_percohort']['tcga_to_orien']:.4f}, "
                f"O→T={perf['cox_percohort']['orien_to_tcga']:.4f}, "
                f"mean={perf['cox_percohort']['mean_bidirectional']:.4f}")
    logger.info(f"    Cox union:       T→O={perf['cox_union']['tcga_to_orien']:.4f}, "
                f"O→T={perf['cox_union']['orien_to_tcga']:.4f}, "
                f"mean={perf['cox_union']['mean_bidirectional']:.4f}")
    logger.info(f"    DeepSurv union:  T→O={perf['deepsurv_union']['tcga_to_orien']:.4f}, "
                f"O→T={perf['deepsurv_union']['orien_to_tcga']:.4f}, "
                f"mean={perf['deepsurv_union']['mean_bidirectional']:.4f}")

    with open(out_dir / "performance_summary.json", 'w') as f:
        json.dump(perf, f, indent=2)

    # -----------------------------------------------------------------
    # 5.1b: Gene overlap
    # -----------------------------------------------------------------
    logger.info("")
    logger.info("  --- 5.1b: Gene Overlap ---")

    union_df = pd.read_csv(SWEEP_DIR / "union_gene_set.csv")

    tcga_stable = phase7['gene_sets']['tcga_stable']
    orien_stable = phase7['gene_sets']['orien_stable']
    overlap_n = phase7['gene_sets']['overlap']

    gene_overlap = {
        'tcga_stable_n': tcga_stable,
        'orien_stable_n': orien_stable,
        'union_n': phase7['gene_sets']['union'],
        'overlap_n': overlap_n,
        'deepsurv_consensus_n': 0,
        'deepsurv_note': (
            'Corrected Phase 4 (randomized lasso, l1_ratio=1.0) produced 0 '
            'DeepSurv consensus genes. This is expected — DeepSurv stability '
            'selection instability is a model property, not a bug. '
            'Cox is used for gene selection; DeepSurv for patient-level interpretation.'
        ),
    }

    logger.info(f"  TCGA stable genes: {tcga_stable}")
    logger.info(f"  ORIEN stable genes: {orien_stable}")
    logger.info(f"  Overlap: {overlap_n}")
    logger.info(f"  Union: {phase7['gene_sets']['union']}")
    logger.info(f"  DeepSurv consensus: 0 (expected)")

    with open(out_dir / "gene_overlap.json", 'w') as f:
        json.dump(gene_overlap, f, indent=2)

    # Return consensus gene list for downstream analyses
    consensus_genes = load_consensus_genes()
    return consensus_genes


# =============================================================================
# Analysis 5.2: Proportional Hazards Assumption
# =============================================================================

def analysis_5_2(consensus_genes: list, out_dir: Path):
    """
    Test PH assumption using Schoenfeld residuals (Grambsch & Therneau, 1994).

    5.2a: Per-gene univariate Cox PH test for each consensus gene.
    5.2b: Overall DeepSurv risk score PH test (out-of-sample).
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("ANALYSIS 5.2: Proportional Hazards Assumption")
    logger.info("=" * 70)

    out_dir.mkdir(parents=True, exist_ok=True)

    from src.data.data_loader import load_cohort

    # Load annotation for gene symbols
    annot = pd.read_csv(ANNOTATION_FILE)
    ensg_to_symbol = dict(zip(annot['gene_id'], annot['gene_symbol']))

    # Load data for both cohorts
    cohort_data = {}
    for cohort_name in ['tcga', 'orien']:
        expr, surv = load_cohort(cohort_name, str(DATA_DIR))
        cohort_data[cohort_name] = {'expr': expr, 'surv': surv}
        logger.info(f"  Loaded {cohort_name.upper()}: {expr.shape[1]} samples, "
                    f"{int(surv['event'].sum())} events")

    # -----------------------------------------------------------------
    # 5.2a: Per-gene Schoenfeld test (univariate Cox)
    # -----------------------------------------------------------------
    logger.info("")
    logger.info("  --- 5.2a: Per-Gene PH Test ---")

    per_gene_results = {}

    for cohort_name in ['tcga', 'orien']:
        expr = cohort_data[cohort_name]['expr']
        surv = cohort_data[cohort_name]['surv']

        # Align samples
        common = sorted(set(expr.columns) & set(surv.index))
        surv_aligned = surv.loc[common]

        gene_results = []

        for gene in consensus_genes:
            symbol = ensg_to_symbol.get(gene, gene)

            if gene not in expr.index:
                logger.warning(f"    {gene} ({symbol}) not in {cohort_name.upper()} expression")
                continue

            # Build univariate Cox dataframe
            gene_vals = expr.loc[gene, common].values.astype(float)
            df = pd.DataFrame({
                'gene_expr': gene_vals,
                'time': surv_aligned['time'].values,
                'event': surv_aligned['event'].values,
            })
            df = df[df['time'] > 0]

            try:
                cph = CoxPHFitter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cph.fit(df, duration_col='time', event_col='event')

                ph_test = proportional_hazard_test(cph, df, time_transform='rank')
                test_stat = float(ph_test.summary['test_statistic'].iloc[0])
                p_value = float(ph_test.summary['p'].iloc[0])

                gene_results.append({
                    'gene': gene,
                    'symbol': symbol,
                    'schoenfeld_stat': test_stat,
                    'p_value': p_value,
                    'n_samples': len(df),
                    'n_events': int(df['event'].sum()),
                    'coef': float(cph.summary['coef'].iloc[0]),
                    'hr': float(cph.summary['exp(coef)'].iloc[0]),
                    'ph_violated_005': p_value < 0.05,
                })

            except Exception as e:
                gene_results.append({
                    'gene': gene,
                    'symbol': symbol,
                    'error': str(e),
                    'ph_violated_005': None,
                })
                logger.warning(f"    PH test failed for {symbol} in {cohort_name.upper()}: {e}")

        gene_df = pd.DataFrame(gene_results)
        gene_df.to_csv(out_dir / f"schoenfeld_per_gene_{cohort_name}.csv", index=False)
        per_gene_results[cohort_name] = gene_df

        n_violated = gene_df['ph_violated_005'].sum() if 'ph_violated_005' in gene_df else 0
        n_tested = gene_df['ph_violated_005'].notna().sum()
        logger.info(f"  {cohort_name.upper()}: {n_violated}/{n_tested} genes "
                    f"with PH violation (p < 0.05)")

    # -----------------------------------------------------------------
    # 5.2b: Overall DeepSurv risk score PH test
    # -----------------------------------------------------------------
    logger.info("")
    logger.info("  --- 5.2b: Overall Risk Score PH Test ---")

    risk_score_ph = {}

    for cohort_name in ['tcga', 'orien']:
        # Out-of-sample: TCGA patients tested with ORIEN-trained model
        if cohort_name == 'tcga':
            direction = 'orien_to_tcga'
        else:
            direction = 'tcga_to_orien'

        logger.info(f"  {cohort_name.upper()} (model: {direction})")

        surv = cohort_data[cohort_name]['surv']

        # Derive risk scores from saved IG attributions (sum across genes)
        risk_scores = load_ig_risk_scores(direction)

        # Align with survival
        common = risk_scores.index.intersection(surv.index)
        rs_df = pd.DataFrame({
            'risk_score': risk_scores.loc[common].values,
            'time': surv.loc[common, 'time'].values,
            'event': surv.loc[common, 'event'].values,
        })
        rs_df = rs_df[rs_df['time'] > 0]

        try:
            cph = CoxPHFitter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph.fit(rs_df, duration_col='time', event_col='event')

            ph_test = proportional_hazard_test(cph, rs_df, time_transform='rank')
            test_stat = float(ph_test.summary['test_statistic'].iloc[0])
            p_value = float(ph_test.summary['p'].iloc[0])

            risk_score_ph[cohort_name] = {
                'schoenfeld_stat': test_stat,
                'p_value': p_value,
                'n_samples': len(rs_df),
                'n_events': int(rs_df['event'].sum()),
                'model_direction': direction,
                'ph_violated_005': p_value < 0.05,
            }

            logger.info(f"    Schoenfeld stat={test_stat:.3f}, p={p_value:.4f}")

        except Exception as e:
            risk_score_ph[cohort_name] = {'error': str(e)}
            logger.warning(f"    PH test failed: {e}")

    risk_score_ph['note'] = (
        'Risk scores derived from Phase 7 IG attributions (sum across genes '
        '≈ log-hazard by IG completeness axiom). Out-of-sample: model trained '
        'on opposite cohort. Single run, no multi-seed sensitivity.'
    )

    with open(out_dir / "schoenfeld_risk_score.json", 'w') as f:
        json.dump(risk_score_ph, f, indent=2)

    # -----------------------------------------------------------------
    # Combined PH summary
    # -----------------------------------------------------------------
    logger.info("")
    logger.info("  --- PH Summary ---")

    ph_summary = {
        'per_gene': {},
        'risk_score': {},
        'note': (
            'All PH tests are descriptive. Gene set was selected using '
            'these cohorts\' outcomes (post-selection inference). '
            'PH violations at p<0.05 may reflect time-varying effects.'
        ),
    }

    for cohort_name in ['tcga', 'orien']:
        gene_df = per_gene_results[cohort_name]
        n_tested = gene_df['ph_violated_005'].notna().sum()
        n_violated = int(gene_df['ph_violated_005'].sum())
        ph_summary['per_gene'][cohort_name] = {
            'n_tested': int(n_tested),
            'n_violated_005': n_violated,
            'proportion_violated': float(n_violated / n_tested) if n_tested > 0 else 0,
            'violated_genes': gene_df.loc[
                gene_df['ph_violated_005'] == True, 'symbol'
            ].tolist(),
        }

        if cohort_name in risk_score_ph and 'p_value' in risk_score_ph[cohort_name]:
            ph_summary['risk_score'][cohort_name] = {
                'p_value': risk_score_ph[cohort_name]['p_value'],
                'violated': risk_score_ph[cohort_name]['ph_violated_005'],
            }

        logger.info(f"  {cohort_name.upper()}: "
                    f"{n_violated}/{n_tested} genes with PH violation; "
                    f"risk score p={risk_score_ph.get(cohort_name, {}).get('p_value', 'N/A')}")

    with open(out_dir / "ph_summary.json", 'w') as f:
        json.dump(ph_summary, f, indent=2)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Log to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = RESULTS_DIR / f"phase5a_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(file_handler)

    logger.info("Phase 5a: Performance Comparison and PH Testing (Corrected Pipeline)")
    logger.info(f"Phase 7 results: {PHASE7_DIR}")
    logger.info(f"Consensus genes: {CONSENSUS_PATH}")
    logger.info(f"Output: {RESULTS_DIR}")

    # Run Analysis 5.1
    consensus_genes = analysis_5_1(RESULTS_DIR / "analysis_5_1")

    # Run Analysis 5.2
    analysis_5_2(consensus_genes, RESULTS_DIR / "analysis_5_2")

    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 5a COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results saved to {RESULTS_DIR}")
    logger.info(f"Log saved to {log_file}")
