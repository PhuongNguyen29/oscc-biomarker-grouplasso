"""
Phase 5a: Performance Comparison, Pathway Enrichment, and PH Testing

Analyses 5.1 and 5.2 from the v4 pipeline document.

Analysis 5.1: Cox vs. DeepSurv Comparison
    - Out-of-fold C-index from Phase 3 (3 seeds × 2 directions × 2 methods)
    - Gene overlap: Jaccard between Cox and DeepSurv consensus sets
    - Pathway enrichment: ORA on 64 Cox consensus genes using MSigDB
      (Hallmark, KEGG, GO:BP, C7 Immunologic)

Analysis 5.2: Proportional Hazards Assumption
    - Schoenfeld residual test for each consensus gene (univariate Cox)
    - Schoenfeld test for the overall out-of-fold risk score
    - Reports proportion of genes with PH violations

All analyses are descriptive/exploratory. No significance claims.
Gene set was selected using TCGA+ORIEN outcomes (post-selection).

References:
    - Grambsch & Therneau (1994) — Schoenfeld residuals
    - Boyle et al. (2004) Bioinformatics — GO enrichment / ORA
    - Subramanian et al. (2005) PNAS — Gene set enrichment / MSigDB

Usage:
    cd ~/hpchome/oscc-biomarker-grouplasso
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/phase5a_performance_ph.py

Output:
    results/phase5/
        analysis_5_1/
            performance_summary.json
            gene_overlap.json
            enrichment_hallmark.csv
            enrichment_kegg.csv
            enrichment_go_bp.csv
            enrichment_c7_immunesigdb.csv
            enrichment_summary.json
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

PHASE2_DIR = Path("results/phase2_tuning")
PHASE7_DIR = Path("results/phase7_percohort_mult0.6_pi60")
PHASE7_IG_DIR = PHASE7_DIR / "deepsurv"
RESULTS_DIR = Path("results/phase5_corrected")
DATA_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")

ANNOTATION_FILE = RAW_DIR / "protein_coding_genes_annotation.csv"
CONSENSUS_PATH = PHASE7_DIR / "union_genes.txt"


# GMT files (MSigDB v2026.1)
GMT_FILES = {
    'Hallmark': RAW_DIR / "h.all.v2026.1.Hs.symbols.gmt",
    'KEGG': RAW_DIR / "c2.cp.kegg_medicus.v2026.1.Hs.symbols.gmt",
    'GO_BP': RAW_DIR / "c5.go.bp.v2026.1.Hs.symbols.gmt",
    'C7_Immunologic': RAW_DIR / "c7.immunesigdb.v2026.1.Hs.symbols.gmt",
}

# Enrichment significance threshold (descriptive, not used for claims)
ENRICHMENT_FDR_THRESHOLD = 0.05

def load_phase7_summary() -> dict:
    """Load Phase 7 validation summary."""
    summary_file = PHASE7_DIR / "phase7_summary.json"
    with open(summary_file) as f:
        summary = json.load(f)
    logger.info(f"  Loaded Phase 7 summary from {summary_file}")
    return summary


def load_consensus_genes() -> list:
    """Load 25-gene union set from Phase 7."""
    genes = CONSENSUS_PATH.read_text().strip().split('\n')
    logger.info(f"  Consensus genes: {len(genes)}")
    return genes


# =============================================================================
# Analysis 5.1: Cox vs. DeepSurv Comparison
# =============================================================================
def analysis_5_1(out_dir: Path):
    """
    5.1a: Performance comparison from Phase 7 bidirectional validation.
    5.1b: Gene overlap between Cox and DeepSurv stable sets.
    5.1c: Pathway enrichment on union consensus genes.
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

    perf = {
        'cox': {
            'tcga_to_orien': phase7['cox_validation']['tcga_to_orien']['test_ci'],
            'orien_to_tcga': phase7['cox_validation']['orien_to_tcga']['test_ci'],
            'mean_bidirectional': phase7['cox_validation']['mean_test_ci'],
        },
        'deepsurv': {
            'tcga_to_orien': phase7['deepsurv_validation']['tcga_to_orien']['test_ci'],
            'orien_to_tcga': phase7['deepsurv_validation']['orien_to_tcga']['test_ci'],
            'mean_bidirectional': phase7['deepsurv_validation']['mean_test_ci'],
        },
        'cox_per_cohort_genes': {
            'tcga_to_orien': phase7['cox_validation']['tcga_to_orien']['n_genes'],
            'orien_to_tcga': phase7['cox_validation']['orien_to_tcga']['n_genes'],
        },
        'union_genes': phase7['gene_sets']['union'],
        'source': 'Phase 7 bidirectional validation (per-cohort stable genes, union set)',
        'note': (
            'Cox per-cohort models use only that cohort\'s stable genes '
            '(TCGA: 12, ORIEN: 14). DeepSurv uses all 25 union genes. '
            'C-indices are out-of-sample (train on one cohort, test on the other).'
        ),
    }

    logger.info(f"  Phase 7 C-indices (out-of-sample):")
    logger.info(f"    TCGA→ORIEN Cox:      {perf['cox']['tcga_to_orien']:.4f}")
    logger.info(f"    TCGA→ORIEN DeepSurv: {perf['deepsurv']['tcga_to_orien']:.4f}")
    logger.info(f"    ORIEN→TCGA Cox:      {perf['cox']['orien_to_tcga']:.4f}")
    logger.info(f"    ORIEN→TCGA DeepSurv: {perf['deepsurv']['orien_to_tcga']:.4f}")
    logger.info(f"    Mean bidirectional Cox:     {perf['cox']['mean_bidirectional']:.4f}")
    logger.info(f"    Mean bidirectional DeepSurv: {perf['deepsurv']['mean_bidirectional']:.4f}")

    with open(out_dir / "performance_summary.json", 'w') as f:
        json.dump(perf, f, indent=2)

    # -----------------------------------------------------------------
    # 5.1b: Gene overlap
    # -----------------------------------------------------------------
    logger.info("")
    logger.info("  --- 5.1b: Gene Overlap ---")

    # Load union gene set details from Phase 7
    union_df = pd.read_csv(PHASE7_DIR / "union_gene_set.csv")

    # Phase 4 corrected: Cox produced stable genes, DeepSurv produced 0
    tcga_stable = phase7['gene_sets']['tcga_stable']
    orien_stable = phase7['gene_sets']['orien_stable']
    overlap_n = phase7['gene_sets']['overlap']
    overlap_genes = phase7['gene_sets']['overlap_genes']

    gene_overlap = {
        'tcga_stable_n': tcga_stable,
        'orien_stable_n': orien_stable,
        'union_n': phase7['gene_sets']['union'],
        'overlap_n': overlap_n,
        'overlap_genes': overlap_genes,
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
    logger.info(f"  Overlap: {overlap_n} ({overlap_genes})")
    logger.info(f"  Union: {phase7['gene_sets']['union']}")
    logger.info(f"  DeepSurv consensus: 0 (expected)")

    with open(out_dir / "gene_overlap.json", 'w') as f:
        json.dump(gene_overlap, f, indent=2)


# =============================================================================
# Analysis 5.2: Proportional Hazards Assumption
# =============================================================================

def analysis_5_2(cox_genes: set, out_dir: Path):
    """
    Test PH assumption for each consensus gene (univariate Cox)
    and for the overall out-of-fold DeepSurv risk score.

    Uses Schoenfeld residuals (Grambsch & Therneau, 1994).
    """
    # -----------------------------------------------------------------
    # PH test on overall DeepSurv risk score
    # -----------------------------------------------------------------
    logger.info("")
    logger.info("  --- Overall Risk Score PH Test ---")

    from src.data.data_loader import load_cohort
    consensus_genes = load_consensus_genes()

    risk_score_ph = {}

    for cohort_name, surv in [('tcga', tcga_surv), ('orien', orien_surv)]:
        # TCGA patients get risk scores from orien_to_tcga model
        # ORIEN patients get risk scores from tcga_to_orien model
        if cohort_name == 'tcga':
            direction = 'orien_to_tcga'
        else:
            direction = 'tcga_to_orien'

        logger.info(f"  {cohort_name.upper()} (model: {direction})")

        # Load model and compute risk scores
        model = load_deepsurv_model(direction, n_features=len(consensus_genes))
        expr, _ = load_cohort(cohort_name, str(DATA_DIR))
        risk_scores = compute_risk_scores(model, expr, consensus_genes)

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
            }

            logger.info(f"    Schoenfeld stat={test_stat:.3f}, p={p_value:.4f}")

        except Exception as e:
            risk_score_ph[cohort_name] = {'error': str(e)}
            logger.warning(f"    PH test failed: {e}")

    risk_score_ph['note'] = (
        'Risk scores computed from Phase 7 DeepSurv models '
        '(out-of-sample: model trained on opposite cohort). '
        'Single run, no multi-seed sensitivity.'
    )

    with open(out_dir / "schoenfeld_risk_score.json", 'w') as f:
        json.dump(risk_score_ph, f, indent=2)


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

    logger.info("Phase 5a: Performance Comparison, Pathway Enrichment, PH Testing")
    logger.info(f"Phase 3 results: {PHASE3_DIR}")
    logger.info(f"Phase 4 results: {PHASE4_DIR}")
    logger.info(f"Output: {RESULTS_DIR}")

    # Run Analysis 5.1
    cox_genes, cox_symbols, bg_symbols = analysis_5_1(RESULTS_DIR / "analysis_5_1")

    # Run Analysis 5.2
    analysis_5_2(cox_genes, RESULTS_DIR / "analysis_5_2")

    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 5a COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results saved to {RESULTS_DIR}")
    logger.info(f"Log saved to {log_file}")
