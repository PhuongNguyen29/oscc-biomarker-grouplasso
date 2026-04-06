"""
Phase 5b: IG Sign Concordance and Stratified Survival Analysis

Analyses 5.3 and 5.5 from the v4 pipeline document.

Analysis 5.3: Gene-Level Sign Concordance Between Cohorts
    For each consensus gene:
      - Direction 1 (TCGA-trained): proportion of ORIEN patients with positive IG
      - Direction 2 (ORIEN-trained): proportion of TCGA patients with positive IG
    Compare proportions. Similar values = consistent effect direction across
    independently trained models.

Analysis 5.5: IG-Based vs. Expression-Based Stratification (Exploratory)
    For each consensus gene × clinical stratum:
      - Method 1 (Expression): split by z-score > 0 vs <= 0
      - Method 2 (IG): split by IG sign (positive vs negative)
    KM + log-rank within each stratum. Minimum 5 events per group.
    Concordance between methods is the primary descriptive outcome.

All analyses are descriptive/exploratory. No significance claims.
Gene set was selected using TCGA+ORIEN outcomes (post-selection).

References:
    - Sundararajan et al. (2017) ICML — Integrated Gradients
    - Kaplan & Meier (1958) JASA — KM estimator
    - Mantel (1966) Cancer Chemother Rep — Log-rank test

Usage:
    cd ~/hpchome/oscc-biomarker-grouplasso
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/phase5b_ig_analysis.py

Output:
    results/phase5/
        analysis_5_3/
            sign_concordance_per_gene.csv
            sign_concordance_summary.json
        analysis_5_5/
            stratified_expression.csv
            stratified_ig.csv
            method_concordance.json
            strata_summary.json
"""

import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import warnings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

PHASE7_DIR = Path("results/phase7_percohort_mult0.6_pi60/deepsurv")
RESULTS_DIR = Path("results/phase5")
DATA_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")

ANNOTATION_FILE = RAW_DIR / "protein_coding_genes_annotation.csv"

# Clinical strata definitions (v4: 12 strata = 6 variables × 2 levels)
CLINICAL_STRATA = {
    'age': {'column': 'age', 'type': 'continuous', 'threshold': 65,
            'labels': ['≤65', '>65']},
    'gender': {'column': 'gender', 'type': 'binary', 'values': [0, 1],
               'labels': ['Female', 'Male']},
    'smoking': {'column': 'smoking', 'type': 'categorical',
                'values': ['Never', 'Ever'], 'labels': ['Never', 'Ever']},
    'alcohol': {'column': 'alcohol', 'type': 'categorical',
                'values': ['Never', 'Ever'], 'labels': ['Never', 'Ever']},
    'N_stage': {'column': 'N_stage', 'type': 'categorical',
                'values': ['N0', 'N+'], 'labels': ['N0', 'N+']},
    'T_stage': {'column': 'T_stage', 'type': 'categorical',
                'values': ['T1-T2', 'T3-T4'], 'labels': ['T1-T2', 'T3-T4']},
}

MIN_EVENTS_PER_GROUP = 5


# =============================================================================
# Helper Functions
# =============================================================================

def load_clinical(cohort_name: str) -> pd.DataFrame:
    """Load clinical data and encode N_stage as N0 vs N+."""
    clin = pd.read_csv(DATA_DIR / f"clinical_{cohort_name}_aligned_v4.csv")
    clin = clin.set_index('sampleID')

    # Collapse N1, N2-3 into N+
    clin['N_stage'] = clin['N_stage'].replace({'N1': 'N+', 'N2-3': 'N+'})

    return clin


def load_ig_attributions(direction: str) -> pd.DataFrame:
    """Load out-of-sample IG attributions from Phase 7 bidirectional validation.
    
    Args:
        direction: 'tcga_to_orien' or 'orien_to_tcga'
        
    Returns:
        DataFrame with sampleID as index, genes as columns.
    """
    ig_file = PHASE7_IG_DIR / f"deepsurv_ig_{direction}_test.csv"
    ig = pd.read_csv(ig_file, index_col=0)
    logger.info(f"  Loaded IG attributions: {ig_file.name} "
                f"({ig.shape[0]} patients × {ig.shape[1]} genes)")
    return ig


def load_survival(cohort_name: str) -> pd.DataFrame:
    """Load survival data."""
    surv = pd.read_csv(DATA_DIR / f"surv_{cohort_name}_aligned_v4.csv")
    surv = surv.set_index('sampleID')
    if surv.columns[0] == '' or 'Unnamed: 0' in str(surv.columns[0]):
        surv = surv.drop(columns=surv.columns[0])
    return surv


def load_expression(cohort_name: str) -> pd.DataFrame:
    """Load preprocessed expression data (genes × samples)."""
    expr = pd.read_csv(DATA_DIR / f"{cohort_name}_preprocessed_v4.csv",
                       index_col=0)
    return expr


def stratify_patients(clinical_df: pd.DataFrame, stratum_name: str):
    """
    Split patients into two groups based on a clinical variable.
    Returns (group0_mask, group1_mask, group0_label, group1_label, n_excluded).
    Patients with Unknown values are excluded.
    """
    spec = CLINICAL_STRATA[stratum_name]
    col = spec['column']
    labels = spec['labels']

    if spec['type'] == 'continuous':
        threshold = spec['threshold']
        valid = clinical_df[col].notna()
        group0 = valid & (clinical_df[col] <= threshold)
        group1 = valid & (clinical_df[col] > threshold)
    elif spec['type'] == 'binary':
        vals = spec['values']
        valid = clinical_df[col].isin(vals)
        group0 = valid & (clinical_df[col] == vals[0])
        group1 = valid & (clinical_df[col] == vals[1])
    elif spec['type'] == 'categorical':
        vals = spec['values']
        # Exclude Unknown
        valid = clinical_df[col].isin(vals)
        group0 = valid & (clinical_df[col] == vals[0])
        group1 = valid & (clinical_df[col] == vals[1])

    n_excluded = (~valid).sum() if 'valid' in dir() else 0

    return group0, group1, labels[0], labels[1], int(n_excluded)


# =============================================================================
# Analysis 5.3: Gene-Level Sign Concordance
# =============================================================================

def analysis_5_3(consensus_genes: list, out_dir: Path):
    """
    For each consensus gene, compute:
      - Direction 1 (TCGA→ORIEN): proportion of ORIEN patients with IG > 0
      - Direction 2 (ORIEN→TCGA): proportion of TCGA patients with IG > 0

    Similar proportions = consistent effect direction across independently
    trained models.

    Uses out-of-sample IG attributions from Phase 7 bidirectional validation.
    Cross-cohort consistency (TCGA-trained vs ORIEN-trained) serves as the
    robustness check in place of the old multi-seed sensitivity analysis.
    """
    logger.info("=" * 70)
    logger.info("ANALYSIS 5.3: Gene-Level IG Sign Concordance")
    logger.info("=" * 70)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load gene annotation for symbols
    annot = pd.read_csv(ANNOTATION_FILE)
    ensg_to_symbol = dict(zip(annot['gene_id'], annot['gene_symbol']))

    # Direction 1: TCGA-trained model → ORIEN patients (out-of-sample)
    ig_t2o = load_ig_attributions('tcga_to_orien')
    # Direction 2: ORIEN-trained model → TCGA patients (out-of-sample)
    ig_o2t = load_ig_attributions('orien_to_tcga')

    gene_results = []

    for gene in consensus_genes:
        if gene not in ig_t2o.columns or gene not in ig_o2t.columns:
            logger.warning(f"  Gene {gene} not found in IG files, skipping")
            continue

        # Proportion with positive IG in each direction
        orien_positive = (ig_t2o[gene] > 0).mean()
        tcga_positive = (ig_o2t[gene] > 0).mean()

        # Absolute difference captures divergence between cohorts
        abs_diff = abs(orien_positive - tcga_positive)

        # Direction agreement: both > 0.5 or both < 0.5
        same_majority = (orien_positive > 0.5) == (tcga_positive > 0.5)

        symbol = ensg_to_symbol.get(gene, gene)

        gene_results.append({
            'gene': gene,
            'symbol': symbol,
            'orien_prop_positive': float(orien_positive),
            'tcga_prop_positive': float(tcga_positive),
            'abs_diff': float(abs_diff),
            'same_majority_direction': bool(same_majority),
            'orien_n': len(ig_t2o),
            'tcga_n': len(ig_o2t),
        })

    df = pd.DataFrame(gene_results)
    df.to_csv(out_dir / "sign_concordance_per_gene.csv", index=False)

    n_concordant = df['same_majority_direction'].sum()
    n_total = len(df)
    median_diff = df['abs_diff'].median()
    mean_orien = df['orien_prop_positive'].mean()
    mean_tcga = df['tcga_prop_positive'].mean()

    logger.info(f"  Genes with same majority direction: "
                f"{n_concordant}/{n_total} ({100*n_concordant/n_total:.1f}%)")
    logger.info(f"  Median |prop_diff| across genes: {median_diff:.3f}")
    logger.info(f"  Mean proportion positive (ORIEN patients): {mean_orien:.3f}")
    logger.info(f"  Mean proportion positive (TCGA patients): {mean_tcga:.3f}")

    summary = {
        'n_consensus_genes': int(n_total),
        'n_same_majority_direction': int(n_concordant),
        'proportion_concordant': float(n_concordant / n_total) if n_total > 0 else 0,
        'median_abs_diff': float(median_diff),
        'mean_abs_diff': float(df['abs_diff'].mean()),
        'mean_orien_prop_positive': float(mean_orien),
        'mean_tcga_prop_positive': float(mean_tcga),
        'ig_source': 'Phase 7 bidirectional validation (out-of-sample test splits)',
        'orien_n_patients': int(len(ig_t2o)),
        'tcga_n_patients': int(len(ig_o2t)),
        'note': (
            'Proportions near 0.5 indicate ~50% of patients have positive IG for that gene. '
            'This reflects patient-level heterogeneity (z-scored expression, mean baseline), '
            'not necessarily noise. Same_majority_direction indicates whether the majority '
            'sign is consistent across independently trained models.'
        ),
    }

    with open(out_dir / "sign_concordance_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("")
    logger.info("  Analysis 5.3 complete.")

    return df

# =============================================================================
# Analysis 5.5: IG-Based vs Expression-Based Stratification
# =============================================================================

def analysis_5_5(consensus_genes: list, out_dir: Path):
    """
    For each consensus gene × clinical stratum:
      Method 1: Split by expression (z-score > 0 vs <= 0)
      Method 2: Split by IG sign (positive vs negative)

    Run KM + log-rank within each stratum.
    Minimum 5 events per group enforced.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("ANALYSIS 5.5: IG-Based vs Expression-Based Stratification")
    logger.info("=" * 70)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load gene annotation
    annot = pd.read_csv(ANNOTATION_FILE)
    ensg_to_symbol = dict(zip(annot['gene_id'], annot['gene_symbol']))

    # Load data for both cohorts
    cohort_data = {}
    for cohort_name in ['tcga', 'orien']:
        expr = load_expression(cohort_name)
        surv = load_survival(cohort_name)
        clin = load_clinical(cohort_name)

        # Determine IG direction (TCGA patients get IG from orien_to_tcga)
        if cohort_name == 'tcga':
            ig_direction = 'orien_to_tcga'
        else:
            ig_direction = 'tcga_to_orien'

        ig = load_ig_attributions(ig_direction)

        # Align all data by sample ID
        common_ids = (expr.columns
                      .intersection(surv.index)
                      .intersection(clin.index)
                      .intersection(ig.index))

        cohort_data[cohort_name] = {
            'expr': expr[common_ids],       # genes × samples
            'surv': surv.loc[common_ids],
            'clin': clin.loc[common_ids],
            'ig': ig.loc[common_ids],
            'n': len(common_ids),
        }

        logger.info(f"  {cohort_name.upper()}: {len(common_ids)} patients aligned")

    # Run stratified analysis
    expr_results = []
    ig_results = []

    for cohort_name, data in cohort_data.items():
        logger.info(f"")
        logger.info(f"  --- {cohort_name.upper()} ---")

        for stratum_name, spec in CLINICAL_STRATA.items():
            group0, group1, label0, label1, n_excluded = stratify_patients(
                data['clin'], stratum_name
            )

            if n_excluded > 0:
                logger.info(f"    {stratum_name}: {n_excluded} patients excluded (Unknown)")

            # Process each group separately
            for group_mask, group_label in [(group0, label0), (group1, label1)]:
                group_ids = data['clin'].index[group_mask]

                if len(group_ids) < 10:
                    logger.info(f"    {stratum_name}/{group_label}: "
                                f"too few patients ({len(group_ids)}), skipping")
                    continue

                surv_group = data['surv'].loc[group_ids]
                n_events = int(surv_group['event'].sum())

                for gene in consensus_genes:
                    symbol = ensg_to_symbol.get(gene, gene)

                    # ---- Method 1: Expression-based split ----
                    expr_vals = data['expr'].loc[gene, group_ids]
                    expr_high = expr_vals > 0  # z-score > 0 = above mean
                    expr_low = ~expr_high

                    n_high_events = int(surv_group.loc[expr_high[expr_high].index, 'event'].sum())
                    n_low_events = int(surv_group.loc[expr_low[expr_low].index, 'event'].sum())

                    if n_high_events >= MIN_EVENTS_PER_GROUP and n_low_events >= MIN_EVENTS_PER_GROUP:
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                lr = logrank_test(
                                    surv_group.loc[expr_high[expr_high].index, 'time'],
                                    surv_group.loc[expr_low[expr_low].index, 'time'],
                                    surv_group.loc[expr_high[expr_high].index, 'event'],
                                    surv_group.loc[expr_low[expr_low].index, 'event'],
                                )

                            # Compute HR via univariate Cox on this subgroup
                            sub_df = pd.DataFrame({
                                'high_expr': expr_high.astype(float),
                                'time': surv_group['time'],
                                'event': surv_group['event'],
                            }).loc[group_ids]
                            sub_df = sub_df[sub_df['time'] > 0]

                            cph = CoxPHFitter()
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                cph.fit(sub_df, duration_col='time', event_col='event')

                            expr_results.append({
                                'cohort': cohort_name,
                                'stratum': stratum_name,
                                'stratum_level': group_label,
                                'gene': gene,
                                'symbol': symbol,
                                'n_patients': int(group_mask.sum()),
                                'n_high': int(expr_high.sum()),
                                'n_low': int(expr_low.sum()),
                                'n_high_events': n_high_events,
                                'n_low_events': n_low_events,
                                'logrank_stat': float(lr.test_statistic),
                                'logrank_p': float(lr.p_value),
                                'hr': float(cph.hazard_ratios_.iloc[0]),
                                'hr_lower': float(cph.confidence_intervals_.iloc[0, 0]),
                                'hr_upper': float(cph.confidence_intervals_.iloc[0, 1]),
                                'sufficient_events': True,
                            })

                        except Exception as e:
                            expr_results.append({
                                'cohort': cohort_name,
                                'stratum': stratum_name,
                                'stratum_level': group_label,
                                'gene': gene,
                                'symbol': symbol,
                                'n_patients': int(group_mask.sum()),
                                'sufficient_events': True,
                                'error': str(e),
                            })
                    else:
                        expr_results.append({
                            'cohort': cohort_name,
                            'stratum': stratum_name,
                            'stratum_level': group_label,
                            'gene': gene,
                            'symbol': symbol,
                            'n_patients': int(group_mask.sum()),
                            'n_high_events': n_high_events,
                            'n_low_events': n_low_events,
                            'sufficient_events': False,
                        })

                    # ---- Method 2: IG-based split ----
                    ig_vals = data['ig'].loc[group_ids, gene]
                    ig_pos = ig_vals > 0
                    ig_neg = ~ig_pos

                    n_pos_events = int(surv_group.loc[ig_pos[ig_pos].index, 'event'].sum())
                    n_neg_events = int(surv_group.loc[ig_neg[ig_neg].index, 'event'].sum())

                    if n_pos_events >= MIN_EVENTS_PER_GROUP and n_neg_events >= MIN_EVENTS_PER_GROUP:
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                lr = logrank_test(
                                    surv_group.loc[ig_pos[ig_pos].index, 'time'],
                                    surv_group.loc[ig_neg[ig_neg].index, 'time'],
                                    surv_group.loc[ig_pos[ig_pos].index, 'event'],
                                    surv_group.loc[ig_neg[ig_neg].index, 'event'],
                                )

                            sub_df = pd.DataFrame({
                                'ig_positive': ig_pos.astype(float),
                                'time': surv_group['time'],
                                'event': surv_group['event'],
                            }).loc[group_ids]
                            sub_df = sub_df[sub_df['time'] > 0]

                            cph = CoxPHFitter()
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                cph.fit(sub_df, duration_col='time', event_col='event')

                            ig_results.append({
                                'cohort': cohort_name,
                                'stratum': stratum_name,
                                'stratum_level': group_label,
                                'gene': gene,
                                'symbol': symbol,
                                'n_patients': int(group_mask.sum()),
                                'n_pos': int(ig_pos.sum()),
                                'n_neg': int(ig_neg.sum()),
                                'n_pos_events': n_pos_events,
                                'n_neg_events': n_neg_events,
                                'logrank_stat': float(lr.test_statistic),
                                'logrank_p': float(lr.p_value),
                                'hr': float(cph.hazard_ratios_.iloc[0]),
                                'hr_lower': float(cph.confidence_intervals_.iloc[0, 0]),
                                'hr_upper': float(cph.confidence_intervals_.iloc[0, 1]),
                                'sufficient_events': True,
                            })

                        except Exception as e:
                            ig_results.append({
                                'cohort': cohort_name,
                                'stratum': stratum_name,
                                'stratum_level': group_label,
                                'gene': gene,
                                'symbol': symbol,
                                'n_patients': int(group_mask.sum()),
                                'sufficient_events': True,
                                'error': str(e),
                            })
                    else:
                        ig_results.append({
                            'cohort': cohort_name,
                            'stratum': stratum_name,
                            'stratum_level': group_label,
                            'gene': gene,
                            'symbol': symbol,
                            'n_patients': int(group_mask.sum()),
                            'n_pos_events': n_pos_events,
                            'n_neg_events': n_neg_events,
                            'sufficient_events': False,
                        })

    # Save raw results
    expr_df = pd.DataFrame(expr_results)
    ig_df = pd.DataFrame(ig_results)

    expr_df.to_csv(out_dir / "stratified_expression.csv", index=False)
    ig_df.to_csv(out_dir / "stratified_ig.csv", index=False)

    # -----------------------------------------------------------------
    # Method concordance analysis
    # -----------------------------------------------------------------
    logger.info("")
    logger.info("  --- Method Concordance ---")

    # For tests where both methods have sufficient events, compare HR directions
    concordance_results = {}

    for cohort_name in ['tcga', 'orien']:
        expr_sub = expr_df[(expr_df['cohort'] == cohort_name) &
                           (expr_df['sufficient_events'] == True) &
                           (expr_df['hr'].notna())]
        ig_sub = ig_df[(ig_df['cohort'] == cohort_name) &
                       (ig_df['sufficient_events'] == True) &
                       (ig_df['hr'].notna())]

        # Merge on gene × stratum × level
        merged = expr_sub.merge(
            ig_sub,
            on=['cohort', 'stratum', 'stratum_level', 'gene', 'symbol'],
            suffixes=('_expr', '_ig'),
            how='inner',
        )

        if len(merged) > 0:
            # HR direction concordance: both > 1 or both < 1
            same_dir = ((merged['hr_expr'] > 1) == (merged['hr_ig'] > 1))
            n_same = same_dir.sum()
            n_total = len(merged)

            # Correlation of log-HR
            log_hr_corr = np.corrcoef(
                np.log(merged['hr_expr'].clip(1e-6)),
                np.log(merged['hr_ig'].clip(1e-6))
            )[0, 1]

            concordance_results[cohort_name] = {
                'n_testable_pairs': int(n_total),
                'n_same_hr_direction': int(n_same),
                'proportion_concordant': float(n_same / n_total) if n_total > 0 else 0,
                'log_hr_correlation': float(log_hr_corr) if not np.isnan(log_hr_corr) else None,
            }

            logger.info(f"  {cohort_name.upper()}: {n_same}/{n_total} "
                        f"({100*n_same/n_total:.1f}%) concordant HR direction, "
                        f"log-HR corr={log_hr_corr:.3f}")
        else:
            concordance_results[cohort_name] = {
                'n_testable_pairs': 0,
                'note': 'No gene-stratum pairs with sufficient events in both methods',
            }

    with open(out_dir / "method_concordance.json", 'w') as f:
        json.dump(concordance_results, f, indent=2)

    # -----------------------------------------------------------------
    # Strata summary
    # -----------------------------------------------------------------
    strata_summary = {
        'expression_based': {
            'total_tests': len(expr_df),
            'sufficient_events': int(expr_df['sufficient_events'].sum()),
            'insufficient_events': int((~expr_df['sufficient_events']).sum()),
        },
        'ig_based': {
            'total_tests': len(ig_df),
            'sufficient_events': int(ig_df['sufficient_events'].sum()),
            'insufficient_events': int((~ig_df['sufficient_events']).sum()),
        },
        'min_events_threshold': MIN_EVENTS_PER_GROUP,
        'clinical_strata': list(CLINICAL_STRATA.keys()),
        'n_consensus_genes': len(consensus_genes),
        'primary_seed': PRIMARY_SEED,
        'method': 'KM + log-rank per gene per stratum',
        'expression_split': 'z-score > 0 (above population mean) vs <= 0',
        'ig_split': 'IG attribution > 0 (positive) vs <= 0 (negative)',
        'note': (
            'All analyses are descriptive. Gene set was selected using '
            'TCGA+ORIEN outcomes (post-selection inference). '
            'Strata with Unknown clinical values are excluded per-variable.'
        ),
    }

    # Per-stratum event counts
    for cohort_name in ['tcga', 'orien']:
        strata_summary[f'{cohort_name}_strata'] = {}
        for stratum_name in CLINICAL_STRATA:
            clin = cohort_data[cohort_name]['clin']
            surv = cohort_data[cohort_name]['surv']
            g0, g1, l0, l1, n_excl = stratify_patients(clin, stratum_name)
            strata_summary[f'{cohort_name}_strata'][stratum_name] = {
                f'{l0}_n': int(g0.sum()),
                f'{l0}_events': int(surv.loc[clin.index[g0], 'event'].sum()),
                f'{l1}_n': int(g1.sum()),
                f'{l1}_events': int(surv.loc[clin.index[g1], 'event'].sum()),
                'n_excluded': n_excl,
            }

    with open(out_dir / "strata_summary.json", 'w') as f:
        json.dump(strata_summary, f, indent=2)

    logger.info("")
    logger.info(f"  Expression-based: {expr_df['sufficient_events'].sum()}/{len(expr_df)} "
                f"tests with sufficient events")
    logger.info(f"  IG-based: {ig_df['sufficient_events'].sum()}/{len(ig_df)} "
                f"tests with sufficient events")
    logger.info("")
    logger.info("  Analysis 5.5 complete.")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Log to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = RESULTS_DIR / f"phase5b_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(file_handler)

    logger.info("Phase 5b: IG Sign Concordance and Stratified Analysis")
    logger.info(f"Phase 3 results: {PHASE3_DIR}")
    logger.info(f"Phase 4 results: {PHASE4_DIR}")
    logger.info(f"Output: {RESULTS_DIR}")

    # Load consensus genes
    consensus_file = PHASE4_DIR / "consensus" / "cox_consensus_pi080.txt"
    consensus_genes = consensus_file.read_text().strip().split('\n')
    logger.info(f"Consensus genes: {len(consensus_genes)}")

    # Run Analysis 5.3
    concordance_df = analysis_5_3(consensus_genes, RESULTS_DIR / "analysis_5_3")

    # Run Analysis 5.5
    analysis_5_5(consensus_genes, RESULTS_DIR / "analysis_5_5")

    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 5b COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results saved to {RESULTS_DIR}")
    logger.info(f"Log saved to {log_file}")
