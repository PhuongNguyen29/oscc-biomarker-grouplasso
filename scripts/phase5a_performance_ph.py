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

PHASE3_DIR = Path("results/phase3_cross_fitted")
PHASE4_DIR = Path("results/phase4_stability_mad3000")
RESULTS_DIR = Path("results/phase5")
DATA_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")

SEEDS = [42, 123, 456]
ANNOTATION_FILE = RAW_DIR / "protein_coding_genes_annotation.csv"

# GMT files (MSigDB v2026.1)
GMT_FILES = {
    'Hallmark': RAW_DIR / "h.all.v2026.1.Hs.symbols.gmt",
    'KEGG': RAW_DIR / "c2.cp.kegg_medicus.v2026.1.Hs.symbols.gmt",
    'GO_BP': RAW_DIR / "c5.go.bp.v2026.1.Hs.symbols.gmt",
    'C7_Immunologic': RAW_DIR / "c7.immunesigdb.v2026.1.Hs.symbols.gmt",
}

# Enrichment significance threshold (descriptive, not used for claims)
ENRICHMENT_FDR_THRESHOLD = 0.05


# =============================================================================
# Analysis 5.1: Cox vs. DeepSurv Comparison
# =============================================================================

def analysis_5_1(out_dir: Path):
    """
    5.1a: Performance comparison from Phase 3 cross-fitted C-indices.
    5.1b: Gene overlap between Cox and DeepSurv consensus sets.
    5.1c: Pathway enrichment on Cox consensus genes.
    """
    logger.info("=" * 70)
    logger.info("ANALYSIS 5.1: Cox vs. DeepSurv Comparison")
    logger.info("=" * 70)

    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # 5.1a: Performance summary
    # -----------------------------------------------------------------
    logger.info("")
    logger.info("  --- 5.1a: Performance Summary ---")

    summary_file = PHASE3_DIR / "overall_summary.json"
    with open(summary_file) as f:
        phase3_summary = json.load(f)

    # Extract C-indices
    perf = {}
    for key in ['tcga_to_orien_cox', 'tcga_to_orien_deepsurv',
                'orien_to_tcga_cox', 'orien_to_tcga_deepsurv']:
        info = phase3_summary[key]
        perf[key] = {
            'mean_cindex': info['mean_cindex'],
            'std_across_seeds': info['std_across_seeds'],
            'per_seed': info['per_seed'],
        }

    # Compute mean bidirectional C-index per method
    for method in ['cox', 'deepsurv']:
        t2o = perf[f'tcga_to_orien_{method}']['mean_cindex']
        o2t = perf[f'orien_to_tcga_{method}']['mean_cindex']
        perf[f'mean_bidirectional_{method}'] = (t2o + o2t) / 2

    logger.info(f"  Cross-fitted C-indices (mean across 3 seeds):")
    logger.info(f"    TCGA→ORIEN Cox:     {perf['tcga_to_orien_cox']['mean_cindex']:.4f} "
                f"± {perf['tcga_to_orien_cox']['std_across_seeds']:.4f}")
    logger.info(f"    TCGA→ORIEN DeepSurv: {perf['tcga_to_orien_deepsurv']['mean_cindex']:.4f} "
                f"± {perf['tcga_to_orien_deepsurv']['std_across_seeds']:.4f}")
    logger.info(f"    ORIEN→TCGA Cox:     {perf['orien_to_tcga_cox']['mean_cindex']:.4f} "
                f"± {perf['orien_to_tcga_cox']['std_across_seeds']:.4f}")
    logger.info(f"    ORIEN→TCGA DeepSurv: {perf['orien_to_tcga_deepsurv']['mean_cindex']:.4f} "
                f"± {perf['orien_to_tcga_deepsurv']['std_across_seeds']:.4f}")
    logger.info(f"    Mean bidirectional Cox:     {perf['mean_bidirectional_cox']:.4f}")
    logger.info(f"    Mean bidirectional DeepSurv: {perf['mean_bidirectional_deepsurv']:.4f}")

    # Note: these are full-genome C-indices (14,675 genes), not consensus-gene C-indices
    perf['note'] = (
        "These C-indices are from Phase 3 cross-fitted evaluation on all 14,675 genes. "
        "They are NOT consensus-gene-specific C-indices. Post-selection C-indices on the "
        "64 consensus genes would be biased (v4 document, Inferential Framework)."
    )

    with open(out_dir / "performance_summary.json", 'w') as f:
        json.dump(perf, f, indent=2)

    # -----------------------------------------------------------------
    # 5.1b: Gene overlap
    # -----------------------------------------------------------------
    logger.info("")
    logger.info("  --- 5.1b: Gene Overlap ---")

    # Load consensus gene lists from Phase 4
    consensus_dir = PHASE4_DIR / "consensus"

    cox_genes = set()
    cox_file = consensus_dir / "cox_consensus_pi080.txt"
    if cox_file.exists():
        cox_genes = set(cox_file.read_text().strip().split('\n'))

    ds_genes = set()
    ds_file = consensus_dir / "deepsurv_consensus_pi080.txt"
    if ds_file.exists():
        content = ds_file.read_text().strip()
        if content:
            ds_genes = set(content.split('\n'))

    overlap = cox_genes & ds_genes
    union = cox_genes | ds_genes
    jaccard = len(overlap) / len(union) if union else 0

    gene_overlap = {
        'cox_n': len(cox_genes),
        'deepsurv_n': len(ds_genes),
        'overlap_n': len(overlap),
        'overlap_genes': sorted(overlap),
        'jaccard': jaccard,
        'cox_only_n': len(cox_genes - ds_genes),
        'deepsurv_only_n': len(ds_genes - cox_genes),
        'deepsurv_subset_of_cox': ds_genes.issubset(cox_genes),
    }

    logger.info(f"  Cox consensus: {len(cox_genes)} genes")
    logger.info(f"  DeepSurv consensus: {len(ds_genes)} genes")
    logger.info(f"  Overlap: {len(overlap)} genes")
    logger.info(f"  Jaccard: {jaccard:.3f}")
    logger.info(f"  DeepSurv is subset of Cox: {ds_genes.issubset(cox_genes)}")

    with open(out_dir / "gene_overlap.json", 'w') as f:
        json.dump(gene_overlap, f, indent=2)

    # -----------------------------------------------------------------
    # 5.1c: Pathway enrichment on Cox consensus genes
    # -----------------------------------------------------------------
    logger.info("")
    logger.info("  --- 5.1c: Pathway Enrichment ---")

    # Load gene annotation for ENSG → symbol mapping
    annot = pd.read_csv(ANNOTATION_FILE)
    ensg_to_symbol = dict(zip(annot['gene_id'], annot['gene_symbol']))

    # Map consensus genes to symbols
    cox_symbols = []
    unmapped = []
    for ensg in sorted(cox_genes):
        symbol = ensg_to_symbol.get(ensg)
        if symbol:
            cox_symbols.append(symbol)
        else:
            unmapped.append(ensg)

    logger.info(f"  Mapped {len(cox_symbols)}/{len(cox_genes)} consensus genes to symbols")
    if unmapped:
        logger.info(f"  Unmapped: {unmapped}")

    # Build background gene list (all 16,353 genes mapped to symbols)
    from src.data.data_loader import load_cohort
    tcga_expr, _ = load_cohort('tcga', str(DATA_DIR))
    all_ensg = tcga_expr.index.tolist()
    background_symbols = [ensg_to_symbol[e] for e in all_ensg if e in ensg_to_symbol]

    logger.info(f"  Background: {len(background_symbols)} genes with symbols "
                f"(of {len(all_ensg)} total)")

    # Run ORA for each GMT file
    import gseapy as gp

    enrichment_summary = {}

    for lib_name, gmt_path in GMT_FILES.items():
        if not gmt_path.exists():
            logger.warning(f"  GMT file not found: {gmt_path}")
            continue

        logger.info(f"  Running ORA: {lib_name}...")

        try:
            enr = gp.enrich(
                gene_list=cox_symbols,
                gene_sets=str(gmt_path),
                background=background_symbols,
                outdir=None,  # don't write gseapy's default output
                no_plot=True,
                verbose=False,
            )

            results_df = enr.results

            if results_df is not None and len(results_df) > 0:
                # Filter to significant (descriptive threshold)
                sig = results_df[results_df['Adjusted P-value'] < ENRICHMENT_FDR_THRESHOLD]

                results_df.to_csv(out_dir / f"enrichment_{lib_name.lower()}.csv", index=False)

                enrichment_summary[lib_name] = {
                    'total_gene_sets_tested': len(results_df),
                    'significant_fdr05': len(sig),
                    'top_5': [],
                }

                if len(sig) > 0:
                    top5 = sig.nsmallest(5, 'Adjusted P-value')
                    for _, row in top5.iterrows():
                        enrichment_summary[lib_name]['top_5'].append({
                            'term': row['Term'],
                            'overlap': row['Overlap'],
                            'p_value': float(row['P-value']),
                            'fdr': float(row['Adjusted P-value']),
                            'genes': row['Genes'],
                        })

                logger.info(f"    {lib_name}: {len(sig)} significant terms "
                            f"(FDR < {ENRICHMENT_FDR_THRESHOLD}) "
                            f"out of {len(results_df)} tested")

                if len(sig) > 0:
                    for _, row in sig.nsmallest(3, 'Adjusted P-value').iterrows():
                        logger.info(f"      {row['Term']}: "
                                    f"FDR={row['Adjusted P-value']:.2e}, "
                                    f"overlap={row['Overlap']}")
            else:
                logger.info(f"    {lib_name}: no results returned")
                enrichment_summary[lib_name] = {
                    'total_gene_sets_tested': 0,
                    'significant_fdr05': 0,
                    'top_5': [],
                }

        except Exception as e:
            logger.error(f"    {lib_name} enrichment failed: {e}")
            enrichment_summary[lib_name] = {'error': str(e)}

    enrichment_summary['consensus_genes_mapped'] = len(cox_symbols)
    enrichment_summary['consensus_genes_unmapped'] = len(unmapped)
    enrichment_summary['background_size'] = len(background_symbols)
    enrichment_summary['msigdb_version'] = 'v2026.1'
    enrichment_summary['method'] = 'ORA (Fisher exact test) via gseapy.enrich()'
    enrichment_summary['fdr_method'] = 'Benjamini-Hochberg'
    enrichment_summary['fdr_threshold'] = ENRICHMENT_FDR_THRESHOLD
    enrichment_summary['note'] = (
        'Enrichment is post-hoc characterization of the consensus gene set. '
        'It is not part of gene selection and does not affect the gene panel.'
    )

    # Save consensus gene symbol list for reference
    gene_list_df = pd.DataFrame({
        'ensg_id': sorted(cox_genes),
        'symbol': [ensg_to_symbol.get(e, 'UNMAPPED') for e in sorted(cox_genes)],
    })
    gene_list_df.to_csv(out_dir / "consensus_genes_annotated.csv", index=False)

    with open(out_dir / "enrichment_summary.json", 'w') as f:
        json.dump(enrichment_summary, f, indent=2, default=str)

    logger.info("")
    logger.info("  Analysis 5.1 complete.")

    return cox_genes, cox_symbols, background_symbols


# =============================================================================
# Analysis 5.2: Proportional Hazards Assumption
# =============================================================================

def analysis_5_2(cox_genes: set, out_dir: Path):
    """
    Test PH assumption for each consensus gene (univariate Cox)
    and for the overall out-of-fold DeepSurv risk score.

    Uses Schoenfeld residuals (Grambsch & Therneau, 1994).
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("ANALYSIS 5.2: Proportional Hazards Assumption")
    logger.info("=" * 70)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load expression and survival data
    from src.data.data_loader import load_cohort
    tcga_expr, tcga_surv = load_cohort('tcga', str(DATA_DIR))
    orien_expr, orien_surv = load_cohort('orien', str(DATA_DIR))

    consensus_genes = sorted(cox_genes)

    results_per_cohort = {}

    for cohort_name, expr, surv in [('tcga', tcga_expr, tcga_surv),
                                     ('orien', orien_expr, orien_surv)]:
        logger.info(f"")
        logger.info(f"  --- {cohort_name.upper()} ---")

        # Build DataFrame: gene expression + survival
        # expr is (genes × samples), transpose to (samples × genes)
        gene_df = expr.loc[consensus_genes].T
        gene_df.index.name = 'sampleID'

        # Align with survival
        common_ids = gene_df.index.intersection(surv.index)
        gene_df = gene_df.loc[common_ids]
        surv_aligned = surv.loc[common_ids]

        analysis_df = gene_df.copy()
        analysis_df['time'] = surv_aligned['time'].values
        analysis_df['event'] = surv_aligned['event'].values

        # Filter: need time > 0 for Cox
        analysis_df = analysis_df[analysis_df['time'] > 0]

        logger.info(f"  Samples: {len(analysis_df)}, Events: {int(analysis_df['event'].sum())}")

        # Per-gene univariate Schoenfeld test
        gene_results = []

        for gene in consensus_genes:
            try:
                cph = CoxPHFitter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cph.fit(
                        analysis_df[[gene, 'time', 'event']],
                        duration_col='time',
                        event_col='event',
                    )

                # Schoenfeld test
                ph_test = proportional_hazard_test(cph, analysis_df[[gene, 'time', 'event']],
                                                   time_transform='rank')

                test_stat = float(ph_test.summary['test_statistic'].iloc[0])
                p_value = float(ph_test.summary['p'].iloc[0])

                gene_results.append({
                    'gene': gene,
                    'schoenfeld_stat': test_stat,
                    'p_value': p_value,
                    'hr': float(cph.hazard_ratios_.iloc[0]),
                    'hr_lower': float(cph.confidence_intervals_.iloc[0, 0]),
                    'hr_upper': float(cph.confidence_intervals_.iloc[0, 1]),
                })

            except Exception as e:
                gene_results.append({
                    'gene': gene,
                    'schoenfeld_stat': np.nan,
                    'p_value': np.nan,
                    'hr': np.nan,
                    'hr_lower': np.nan,
                    'hr_upper': np.nan,
                    'error': str(e),
                })

        gene_results_df = pd.DataFrame(gene_results)

        # Multiple testing correction (descriptive)
        from statsmodels.stats.multitest import multipletests
        valid_mask = gene_results_df['p_value'].notna()
        if valid_mask.sum() > 0:
            _, fdr_values, _, _ = multipletests(
                gene_results_df.loc[valid_mask, 'p_value'],
                method='fdr_bh'
            )
            gene_results_df.loc[valid_mask, 'fdr_q'] = fdr_values
        else:
            gene_results_df['fdr_q'] = np.nan

        gene_results_df.to_csv(out_dir / f"schoenfeld_per_gene_{cohort_name}.csv", index=False)

        n_violation_005 = (gene_results_df['p_value'] < 0.05).sum()
        n_violation_fdr = (gene_results_df['fdr_q'] < 0.05).sum() if 'fdr_q' in gene_results_df else 0
        n_tested = valid_mask.sum()

        logger.info(f"  Univariate PH test: {n_violation_005}/{n_tested} genes "
                    f"with p < 0.05 ({n_violation_fdr} with FDR < 0.05)")

        results_per_cohort[cohort_name] = {
            'n_genes_tested': int(n_tested),
            'n_violation_p005': int(n_violation_005),
            'n_violation_fdr005': int(n_violation_fdr),
            'proportion_violation_p005': float(n_violation_005 / n_tested) if n_tested > 0 else 0,
        }

    # -----------------------------------------------------------------
    # PH test on overall out-of-fold risk score
    # -----------------------------------------------------------------
    logger.info("")
    logger.info("  --- Overall Risk Score PH Test ---")

    risk_score_ph = {}

    for cohort_name, surv in [('tcga', tcga_surv), ('orien', orien_surv)]:
        # Determine which direction produces risk scores for this cohort
        # TCGA patients get risk scores from orien_to_tcga direction
        # ORIEN patients get risk scores from tcga_to_orien direction
        if cohort_name == 'tcga':
            direction = 'orien_to_tcga'
        else:
            direction = 'tcga_to_orien'

        # Use seed 42 as primary (report sensitivity across seeds)
        seed_results = {}
        for seed in SEEDS:
            score_file = (PHASE3_DIR / f"seed_{seed}" / "aggregated" /
                         f"{direction}_deepsurv_all_scores.csv")
            if not score_file.exists():
                logger.warning(f"  Risk scores not found: {score_file}")
                continue

            scores = pd.read_csv(score_file)
            scores = scores.set_index('sampleID')

            # Align with survival
            common = scores.index.intersection(surv.index)
            rs_df = pd.DataFrame({
                'risk_score': scores.loc[common, 'risk_score'].values,
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

                seed_results[str(seed)] = {
                    'schoenfeld_stat': test_stat,
                    'p_value': p_value,
                    'n_samples': len(rs_df),
                    'n_events': int(rs_df['event'].sum()),
                }

                logger.info(f"  {cohort_name.upper()} seed {seed}: "
                            f"Schoenfeld stat={test_stat:.3f}, p={p_value:.4f}")

            except Exception as e:
                seed_results[str(seed)] = {'error': str(e)}
                logger.warning(f"  {cohort_name.upper()} seed {seed}: PH test failed: {e}")

        risk_score_ph[cohort_name] = seed_results

    with open(out_dir / "schoenfeld_risk_score.json", 'w') as f:
        json.dump(risk_score_ph, f, indent=2)

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    ph_summary = {
        'per_gene': results_per_cohort,
        'risk_score': risk_score_ph,
        'method': 'Schoenfeld residuals (Grambsch & Therneau 1994)',
        'time_transform': 'rank',
        'multiple_testing': 'BH FDR (descriptive, reported in supplement)',
        'note': (
            'PH violations indicate time-varying effects. If common, '
            'IG attributions reflect time-averaged hazard contributions '
            'and may mask time-varying gene effects (v4 Limitations).'
        ),
    }

    with open(out_dir / "ph_summary.json", 'w') as f:
        json.dump(ph_summary, f, indent=2)

    logger.info("")
    logger.info("  Analysis 5.2 complete.")


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
