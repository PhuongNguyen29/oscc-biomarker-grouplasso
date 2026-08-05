"""
Phase 4: Stability Selection with Randomized Lasso — Complementary Pairs (CPSS)

Revision of phase4_stability_randomized_lasso.py for the corrected pipeline.

WHAT CHANGED AND WHY
--------------------
1. SUBSAMPLING DESIGN (substantive)
   Old: B independent stratified subsamples at fraction 0.80.
   New: B/2 complementary PAIRS of disjoint stratified half-samples
        (fraction 0.50), union of each pair = full dataset.

   Reason: Shah & Samworth (2013) JRSS-B 75:55-80 derive their error bounds
   for complementary pairs stability selection. Applying their bound to
   output generated under a different subsampling scheme would cite a
   theorem whose conditions are not met. Total fits are unchanged
   (B/2 pairs x 2 halves = B fits).

   Consequence: each fit sees ~half the samples instead of ~80%, so fewer
   events per fit, so the lasso selects fewer genes per fit. q is expected
   to DROP relative to the old grid. Since E(V) scales with q^2, the bound
   should tighten.

2. ODD-COUNT HANDLING (correctness)
   Old: n_event = int(len(event_idx) * fraction) truncates. At fraction 0.5
        with an odd count, one sample is silently dropped from both halves,
        breaking the disjoint-union property required by CPSS.
   New: the remainder is assigned explicitly, alternating between halves
        across pairs so neither half is systematically favoured.

3. PI GRID EXTENDED
   Old: [0.60, 0.70, 0.80, 0.90]
   New: [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
   Reason: the Shah-Samworth bounds have different regimes above and below
   tau = 0.75. The grid now spans both.

4. ERROR BOUNDS COMPUTED PER CELL
   M&B (2010) bound implemented. Shah-Samworth left as a STUB — see
   shah_samworth_bound() below. Both are written to summary.json.

5. PI_PRIMARY REMOVED
   The old PI_PRIMARY = 0.80 was never the adopted operating point (pi 0.60
   was used) and was mistakable for a selection criterion. Cell selection is
   now made by the frozen A4.5 rule, applied in select_operating_point().

6. FILTER DIRECTORY DEFAULT
   Now points at the corrected post-ComBat pre-z filter output, not the
   frozen old post-z set.

FROZEN SELECTION RULE (B / A4.5)
--------------------------------
Among cells satisfying E(V) <= EV_TARGET in BOTH cohorts, select the one with
the smallest alpha multiplier; ties broken by the lowest pi threshold.
No test-cohort C-index, no external validation result, and no gene-set size
enters this rule. If no cell is admissible, that is the reported result; the
target is not relaxed.

References:
    - Meinshausen & Buhlmann (2010) JRSS-B 72:417-473 — stability selection,
      randomized lasso (Section 3.1)
    - Shah & Samworth (2013) JRSS-B 75:55-80 — complementary pairs, sharper bounds
    - Bourgon et al. (2010) PNAS — independent filtering

Usage:
    cd ~/hpchome/oscc-biomarker-grouplasso
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/phase4_stability_cpss.py \
        --filter-dir iqr_filtered_v4_prez \
        --results-dir results/phase4_cpss
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxnetSurvivalAnalysis

from src.data.data_loader import load_cohort

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

B_FITS = 50                  # total fits = N_PAIRS * 2
N_PAIRS = B_FITS // 2        # 25 complementary pairs
SUBSAMPLE_FRACTION = 0.50    # required by CPSS; NOT a tunable
RANDOM_SEED = 42
DATA_DIR = "data/processed"
RESULTS_BASE = "results/phase4_cpss"

PI_THRESHOLDS = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# Frozen error-control target (B / A4.4). Applied symmetrically to both cohorts.
EV_TARGET = 10.0

# Which bound governs cell selection. "mb" or "shah_samworth".
# Per A4.6, falls back to "mb" if the CPSS bound cannot be validated.
PRIMARY_BOUND = "mb"

# Randomized lasso weakness: W_j ~ Uniform(WEAKNESS_ALPHA, 1.0). Frozen.
WEAKNESS_ALPHA = 0.5

ALPHA_MULTIPLIERS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Phase 2 anchoring alphas. MUST be re-derived from the corrected Phase 2 run
# on the 4,835-gene universe before this script is run for the record.
# Values below are from the OLD 4,448-gene tuning and are placeholders.
PHASE2_COX_PARAMS = {
    'tcga':  {'l1_ratio': 1.0, 'alpha': 0.10217330634724603},
    'orien': {'l1_ratio': 1.0, 'alpha': 0.06833270705221954},
}
PHASE2_PARAMS_ARE_CORRECTED = False   # set True once re-derived; asserted below


# =============================================================================
# Error bounds
# =============================================================================

def mb_bound(q, pi, p):
    """
    Meinshausen & Buhlmann (2010) Theorem 1.

        E(V) <= q^2 / [ (2*pi - 1) * p ]

    q   : average number of variables selected per fit
    pi  : stability threshold
    p   : number of candidate variables

    Valid for pi > 0.5. Returns inf otherwise.
    """
    if pi <= 0.5:
        return float('inf')
    return (q ** 2) / ((2.0 * pi - 1.0) * p)


def shah_samworth_bound(q, pi, p, B_pairs):
    """
    Shah & Samworth (2013) bound for complementary pairs stability selection.

    NOT IMPLEMENTED. Returns None.

    This function is deliberately left as a stub. The CPSS bounds involve a
    function of (theta, tau, B) whose exact form must be taken from the paper
    (Theorem 1 for the assumption-free bound; Theorem 2 under r-concavity,
    with distinct regimes for tau <= 3/4 and tau > 3/4). Implementing it from
    memory risks a subtly wrong error guarantee, which is worse than none.

    TWO ROUTES TO FILL THIS IN:

    (a) Transcribe from the paper. Write the equation as a comment directly
        above the implementation so it can be checked against the source.
        Verify against the worked examples in the paper before trusting it.

    (b) Use the `stabs` R package (Hofner & Hothorn), which implements the
        Shah-Samworth minD function. Export q, pi and p per cell from
        summary.json, compute the bound in R, and join the result back.
        This is the lower-risk route: the implementation is peer-reviewed
        and maintained.

    Until this is implemented, PRIMARY_BOUND must remain "mb" (A4.6
    implementation contingency).
    """
    return None


def compute_bounds(q, pi, p, B_pairs):
    """Return both bounds for one (cohort, mult, pi) cell."""
    return {
        'mb': mb_bound(q, pi, p),
        'shah_samworth': shah_samworth_bound(q, pi, p, B_pairs),
    }


# =============================================================================
# Complementary pairs subsampling
# =============================================================================

def generate_complementary_pairs(y_event, n_pairs, seed):
    """
    Generate n_pairs complementary pairs of stratified half-samples.

    For each pair, event indices and censored indices are each split in half,
    so both halves preserve the cohort event rate. The two halves are disjoint
    and their union is the full sample set — the design Shah & Samworth (2013)
    assume.

    Odd counts: the leftover sample is assigned to half A on even-numbered
    pairs and half B on odd-numbered pairs, so neither half is systematically
    larger. This is what the old int() truncation got wrong — it dropped the
    leftover from both halves, so the union was not the full dataset.

    Returns a list of n_pairs dicts, each with 'A' and 'B' index arrays.
    """
    rng = np.random.RandomState(seed)
    event_idx = np.where(y_event == 1)[0]
    censor_idx = np.where(y_event == 0)[0]

    pairs = []
    for k in range(n_pairs):
        ev = rng.permutation(event_idx)
        cn = rng.permutation(censor_idx)

        n_ev_half = len(ev) // 2
        n_cn_half = len(cn) // 2

        # Alternate which half receives the odd leftover
        ev_extra = 1 if (len(ev) % 2 and k % 2 == 0) else 0
        cn_extra = 1 if (len(cn) % 2 and k % 2 == 0) else 0

        A = np.sort(np.concatenate([ev[:n_ev_half + ev_extra],
                                    cn[:n_cn_half + cn_extra]]))
        B = np.sort(np.concatenate([ev[n_ev_half + ev_extra:],
                                    cn[n_cn_half + cn_extra:]]))

        # CPSS requires disjoint halves whose union is everything
        assert len(np.intersect1d(A, B)) == 0, "halves overlap"
        assert len(A) + len(B) == len(y_event), "halves do not cover all samples"

        pairs.append({
            'A': A, 'B': B,
            'n_A': len(A), 'n_B': len(B),
            'n_A_events': int(y_event[A].sum()),
            'n_B_events': int(y_event[B].sum()),
        })

    return pairs


# =============================================================================
# Randomized lasso Cox on one half-sample
# =============================================================================

def fit_randomized_lasso_cox(X, y_event, y_time, gene_names,
                             alpha, l1_ratio, weakness, rng):
    """
    Fit lasso Cox with randomized penalty weights on one half-sample.

    Randomization (M&B 2010 Section 3.1): draw W_j ~ Uniform(weakness, 1),
    scale column j by 1/W_j, so the effective penalty on gene j is alpha*W_j.
    Unchanged from the original script.
    """
    n_features = X.shape[1]
    weights = rng.uniform(weakness, 1.0, size=n_features)
    X_scaled = X / weights[np.newaxis, :]

    y_struct = np.array(
        [(bool(e), t) for e, t in zip(y_event, y_time)],
        dtype=[('event', bool), ('time', float)]
    )

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
            current_alpha *= 2
            if attempt == 4:
                logger.warning(f"    Lasso fit failed after 5 attempts "
                               f"(alpha={current_alpha:.6f})")
                return [], {}, 0

    coefs_scaled = model.coef_[:, 0]
    coefs_original = coefs_scaled / weights

    nonzero = np.abs(coefs_scaled) > 1e-10
    selected = [gene_names[j] for j in range(n_features) if nonzero[j]]
    signed = dict(zip(gene_names, coefs_original))

    return selected, signed, int(nonzero.sum())


# =============================================================================
# Per-cohort stability selection
# =============================================================================

def run_stability_for_cohort(cohort, X, y_event, y_time, gene_names,
                             alpha, l1_ratio, results_dir):
    """Run CPSS randomized lasso across the alpha multiplier grid."""

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"CPSS RANDOMIZED LASSO: {cohort.upper()}")
    logger.info("=" * 70)
    logger.info(f"  Samples: {len(y_event)}, Events: {int(y_event.sum())}")
    logger.info(f"  Features: {X.shape[1]}")
    logger.info(f"  Anchor alpha: {alpha:.6f}, l1_ratio: {l1_ratio}")
    logger.info(f"  Weakness: {WEAKNESS_ALPHA}")
    logger.info(f"  Pairs: {N_PAIRS} (= {N_PAIRS*2} fits), fraction 0.50")

    pairs = generate_complementary_pairs(y_event, N_PAIRS, RANDOM_SEED)
    logger.info(f"  Half sizes: A~{pairs[0]['n_A']} ({pairs[0]['n_A_events']} ev), "
                f"B~{pairs[0]['n_B']} ({pairs[0]['n_B_events']} ev)")

    sel_dir = results_dir / "subsample_selections" / cohort
    sel_dir.mkdir(parents=True, exist_ok=True)
    coef_dir = results_dir / "signed_coefficients" / cohort
    coef_dir.mkdir(parents=True, exist_ok=True)
    prob_dir = results_dir / "selection_probabilities"
    prob_dir.mkdir(parents=True, exist_ok=True)

    all_alpha_results = {}

    for mult in ALPHA_MULTIPLIERS:
        current_alpha = alpha * mult
        logger.info("")
        logger.info(f"  --- multiplier {mult}x (alpha={current_alpha:.6f}) ---")

        # Reset weight RNG per multiplier so cells are comparable
        weight_rng = np.random.RandomState(RANDOM_SEED + 1000)

        selections, signed_coefs, n_nonzero = [], [], []
        fit_labels = []

        for k, pr in enumerate(pairs):
            for half in ('A', 'B'):
                sel, sgn, nnz = fit_randomized_lasso_cox(
                    X[pr[half]], y_event[pr[half]], y_time[pr[half]],
                    gene_names, current_alpha, l1_ratio,
                    WEAKNESS_ALPHA, weight_rng,
                )
                selections.append(sel)
                signed_coefs.append(sgn)
                n_nonzero.append(nnz)
                fit_labels.append(f"pair{k+1:02d}{half}")

        q = float(np.mean(n_nonzero))
        logger.info(f"    q (mean selected per fit): {q:.1f} "
                    f"+/- {np.std(n_nonzero):.1f}  "
                    f"[{min(n_nonzero)}, {max(n_nonzero)}]")

        # Selection probability = fraction of ALL fits selecting the gene
        counts = pd.Series(0.0, index=gene_names)
        for sel in selections:
            counts[sel] += 1
        sel_probs = counts / len(selections)

        for pi in PI_THRESHOLDS:
            n_stable = int((sel_probs >= pi).sum())
            b = compute_bounds(q, pi, len(gene_names), N_PAIRS)
            ss = "n/a" if b['shah_samworth'] is None else f"{b['shah_samworth']:.2f}"
            logger.info(f"    pi>={pi:.2f}: {n_stable:>4} stable | "
                        f"E(V) M&B={b['mb']:.2f}, SS={ss}")

        all_alpha_results[mult] = {
            'alpha': current_alpha,
            'sel_probs': sel_probs,
            'q': q,
            'std_nonzero': float(np.std(n_nonzero)),
            'n_nonzero_per_fit': n_nonzero,
        }

        # Persist
        tag = f"mult{mult:.2f}".replace(".", "p")
        for lbl, sel in zip(fit_labels, selections):
            pd.DataFrame({'gene': sel}).to_csv(
                sel_dir / f"{tag}_{lbl}.csv", index=False)
        sc = pd.DataFrame(signed_coefs, columns=gene_names).T
        sc.columns = fit_labels
        sc.to_csv(coef_dir / f"cox_signed_coefficients_{tag}.csv")
        sel_probs.to_csv(prob_dir / f"{cohort}_selection_probs_{tag}.csv",
                         header=["selection_prob"])

    return all_alpha_results


# =============================================================================
# Frozen selection rule (A4.5)
# =============================================================================

def select_operating_point(grid, ev_target, bound_key):
    """
    Apply the frozen A4.5 rule.

    Among cells with E(V) <= ev_target in BOTH cohorts, take the smallest
    alpha multiplier; ties broken by the lowest pi.

    grid: list of dicts with mult, pi, ev_tcga, ev_orien
    Returns the selected cell dict, or None if no cell is admissible.
    """
    admissible = [
        c for c in grid
        if c[f'ev_tcga_{bound_key}'] is not None
        and c[f'ev_orien_{bound_key}'] is not None
        and c[f'ev_tcga_{bound_key}'] <= ev_target
        and c[f'ev_orien_{bound_key}'] <= ev_target
    ]
    if not admissible:
        return None
    admissible.sort(key=lambda c: (c['mult'], c['pi']))
    return admissible[0]


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 4: CPSS randomized lasso")
    ap.add_argument('--filter-dir', type=str, default='iqr_filtered_v4_prez',
                    help="Subdirectory under data/processed/ with retained_genes.txt")
    ap.add_argument('--cohort', type=str, default='both',
                    choices=['tcga', 'orien', 'both'])
    ap.add_argument('--results-dir', type=str, default=None)
    ap.add_argument('--allow-stale-phase2', action='store_true',
                    help="Run with the old 4,448-gene Phase 2 alphas (diagnostic only)")
    args = ap.parse_args()

    if not PHASE2_PARAMS_ARE_CORRECTED and not args.allow_stale_phase2:
        raise RuntimeError(
            "PHASE2_COX_PARAMS are from the old 4,448-gene tuning. Re-derive them "
            "from the corrected Phase 2 run on the 4,835-gene universe and set "
            "PHASE2_PARAMS_ARE_CORRECTED = True, or pass --allow-stale-phase2 "
            "for a diagnostic run whose output must not be used for the record."
        )

    np.random.seed(RANDOM_SEED)
    results_dir = Path(args.results_dir) if args.results_dir else Path(RESULTS_BASE)
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(results_dir / f"phase4_cpss_{ts}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(fh)

    logger.info("Phase 4: CPSS Randomized Lasso Stability Selection")
    logger.info(f"Results: {results_dir}")
    logger.info(f"EV_TARGET={EV_TARGET}, PRIMARY_BOUND={PRIMARY_BOUND}")
    if args.allow_stale_phase2:
        logger.warning("RUNNING WITH STALE PHASE 2 ALPHAS — diagnostic only")

    # --- Load ---------------------------------------------------------------
    tcga_expr, tcga_surv = load_cohort('tcga', DATA_DIR)
    orien_expr, orien_surv = load_cohort('orien', DATA_DIR)

    gene_names_full = tcga_expr.index.tolist()
    if list(orien_expr.index) != gene_names_full:
        raise ValueError("Gene sets/order differ between TCGA and ORIEN.")

    iqr_gene_file = Path(DATA_DIR) / args.filter_dir / "retained_genes.txt"
    if not iqr_gene_file.exists():
        raise FileNotFoundError(f"Retained gene list not found: {iqr_gene_file}")
    with open(iqr_gene_file) as f:
        retained_genes = [ln.strip() for ln in f if ln.strip()]

    iqr_meta_file = Path(DATA_DIR) / args.filter_dir / "filter_metadata.json"
    iqr_metadata = json.load(open(iqr_meta_file)) if iqr_meta_file.exists() else {}

    missing = set(retained_genes) - set(gene_names_full)
    if missing:
        raise ValueError(f"{len(missing)} retained genes absent from expression matrix")

    logger.info(f"Universe: {len(gene_names_full)} -> retained {len(retained_genes)}")

    gene_names = retained_genes
    p = len(gene_names)

    cohort_data = {
        'tcga': (tcga_expr.loc[retained_genes].values.T.astype(np.float32),
                 tcga_surv['event'].values.astype(np.float32),
                 tcga_surv['time'].values.astype(np.float32)),
        'orien': (orien_expr.loc[retained_genes].values.T.astype(np.float32),
                  orien_surv['event'].values.astype(np.float32),
                  orien_surv['time'].values.astype(np.float32)),
    }

    # --- Run ----------------------------------------------------------------
    cohorts = ['tcga', 'orien'] if args.cohort == 'both' else [args.cohort]
    all_results = {}
    for c in cohorts:
        X, ev, tm = cohort_data[c]
        all_results[c] = run_stability_for_cohort(
            c, X, ev, tm, gene_names,
            alpha=PHASE2_COX_PARAMS[c]['alpha'],
            l1_ratio=PHASE2_COX_PARAMS[c]['l1_ratio'],
            results_dir=results_dir,
        )

    if len(all_results) < 2:
        logger.info("Single cohort only — no cell selection. Use --cohort both.")
        raise SystemExit(0)

    # --- Build the grid with bounds ----------------------------------------
    annot_file = Path("data/raw/protein_coding_genes_annotation.csv")
    if annot_file.exists():
        annot = pd.read_csv(annot_file)
        ensg_to_symbol = dict(zip(annot["gene_id"], annot["gene_symbol"]))
    else:
        ensg_to_symbol = {}

    consensus_dir = results_dir / "consensus"
    consensus_dir.mkdir(parents=True, exist_ok=True)

    grid = []
    logger.info("")
    logger.info("=" * 70)
    logger.info("GRID WITH ERROR BOUNDS")
    logger.info("=" * 70)
    logger.info(f"{'mult':>5} {'pi':>5} {'qT':>7} {'qO':>7} "
                f"{'EV_T':>8} {'EV_O':>8} {'nT':>5} {'nO':>5} {'ovlp':>5}")

    for mult in ALPHA_MULTIPLIERS:
        tp = all_results['tcga'][mult]['sel_probs']
        op = all_results['orien'][mult]['sel_probs']
        qt = all_results['tcga'][mult]['q']
        qo = all_results['orien'][mult]['q']

        for pi in PI_THRESHOLDS:
            st = set(tp.index[tp >= pi])
            so = set(op.index[op >= pi])
            ov = st & so

            bt = compute_bounds(qt, pi, p, N_PAIRS)
            bo = compute_bounds(qo, pi, p, N_PAIRS)

            cell = {
                'mult': mult, 'pi': pi,
                'q_tcga': qt, 'q_orien': qo,
                'ev_tcga_mb': bt['mb'], 'ev_orien_mb': bo['mb'],
                'ev_tcga_shah_samworth': bt['shah_samworth'],
                'ev_orien_shah_samworth': bo['shah_samworth'],
                'n_tcga_stable': len(st), 'n_orien_stable': len(so),
                'n_overlap': len(ov),
                'tcga_stable_genes': sorted(st),
                'orien_stable_genes': sorted(so),
                'overlap_genes': sorted(ov),
                'union_genes': sorted(st | so),
            }
            grid.append(cell)

            logger.info(f"{mult:>5} {pi:>5.2f} {qt:>7.1f} {qo:>7.1f} "
                        f"{bt['mb']:>8.2f} {bo['mb']:>8.2f} "
                        f"{len(st):>5} {len(so):>5} {len(ov):>5}")

            tag = f"mult{mult}_pi{int(pi*100):02d}"
            for name, genes in (('tcga', sorted(st)), ('orien', sorted(so))):
                pd.DataFrame(
                    [(g, ensg_to_symbol.get(g, "")) for g in genes],
                    columns=['gene', 'symbol']
                ).to_csv(consensus_dir / f"{name}_stable_{tag}.csv", index=False)

    # --- Apply the frozen rule ---------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"FROZEN SELECTION RULE (A4.5): E(V) <= {EV_TARGET} in both "
                f"cohorts, bound = {PRIMARY_BOUND}")
    logger.info("=" * 70)

    selected = select_operating_point(grid, EV_TARGET, PRIMARY_BOUND)

    if selected is None:
        logger.info("  NO ADMISSIBLE CELL. Reported as the result; target not relaxed.")
    else:
        logger.info(f"  Selected: mult={selected['mult']}, pi={selected['pi']:.2f}")
        logger.info(f"    E(V): TCGA={selected[f'ev_tcga_{PRIMARY_BOUND}']:.2f}, "
                    f"ORIEN={selected[f'ev_orien_{PRIMARY_BOUND}']:.2f}")
        logger.info(f"    Stable: TCGA={selected['n_tcga_stable']}, "
                    f"ORIEN={selected['n_orien_stable']}, "
                    f"overlap={selected['n_overlap']}, "
                    f"union={len(selected['union_genes'])}")
        for g in selected['union_genes']:
            logger.info(f"      {g} ({ensg_to_symbol.get(g, '')})")

        pd.DataFrame(
            [(g, ensg_to_symbol.get(g, ""),
              'both' if g in selected['overlap_genes']
              else ('tcga' if g in selected['tcga_stable_genes'] else 'orien'))
             for g in selected['union_genes']],
            columns=['gene', 'symbol', 'in']
        ).to_csv(results_dir / "union_gene_set.csv", index=False)
        with open(results_dir / "union_genes.txt", 'w') as f:
            f.write("\n".join(selected['union_genes']) + "\n")

    # --- Summary ------------------------------------------------------------
    summary = {
        'method': 'Complementary Pairs Stability Selection, randomized lasso Cox',
        'design': f'{N_PAIRS} complementary pairs (fraction 0.50), {N_PAIRS*2} fits',
        'base_procedure': 'CoxnetSurvivalAnalysis, l1_ratio=1.0',
        'randomization': f'W_j ~ Uniform({WEAKNESS_ALPHA}, 1.0)',
        'references': [
            'Meinshausen & Buhlmann (2010) JRSS-B 72:417-473',
            'Shah & Samworth (2013) JRSS-B 75:55-80',
        ],
        'selection_rule': (
            f'Frozen A4.5: among cells with E(V) <= {EV_TARGET} in BOTH cohorts, '
            f'smallest alpha multiplier; ties broken by lowest pi. '
            f'No performance quantity enters this rule.'
        ),
        'ev_target': EV_TARGET,
        'primary_bound': PRIMARY_BOUND,
        'shah_samworth_implemented': shah_samworth_bound(1, 0.8, 100, 25) is not None,
        'phase2_params': PHASE2_COX_PARAMS,
        'phase2_params_are_corrected': PHASE2_PARAMS_ARE_CORRECTED,
        'iqr_filter': iqr_metadata,
        'n_genes': p,
        'config': {
            'n_pairs': N_PAIRS,
            'n_fits': N_PAIRS * 2,
            'subsample_fraction': SUBSAMPLE_FRACTION,
            'weakness_alpha': WEAKNESS_ALPHA,
            'alpha_multipliers': ALPHA_MULTIPLIERS,
            'pi_thresholds': PI_THRESHOLDS,
            'random_seed': RANDOM_SEED,
        },
        'grid': grid,
        'selected_cell': selected,
    }
    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("")
    logger.info(f"Phase 4 (CPSS) complete. Results: {results_dir}")
