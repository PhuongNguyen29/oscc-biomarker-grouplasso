"""
Phase 4b: Subsampling-fraction stability sweep

Measures how selection stability depends on subsample fraction, per cohort,
and produces the fraction-0.8 gene sets alongside.

CHANGELOG
---------
rev 2026-08 (b) -- WEIGHT DIRECTION + SEED DECOUPLING (behaviour-changing)
    Supersedes the run that produced the recorded fraction-stability table
    (plan section D5). Three defects fixed; ALL NUMBERS CHANGE ON RERUN:

    1. RANDOMISED-LASSO WEIGHT DIRECTION. fit_one used Z = X / W -- the same
       inverted direction fixed in the CPSS script (phase4_cpss RULEFIX,
       plan D2b). M&B (2010) Sec 3.1 penalise lambda * |beta_j| / W_j, which an
       ordinary lasso realises via Z = X * W (substitute gamma = beta / W).
       Division gives the OPPOSITE: small-W genes were shrunk LESS. Corrected
       to Z = X * W. Empirically (D2b), the corrected direction also SHRINKS
       columns (W <= 1), improving conditioning and sparsity.

    2. RANKING SCALE. Top-k sets for Kuncheva proper were ranked by |gamma|
       (coefficients on the scaled design), which mixes the random weights
       into the ranking. Coefficients are now back-transformed to the original
       scale (beta = gamma * W) before ranking. The SELECTED SET is unchanged
       by this (W > 0 preserves the zero pattern); only the top-k ordering is.

    3. CROSS-COHORT RNG COUPLING. Weight and subsample RNGs used the same
       seeds for both cohorts, coupling the Monte Carlo noise that enters the
       cross-cohort overlap table. Seeds now carry a per-cohort offset.
       (Directionally, coupling could only inflate apparent cross-cohort
       agreement; the recorded zero-overlap findings were therefore
       conservative -- disclosed, not invalidated, on this count.)

    Also fixed: the stratified_subsamples docstring claimed subsamples are
    "held fixed across all fractions" -- false; the SCHEME is fixed, the draws
    necessarily differ per fraction (different n). Interpretation of the sweep
    is unaffected.

    Interpretive consequence: the recorded D5 numbers (Kuncheva 0.084 -> 0.276,
    ratio ~0.33 * f^2) were produced by a randomised lasso with inverted
    weights. The perturbation was still exchangeable and random, so the
    QUALITATIVE finding -- stability scales with subsample overlap -- is
    expected to survive; the specific values are not citable. The plan's D5
    table must be refreshed from this script's rerun, and the D3
    design-attribution row rechecked.

WHAT THIS ANSWERS
-----------------
Two DIFFERENT quantities have been conflated in discussion so far:

  (a) WITHIN-cohort stability -- does repeated subsampling of the SAME cohort
      select the same genes? Measured here by Kuncheva's index.
  (b) CROSS-cohort overlap -- do TCGA and ORIEN select the same genes?
      Measured by set intersection.

Zero cross-cohort overlap has two competing explanations. If within-cohort
stability is itself near zero, the cohorts cannot agree because neither can
agree with itself -- the Ein-Dor et al. (2006) small-sample regime. If
within-cohort stability is high in both and they still share nothing, that is
genuine divergence. These have different implications, and (b) alone cannot
distinguish them. This script measures (a).

DESIGN NOTE -- WHY INDEPENDENT SUBSAMPLING AT EVERY FRACTION
------------------------------------------------------------
The CPSS run (fraction 0.5, complementary pairs) and the old run (fraction 0.8,
independent) differ in TWO ways. A sweep that used pairs at 0.5 and independent
elsewhere would confound fraction with sampling scheme. Here the scheme is held
fixed at independent stratified subsampling for all fractions, so the curve
isolates fraction. The 0.5 point will therefore NOT reproduce the CPSS numbers
exactly; CPSS remains a separate, error-controlled arm. As of rev (b) the BASE
LEARNER (corrected randomised lasso) matches the CPSS arm, so the remaining
difference at fraction 0.5 is the sampling scheme alone.

ERROR BOUNDS
------------
M&B (2010) and Shah & Samworth (2013) both derive bounds for subsamples of size
floor(n/2). At fraction != 0.5 NO bound applies under either theory. E(V) is
computed for reference and flagged `mb_bound_applicable = False` outside 0.5.
It must not be reported as an error guarantee at other fractions. Reporting
selection frequencies without a bound is standard practice (He & Yu 2010,
Table 1) -- the requirement is to say so, not to avoid it.

STABILITY MEASURES (He & Yu 2010, Table 4)
------------------------------------------
  kuncheva_topk  -- Kuncheva (2007) MS7, corrected for chance given subset size.
                    His formula assumes |S| = |S'| = c, so it is computed on
                    top-k sets ranked by |coefficient| (original beta scale,
                    rev b), k = median selection size in that cell. This is
                    Kuncheva's index proper.
  kuncheva_gen   -- chance-corrected index for the NATIVE variable-size sets:
                    (|S∩S'| - |S||S'|/m) / (min(|S|,|S'|) - |S||S'|/m).
                    A generalization, NOT Kuncheva's published statistic.
                    Labelled as such wherever reported.
  jaccard        -- |S∩S'| / |S∪S'|. Reported because it is what most papers
                    use; NOT corrected for chance, so it is confounded with
                    selection size across the sweep. Do not use it as the
                    headline measure here.

All are averaged over all pairs of usable fits within a cell.

References:
    Meinshausen & Buhlmann (2010) JRSS-B 72:417-473
    Shah & Samworth (2013) JRSS-B 75:55-80
    Kuncheva (2007) Proc. IASTED AIA, pp. 390-395
    He & Yu (2010) Comput Biol Chem 34:215-225
    Ein-Dor, Zuk & Domany (2006) PNAS 103:5923-5928

Usage:
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/phase4b_fraction_stability.py \
        --filter-dir iqr_filtered_v4_prez \
        --phase2-tcga results/phase2_tuning/cox_tuning_20260804_231755/tcga_best_params.json \
        --phase2-orien results/phase2_tuning/cox_tuning_20260804_225946/orien_best_params.json \
        --results-dir results/phase4b_fraction_stability_v2
"""

import argparse
import itertools
import json
import logging
import warnings
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

FRACTIONS = [0.5, 0.6, 0.7, 0.8, 0.9]
B_FITS = 100                 # matches the 100 fits in the CPSS run
ALPHA_MULTIPLIERS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
PI_THRESHOLDS = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
RANDOM_SEED = 42
WEAKNESS_ALPHA = 0.5         # frozen, as in Phase 4 (corrected direction, rev b)
REQUIRED_L1_RATIO = 1.0
MAX_ITER = 100000
TOL = 1e-7
DATA_DIR = "data/processed"

# Per-cohort seed offsets (rev b): decouple the Monte Carlo randomness that
# enters the cross-cohort overlap table. Offsets are arbitrary constants far
# apart; what matters is that TCGA and ORIEN no longer share weight or
# subsample streams.
COHORT_SEED_OFFSET = {'tcga': 0, 'orien': 500000}

# Descriptive analysis, not error-controlled: a cell is reported if at least
# this fraction of fits converged. Non-converged fits are excluded and counted.
MIN_USABLE_FIT_FRACTION = 0.95


# =============================================================================
# Stability measures
# =============================================================================

def kuncheva_index(s1, s2, m):
    """
    Kuncheva (2007) stability index, He & Yu (2010) Table 4 MS7:

        (|S1 n S2| * m - c^2) / (c * (m - c))     with c = |S1| = |S2|

    Corrected for the overlap expected by chance given subset size c. Ranges
    from ~-1 (complementary) through 0 (chance) to 1 (identical).

    REQUIRES equal cardinality. Returns nan if sizes differ or c is degenerate.
    """
    c = len(s1)
    if len(s2) != c or c == 0 or c == m:
        return np.nan
    inter = len(s1 & s2)
    return (inter * m - c * c) / (c * (m - c))


def kuncheva_generalized(s1, s2, m):
    """
    Chance-corrected index for variable-size sets:

        (|S1 n S2| - |S1||S2|/m) / (min(|S1|,|S2|) - |S1||S2|/m)

    Same chance correction logic as Kuncheva but admits |S1| != |S2|, which is
    what a lasso actually produces. This is a GENERALIZATION, not Kuncheva's
    published statistic, and must be labelled as such when reported.
    """
    n1, n2 = len(s1), len(s2)
    if n1 == 0 or n2 == 0:
        return np.nan
    expected = n1 * n2 / m
    denom = min(n1, n2) - expected
    if abs(denom) < 1e-12:
        return np.nan
    return (len(s1 & s2) - expected) / denom


def jaccard(s1, s2):
    """|S1 n S2| / |S1 u S2|. NOT chance-corrected."""
    u = len(s1 | s2)
    return len(s1 & s2) / u if u else np.nan


def pairwise_stability(selections, ranked_selections, m, max_pairs=None, rng=None):
    """
    Mean pairwise stability across fits.

    selections        : list of sets (native, variable-size)
    ranked_selections : list of lists, genes ordered by descending |beta|
                        (original scale, rev b), used to build equal-size
                        top-k sets for Kuncheva proper
    """
    n = len(selections)
    if n < 2:
        return {'kuncheva_topk': np.nan, 'kuncheva_gen': np.nan,
                'jaccard': np.nan, 'n_pairs': 0, 'topk': np.nan}

    sizes = [len(s) for s in selections]
    k = int(np.median(sizes))
    k = max(1, min(k, min(sizes)))          # top-k must exist in every fit
    topk = [set(r[:k]) for r in ranked_selections]

    pairs = list(itertools.combinations(range(n), 2))
    if max_pairs is not None and len(pairs) > max_pairs:
        idx = (rng or np.random).choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]

    kt, kg, jc = [], [], []
    for i, j in pairs:
        kt.append(kuncheva_index(topk[i], topk[j], m))
        kg.append(kuncheva_generalized(selections[i], selections[j], m))
        jc.append(jaccard(selections[i], selections[j]))

    return {
        'kuncheva_topk': float(np.nanmean(kt)),
        'kuncheva_topk_sd': float(np.nanstd(kt)),
        'kuncheva_gen': float(np.nanmean(kg)),
        'kuncheva_gen_sd': float(np.nanstd(kg)),
        'jaccard': float(np.nanmean(jc)),
        'n_pairs': len(pairs),
        'topk': k,
    }


def mb_bound(q, pi, p):
    """M&B (2010) E(V) <= q^2 / [(2*pi - 1)*p]. Valid only at fraction 0.5."""
    if pi <= 0.5:
        return float('inf')
    return (q ** 2) / ((2.0 * pi - 1.0) * p)


# =============================================================================
# Subsampling and fitting
# =============================================================================

def stratified_subsamples(y_event, fraction, n_fits, seed):
    """
    Independent stratified subsamples at the given fraction.

    The sampling SCHEME (independent, event-stratified) is held fixed across
    the whole sweep so the curve isolates fraction rather than scheme. The
    draws themselves necessarily differ per fraction -- different subsample
    sizes cannot share index sets. (Docstring corrected in rev b; the previous
    text claimed the subsamples themselves were fixed across fractions.)
    """
    rng = np.random.RandomState(seed)
    ev = np.where(y_event == 1)[0]
    cn = np.where(y_event == 0)[0]
    n_ev = max(1, int(round(len(ev) * fraction)))
    n_cn = max(1, int(round(len(cn) * fraction)))

    out = []
    for _ in range(n_fits):
        idx = np.concatenate([
            rng.choice(ev, n_ev, replace=False),
            rng.choice(cn, n_cn, replace=False),
        ])
        out.append(np.sort(idx))
    return out


def fit_one(X, y_event, y_time, gene_names, alpha, l1_ratio, weakness, rng):
    """
    Randomized lasso Cox on one subsample.

    Randomisation (M&B 2010 Sec 3.1): W_j ~ Uniform(weakness, 1), ordinary
    lasso on Z = X * W. Equivalent to effective penalty alpha * |beta_j| / W_j
    on the original scale: smaller W_j means STRONGER penalisation.
    (rev b: the previous version used Z = X / W -- the inverted direction --
    and ranked genes by |gamma| on the scaled design. Corrected to Z = X * W
    with back-transform beta = gamma * W before ranking. The zero pattern,
    and hence the selected SET, is invariant to the diagonal scaling; the
    top-k ordering is not.)

    Returns (selected_set, ranked_gene_list, n_nonzero, status).
    Convergence is captured explicitly: coordinate descent returns a partially
    optimised path at max_iter WITHOUT raising, and unconverged paths are less
    sparse, which inflates q and distorts every stability measure.
    ArithmeticError is a failure, not retried at a different alpha.
    """
    p = X.shape[1]
    w = rng.uniform(weakness, 1.0, size=p)
    Xs = X * w[np.newaxis, :]
    y = np.array([(bool(e), t) for e, t in zip(y_event, y_time)],
                 dtype=[('event', bool), ('time', float)])

    model = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=[alpha],
                                   normalize=False, tol=TOL, max_iter=MAX_ITER)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model.fit(Xs, y)
        except ArithmeticError:
            return set(), [], 0, "arithmetic_error"

    if any(w_.category.__name__ == "ConvergenceWarning"
           or "terminated early" in str(w_.message).lower() for w_ in caught):
        return set(), [], 0, "nonconverged"

    gamma = model.coef_[:, 0]
    beta = gamma * w                        # back-transform to original scale
    nz = np.abs(gamma) > 1e-10              # zero pattern: scale-invariant
    sel_idx = np.where(nz)[0]
    order = sel_idx[np.argsort(-np.abs(beta[sel_idx]))]
    ranked = [gene_names[j] for j in order]
    return set(ranked), ranked, int(nz.sum()), "ok"


# =============================================================================
# Phase 2 anchor
# =============================================================================

def load_phase2_anchor(path, cohort, filter_dir, n_genes):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Phase 2 results not found for {cohort}: {path}")
    p2 = json.load(open(path))
    l1, alpha = p2.get('best_l1_ratio'), p2.get('actual_path_alpha')
    if alpha is None or alpha <= 0:
        raise ValueError(f"{cohort}: invalid actual_path_alpha in {path}")
    if l1 != REQUIRED_L1_RATIO:
        raise ValueError(
            f"{cohort}: Phase 2 tuned at l1_ratio={l1}; randomized lasso needs "
            f"{REQUIRED_L1_RATIO}. Alpha does not transfer across l1_ratio."
        )
    if p2.get('filter_dir') not in (None, filter_dir):
        raise ValueError(f"{cohort}: Phase 2 filter_dir={p2.get('filter_dir')} "
                         f"!= {filter_dir}")
    if p2.get('n_genes') not in (None, n_genes):
        raise ValueError(f"{cohort}: Phase 2 n_genes={p2.get('n_genes')} "
                         f"!= {n_genes}")
    return {'l1_ratio': l1, 'alpha': alpha}, {
        'source_file': str(path), 'alpha': alpha, 'l1_ratio': l1,
        'phase2_filter_dir': p2.get('filter_dir'),
        'phase2_n_genes': p2.get('n_genes'),
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Subsampling-fraction stability sweep with Kuncheva index")
    ap.add_argument('--data-dir', default=DATA_DIR)
    ap.add_argument('--filter-dir', default='iqr_filtered_v4_prez')
    ap.add_argument('--phase2-tcga', required=True)
    ap.add_argument('--phase2-orien', required=True)
    ap.add_argument('--results-dir', default='results/phase4b_fraction_stability_v2')
    ap.add_argument('--fractions', type=float, nargs='+', default=FRACTIONS)
    ap.add_argument('--multipliers', type=float, nargs='+', default=ALPHA_MULTIPLIERS)
    ap.add_argument('--n-fits', type=int, default=B_FITS)
    ap.add_argument('--max-pairs', type=int, default=None,
                    help="Subsample the pairwise comparisons (default: all)")
    ap.add_argument('--seed', type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(results_dir / f"phase4b_{ts}.log")
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(fh)

    logger.info("Phase 4b: subsampling-fraction stability sweep (rev b)")
    logger.info("  base learner: corrected randomized lasso (Z = X*W), matching CPSS arm")
    logger.info(f"  fractions:   {args.fractions}")
    logger.info(f"  multipliers: {args.multipliers}")
    logger.info(f"  fits/cell:   {args.n_fits}")
    logger.info("  sampling:    INDEPENDENT stratified at every fraction "
                "(isolates fraction; CPSS is a separate arm)")
    logger.info("  RNG:         per-cohort seed offsets "
                f"{COHORT_SEED_OFFSET} (weights and subsamples decoupled)")
    logger.info("  NOTE: no M&B / Shah-Samworth bound applies at fraction != 0.5")

    # ---- Data --------------------------------------------------------------
    tcga_expr, tcga_surv = load_cohort('tcga', args.data_dir)
    orien_expr, orien_surv = load_cohort('orien', args.data_dir)

    gene_file = Path(args.data_dir) / args.filter_dir / "retained_genes.txt"
    retained = [ln.strip() for ln in open(gene_file) if ln.strip()]
    missing = set(retained) - set(tcga_expr.index)
    if missing:
        raise ValueError(f"{len(missing)} retained genes absent from expression matrix")
    gene_names = retained
    p = len(gene_names)
    logger.info(f"  genes: {p}")

    anchors, provenance = {}, {}
    for c, path in (('tcga', args.phase2_tcga), ('orien', args.phase2_orien)):
        anchors[c], provenance[c] = load_phase2_anchor(path, c, args.filter_dir, p)
        logger.info(f"  anchor {c.upper()}: alpha={anchors[c]['alpha']:.8f}")

    cohort_data = {
        'tcga': (tcga_expr.loc[retained].values.T.astype(np.float32),
                 tcga_surv['event'].values.astype(np.float32),
                 tcga_surv['time'].values.astype(np.float32)),
        'orien': (orien_expr.loc[retained].values.T.astype(np.float32),
                  orien_surv['event'].values.astype(np.float32),
                  orien_surv['time'].values.astype(np.float32)),
    }

    annot_file = Path("data/raw/protein_coding_genes_annotation.csv")
    sym = {}
    if annot_file.exists():
        a = pd.read_csv(annot_file)
        sym = dict(zip(a["gene_id"], a["gene_symbol"]))

    # ---- Sweep -------------------------------------------------------------
    stability_rows, selection_rows = [], []
    sel_probs_store = {}

    for cohort in ('tcga', 'orien'):
        X, ev, tm = cohort_data[cohort]
        alpha0 = anchors[cohort]['alpha']
        off = COHORT_SEED_OFFSET[cohort]
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"{cohort.upper()}: {X.shape[0]} samples, {int(ev.sum())} events")
        logger.info("=" * 70)

        for frac in args.fractions:
            # rev b: subsample stream decoupled across cohorts via offset
            subs = stratified_subsamples(ev, frac, args.n_fits, args.seed + off)
            logger.info(f"\n  fraction {frac}: n={len(subs[0])}, "
                        f"events={int(ev[subs[0]].sum())}")

            for mult in args.multipliers:
                alpha = alpha0 * mult
                # rev b: weight stream decoupled across cohorts via offset.
                # Reset per (fraction, multiplier) so cells within a cohort
                # remain comparable (same weight sequence per fit index).
                wr = np.random.RandomState(args.seed + 1000 + off)

                sels, ranked, nnz = [], [], []
                status = {"ok": 0, "nonconverged": 0, "arithmetic_error": 0}
                for idx in subs:
                    s, r, n, st = fit_one(X[idx], ev[idx], tm[idx], gene_names,
                                          alpha, REQUIRED_L1_RATIO,
                                          WEAKNESS_ALPHA, wr)
                    status[st] += 1
                    if st == "ok":
                        sels.append(s); ranked.append(r); nnz.append(n)

                usable = len(sels) / args.n_fits
                estimable = usable >= MIN_USABLE_FIT_FRACTION
                if not estimable:
                    logger.warning(f"    mult {mult}: only {len(sels)}/{args.n_fits} "
                                   f"fits usable -> NON-ESTIMABLE")
                    stability_rows.append({
                        'cohort': cohort, 'fraction': frac, 'multiplier': mult,
                        'estimable': False, 'n_usable_fits': len(sels),
                        **{k: status[k] for k in status},
                    })
                    continue

                q = float(np.mean(nnz))
                rng = np.random.RandomState(args.seed)
                stab = pairwise_stability(sels, ranked, p, args.max_pairs, rng)

                logger.info(
                    f"    mult {mult}: q={q:.1f} | "
                    f"Kuncheva(top{stab['topk']})={stab['kuncheva_topk']:.3f} | "
                    f"Kuncheva_gen={stab['kuncheva_gen']:.3f} | "
                    f"Jaccard={stab['jaccard']:.3f}"
                )

                stability_rows.append({
                    'cohort': cohort, 'fraction': frac, 'multiplier': mult,
                    'estimable': True, 'alpha': alpha, 'q': q,
                    'q_sd': float(np.std(nnz)),
                    'n_usable_fits': len(sels), **stab,
                    **{k: status[k] for k in status},
                })

                counts = pd.Series(0.0, index=gene_names)
                for s in sels:
                    counts[list(s)] += 1
                probs = counts / len(sels)
                sel_probs_store[(cohort, frac, mult)] = probs

                for pi in PI_THRESHOLDS:
                    stable = sorted(probs.index[probs >= pi])
                    selection_rows.append({
                        'cohort': cohort, 'fraction': frac, 'multiplier': mult,
                        'pi': pi, 'n_stable': len(stable), 'q': q,
                        'mb_ev': mb_bound(q, pi, p),
                        # Bounds are derived for subsamples of size n/2 only.
                        'mb_bound_applicable': abs(frac - 0.5) < 1e-9,
                        'genes': ";".join(stable),
                        'symbols': ";".join(sym.get(g, "") for g in stable),
                    })

    stab_df = pd.DataFrame(stability_rows)
    sel_df = pd.DataFrame(selection_rows)
    stab_df.to_csv(results_dir / "stability_by_fraction.csv", index=False)
    sel_df.to_csv(results_dir / "selection_by_fraction.csv", index=False)

    for (c, f, m), s in sel_probs_store.items():
        tag = f"{c}_frac{str(f).replace('.','p')}_mult{str(m).replace('.','p')}"
        s[s > 0].sort_values(ascending=False).to_csv(
            results_dir / f"selprobs_{tag}.csv", header=["selection_prob"])

    # ---- Cross-cohort overlap ---------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("CROSS-COHORT OVERLAP (a DIFFERENT quantity from Kuncheva)")
    logger.info("=" * 70)
    overlap_rows = []
    for frac in args.fractions:
        for mult in args.multipliers:
            for pi in PI_THRESHOLDS:
                t = sel_df[(sel_df.cohort == 'tcga') & (sel_df.fraction == frac)
                           & (sel_df.multiplier == mult) & (sel_df.pi == pi)]
                o = sel_df[(sel_df.cohort == 'orien') & (sel_df.fraction == frac)
                           & (sel_df.multiplier == mult) & (sel_df.pi == pi)]
                if t.empty or o.empty:
                    continue
                ts_ = set(filter(None, t.iloc[0]['genes'].split(";")))
                os_ = set(filter(None, o.iloc[0]['genes'].split(";")))
                ov = ts_ & os_
                overlap_rows.append({
                    'fraction': frac, 'multiplier': mult, 'pi': pi,
                    'n_tcga': len(ts_), 'n_orien': len(os_),
                    'n_overlap': len(ov), 'n_union': len(ts_ | os_),
                    'overlap_genes': ";".join(sorted(ov)),
                    'union_genes': ";".join(sorted(ts_ | os_)),
                })
    ov_df = pd.DataFrame(overlap_rows)
    ov_df.to_csv(results_dir / "cross_cohort_overlap.csv", index=False)

    logger.info(f"{'frac':>5} {'mult':>5} {'pi':>5} {'nT':>4} {'nO':>4} "
                f"{'ovlp':>5} {'union':>6}")
    for _, r in ov_df[ov_df.pi == 0.60].iterrows():
        logger.info(f"{r.fraction:>5} {r.multiplier:>5} {r.pi:>5.2f} "
                    f"{r.n_tcga:>4} {r.n_orien:>4} {r.n_overlap:>5} {r.n_union:>6}")

    # ---- Stability curve ---------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("WITHIN-COHORT STABILITY BY FRACTION (Kuncheva, top-k)")
    logger.info("=" * 70)
    est = stab_df[stab_df.estimable == True]
    if not est.empty:
        piv = est.pivot_table(index='fraction', columns='cohort',
                              values='kuncheva_topk', aggfunc='mean')
        logger.info("\n" + piv.round(3).to_string())
        logger.info("\nInterpretation: ~0 means selections agree no better than "
                    "chance given subset size. If within-cohort stability is near "
                    "zero, cross-cohort disagreement does not require a "
                    "heterogeneity explanation (Ein-Dor et al. 2006).")

    with open(results_dir / "config.json", 'w') as f:
        json.dump({
            'revision': ('b (2026-08): corrected weight direction Z=X*W, '
                         'beta-scale ranking, per-cohort RNG offsets; '
                         'supersedes the pre-audit fraction sweep'),
            'design': 'independent stratified subsampling at every fraction',
            'fractions': args.fractions, 'multipliers': args.multipliers,
            'n_fits': args.n_fits, 'pi_thresholds': PI_THRESHOLDS,
            'weakness_alpha': WEAKNESS_ALPHA, 'seed': args.seed,
            'cohort_seed_offsets': COHORT_SEED_OFFSET,
            'n_genes': p, 'phase2_provenance': provenance,
            'bound_note': ('M&B and Shah-Samworth bounds are derived for '
                           'subsamples of size n/2; no bound applies at '
                           'fraction != 0.5'),
            'stability_measures': {
                'kuncheva_topk': ('Kuncheva (2007) MS7 on equal-size top-k sets, '
                                  'ranked by |beta| on the original scale'),
                'kuncheva_gen': 'chance-corrected generalization, NOT Kuncheva',
                'jaccard': 'not chance-corrected; confounded with subset size',
            },
        }, f, indent=2, default=str)

    logger.info(f"\nDone. Results: {results_dir}")
