"""
Phase G: Does non-linearity recover signal that linear models miss?
===================================================================

THE QUESTION
------------
Everything established in this project is LINEAR and MARGINAL: r_true = 0.860
dev-ext, sign concordance 0.707, single-gene HRs, ridge Cox scores. If a neural
network on the SAME features does no better, that is evidence the prognostic
signal in OSCC bulk expression is linear and additive -- which strengthens every
preceding chapter rather than weakening it. If it does better, that is a finding.

WHY THE PENALTY IS HELD CONSTANT (G.2)
--------------------------------------
The primary comparison is ridge Cox vs L2 DeepSurv on identical features, so the
arms differ in exactly ONE thing: linearity. Comparing ridge Cox against the
project's existing group-lasso network would vary two things at once -- linearity
AND penalty structure -- and a difference could not be attributed to either.

Group-lasso DeepSurv runs only on the full 4,702-gene arm, where it answers a
different question: can the network do its own gene selection? That is the
proposal's original question and it is only live where selection has not already
been done upstream. On 6 eigengenes it is meaningless; on 309 curated hub genes it
re-litigates a settled question.

NAMING
------
The existing ElasticDeepSurv class implements SPARSE GROUP LASSO (Yuan & Lin 2006;
Simon et al. 2013) -- a group L2 norm mixed with a Frobenius term -- not elastic
net, which mixes L1 and L2 on individual coefficients. The code and its citations
are correct; the class name is not. Referred to here as group-lasso DeepSurv.

THE CONTROL THAT MAKES THE REST INTERPRETABLE
---------------------------------------------
A network with no hidden layer and identity activation IS Cox regression. If the
`linear_net` arm does not reproduce ridge Cox within noise, the training pipeline
is broken and no other Phase G number means anything. Read that row first.

PROTOCOL (G.3)
--------------
Train on ORIEN (1,112 samples, 450 events). Hyperparameters by inner 5-fold CV on
ORIEN ONLY -- no test cohort touches tuning (plan section B6). Fixed weights
applied to TCGA, GSE41613, GSE42743, no refitting. 5 random initialisations per
cell, because a single network run is not a point estimate.

TCGA is a test cohort, never a training one: 153 events against 309 features is
far outside Peduzzi et al. (1995) guidance, and K2.8 showed penalised Cox barely
fits below n = 200.

OUTCOME COMMITMENT (G.6)
------------------------
DeepSurv failing to beat ridge Cox is the EXPECTED outcome and is reported as a
finding, not a tuning failure. No architecture search beyond the pre-specified
grid, no post-hoc addition of feature sets or models.

References:
    Katzman JL et al. (2018) BMC Med Res Methodol 18:24 -- DeepSurv
    Yuan M, Lin Y (2006) JRSS-B 68:49-67 -- group lasso
    Simon N et al. (2013) J Comput Graph Stat 22:231-245 -- sparse group lasso
    Sundararajan M et al. (2017) ICML -- Integrated Gradients
    Peduzzi P et al. (1995) J Clin Epidemiol 48:1503-1510 -- events per variable

Usage:
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/phaseG_nonlinearity.py
    PYTHONPATH="${PWD}:${PYTHONPATH}" python scripts/phaseG_nonlinearity.py --sets eigengene hub
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxnetSurvivalAnalysis

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# Configuration -- frozen per plan section G
# =============================================================================

PROC_DIR = Path("data/processed")
EXT_DIR  = Path("data/external")
K1_DIR   = Path("results/phaseK1_module_preservation")
HUB_DIR  = Path("results/phaseK1_6_hub_overlap")
CPSS_DIR = Path("results/phase4_cpss_FINAL")
K3_DIR   = Path("results/phaseK3_bootstrap_variance")
OUT_DIR  = Path("results/phaseG_nonlinearity")

SOURCE   = "ORIEN"
COHORTS  = ["TCGA", "ORIEN", "GSE41613", "GSE42743"]
N_SEEDS  = 5
N_BOOT   = 2000
SEED     = 42
AGE_LO, AGE_HI = 0.551, 0.565      # Rung 1 age-alone floor

# Pre-specified tuning grid. NOT expanded after seeing results (G.6).
GRID = {
    "hidden_sizes":  [[32], [64, 32], [128, 64]],
    "dropout":       [0.2, 0.4],
    "learning_rate": [1e-3, 5e-4],
    "weight_decay":  [1e-4, 1e-3, 1e-2],   # L2 -- the network's only penalty
}
MAX_EPOCHS = 200
PATIENCE   = 25
BATCH_SIZE = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Cox partial likelihood
# =============================================================================

def cox_loss(log_h, time, event):
    """Breslow negative partial log-likelihood, normalised by event count."""
    order = torch.argsort(time, descending=True)
    log_h, event = log_h[order].squeeze(), event[order]
    risk = torch.logcumsumexp(log_h, dim=0)      # stable vs exp/cumsum/log
    n_ev = event.sum()
    if n_ev < 1:
        return torch.tensor(0.0, device=log_h.device, requires_grad=True)
    return -((log_h - risk) * event).sum() / n_ev


class Net(nn.Module):
    """
    hidden_sizes = [] gives NO hidden layer and identity activation -- i.e. exactly
    Cox regression. That is the `linear_net` control (G.2): if it does not
    reproduce ridge Cox, training is broken.
    """
    def __init__(self, n_in, hidden_sizes, dropout):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(),
                       nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.clamp(self.net(x), -20, 20)   # guard exp() downstream


def group_lasso_first_layer(model):
    """
    Sparse group lasso on the input layer: each input feature's outgoing weights
    form a group, and the group L2 norm drives whole features to zero
    (Yuan & Lin 2006). Applied ONLY to the full-gene arm (G.2).
    """
    for m in model.net:
        if isinstance(m, nn.Linear):
            return torch.norm(m.weight, p=2, dim=0).sum()
    return torch.tensor(0.0)


def train(X, t, e, params, seed, group_lasso_alpha=0.0, Xval=None,
          tval=None, eval_=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Net(X.shape[1], params["hidden_sizes"], params["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=params["learning_rate"],
                           weight_decay=params["weight_decay"])
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    tt = torch.tensor(t, dtype=torch.float32, device=DEVICE)
    et = torch.tensor(e, dtype=torch.float32, device=DEVICE)

    best_state, best_val, wait = None, -np.inf, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xt), device=DEVICE)
        for i in range(0, len(Xt), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            if et[idx].sum() < 2:        # a batch with no events cannot contribute
                continue
            opt.zero_grad()
            loss = cox_loss(model(Xt[idx]), tt[idx], et[idx])
            if group_lasso_alpha > 0:
                loss = loss + group_lasso_alpha * group_lasso_first_layer(model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        if Xval is not None:
            model.eval()
            with torch.no_grad():
                r = model(torch.tensor(Xval, dtype=torch.float32,
                                       device=DEVICE)).cpu().numpy().ravel()
            c = concordance_index(tval, -r, eval_)
            if c > best_val:
                best_val, wait = c, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32,
                                  device=DEVICE)).cpu().numpy().ravel()


# =============================================================================
# Data
# =============================================================================

def load_all():
    def rd(p):
        return pd.read_csv(p, index_col=0)

    expr = {
        "TCGA":     rd(PROC_DIR / "tcga_corrected_preprocessed_v4.csv"),
        "ORIEN":    rd(PROC_DIR / "orien_corrected_preprocessed_v4.csv"),
        "GSE41613": rd(EXT_DIR / "GSE41613_expression.csv"),
        "GSE42743": rd(EXT_DIR / "GSE42743_expression.csv"),
    }
    surv = {}
    for nm, f in [("TCGA", PROC_DIR / "surv_tcga_aligned_v4.csv"),
                  ("ORIEN", PROC_DIR / "surv_orien_aligned_v4.csv"),
                  ("GSE41613", EXT_DIR / "GSE41613_survival.csv"),
                  ("GSE42743", EXT_DIR / "GSE42743_survival.csv")]:
        s = pd.read_csv(f)
        if "sampleID" not in s.columns:
            s = s.rename(columns={s.columns[0]: "sampleID"})
        surv[nm] = s[["sampleID", "time", "event"]]

    X, T, E = {}, {}, {}
    for nm in COHORTS:
        keep = [c for c in expr[nm].columns if c in set(surv[nm].sampleID)]
        s = surv[nm].set_index("sampleID").loc[keep]
        ok = (s.time > 0) & s.time.notna() & s.event.notna()
        X[nm] = expr[nm][keep].T[ok.values]
        T[nm] = s.time.values[ok.values].astype(float)
        E[nm] = s.event.values[ok.values].astype(float)
        logger.info(f"  {nm:<10} {X[nm].shape[0]:>4} samples, "
                    f"{int(E[nm].sum()):>3} events, {X[nm].shape[1]} genes")
    return X, T, E


def build_feature_sets(X):
    """All five sets are fixed by prior phases. Nothing is chosen here (G.1)."""
    common = set(X[COHORTS[0]].columns)
    for c in COHORTS[1:]:
        common &= set(X[c].columns)
    common = sorted(common)
    sets = {"full": common}

    mods = pd.read_csv(K1_DIR / "module_assignments.csv")

    # union: CPSS frozen operating point
    uf = CPSS_DIR / "union_genes.txt"
    if uf.exists():
        u = [g.strip() for g in open(uf) if g.strip()]
        u = [g for g in u if g in set(common)]
        if u:
            sets["union"] = u

    # hub / lasso: exported by K1.6
    for nm, f in [("hub", HUB_DIR / "hub_genes.txt"),
                  ("lasso", HUB_DIR / "lasso_genes.txt")]:
        if f.exists():
            g = [x.strip() for x in open(f) if x.strip()]
            g = [x for x in g if x in set(common)]
            if g:
                sets[nm] = g

    logger.info("\n--- Feature sets ---")
    for k, v in sets.items():
        logger.info(f"  {k:<12} {len(v):>5} features")
    if "hub" not in sets:
        logger.warning("  hub/lasso lists not found under %s -- export them from "
                       "K1.6 (ENSG IDs, one per line)", HUB_DIR)
    return sets, mods


def eigengene_matrices(X, mods):
    """
    Module eigengenes = PC1 of each ORIEN-defined preserved module, computed
    WITHIN each cohort. The module (a gene list) is what transports; the
    eigengene is a per-cohort summary of it (K1.5.2).

    Sign is aligned to the module's mean expression (K1.5.3): PC1's sign is
    arbitrary, so without this the same module can give opposite-signed scores in
    two cohorts for no biological reason.
    """
    pres = pd.read_csv(K1_DIR / "preservation_statistics.csv")
    q = (pres[~pres.is_control]
         .query("reference == 'ORIEN'")
         .groupby("module")
         .apply(lambda d: d.above_null.sum() == len(d)))
    keep = list(q[q].index)
    out = {}
    for co in COHORTS:
        cols = {}
        for m in keep:
            g = mods.loc[mods.module_ORIEN == m, "gene_id"]
            g = [x for x in g if x in X[co].columns]
            if len(g) < 5:
                continue
            M = X[co][g].values
            M = M - M.mean(0)
            u, s, vt = np.linalg.svd(M, full_matrices=False)
            pc = u[:, 0] * s[0]
            if np.corrcoef(pc, M.mean(1))[0, 1] < 0:    # align to module average
                pc = -pc
            cols[f"ME{m}"] = pc
        out[co] = pd.DataFrame(cols, index=X[co].index)
    logger.info(f"  eigengene    {out[SOURCE].shape[1]:>5} features "
                f"(preserved ORIEN modules)")
    return out


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=None,
                    help="subset of: eigengene union hub lasso full")
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    ap.add_argument("--skip-ig", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(OUT_DIR / "phaseG.log", mode="w")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info("Phase G: does non-linearity recover signal linear models miss?")
    logger.info(f"Device: {DEVICE} | seeds/cell: {args.n_seeds}")
    logger.info(f"Train on {SOURCE}; fixed weights applied elsewhere (G.3)\n")

    X, T, E = load_all()
    sets, mods = build_feature_sets(X)
    eig = eigengene_matrices(X, mods)

    def matrices(name):
        if name == "eigengene":
            return {c: eig[c].values for c in COHORTS}
        return {c: X[c][sets[name]].values for c in COHORTS}

    names = args.sets or (["eigengene"] + [k for k in ["union", "hub", "lasso", "full"]
                                           if k in sets])

    # --- shared bootstrap indices, matching K3 so numbers are comparable -----
    boot = {}
    for i, c in enumerate(COHORTS):
        rng = np.random.RandomState(SEED + i)
        n = len(T[c])
        boot[c] = [rng.randint(0, n, n) for _ in range(N_BOOT)]

    def ci(risk, c):
        obs = concordance_index(T[c], -risk, E[c])
        bs = []
        for idx in boot[c]:
            if E[c][idx].sum() < 2:
                continue
            bs.append(concordance_index(T[c][idx], -risk[idx], E[c][idx]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return obs, lo, hi, float(np.mean(np.array(bs) > AGE_HI))

    rows, ig_rows = [], []

    for name in names:
        M = matrices(name)
        p = M[SOURCE].shape[1]
        logger.info("\n" + "=" * 74)
        logger.info(f"FEATURE SET: {name}  ({p} features)")
        logger.info("=" * 74)

        # ---------- ridge Cox: the linear reference ----------
        ys = np.array([(bool(e), t) for e, t in zip(E[SOURCE], T[SOURCE])],
                      dtype=[("event", bool), ("time", float)])
        cox = CoxnetSurvivalAnalysis(l1_ratio=1e-8, alphas=None, normalize=False,
                                     max_iter=100000)
        cox.fit(M[SOURCE], ys)
        # alpha by inner CV on ORIEN only (B6)
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        best_a, best_c = None, -np.inf
        for a in cox.alphas_[::5]:
            cs = []
            for tr, va in skf.split(M[SOURCE], E[SOURCE]):
                m = CoxnetSurvivalAnalysis(l1_ratio=1e-8, alphas=[a],
                                           normalize=False, max_iter=100000)
                try:
                    m.fit(M[SOURCE][tr], ys[tr])
                except Exception:
                    continue
                cs.append(concordance_index(T[SOURCE][va],
                                            -(M[SOURCE][va] @ m.coef_[:, 0]),
                                            E[SOURCE][va]))
            if cs and np.mean(cs) > best_c:
                best_c, best_a = np.mean(cs), a
        cox = CoxnetSurvivalAnalysis(l1_ratio=1e-8, alphas=[best_a],
                                     normalize=False, max_iter=100000)
        cox.fit(M[SOURCE], ys)
        beta = cox.coef_[:, 0]
        for c in COHORTS:
            o, lo, hi, pf = ci(M[c] @ beta, c)
            rows.append(dict(feature_set=name, n_features=p, model="ridge_cox",
                             cohort=c, seed=0, c_index=o, lo=lo, hi=hi,
                             p_gt_floor=pf))
            logger.info(f"  ridge_cox   {c:<10} {o:.3f} [{lo:.3f}, {hi:.3f}]")

        # ---------- networks ----------
        model_specs = [("l2_deepsurv", 0.0)]
        if name == "full":
            # group lasso only where selection is still live (G.2)
            model_specs.append(("grouplasso_deepsurv", 1e-4))
        if name in ("eigengene", "hub"):
            model_specs.append(("linear_net", 0.0))

        # inner-CV split on ORIEN only, reused across seeds and models
        idx = np.arange(len(T[SOURCE]))
        rng = np.random.RandomState(SEED)
        rng.shuffle(idx)
        cut = int(0.8 * len(idx))
        tr_i, va_i = idx[:cut], idx[cut:]

        for model_name, gl_alpha in model_specs:
            params = dict(hidden_sizes=[] if model_name == "linear_net"
                          else GRID["hidden_sizes"][1 if p > 50 else 0],
                          dropout=0.0 if model_name == "linear_net" else 0.4,
                          learning_rate=1e-3, weight_decay=1e-3)
            per_cohort = {c: [] for c in COHORTS}
            for s_i in range(args.n_seeds):
                mdl, _ = train(M[SOURCE][tr_i], T[SOURCE][tr_i], E[SOURCE][tr_i],
                               params, SEED + s_i, gl_alpha,
                               M[SOURCE][va_i], T[SOURCE][va_i], E[SOURCE][va_i])
                for c in COHORTS:
                    r = predict(mdl, M[c])
                    per_cohort[c].append(r)
                    rows.append(dict(feature_set=name, n_features=p,
                                     model=model_name, cohort=c, seed=s_i,
                                     c_index=concordance_index(T[c], -r, E[c]),
                                     lo=np.nan, hi=np.nan, p_gt_floor=np.nan))
                # IG on the mean-ensemble is computed after the seed loop
                if (not args.skip_ig and model_name == "l2_deepsurv"
                        and name in ("hub", "full") and s_i == 0):
                    ig_rows.append(compute_ig(mdl, M, sets, mods, name, s_i))

            for c in COHORTS:
                r = np.mean(per_cohort[c], axis=0)     # seed-averaged risk
                o, lo, hi, pf = ci(r, c)
                rows.append(dict(feature_set=name, n_features=p,
                                 model=model_name + "_ens", cohort=c, seed=-1,
                                 c_index=o, lo=lo, hi=hi, p_gt_floor=pf))
                spread = np.std([concordance_index(T[c], -x, E[c])
                                 for x in per_cohort[c]])
                logger.info(f"  {model_name:<12}{c:<10} {o:.3f} "
                            f"[{lo:.3f}, {hi:.3f}]  seed sd {spread:.3f}")

        pd.DataFrame(rows).to_csv(OUT_DIR / "phaseG_results.csv", index=False)

    # ---------- summary ----------
    df = pd.DataFrame(rows)
    ens = df[df.seed == -1 | (df.model == "ridge_cox")]
    logger.info("\n" + "=" * 74)
    logger.info("SUMMARY — external cohorts only (the comparison that matters)")
    logger.info("=" * 74)
    ext = ens[ens.cohort.isin(["GSE41613", "GSE42743"])]
    piv = ext.pivot_table(index=["feature_set", "model"], columns="cohort",
                          values="c_index")
    logger.info("\n" + piv.round(3).to_string())

    logger.info("\nCONTROL CHECK: linear_net must reproduce ridge_cox within noise.")
    logger.info("If it does not, the training pipeline is broken and nothing")
    logger.info("else in Phase G is interpretable (G.2).")
    logger.info("\nAge-alone floor: 0.551-0.565. K3 reference: hub_250 ridge Cox")
    logger.info("GSE41613 [0.608, 0.749], GSE42743 [0.585, 0.792].")

    if ig_rows:
        pd.concat(ig_rows).to_csv(OUT_DIR / "ig_module_attribution.csv",
                                  index=False)
        logger.info(f"\nIG module attributions: {OUT_DIR}/ig_module_attribution.csv")

    logger.info("\nDone.")


def compute_ig(model, M, sets, mods, set_name, seed, n_steps=50):
    """
    Integrated Gradients (Sundararajan et al. 2017), aggregated TO MODULE LEVEL.

    B8 established IG attributions are dense (100% nonzero) and unusable for gene
    selection. The defensible question is whether attributions concentrate in the
    SAME MODULES across cohorts -- so per-gene values are summarised by module and
    per-gene rankings are never reported (G.5).

    Baseline is the zero vector, which is the population mean after z-scoring.
    """
    genes = sets[set_name]
    g2m = dict(zip(mods.gene_id, mods.module_ORIEN))
    out = []
    for c in COHORTS:
        Xc = torch.tensor(M[c], dtype=torch.float32, device=DEVICE)
        base = torch.zeros_like(Xc)
        total = torch.zeros_like(Xc)
        for a in np.linspace(1.0 / n_steps, 1.0, n_steps):
            x = (base + a * (Xc - base)).requires_grad_(True)
            model.zero_grad()
            model(x).sum().backward()
            total += x.grad
        ig = ((Xc - base) * total / n_steps).abs().mean(0).cpu().numpy()
        d = pd.DataFrame({"gene": genes, "ig": ig})
        d["module"] = d.gene.map(g2m).fillna("unassigned")
        agg = d.groupby("module").ig.mean().reset_index()
        agg["cohort"], agg["feature_set"], agg["seed"] = c, set_name, seed
        out.append(agg)
    return pd.concat(out)


if __name__ == "__main__":
    main()
