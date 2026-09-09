#!/usr/bin/env Rscript
# ==============================================================================
# Phase K2.8: Transfer learning curve — when is borrowing worth it?
# ==============================================================================
#
# THE QUESTION
# ------------
# K2 produced an apparent contradiction:
#   - In 5x3 CV on TCGA (n_train = 271), Trans-Lasso beat target-only by +0.063
#     (paired p = 0.002), and source-only beat it by +0.050.
#   - Refit on the FULL 339 TCGA samples and evaluated externally, target-only
#     matched or beat both (GSE42743: 0.657 vs 0.626 / 0.628).
#
# These are not in conflict. The benefit of transfer is a FUNCTION OF TARGET
# SAMPLE SIZE, and the two evaluations sit on opposite sides of a crossover.
# This script measures that function directly.
#
# THE QUANTITY OF INTEREST: TRANSFER BREAK-EVEN SAMPLE SIZE
# ---------------------------------------------------------
#   source_only  is FLAT in target n -- it never sees the target at all. On the
#                external cohorts its C-index is a constant.
#   target_only  RISES with n.
# They cross. Below the crossing, borrowing a source model beats fitting your
# own; above it, borrowing is not worth the assumption. The K2 results bracket
# the crossing between n = 271 and n = 339 -- which is the size of most
# published OSCC cohorts, and therefore a number a reader can act on.
#
# This also TESTS Trans-Lasso rather than demonstrating it. The theory predicts
# it should track source_only at low n, track target_only at high n, and beat
# both in between. Whether it does is an empirical question.
#
# DESIGN
# ------
# For each draw: sample n TCGA patients (stratified on event status) as the
# training set; the REMAINDER of TCGA is the internal held-out set. Both
# external cohorts are fixed and are never touched by any training step, so
# they provide a clean evaluation at every n.
#
#   arms      : target_only, source_only, translasso  (pooled dropped -- straw man)
#   source    : ORIEN, fitted ONCE on all 1,112 samples (w_hat is constant across
#               the whole experiment, which is exactly why its curve is flat)
#   evaluation: internal held-out TCGA (n_test shrinks as n grows -- reported,
#               and noisy at the top end) + GSE41613 + GSE42743 (fixed)
#
# lambda is always selected by inner CV on the TRAINING SUBSAMPLE ONLY (B6).
#
# GRACEFUL DEGRADATION
# --------------------
# The loop is draws-OUTER, sizes-INNER, and results are written after every
# draw. One completed draw already gives the full curve; further draws only
# tighten it. A run killed part-way is still usable -- the previous K2 run died
# mid-loop and left nothing.
#
# References:
#   Li S, Cai TT, Li H (2022) JRSS-B 84:149-173 -- Trans-Lasso
#   Simon N et al. (2011) J Stat Softw 39:1-13 -- glmnet Cox
#
# Usage:
#   nohup Rscript scripts/phaseK2_8_learning_curve.R > logs/k2_8.log 2>&1 &
# ==============================================================================

suppressPackageStartupMessages({
  library(glmnet); library(survival); library(dplyr); library(doParallel)
})
options(stringsAsFactors = FALSE)

PROC_DIR <- "data/processed"
EXT_DIR  <- "data/external"
OUT_DIR  <- "results/phaseK2_8_learning_curve"

SOURCE   <- "ORIEN"
TARGET   <- "TCGA"
EXTERNAL <- c("GSE41613", "GSE42743")

N_GRID       <- c(50, 100, 150, 200, 250, 300)
N_DRAWS      <- 10
INNER_FOLDS  <- 10      # raised from 5: at n=50 (23 events) 5-fold CV leaves
                        # ~4 events per held-out fold, the partial likelihood is
                        # nearly flat, and lambda.min collapses to the null model
N_LAMBDA     <- 50
ALPHA_LASSO  <- 1.0
N_CORES      <- 8
SEED         <- 42

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create("logs", showWarnings = FALSE)

cat("Phase K2.8: transfer learning curve\n")
cat("Started:", format(Sys.time()), "\n")
cat("n grid:", paste(N_GRID, collapse = ", "), "| draws:", N_DRAWS, "\n\n")

set.seed(SEED)
cl <- makeCluster(N_CORES); registerDoParallel(cl)

# ==============================================================================
# Data
# ==============================================================================

load_expr <- function(p) as.matrix(read.csv(p, row.names = 1, check.names = FALSE))
read_surv <- function(p) {
  s <- read.csv(p); if (!"sampleID" %in% names(s)) names(s)[1] <- "sampleID"
  s[, c("sampleID", "time", "event")]
}

expr <- list(
  TCGA     = load_expr(file.path(PROC_DIR, "tcga_corrected_preprocessed_v4.csv")),
  ORIEN    = load_expr(file.path(PROC_DIR, "orien_corrected_preprocessed_v4.csv")),
  GSE41613 = load_expr(file.path(EXT_DIR,  "GSE41613_expression.csv")),
  GSE42743 = load_expr(file.path(EXT_DIR,  "GSE42743_expression.csv"))
)
surv <- list(
  TCGA     = read_surv(file.path(PROC_DIR, "surv_tcga_aligned_v4.csv")),
  ORIEN    = read_surv(file.path(PROC_DIR, "surv_orien_aligned_v4.csv")),
  GSE41613 = read_surv(file.path(EXT_DIR,  "GSE41613_survival.csv")),
  GSE42743 = read_surv(file.path(EXT_DIR,  "GSE42743_survival.csv"))
)

genes <- Reduce(intersect, lapply(expr, rownames))
cat("Common genes:", length(genes), "\n")

X <- list(); Y <- list()
for (nm in names(expr)) {
  keep <- intersect(colnames(expr[[nm]]), surv[[nm]]$sampleID)
  Xn <- t(expr[[nm]][genes, keep, drop = FALSE])
  s  <- surv[[nm]][match(keep, surv[[nm]]$sampleID), ]
  ok <- s$time > 0 & !is.na(s$time) & !is.na(s$event)
  X[[nm]] <- Xn[ok, , drop = FALSE]; s <- s[ok, ]
  Y[[nm]] <- Surv(s$time, s$event)
  cat(sprintf("  %-10s %4d samples, %3d events\n", nm, nrow(X[[nm]]), sum(s$event)))
}
rm(expr); gc()      # the full matrices are dead weight once X is built
cat("\n")

cindex <- function(risk, y)
  as.numeric(survival::concordance(y ~ risk, reverse = TRUE)$concordance)

# ------------------------------------------------------------------------------
# EMPTY-MODEL FALLBACK -- added after draw 1 of the first run
#
# In that run target_only selected ZERO genes at n = 50, 150 and 250, scoring
# C = 0.5 by default, with no monotone trend in n. That is not small-sample
# behaviour of the lasso; it is lambda selection collapsing. With few events the
# cross-validated partial likelihood is nearly flat and its minimum lands on the
# null model.
#
# An arm that returns no model is not a competitor, and the apparent transfer
# advantage in K2 was largely target_only failing to fit rather than transfer
# adding information. Fallback: if lambda.min gives an empty model, take the
# LARGEST lambda (strongest penalty) retaining at least one gene -- the sparsest
# non-trivial model the path contains.
#
# This is a CONVENTION, not a better estimator. It is recorded per fit in
# `lambda_rule` so the frequency of fallback is reportable: if an arm relies on
# it often, that is itself the finding about fittability at that sample size.
# ------------------------------------------------------------------------------
fit_lasso <- function(x, y, offset = NULL, seed = SEED) {
  set.seed(seed)
  cv <- cv.glmnet(x, y, family = "cox", alpha = ALPHA_LASSO,
                  nfolds = INNER_FOLDS, offset = offset, standardize = FALSE,
                  nlambda = N_LAMBDA, parallel = TRUE, cox.ties = "breslow")
  b <- as.numeric(coef(cv, s = "lambda.min"))
  rule <- "lambda.min"

  if (all(b == 0)) {
    df <- cv$glmnet.fit$df                 # nonzero count along the path
    ok <- which(df >= 1)
    if (length(ok)) {
      # lambda is stored in decreasing order, so the first index with df >= 1
      # is the largest lambda giving a non-empty model
      b <- as.numeric(coef(cv$glmnet.fit, s = cv$glmnet.fit$lambda[ok[1]]))
      rule <- "fallback_min_nonzero"
    } else {
      rule <- "empty_path"                 # no lambda on the path selects anything
    }
  }
  list(beta = b, rule = rule)
}

# ==============================================================================
# Source fit: ONCE. w_hat is constant for the entire experiment, which is
# precisely why the source_only curve is flat in target n.
# ==============================================================================

cat("--- Source fit on", SOURCE, "(once) ---\n")
src_fit <- fit_lasso(X[[SOURCE]], Y[[SOURCE]])
w_hat <- src_fit$beta
cat("  nonzero:", sum(w_hat != 0), "/", length(w_hat),
    "| rule:", src_fit$rule, "\n\n")

# ==============================================================================
# Learning curve
# ==============================================================================

xt <- X[[TARGET]]; yt <- Y[[TARGET]]
ev <- yt[, "status"]
rows <- list()

for (draw in seq_len(N_DRAWS)) {
  for (n in N_GRID) {
    if (n >= nrow(xt)) next
    set.seed(SEED + draw * 1000 + n)

    # Stratify on event status so the event RATE is held constant as n varies.
    # Without this, small draws would vary in event count as well as size and
    # the curve would confound the two.
    n_ev <- max(2, round(n * mean(ev)))
    n_cn <- n - n_ev
    idx <- c(sample(which(ev == 1), min(n_ev, sum(ev == 1))),
             sample(which(ev == 0), min(n_cn, sum(ev == 0))))
    te <- setdiff(seq_len(nrow(xt)), idx)
    if (sum(ev[te]) < 5) next

    xtr <- xt[idx, , drop = FALSE]; ytr <- yt[idx]

    f_target <- fit_lasso(xtr, ytr, seed = SEED + draw)
    f_delta  <- fit_lasso(xtr, ytr, offset = as.numeric(xtr %*% w_hat),
                          seed = SEED + draw)
    b_target <- f_target$beta
    b_trans  <- w_hat + f_delta$beta
    rules <- c(target_only = f_target$rule, source_only = src_fit$rule,
               translasso = f_delta$rule)

    evals <- list(
      TCGA_heldout = list(x = xt[te, , drop = FALSE], y = yt[te]),
      GSE41613     = list(x = X$GSE41613, y = Y$GSE41613),
      GSE42743     = list(x = X$GSE42743, y = Y$GSE42743)
    )

    for (arm in c("target_only", "source_only", "translasso")) {
      b <- switch(arm, target_only = b_target, source_only = w_hat,
                  translasso = b_trans)
      for (ev_nm in names(evals)) {
        e <- evals[[ev_nm]]
        rows[[paste(draw, n, arm, ev_nm)]] <- data.frame(
          draw = draw, n_train = n, n_events_train = sum(ev[idx]),
          arm = arm, eval_set = ev_nm,
          n_test = nrow(e$x), n_events_test = sum(e$y[, "status"]),
          n_nonzero = sum(b != 0),
          lambda_rule = rules[[arm]],
          c_index = cindex(as.numeric(e$x %*% b), e$y)
        )
      }
    }
  }

  # Written after EVERY draw: one draw already gives the full curve.
  write.csv(bind_rows(rows), file.path(OUT_DIR, "learning_curve.csv"),
            row.names = FALSE)
  cat("  draw", draw, "of", N_DRAWS, "done  (", format(Sys.time(), "%H:%M:%S"),
      ") — results written\n")
  flush.console()
}

lc <- bind_rows(rows)

# ==============================================================================
# Report
# ==============================================================================

cat("\n", rep("=", 70), "\n", sep = "")
cat("LEARNING CURVE: mean C-index by training size\n")
cat(rep("=", 70), "\n", sep = "")

for (ev_nm in unique(lc$eval_set)) {
  cat("\n### evaluated on", ev_nm, "###\n")
  tab <- lc %>% filter(eval_set == ev_nm) %>%
    group_by(n_train, arm) %>%
    summarise(C = mean(c_index), sd = sd(c_index), .groups = "drop") %>%
    tidyr::pivot_wider(names_from = arm, values_from = c(C, sd))
  print(as.data.frame(tab), row.names = FALSE, digits = 3)
}

cat("\n--- How often did the empty-model fallback fire? ---\n")
cat("An arm relying on it often is not reliably fittable at that sample size,\n")
cat("which qualifies any comparison against it.\n\n")
fb <- lc %>% filter(eval_set == "GSE41613") %>%
  group_by(n_train, arm) %>%
  summarise(n_fits = n(), fallback = sum(lambda_rule == "fallback_min_nonzero"),
            empty = sum(lambda_rule == "empty_path"),
            mean_nonzero = mean(n_nonzero), .groups = "drop")
print(as.data.frame(fb), row.names = FALSE, digits = 3)

cat("\n", rep("=", 70), "\n", sep = "")
cat("TRANSFER BREAK-EVEN SAMPLE SIZE\n")
cat(rep("=", 70), "\n", sep = "")
cat("The n at which target_only overtakes source_only. Below it, borrowing a\n")
cat("source model beats fitting your own.\n\n")

for (ev_nm in unique(lc$eval_set)) {
  m <- lc %>% filter(eval_set == ev_nm) %>%
    group_by(n_train, arm) %>% summarise(C = mean(c_index), .groups = "drop") %>%
    tidyr::pivot_wider(names_from = arm, values_from = C) %>% arrange(n_train)
  d <- m$target_only - m$source_only
  cross <- which(d[-length(d)] < 0 & d[-1] > 0)
  cat(sprintf("  %-14s ", ev_nm))
  if (length(cross)) {
    lo <- m$n_train[cross[1]]; hi <- m$n_train[cross[1] + 1]
    # linear interpolation between the bracketing grid points
    be <- lo + (hi - lo) * (-d[cross[1]]) / (d[cross[1] + 1] - d[cross[1]])
    cat(sprintf("break-even n ~ %.0f  (between %d and %d)\n", be, lo, hi))
  } else if (all(d < 0)) {
    cat("target_only never overtakes source_only in this range (n <= ",
        max(m$n_train), ")\n", sep = "")
  } else {
    cat("target_only already ahead at the smallest n tested\n")
  }
}

cat("\n--- Does Trans-Lasso beat BOTH single-source arms? ---\n")
cat("Theory predicts: tracks source_only at low n, target_only at high n,\n")
cat("dominates both in between. Testing that, not assuming it.\n\n")
for (ev_nm in unique(lc$eval_set)) {
  m <- lc %>% filter(eval_set == ev_nm) %>%
    group_by(n_train, arm) %>% summarise(C = mean(c_index), .groups = "drop") %>%
    tidyr::pivot_wider(names_from = arm, values_from = C) %>% arrange(n_train)
  cat("  ", ev_nm, "\n")
  for (i in seq_len(nrow(m))) {
    best <- c("target_only", "source_only", "translasso")[
      which.max(c(m$target_only[i], m$source_only[i], m$translasso[i]))]
    cat(sprintf("    n=%3d  target %.3f  source %.3f  trans %.3f   best: %s\n",
                m$n_train[i], m$target_only[i], m$source_only[i],
                m$translasso[i], best))
  }
}

# ---- Figure -----------------------------------------------------------------
try({
  png(file.path(OUT_DIR, "learning_curve.png"), width = 1500, height = 520, res = 130)
  par(mfrow = c(1, 3), mar = c(4.2, 4.2, 3, 1))
  cols <- c(target_only = "#1b6ca8", source_only = "#c1272d", translasso = "#2e8b57")
  for (ev_nm in unique(lc$eval_set)) {
    m <- lc %>% filter(eval_set == ev_nm) %>%
      group_by(n_train, arm) %>% summarise(C = mean(c_index), .groups = "drop")
    plot(NA, xlim = range(N_GRID), ylim = c(0.45, 0.75),
         xlab = "TCGA training samples", ylab = "C-index", main = ev_nm)
    abline(h = c(0.551, 0.565), col = "grey70", lty = 3)
    for (a in names(cols)) {
      d <- m[m$arm == a, ]
      lines(d$n_train, d$C, col = cols[a], lwd = 2, type = "b", pch = 19)
    }
    if (ev_nm == unique(lc$eval_set)[1])
      legend("bottomright", names(cols), col = cols, lwd = 2, bty = "n", cex = 0.8)
  }
  dev.off()
  cat("\nFigure:", file.path(OUT_DIR, "learning_curve.png"), "\n")
}, silent = TRUE)

stopCluster(cl)
cat("\nFinished:", format(Sys.time()), "\n")
