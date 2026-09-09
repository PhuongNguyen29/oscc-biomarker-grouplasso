# ==============================================================================
# Hub-set size (k) sweep — selection inside ORIEN, evaluation once
#
# WHY THIS EXISTS. k = 40 per module per cohort was chosen so the hub set would
# be size-matched to the penalized comparator, not by any criterion of its own.
# This script replaces that with a stated criterion, WITHOUT using the external
# cohorts to choose.
#
# PRE-REGISTERED PROTOCOL (fix before running; do not amend after seeing output)
#   1. SELECTION uses ORIEN only: for each k, repeated 5-fold cross-validation
#      of the ridge Cox model within ORIEN (N_REPEATS repeats, fixed seeds).
#      The chosen k is the SMALLEST k whose mean CV C-index is within one
#      standard error of the best (the usual parsimony rule). Fixing the
#      tie-break now is the point: "best mean" and "1-SE" can differ, and
#      choosing between them afterwards would be selection by inspection.
#   2. EVALUATION happens once, at the chosen k, on TCGA and the two GEO
#      cohorts, with coefficients fitted on full ORIEN and applied unchanged.
#   3. The full sweep of EXTERNAL performance across all k is also written, and
#      is labelled POST-HOC. It is reported for transparency and must not be
#      used to revise the choice of k.
#   Expected reading: external performance broadly flat across k, degrading at
#   very small k (too few genes) and very large k (peripheral genes diluting
#   hubs). If instead it rises monotonically with k, that is reportable and
#   would indicate the size-matching constraint cost predictive performance.
#
# HOW TO RUN. This script assumes the phaseK3 environment is already loaded:
# X, Y, kme, mods, pairs, hub_set(), SEED, OUT_DIR. Run phaseK3 first (or
# source it), then source this file. It writes to its own directory and does
# not modify any phaseK3 artifact.
# ==============================================================================

suppressPackageStartupMessages({
  library(glmnet); library(survival); library(dplyr)
})

stopifnot(exists("X"), exists("Y"), exists("kme"), exists("pairs"),
          exists("hub_set"), exists("SEED"))

K_GRID     <- c(20, 30, 40, 50, 60, 70)
N_REPEATS  <- 20          # CV repeats inside ORIEN
N_FOLDS    <- 5
N_BOOT     <- 2000        # matches phaseK3
SWEEP_DIR  <- file.path(OUT_DIR, "k_sweep")
dir.create(SWEEP_DIR, recursive = TRUE, showWarnings = FALSE)

cidx <- function(risk, y) as.numeric(concordance(y ~ risk, reverse = TRUE)$concordance)

fit_hub <- function(g, seed = SEED) {
  set.seed(seed)
  cv <- cv.glmnet(X$ORIEN[, g, drop = FALSE], Y$ORIEN, family = "cox",
                  alpha = 0, nfolds = 10, standardize = FALSE,
                  cox.ties = "breslow")
  as.numeric(coef(cv, s = "lambda.min"))
}

# ==============================================================================
# STEP 1 — selection: repeated CV inside ORIEN only
# ==============================================================================

cat("STEP 1 — selecting k by repeated", N_FOLDS, "-fold CV inside ORIEN\n")
cv_rows <- list()

for (k in K_GRID) {
  g <- hub_set(k)
  g <- Reduce(intersect, c(list(g), lapply(X, colnames)))
  n_ev <- sum(Y$ORIEN[, "status"])

  reps <- numeric(N_REPEATS)
  for (r in seq_len(N_REPEATS)) {
    set.seed(1000 * r + k)
    fold <- sample(rep(seq_len(N_FOLDS), length.out = nrow(X$ORIEN)))
    oof <- rep(NA_real_, nrow(X$ORIEN))
    for (f in seq_len(N_FOLDS)) {
      tr <- fold != f; te <- !tr
      set.seed(SEED)
      cvf <- cv.glmnet(X$ORIEN[tr, g, drop = FALSE], Y$ORIEN[tr], family = "cox",
                       alpha = 0, nfolds = 10, standardize = FALSE,
                       cox.ties = "breslow")
      bf <- as.numeric(coef(cvf, s = "lambda.min"))
      oof[te] <- as.numeric(X$ORIEN[te, g, drop = FALSE] %*% bf)
    }
    reps[r] <- cidx(oof, Y$ORIEN)
  }

  cv_rows[[as.character(k)]] <- data.frame(
    k = k, n_genes = length(g),
    cv_mean = mean(reps), cv_sd = sd(reps),
    cv_se = sd(reps) / sqrt(N_REPEATS))
  cat(sprintf("  k = %2d  %3d genes  ORIEN CV C = %.4f (SE %.4f)\n",
              k, length(g), mean(reps), sd(reps) / sqrt(N_REPEATS)))
}

cvdf <- bind_rows(cv_rows)
write.csv(cvdf, file.path(SWEEP_DIR, "k_selection_orien_cv.csv"), row.names = FALSE)

# 1-SE rule: smallest k within one SE of the best mean
best_i    <- which.max(cvdf$cv_mean)
threshold <- cvdf$cv_mean[best_i] - cvdf$cv_se[best_i]
k_star    <- min(cvdf$k[cvdf$cv_mean >= threshold])

cat(sprintf("\n  best mean at k = %d (C = %.4f); 1-SE threshold %.4f\n",
            cvdf$k[best_i], cvdf$cv_mean[best_i], threshold))
cat(sprintf("  SELECTED k = %d (smallest within 1 SE)\n\n", k_star))

# ==============================================================================
# STEP 2 — evaluation at the selected k, once
# ==============================================================================

cat("STEP 2 — evaluating the selected hub set on held-out cohorts\n")
g_star <- Reduce(intersect, c(list(hub_set(k_star)), lapply(X, colnames)))
b_star <- fit_hub(g_star)

write.csv(data.frame(gene = g_star, beta = b_star),
          file.path(SWEEP_DIR, sprintf("coefs_hub_k%d.csv", k_star)), row.names = FALSE)

eval_rows <- list()
for (co in names(X)) {
  r <- as.numeric(X[[co]][, g_star, drop = FALSE] %*% b_star)
  pt <- cidx(r, Y[[co]])
  set.seed(SEED)
  bs <- replicate(N_BOOT, {
    i <- sample(length(r), replace = TRUE)
    if (sum(Y[[co]][i, "status"]) < 2) return(NA_real_)
    cidx(r[i], Y[[co]][i])
  })
  ci <- quantile(bs, c(0.025, 0.975), na.rm = TRUE)
  eval_rows[[co]] <- data.frame(k = k_star, cohort = co, n_genes = length(g_star),
                                C = pt, lo = ci[1], hi = ci[2],
                                role = ifelse(co == "ORIEN", "source (in-sample)",
                                              "held out"))
  cat(sprintf("  %-9s C = %.3f [%.3f, %.3f]  %s\n", co, pt, ci[1], ci[2],
              eval_rows[[co]]$role))
}
write.csv(bind_rows(eval_rows),
          file.path(SWEEP_DIR, sprintf("evaluation_at_k%d.csv", k_star)),
          row.names = FALSE)

# ==============================================================================
# STEP 3 — POST-HOC sweep of external performance across all k
#          Reported for transparency. MUST NOT revise the choice above.
# ==============================================================================

cat("\nSTEP 3 — post-hoc: external performance across all k (NOT used to choose k)\n")
post_rows <- list()
for (k in K_GRID) {
  g <- Reduce(intersect, c(list(hub_set(k)), lapply(X, colnames)))
  b <- fit_hub(g)
  for (co in names(X)) {
    r <- as.numeric(X[[co]][, g, drop = FALSE] %*% b)
    post_rows[[paste(k, co)]] <- data.frame(k = k, cohort = co,
                                            n_genes = length(g), C = cidx(r, Y[[co]]))
  }
}
post <- bind_rows(post_rows)
write.csv(post, file.path(SWEEP_DIR, "POSTHOC_external_by_k.csv"), row.names = FALSE)

cat("\n  C-index by k (post-hoc, for transparency only):\n")
print(tidyr::pivot_wider(post, id_cols = k, names_from = cohort, values_from = C) %>%
        as.data.frame(), row.names = FALSE, digits = 3)

cat(sprintf("\nSelected k = %d by ORIEN-internal CV (1-SE rule), fixed before evaluation.\n", k_star))
cat("Written to:", SWEEP_DIR, "\n")
