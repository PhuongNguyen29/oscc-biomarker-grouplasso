#!/usr/bin/env Rscript
# ==============================================================================
# Phase K2.9: Trans-ELASTIC NET — does the base learner matter?
# ==============================================================================
#
# WHY ELASTIC NET, DECIDED BEFORE RUNNING
# ---------------------------------------
# NOT because Trans-Lasso gave a negative result. The a priori argument comes
# from K1: this data has strong correlated-group structure -- 5 TCGA and 6 ORIEN
# co-expression modules preserve above the null-control ceiling in every test
# cohort, with one-to-one Fisher mapping at p = 5.7e-280. The lasso's selection
# consistency requires the irrepresentable condition (Zhao & Yu 2006), which
# correlated predictors violate, and its within-group choice is arbitrary. The
# elastic net's grouping effect (Zou & Hastie 2005) is designed for exactly this
# case. That argument connects two chapters and predates the K2 result.
#
# PRE-SPECIFIED EXPECTATION: differences under +/-0.02, inside the fold-to-fold
# SD already observed. The K2 evidence says the binding constraint is target
# EVENT COUNT (target_only loses by 0.050 at 153 events, wins by 0.072 at 450),
# not the penalty geometry. Recorded here so the prediction can be scored.
#
# REPORTING RULE: the result is the CURVE over l1_ratio, not the best cell.
# Every value run is reported.
#
# NOT INCLUDED -- randomized lasso. Randomization (M&B 2010 Sec 3.1) does target
# correlated predictors, by weakening the irrepresentable condition. But a
# SINGLE randomized fit is a lasso with perturbed penalties: noisier for
# prediction, not better. The prediction-appropriate form is BAGGED -- average
# coefficients over B randomized draws -- which inherits the stability benefit
# and is a legitimate estimator. Left as future work.
#
# QUESTION
# --------
# Does borrowing strength from ORIEN improve estimation in TCGA, where every
# method so far has failed? K1.5's fixed-coefficient transfer reached C = 0.550
# on TCGA -- exactly the age-alone floor (0.551-0.565).
#
# WHY THIS IS WORTH ATTEMPTING
# ----------------------------
# Two prior results establish that shared signal exists to borrow:
#   E8: marginal effects transport, dev <-> ext r_true = 0.860
#   K1: 5 TCGA and 6 ORIEN co-expression modules preserve above the null-control
#       ceiling in every test cohort
# What fails to transport is SELECTION, not signal. Transfer learning targets
# exactly that gap: it borrows the source's effect structure without requiring
# the two cohorts to agree on which variables are selected.
#
# METHOD AND ITS LIMITS (K2.1) -- READ THIS BEFORE CITING
# -------------------------------------------------------
# Two-step Trans-Lasso (Li, Cai & Li 2022, JRSS-B 84:149-173, Section 2):
#
#   Step 1 (source):  w_hat = argmin  -loglik_source(X beta) + lambda_w ||beta||_1
#   Step 2 (target):  d_hat = argmin  -loglik_target(X(w_hat + delta))
#                                     + lambda_d ||delta||_1
#                     fitted with X %*% w_hat as a fixed OFFSET
#   Estimate:         beta_hat = w_hat + d_hat
#
# The intuition: w_hat carries the source's signal; delta is the target-specific
# CORRECTION. If the cohorts are similar, delta is sparse and cheap to estimate
# from few events -- which is the whole point when the target has 153 events and
# 4,702 features.
#
# *** Li, Cai & Li derive this for LINEAR regression; Tian & Feng (2023, JASA)
# *** extend it to GLMs. I am not aware of a published Cox-specific version
# *** carrying the same guarantees. This script applies the Trans-Lasso
# *** STRUCTURE to the Cox partial likelihood. The theoretical results do NOT
# *** transfer directly and must not be cited as if they do. Describe this as a
# *** method adaptation, not an application of an established survival method.
#
# NEGATIVE CONTROL
# ----------------
# A permuted-source arm shuffles ORIEN's survival labels before step 1. If
# "transfer" helps with a permuted source, the gain comes from the regularisation
# structure (an offset shrinks the target fit) rather than from borrowed signal.
# This is the control that distinguishes the two.
#
# EVALUATION (K2.4)
# -----------------
# 5-fold CV x 5 repeats on the TARGET. Within each fold, lambda is selected by
# inner CV on the TRAINING FOLD ONLY -- no exception to B6, which was written
# after finding this defect in three separate scripts. The source cohort never
# enters an evaluation set, so all of it is used at every fit.
#
# References:
#   Li S, Cai TT, Li H (2022) JRSS-B 84:149-173 -- Trans-Lasso
#   Tian Y, Feng Y (2023) JASA 118:2684-2697 -- transfer learning for GLMs
#   Simon N et al. (2011) J Stat Softw 39:1-13 -- glmnet Cox
#   Ambroise C, McLachlan GJ (2002) PNAS 99:6562-6566 -- selection bias in CV
#
# Usage:
#   Rscript scripts/phaseK2_transfer_learning.R
#   Rscript scripts/phaseK2_transfer_learning.R --direction tcga_to_orien
# ==============================================================================

suppressPackageStartupMessages({
  library(glmnet)
  library(survival)
  library(dplyr)
  library(doParallel)
})
options(stringsAsFactors = FALSE)

# ==============================================================================
# Configuration -- frozen per K2
# ==============================================================================

PROC_DIR <- "data/processed"
EXT_DIR  <- "data/external"
K1_DIR   <- "results/phaseK1_module_preservation"
OUT_DIR  <- "results/phaseK2_9_elasticnet"

N_FOLDS      <- 5
N_REPEATS    <- 3     # reduced: hard cutoff; 15 paired folds still ample
INNER_FOLDS  <- 5
SEED         <- 42
# glmnet's `alpha` IS the elastic net mixing parameter: 1 = lasso, 0 = ridge.
# The same value is used in BOTH steps -- step 1 (source) and step 2 (delta) --
# so a single number describes the estimator. Sweeping them independently would
# be a 2-D search and invites tuning on the outcome.
# alpha = 1.0 is the PURE-LASSO COMPARATOR UNDER K2.9 RULES. It does NOT
# numerically reproduce the original K2 run: K2 used bare lambda.min, whereas
# K2.9 carries the K2.8 empty-model fallback. That distinction matters here
# because K2.8 showed the apparent transfer advantage was partly target-only
# collapsing to a null model -- the very behaviour the fallback removes.
ALPHA_GRID   <- c(0.3, 0.5, 0.7, 0.9, 1.0)
N_LAMBDA     <- 50      # halved from the glmnet default of 100. lambda.min is
                        # chosen from a grid; a coarser grid shifts it slightly
                        # but does not change which arm wins. Recorded as a
                        # speed concession made under a hard time limit.
N_CORES      <- 8       # cv.glmnet parallelises over its INNER folds. More
                        # workers than folds gives nothing, and each worker
                        # holds a copy of the design matrix.
AGE_FLOOR    <- "0.551-0.565"

args <- commandArgs(trailingOnly = TRUE)
direction <- if (length(args) >= 2 && args[1] == "--direction") args[2] else "orien_to_tcga"

if (direction == "orien_to_tcga") {
  SOURCE <- "ORIEN"; TARGET <- "TCGA"
} else if (direction == "tcga_to_orien") {
  SOURCE <- "TCGA"; TARGET <- "ORIEN"
} else stop("--direction must be orien_to_tcga or tcga_to_orien")

EXTERNAL <- c("GSE41613", "GSE42743")

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
LOG_DIR <- "logs"; dir.create(LOG_DIR, showWarnings = FALSE, recursive = TRUE)
log_file <- file.path(LOG_DIR,
  sprintf("phaseK2_9_elasticnet_%s_%s.log", direction, format(Sys.time(), "%Y%m%d_%H%M%S")))
log_con <- file(log_file, open = "wt")
sink(log_con, split = TRUE); sink(log_con, type = "message")

cat("Phase K2: Transfer learning (Trans-Lasso structure, Cox)\n")
cat("Log:", log_file, "\n")
cat("Started:", format(Sys.time()), "\n")
cat("Source:", SOURCE, "-> Target:", TARGET, "\n")
cat("Evaluation:", N_REPEATS, "x", N_FOLDS, "-fold CV on target;",
    "inner", INNER_FOLDS, "-fold for lambda\n")
cat("External held out as VALIDATION, not used as sources (K2.5)\n\n")

set.seed(SEED)

cl <- makeCluster(N_CORES)
registerDoParallel(cl)
cat("Parallel workers:", N_CORES, "\n\n")

# ==============================================================================
# Data
# ==============================================================================

# Phase 1 has been revised more than once and has written both
# *_preprocessed_v4.csv and *_corrected_preprocessed_v4.csv at different times.
# Those are DIFFERENT objects: post-ComBat pre-z vs z-scored + clipped. Loading
# the wrong one would silently change the coefficient scale that x %*% w_hat
# depends on. Report what is actually on disk, and check the loaded matrix
# looks z-scored, rather than trusting the filename.
cand <- list.files(PROC_DIR, pattern = "preprocessed_v4\\.csv$", full.names = TRUE)
cat("--- Candidate preprocessed files in", PROC_DIR, "---\n")
for (f in cand) cat("   ", basename(f), "\n")
cat("\n")

load_expr <- function(p) {
  if (!file.exists(p)) stop("Expression file not found: ", p,
                            "\nSee the candidate list above.")
  m <- as.matrix(read.csv(p, row.names = 1, check.names = FALSE))
  # glmnet is called with standardize = FALSE because Phase 1 already z-scored
  # per gene. If the matrix is NOT z-scored, that assumption is wrong and every
  # coefficient is on the wrong scale.
  gm <- mean(rowMeans(m)); gs <- mean(apply(m, 1, sd))
  cat(sprintf("    %-42s %5d x %4d | mean gene mean %+.3f, mean gene sd %.3f%s\n",
              basename(p), nrow(m), ncol(m), gm, gs,
              if (abs(gm) > 0.1 || abs(gs - 1) > 0.2)
                "  <-- NOT z-scored: check the file" else ""))
  m
}
read_surv <- function(p) {
  s <- read.csv(p)
  if (!"sampleID" %in% names(s)) names(s)[1] <- "sampleID"
  s[, c("sampleID", "time", "event")]
}

expr <- list(
  TCGA     = load_expr(file.path(PROC_DIR, "tcga_preprocessed_v4.csv")),
  ORIEN    = load_expr(file.path(PROC_DIR, "orien_preprocessed_v4.csv")),
  GSE41613 = load_expr(file.path(EXT_DIR,  "GSE41613_expression.csv")),
  GSE42743 = load_expr(file.path(EXT_DIR,  "GSE42743_expression.csv"))
)
surv <- list(
  TCGA     = read_surv(file.path(PROC_DIR, "surv_tcga_aligned_v4.csv")),
  ORIEN    = read_surv(file.path(PROC_DIR, "surv_orien_aligned_v4.csv")),
  GSE41613 = read_surv(file.path(EXT_DIR,  "GSE41613_survival.csv")),
  GSE42743 = read_surv(file.path(EXT_DIR,  "GSE42743_survival.csv"))
)

# K2.6: identical feature space for every arm
genes <- Reduce(intersect, lapply(expr, rownames))
cat("Common genes across all four cohorts:", length(genes), "\n")

X <- list(); Y <- list()
for (nm in names(expr)) {
  keep <- intersect(colnames(expr[[nm]]), surv[[nm]]$sampleID)
  X[[nm]] <- t(expr[[nm]][genes, keep, drop = FALSE])          # samples x genes
  s <- surv[[nm]][match(keep, surv[[nm]]$sampleID), ]
  stopifnot(identical(rownames(X[[nm]]), s$sampleID))
  # Times must be strictly positive for the Cox partial likelihood
  ok <- s$time > 0 & !is.na(s$time) & !is.na(s$event)
  X[[nm]] <- X[[nm]][ok, , drop = FALSE]; s <- s[ok, ]
  Y[[nm]] <- Surv(s$time, s$event)
  cat(sprintf("  %-10s %4d samples, %3d events\n", nm, nrow(X[[nm]]), sum(s$event)))
}
# ORIEN alone is 14,675 x 1,112. Once X holds the transposed common-gene
# subsets, the full matrices are dead weight -- and a previous run was killed,
# most likely on memory, during the CV loop.
rm(expr); gc()
cat("\n")

# ==============================================================================
# Fitting primitives
# ==============================================================================

cindex <- function(risk, y) {
  # Higher risk should mean shorter survival, hence the negation
  as.numeric(survival::concordance(y ~ risk, reverse = TRUE)$concordance)
}

# Empty-model fallback carried over from K2.8: with few events the CV partial
# likelihood is nearly flat and lambda.min collapses to the null model, which is
# not a competitor. Falls back to the largest lambda retaining >=1 gene. The
# rule used is recorded per fit.
fit_en <- function(x, y, alpha, offset = NULL, nfolds = INNER_FOLDS, seed = SEED) {
  set.seed(seed)
  cv <- cv.glmnet(x, y, family = "cox", alpha = alpha,
                  nfolds = nfolds, offset = offset, standardize = FALSE,
                  nlambda = N_LAMBDA, parallel = TRUE)
  b <- as.numeric(coef(cv, s = "lambda.min")); rule <- "lambda.min"
  if (all(b == 0)) {
    ok <- which(cv$glmnet.fit$df >= 1)
    if (length(ok)) {
      b <- as.numeric(coef(cv$glmnet.fit, s = cv$glmnet.fit$lambda[ok[1]]))
      rule <- "fallback_min_nonzero"
    } else rule <- "empty_path"
  }
  list(beta = b, rule = rule)
}

cindex <- function(risk, y)
  as.numeric(survival::concordance(y ~ risk, reverse = TRUE)$concordance)

# ==============================================================================
# Sweep over l1_ratio. Folds are IDENTICAL across alpha values (same seed), so
# every comparison is paired -- unpaired SEs would swamp differences this small.
# ==============================================================================

xt <- X[[TARGET]]; yt <- Y[[TARGET]]
ev <- yt[, "status"]
rows <- list(); src_info <- list()

for (a in ALPHA_GRID) {
  cat("\n", rep("=", 60), "\n", sep = "")
  cat("l1_ratio =", a, ifelse(a == 1.0, " (pure lasso -- reproduces K2)", ""), "\n")
  cat(rep("=", 60), "\n", sep = "")

  # Step 1: source fit, once per alpha
  sf <- fit_en(X[[SOURCE]], Y[[SOURCE]], alpha = a)
  w_hat <- sf$beta
  cat("  source fit:", sum(w_hat != 0), "nonzero | rule:", sf$rule, "\n")

  # NEGATIVE CONTROL, refitted at EVERY alpha.
  # K2 justified dropping this on the grounds that the permuted-source lasso
  # selected zero genes, making its offset identically zero. That reasoning does
  # NOT carry to K2.9: the ridge component at alpha < 1 and the empty-model
  # fallback both make a nonzero permuted source likely. Without it, a positive
  # transfer result cannot be separated from the offset simply acting as
  # shrinkage.
  #   trans_en > target_only AND trans_en > perm_trans  => real borrowing
  #   trans_en ~ perm_trans > target_only               => regularisation artefact
  set.seed(SEED + 999)
  perm <- sample(nrow(X[[SOURCE]]))
  sfp <- fit_en(X[[SOURCE]], Y[[SOURCE]][perm], alpha = a, seed = SEED + 999)
  w_perm <- sfp$beta
  cat("  permuted source:", sum(w_perm != 0), "nonzero | rule:", sfp$rule, "\n")

  src_info[[as.character(a)]] <- data.frame(
    l1_ratio = a, source_nonzero = sum(w_hat != 0), source_rule = sf$rule,
    perm_nonzero = sum(w_perm != 0), perm_rule = sfp$rule)

  for (rep in seq_len(N_REPEATS)) {
    set.seed(SEED + rep)                       # same folds for every alpha
    fold <- integer(length(ev))
    for (lvl in unique(ev)) {
      idx <- which(ev == lvl)
      fold[idx] <- sample(rep_len(seq_len(N_FOLDS), length(idx)))
    }

    for (k in seq_len(N_FOLDS)) {
      tr <- which(fold != k); te <- which(fold == k)
      if (sum(ev[te]) < 2 || sum(ev[tr]) < 5) next
      xtr <- xt[tr, , drop = FALSE]; ytr <- yt[tr]
      xte <- xt[te, , drop = FALSE]; yte <- yt[te]

      f_t  <- fit_en(xtr, ytr, alpha = a, seed = SEED + rep)
      f_d  <- fit_en(xtr, ytr, alpha = a, offset = as.numeric(xtr %*% w_hat),
                     seed = SEED + rep)
      f_dp <- fit_en(xtr, ytr, alpha = a, offset = as.numeric(xtr %*% w_perm),
                     seed = SEED + rep)

      betas <- list(target_only = f_t$beta,
                    source_only = w_hat,
                    trans_en    = w_hat + f_d$beta,
                    perm_trans  = w_perm + f_dp$beta)
      rules <- c(target_only = f_t$rule, source_only = sf$rule,
                 trans_en = f_d$rule, perm_trans = f_dp$rule)

      for (nm in names(betas)) {
        b <- betas[[nm]]
        rows[[paste(a, rep, k, nm)]] <- data.frame(
          l1_ratio = a, repeat_id = rep, fold = k, method = nm,
          n_nonzero = sum(b != 0), lambda_rule = rules[[nm]],
          c_index = cindex(as.numeric(xte %*% b), yte))
      }
    }
  }

  # Written after every alpha: a killed run is still usable
  write.csv(bind_rows(rows),
            file.path(OUT_DIR, sprintf("elasticnet_cv_%s.csv", direction)),
            row.names = FALSE)
  cat("  done (", format(Sys.time(), "%H:%M:%S"), ") — written\n")
  flush.console()
}

cv <- bind_rows(rows)
write.csv(bind_rows(src_info),
          file.path(OUT_DIR, sprintf("elasticnet_source_fits_%s.csv", direction)),
          row.names = FALSE)

# ==============================================================================
# Report — the CURVE over l1_ratio, not the best cell
# ==============================================================================

cat("\n", rep("=", 70), "\n", sep = "")
cat("MEAN C-INDEX BY l1_ratio (", TARGET, "target,", SOURCE, "source )\n")
cat(rep("=", 70), "\n", sep = "")

tab <- cv %>% group_by(l1_ratio, method) %>%
  summarise(C = mean(c_index), sd = sd(c_index),
            nz = mean(n_nonzero), .groups = "drop")
cat(sprintf("  %8s %14s %8s %8s %8s\n",
            "l1_ratio", "method", "C", "sd", "nonzero"))
for (i in seq_len(nrow(tab))) {
  cat(sprintf("  %8.1f %14s %8.4f %8.4f %8.1f\n",
              tab$l1_ratio[i], tab$method[i], tab$C[i], tab$sd[i], tab$nz[i]))
}

cat("\n--- Paired vs l1_ratio = 1.0 (pure lasso, the K2 baseline) ---\n")
w <- cv %>%
  select(l1_ratio, repeat_id, fold, method, c_index) %>%
  tidyr::pivot_wider(names_from = l1_ratio, values_from = c_index,
                     names_prefix = "a")
for (m in unique(cv$method)) {
  d <- w[w$method == m, ]
  cat("  ", m, "\n")
  for (a in setdiff(ALPHA_GRID, 1.0)) {
    col <- paste0("a", a)
    if (!col %in% names(d)) next
    diff <- d[[col]] - d$a1
    cat(sprintf("    l1=%.1f vs 1.0: %+.4f  wins %d/%d  wilcoxon p=%.4f\n",
                a, mean(diff, na.rm = TRUE), sum(diff > 0, na.rm = TRUE),
                sum(!is.na(diff)),
                suppressWarnings(wilcox.test(d[[col]], d$a1, paired = TRUE)$p.value)))
  }
}

cat("\n--- Best method at each l1_ratio ---\n")
for (a in ALPHA_GRID) {
  t <- tab[tab$l1_ratio == a, ]
  g <- function(m) if (m %in% t$method) t$C[t$method == m] else NA_real_
  cat(sprintf("  l1=%.1f  target %.4f  source %.4f  trans %.4f  perm %.4f   best: %s\n",
              a, g("target_only"), g("source_only"), g("trans_en"),
              g("perm_trans"), t$method[which.max(t$C)]))
}

# ---- NEGATIVE CONTROL: the decisive comparison ------------------------------
cat("\n", rep("=", 70), "\n", sep = "")
cat("NEGATIVE CONTROL — is the transfer benefit real borrowing?\n")
cat(rep("=", 70), "\n", sep = "")
cat("perm_trans uses a source fitted to SHUFFLED survival labels. Any gain it\n")
cat("shows over target_only comes from the offset acting as shrinkage, not from\n")
cat("ORIEN information. trans_en must beat BOTH to support real borrowing.\n\n")

pw <- cv %>% select(l1_ratio, repeat_id, fold, method, c_index) %>%
  tidyr::pivot_wider(names_from = method, values_from = c_index)
for (a in ALPHA_GRID) {
  d <- pw[pw$l1_ratio == a, ]
  if (!nrow(d) || !"perm_trans" %in% names(d)) next
  d1 <- d$trans_en - d$perm_trans
  d2 <- d$perm_trans - d$target_only
  cat(sprintf("  l1=%.1f\n", a))
  cat(sprintf("    trans_en - perm_trans : %+.4f  wins %2d/%2d  p=%.4f\n",
              mean(d1, na.rm = TRUE), sum(d1 > 0, na.rm = TRUE), sum(!is.na(d1)),
              suppressWarnings(wilcox.test(d$trans_en, d$perm_trans,
                                           paired = TRUE)$p.value)))
  cat(sprintf("    perm_trans - target   : %+.4f  wins %2d/%2d  p=%.4f\n",
              mean(d2, na.rm = TRUE), sum(d2 > 0, na.rm = TRUE), sum(!is.na(d2)),
              suppressWarnings(wilcox.test(d$perm_trans, d$target_only,
                                           paired = TRUE)$p.value)))
  verdict <- if (mean(d1, na.rm = TRUE) > 0 &&
                 mean(d$trans_en - d$target_only, na.rm = TRUE) > 0)
    "real borrowing supported"
  else if (mean(d2, na.rm = TRUE) > 0)
    "gain attributable to the offset, NOT to source signal"
  else "no transfer benefit"
  cat("    verdict:", verdict, "\n")
}

cat("\n  PRE-SPECIFIED EXPECTATION was |difference| < 0.02 across l1_ratio,\n")
cat("  on the grounds that target event count, not penalty geometry, is the\n")
cat("  binding constraint. Compare against the table above and record whether\n")
cat("  the prediction held.\n")

stopCluster(cl)
cat("\nFinished:", format(Sys.time()), "\n")
cat("--- sessionInfo ---\n"); print(sessionInfo())
sink(type = "message"); sink(); close(log_con)
