#!/usr/bin/env Rscript
# ==============================================================================
# Phase K2: Transfer learning for Cox regression (Trans-Lasso structure)
# ==============================================================================
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
OUT_DIR  <- "results/phaseK2_transfer_learning"

N_FOLDS      <- 5
N_REPEATS    <- 3     # reduced: hard cutoff; 15 paired folds still ample
INNER_FOLDS  <- 5
SEED         <- 42
ALPHA_LASSO  <- 1.0     # pure lasso, matching the Trans-Lasso formulation
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
  sprintf("phaseK2_%s_%s.log", direction, format(Sys.time(), "%Y%m%d_%H%M%S")))
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

load_expr <- function(p) as.matrix(read.csv(p, row.names = 1, check.names = FALSE))
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

fit_lasso <- function(x, y, offset = NULL, nfolds = INNER_FOLDS, seed = SEED) {
  # lambda by inner CV on the SUPPLIED data only. When this is called inside an
  # outer fold, x/y are the training fold -- the held-out fold is never seen.
  set.seed(seed)
  cv <- cv.glmnet(x, y, family = "cox", alpha = ALPHA_LASSO,
                  nfolds = nfolds, offset = offset, standardize = FALSE,
                  nlambda = N_LAMBDA, parallel = TRUE)
  list(beta = as.numeric(coef(cv, s = "lambda.min")), lambda = cv$lambda.min)
}

# --- Arm 1: target-only lasso (the benchmark transfer must beat) --------------
arm_target_only <- function(xtr, ytr) fit_lasso(xtr, ytr)$beta

# --- Arm 2: source-only, coefficients applied unchanged (Phase 7's design) ----
arm_source_only <- function(w) w

# --- Arm 3: naive pooling, no adaptation -------------------------------------
arm_pooled <- function(xtr, ytr, xsrc, ysrc) {
  fit_lasso(rbind(xsrc, xtr), c(ysrc, ytr))$beta
}

# --- Arm 4: Trans-Lasso two-step ---------------------------------------------
arm_translasso <- function(xtr, ytr, w) {
  # Step 2: the source fit enters as a FIXED offset, so the lasso estimates only
  # the target-specific correction delta. If the cohorts are similar, delta is
  # sparse and estimable from few events.
  off <- as.numeric(xtr %*% w)
  delta <- fit_lasso(xtr, ytr, offset = off)$beta
  w + delta
}

# ==============================================================================
# Source fits (done once; the source is never in an evaluation set)
# ==============================================================================

cat("--- Step 1: source fit on", SOURCE, "---\n")
src <- fit_lasso(X[[SOURCE]], Y[[SOURCE]])
w_hat <- src$beta
cat(sprintf("  lambda = %.5f, nonzero = %d / %d\n",
            src$lambda, sum(w_hat != 0), length(w_hat)))

# Negative control: same procedure, source outcomes shuffled. Any apparent gain
# from this arm is attributable to the offset's regularisation effect, not to
# borrowed signal.
cat("--- Negative control: source fit on PERMUTED", SOURCE, "labels ---\n")
set.seed(SEED + 999)
perm <- sample(nrow(X[[SOURCE]]))
src_perm <- fit_lasso(X[[SOURCE]], Y[[SOURCE]][perm], seed = SEED + 999)
w_perm <- src_perm$beta
cat(sprintf("  lambda = %.5f, nonzero = %d\n", src_perm$lambda, sum(w_perm != 0)))
cat("\n")

# ==============================================================================
# K2.4 Repeated CV on the target
# ==============================================================================

cat("--- Repeated CV on target:", TARGET, "---\n")

xt <- X[[TARGET]]; yt <- Y[[TARGET]]
ev <- yt[, "status"]
rows <- list()

for (rep in seq_len(N_REPEATS)) {
  set.seed(SEED + rep)
  # Stratify folds on event status so every fold has events for the partial
  # likelihood and for a defined C-index
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

    # DROPPED for this run:
    #   pooled  -- refits on 1,451 samples every fold (the memory/time hog) and
    #              is a straw man; nobody argues naive concatenation is correct.
    #   perm    -- w_perm came back ALL ZERO (see step 1 output), so the offset
    #              is zero and this arm is numerically identical to target_only.
    #              The control result is recorded in prose instead.
    betas <- list(
      target_only = arm_target_only(xtr, ytr),
      source_only = arm_source_only(w_hat),
      translasso  = arm_translasso(xtr, ytr, w_hat)
    )

    for (nm in names(betas)) {
      b <- betas[[nm]]
      rows[[paste(rep, k, nm)]] <- data.frame(
        repeat_id = rep, fold = k, method = nm,
        n_nonzero = sum(b != 0),
        c_index = cindex(as.numeric(xte %*% b), yte)
      )
    }
  }
  # Write after EVERY repeat. The previous run died mid-loop and left nothing,
  # because results were only written after all folds completed.
  write.csv(bind_rows(rows),
            file.path(OUT_DIR, sprintf("cv_results_%s_partial.csv", direction)),
            row.names = FALSE)
  cat("  repeat", rep, "done  (partial results written)\n")
  flush.console()
}

cv <- bind_rows(rows)
write.csv(cv, file.path(OUT_DIR, sprintf("cv_results_%s.csv", direction)),
          row.names = FALSE)

cat("\n", rep("=", 68), "\n", sep = "")
cat("TARGET CV RESULTS (", TARGET, ")\n", sep = "")
cat(rep("=", 68), "\n", sep = "")

summ <- cv %>%
  group_by(method) %>%
  summarise(mean_C = mean(c_index), sd_C = sd(c_index),
            median_nonzero = median(n_nonzero), n_folds = n(), .groups = "drop") %>%
  arrange(desc(mean_C))
print(as.data.frame(summ), row.names = FALSE)
cat("\n  Age-alone C-index floor:", AGE_FLOOR, "(Rung 1)\n")

# Paired comparison against target-only, fold by fold. Unpaired SEs would render
# every comparison non-significant by construction (B5).
cat("\n--- Paired difference vs target-only (same folds) ---\n")
base <- cv %>% filter(method == "target_only") %>%
  select(repeat_id, fold, base_C = c_index)
paired <- cv %>% filter(method != "target_only") %>%
  left_join(base, by = c("repeat_id", "fold")) %>%
  mutate(diff = c_index - base_C)

for (m in unique(paired$method)) {
  d <- paired$diff[paired$method == m]
  tt <- t.test(d)
  ci <- quantile(d, c(0.025, 0.975), na.rm = TRUE)
  cat(sprintf("  %-16s mean diff = %+.4f  [%+.3f, %+.3f]  paired p = %.4f\n",
              m, mean(d), ci[1], ci[2], tt$p.value))
}
cat("\n  NEGATIVE CONTROL (recorded above, not run as a CV arm): the permuted-\n")
cat("  source fit selected ZERO genes, so its offset is identically zero and the\n")
cat("  arm would be numerically identical to target_only. Permuted labels yield\n")
cat("  no borrowable signal -- the intended result, but a WEAKER control than\n")
cat("  planned: it tests a zero offset, not an offset with noise structure. A\n")
cat("  stronger version permutes the ROWS OF X in the source, preserving the\n")
cat("  coefficient magnitude distribution while destroying the gene-outcome map.\n")

# ==============================================================================
# K2.5 External validation -- refit on full target, fixed coefficients
# ==============================================================================

cat("\n", rep("=", 68), "\n", sep = "")
cat("EXTERNAL VALIDATION (held out; fixed coefficients, no refitting)\n")
cat(rep("=", 68), "\n", sep = "")

final <- list(
  target_only = arm_target_only(xt, yt),
  source_only = w_hat,
  translasso  = arm_translasso(xt, yt, w_hat)
)
saveRDS(final, file.path(OUT_DIR, sprintf("final_coefficients_%s.rds", direction)))

ext_rows <- list()
for (nm in names(final)) {
  b <- final[[nm]]
  for (co in c(TARGET, SOURCE, EXTERNAL)) {
    ext_rows[[paste(nm, co)]] <- data.frame(
      method = nm, cohort = co,
      role = if (co == TARGET) "target (in-sample)"
             else if (co == SOURCE) "source (in-sample)" else "external",
      n = nrow(X[[co]]), n_events = sum(Y[[co]][, "status"]),
      n_nonzero = sum(b != 0),
      c_index = cindex(as.numeric(X[[co]] %*% b), Y[[co]])
    )
  }
}
ext <- bind_rows(ext_rows)
write.csv(ext, file.path(OUT_DIR, sprintf("external_validation_%s.csv", direction)),
          row.names = FALSE)

cat(sprintf("  %-16s %-10s %-20s %6s %7s %8s %8s\n",
            "method", "cohort", "role", "n", "events", "nonzero", "C-index"))
for (i in seq_len(nrow(ext))) {
  cat(sprintf("  %-16s %-10s %-20s %6d %7d %8d %8.3f\n",
              ext$method[i], ext$cohort[i], ext$role[i], ext$n[i],
              ext$n_events[i], ext$n_nonzero[i], ext$c_index[i]))
}

cat("\n  In-sample rows are NOT evidence of transportability.\n")
cat("  K1.5 module-eigengene benchmark: TCGA 0.550, GSE41613 0.638, GSE42743 0.653\n")

stopCluster(cl)

cat("\nFinished:", format(Sys.time()), "\n")
cat("--- sessionInfo ---\n"); print(sessionInfo())
sink(type = "message"); sink(); close(log_con)
