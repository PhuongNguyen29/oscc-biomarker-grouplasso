#!/usr/bin/env Rscript
# ==============================================================================
# Phase K3: Bootstrap intervals + TCGA variance-explained diagnostic
# ==============================================================================
#
# TWO SMALL ANALYSES, BOTH LOAD-BEARING
# -------------------------------------
#
# (1) BOOTSTRAP CONFIDENCE INTERVALS on every external C-index claimed so far.
#
#     The project currently rests on point estimates from 51 and 40 events:
#       K1.5  module eigengene score : TCGA 0.550, GSE41613 0.638, GSE42743 0.653
#       K1.6.7 hub_250 ridge Cox     : TCGA 0.553, GSE41613 0.680, GSE42743 0.692
#       K1.6.7 lasso_250 ridge Cox   : TCGA 0.558, GSE41613 0.675, GSE42743 0.606
#     against an age-alone floor of 0.551-0.565 (Rung 1).
#
#     "0.638 exceeds the floor" is not defensible from a point estimate at 51
#     events. This computes percentile bootstrap intervals over PATIENTS.
#
#     Coefficients are FIXED, fit once on the full source cohort. Only the
#     evaluation set is resampled. Refitting inside the bootstrap would estimate
#     a different quantity -- the variability of the whole fitting procedure --
#     when the question is the precision of THIS score in THIS cohort.
#
#     Also reports the PAIRED difference between gene sets on the same
#     resamples. Unpaired intervals would overlap for almost any comparison at
#     these event counts and would render every contrast "not significant" by
#     construction (plan section B5).
#
# (2) TCGA VARIANCE-EXPLAINED. K1.5 reported ORIEN-defined module eigengenes
#     explaining 0.47-0.61 of variance in ORIEN but only 0.09-0.43 in TCGA. A
#     PC1 explaining 9% is not summarising that module in TCGA at all, and is the
#     most likely mechanical cause of the transfer failure there (C = 0.550, at
#     the age floor). This identifies WHICH module, and whether the modules that
#     transport worst are the ones carrying the score's weight.
#
# References:
#   Efron B, Tibshirani R (1993) An Introduction to the Bootstrap
#   Harrell FE et al. (1996) Stat Med 15:361-387 -- C-index
#
# Usage:
#   Rscript scripts/phaseK3_bootstrap_variance.R
# ==============================================================================

suppressPackageStartupMessages({
  library(WGCNA); library(survival); library(glmnet); library(dplyr)
})
options(stringsAsFactors = FALSE)

PROC_DIR <- "data/processed"
EXT_DIR  <- "data/external"
K1_DIR   <- "results/phaseK1_module_preservation"
HUB_DIR  <- "results/phaseK1_6_hub_overlap"
CPSS_DIR <- "results/phase4_cpss_RULEFIX"
E8_DIR   <- "results/phase8_transportability"
OUT_DIR  <- "results/phaseK3_bootstrap_variance"

N_BOOT    <- 2000
SEED      <- 42
AGE_LO    <- 0.551      # Rung 1 age-alone C-index floor
AGE_HI    <- 0.565

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create("logs", showWarnings = FALSE)
log_file <- file.path("logs",
  sprintf("phaseK3_bootstrap_%s.log", format(Sys.time(), "%Y%m%d_%H%M%S")))
log_con <- file(log_file, open = "wt")
sink(log_con, split = TRUE); sink(log_con, type = "message")

cat("Phase K3: bootstrap intervals + TCGA variance-explained\n")
cat("Started:", format(Sys.time()), "\n")
cat("Bootstrap resamples:", N_BOOT, "| seed:", SEED, "\n")
cat("Age-alone floor:", AGE_LO, "-", AGE_HI, "\n\n")
set.seed(SEED)

# ==============================================================================
# Data
# ==============================================================================

load_expr <- function(p) as.matrix(read.csv(p, row.names = 1, check.names = FALSE))
read_surv <- function(p) {
  s <- read.csv(p); if (!"sampleID" %in% names(s)) names(s)[1] <- "sampleID"
  s[, c("sampleID", "time", "event")]
}

cat("--- Loading ---\n")
EX <- list(
  TCGA     = load_expr(file.path(PROC_DIR, "tcga_corrected_preprocessed_v4.csv")),
  ORIEN    = load_expr(file.path(PROC_DIR, "orien_corrected_preprocessed_v4.csv")),
  GSE41613 = load_expr(file.path(EXT_DIR,  "GSE41613_expression.csv")),
  GSE42743 = load_expr(file.path(EXT_DIR,  "GSE42743_expression.csv")))
SV <- list(
  TCGA     = read_surv(file.path(PROC_DIR, "surv_tcga_aligned_v4.csv")),
  ORIEN    = read_surv(file.path(PROC_DIR, "surv_orien_aligned_v4.csv")),
  GSE41613 = read_surv(file.path(EXT_DIR,  "GSE41613_survival.csv")),
  GSE42743 = read_surv(file.path(EXT_DIR,  "GSE42743_survival.csv")))

X <- list(); Y <- list()
for (nm in names(EX)) {
  keep <- intersect(colnames(EX[[nm]]), SV[[nm]]$sampleID)
  s <- SV[[nm]][match(keep, SV[[nm]]$sampleID), ]
  ok <- s$time > 0 & !is.na(s$time) & !is.na(s$event)
  X[[nm]] <- t(EX[[nm]][, keep, drop = FALSE])[ok, , drop = FALSE]
  Y[[nm]] <- Surv(s$time[ok], s$event[ok])
  cat(sprintf("  %-10s %4d samples, %3d events\n", nm, nrow(X[[nm]]),
              sum(s$event[ok])))
}
cat("\n")

cidx <- function(risk, y) {
  if (sd(risk) < 1e-12 || sum(y[, "status"]) < 2) return(NA_real_)
  as.numeric(concordance(y ~ risk, reverse = TRUE)$concordance)
}

# ==============================================================================
# Build the scores. Coefficients fixed, fit once on the source cohort.
# ==============================================================================

cat("--- Building scores (source = ORIEN, coefficients fixed) ---\n")

mods  <- read.csv(file.path(K1_DIR, "module_assignments.csv"))
pairs <- read.csv(file.path(HUB_DIR, "module_pairs.csv"))
pres  <- read.csv(file.path(K1_DIR, "preservation_statistics.csv"))
b4    <- read.csv(file.path(E8_DIR, "four_cohort_betas.csv"))

# --- kME, needed for hub sets and reused by the variance diagnostic -----------
kme <- list(); ME_by <- list(); VE_by <- list()
for (co in names(X)) {
  g <- intersect(mods$gene_id, colnames(X[[co]]))
  idx <- match(g, mods$gene_id)
  for (ref in c("TCGA", "ORIEN")) {
    me <- moduleEigengenes(X[[co]][, g, drop = FALSE],
                           mods[[paste0("module_", ref)]][idx],
                           align = "along average", excludeGrey = TRUE,
                           verbose = 0)
    ME_by[[paste(ref, co)]] <- me$eigengenes
    VE_by[[paste(ref, co)]] <- me$varExplained
  }
  if (co %in% c("TCGA", "ORIEN")) {
    kme[[co]] <- signedKME(X[[co]][, g, drop = FALSE], ME_by[[paste(co, co)]],
                           outputColumnName = "")
    rownames(kme[[co]]) <- g
  }
}

hub_set <- function(k) {
  out <- character()
  for (i in seq_len(nrow(pairs))) for (co in c("TCGA", "ORIEN")) {
    m <- if (co == "TCGA") pairs$tcga_module[i] else pairs$orien_module[i]
    gg <- rownames(kme[[co]])
    sel <- which(mods[[paste0("module_", co)]][match(gg, mods$gene_id)] == m)
    col <- if (m %in% colnames(kme[[co]])) m else paste0("ME", m)
    out <- c(out, gg[sel][order(-abs(kme[[co]][sel, col]))][seq_len(min(k, length(sel)))])
  }
  unique(out)
}
# Canonical selected-cell artifacts (repointed after audit); see K1.6.4 note.
ft <- file.path(CPSS_DIR, "selected_tcga_selection_probabilities.csv")
fo <- file.path(CPSS_DIR, "selected_orien_selection_probabilities.csv")
lasso_set <- function(n) {
  pt <- read.csv(ft, row.names = 1); po <- read.csv(fo, row.names = 1)
  h <- ceiling(n / 2)
  unique(c(rownames(pt)[order(-pt[, 1])][seq_len(h)],
           rownames(po)[order(-po[, 1])][seq_len(h)]))
}

scores <- list()   # scores[[name]][[cohort]] = numeric risk vector

# --- 1. K1.5 module-eigengene score, ORIEN-defined modules -------------------
qual <- pres %>% filter(!is_control, reference == "ORIEN") %>%
  group_by(module) %>%
  summarise(ok = sum(above_null, na.rm = TRUE) == n(), .groups = "drop") %>%
  filter(ok) %>% pull(module)
me_cols <- intersect(paste0("ME", qual), colnames(ME_by[["ORIEN ORIEN"]]))
Esrc <- ME_by[["ORIEN ORIEN"]][, me_cols, drop = FALSE]
mu <- colMeans(Esrc); sdv <- apply(Esrc, 2, sd)
fit_me <- coxph(Y$ORIEN ~ ., data = as.data.frame(scale(Esrc, mu, sdv)))
cf_me <- coef(fit_me)
for (co in names(X)) {
  E <- ME_by[[paste("ORIEN", co)]][, names(cf_me), drop = FALSE]
  scores[["eigengene_K1.5"]][[co]] <-
    as.numeric(as.matrix(scale(E, mu[names(cf_me)], sdv[names(cf_me)])) %*% cf_me)
}
cat("  eigengene_K1.5:", length(cf_me), "modules\n")

# --- 2/3. hub and lasso ridge Cox, matched size ------------------------------
hub <- hub_set(40); lasso <- lasso_set(length(hub))
for (nm in c("hub_250", "lasso_250")) {
  g <- if (nm == "hub_250") hub else lasso
  g <- Reduce(intersect, c(list(g), lapply(X, colnames)))
  set.seed(SEED)
  cv <- cv.glmnet(X$ORIEN[, g, drop = FALSE], Y$ORIEN, family = "cox",
                  alpha = 0, nfolds = 10, standardize = FALSE)
  b <- as.numeric(coef(cv, s = "lambda.min"))
  for (co in names(X))
    scores[[nm]][[co]] <- as.numeric(X[[co]][, g, drop = FALSE] %*% b)
  cat(sprintf("  %-14s %d genes\n", nm, length(g)))
}
cat("\n")

write.csv(data.frame(gene = hub),
          file.path(OUT_DIR, "hub250_genes.csv"), row.names = FALSE)

# ==============================================================================
# (1) Bootstrap
# ==============================================================================

cat(rep("=", 78), "\n", sep = "")
cat("BOOTSTRAP CONFIDENCE INTERVALS (", N_BOOT, " resamples over patients)\n", sep = "")
cat(rep("=", 78), "\n", sep = "")
cat("Coefficients are FIXED; only the evaluation cohort is resampled.\n")
cat("Refitting inside the bootstrap would estimate the variability of the\n")
cat("fitting procedure, not the precision of this score in this cohort.\n\n")

boot_idx <- list()
for (co in names(X)) {
  n <- nrow(X[[co]])
  set.seed(SEED + which(names(X) == co))
  boot_idx[[co]] <- replicate(N_BOOT, sample.int(n, n, replace = TRUE),
                              simplify = FALSE)
}

rows <- list(); boot_store <- list()
for (nm in names(scores)) {
  for (co in names(X)) {
    r <- scores[[nm]][[co]]; y <- Y[[co]]
    obs <- cidx(r, y)
    bs <- vapply(boot_idx[[co]], function(i) cidx(r[i], y[i]), numeric(1))
    bs <- bs[!is.na(bs)]
    boot_store[[paste(nm, co)]] <- bs
    ci <- quantile(bs, c(0.025, 0.975))
    # Does the interval clear the age-alone floor entirely?
    verdict <- if (ci[1] > AGE_HI) "above floor"
               else if (ci[2] < AGE_LO) "below floor" else "overlaps floor"
    rows[[paste(nm, co)]] <- data.frame(
      score = nm, cohort = co, n = nrow(X[[co]]),
      events = sum(y[, "status"]), c_index = obs,
      lo = ci[1], hi = ci[2], width = ci[2] - ci[1],
      p_gt_floor = mean(bs > AGE_HI), verdict = verdict)
  }
}
bt <- bind_rows(rows)
write.csv(bt, file.path(OUT_DIR, "bootstrap_cis.csv"), row.names = FALSE)

cat(sprintf("  %-16s %-10s %5s %6s %8s %-18s %8s %s\n",
            "score", "cohort", "n", "events", "C-index", "95% CI", "P(>floor)",
            "verdict"))
for (i in seq_len(nrow(bt))) {
  r <- bt[i, ]
  cat(sprintf("  %-16s %-10s %5d %6d %8.3f [%6.3f, %6.3f] %8.3f  %s\n",
              r$score, r$cohort, r$n, r$events, r$c_index, r$lo, r$hi,
              r$p_gt_floor, r$verdict))
}

cat("\n--- Paired differences between scores, SAME resamples ---\n")
cat("Unpaired intervals overlap for almost any comparison at these event\n")
cat("counts and would make every contrast 'not significant' by construction.\n\n")
pair_rows <- list()
combos <- combn(names(scores), 2, simplify = FALSE)
for (cc in combos) for (co in names(X)) {
  a <- boot_store[[paste(cc[1], co)]]; b <- boot_store[[paste(cc[2], co)]]
  n <- min(length(a), length(b)); d <- a[seq_len(n)] - b[seq_len(n)]
  ci <- quantile(d, c(0.025, 0.975))
  cat(sprintf("  %-10s  %-14s - %-14s : %+.4f [%+.4f, %+.4f] %s\n",
              co, cc[1], cc[2], mean(d), ci[1], ci[2],
              if (ci[1] > 0 || ci[2] < 0) "*" else ""))
  pair_rows[[paste(cc[1], cc[2], co)]] <- data.frame(
    cohort = co, score_a = cc[1], score_b = cc[2],
    diff = mean(d), lo = ci[1], hi = ci[2],
    excludes_zero = ci[1] > 0 || ci[2] < 0)
}
write.csv(bind_rows(pair_rows), file.path(OUT_DIR, "bootstrap_paired.csv"),
          row.names = FALSE)

# ==============================================================================
# (2) Variance explained
# ==============================================================================

cat("\n", rep("=", 78), "\n", sep = "")
cat("TCGA VARIANCE-EXPLAINED DIAGNOSTIC\n")
cat(rep("=", 78), "\n", sep = "")
cat("A module eigengene is PC1 of its member genes. If PC1 explains 9% in TCGA\n")
cat("but 60% in ORIEN, it is not summarising that module in TCGA at all -- the\n")
cat("likeliest mechanical cause of the transfer failure (C = 0.550).\n\n")

ve_rows <- list()
for (ref in c("TCGA", "ORIEN")) {
  cat("### modules defined in", ref, "###\n")
  cohorts <- names(X)
  ve <- sapply(cohorts, function(co) {
    v  <- VE_by[[paste(ref, co)]]
    nm <- sub("^ME", "", colnames(ME_by[[paste(ref, co)]]))
    stopifnot(ncol(v) == length(nm))
    setNames(as.numeric(v[1, ]), nm)
  })
  if (is.null(dim(ve))) next
  cat(sprintf("  %-12s %8s %8s %10s %10s %10s\n", "module", "TCGA", "ORIEN",
              "GSE41613", "GSE42743", "TCGA/ORIEN"))
  for (m in rownames(ve)) {
    ratio <- ve[m, "TCGA"] / ve[m, "ORIEN"]
    flag <- if (!is.na(ratio) && ratio < 0.4) "  <-- collapses in TCGA" else ""
    cat(sprintf("  %-12s %8.3f %8.3f %10.3f %10.3f %10.2f%s\n", m,
                ve[m, "TCGA"], ve[m, "ORIEN"], ve[m, "GSE41613"],
                ve[m, "GSE42743"], ratio, flag))
    ve_rows[[paste(ref, m)]] <- data.frame(
      reference = ref, module = m, ve_TCGA = ve[m, "TCGA"],
      ve_ORIEN = ve[m, "ORIEN"], ve_GSE41613 = ve[m, "GSE41613"],
      ve_GSE42743 = ve[m, "GSE42743"], ratio_tcga_orien = ratio)
  }
  cat("\n")
}
ve_df <- bind_rows(ve_rows)
write.csv(ve_df, file.path(OUT_DIR, "variance_explained.csv"), row.names = FALSE)

# Does the score's weight sit on the modules that collapse in TCGA?
cat("--- Does the K1.5 score weight the modules that collapse in TCGA? ---\n")
cat(sprintf("  %-12s %10s %10s %12s\n", "module", "|coef|", "VE ratio", "weight x fail"))
w <- data.frame(module = sub("^ME", "", names(cf_me)), coef = as.numeric(cf_me))
w <- w %>% left_join(ve_df %>% filter(reference == "ORIEN") %>%
                       select(module, ratio_tcga_orien), by = "module")
w$prod <- abs(w$coef) * (1 - pmin(w$ratio_tcga_orien, 1))
for (i in order(-w$prod)) {
  cat(sprintf("  %-12s %10.3f %10.2f %12.3f\n", w$module[i], abs(w$coef[i]),
              w$ratio_tcga_orien[i], w$prod[i]))
}
cat("\n  High |coef| combined with a low VE ratio means the score leans on a\n")
cat("  module that TCGA's eigengene does not represent. If the top rows here\n")
cat("  are also the top-weighted modules, the transfer failure is mechanical\n")
cat("  rather than biological.\n")
write.csv(w, file.path(OUT_DIR, "score_weight_vs_variance.csv"), row.names = FALSE)

cat("\nFinished:", format(Sys.time()), "\n")
cat("--- sessionInfo ---\n"); print(sessionInfo())
sink(type = "message"); sink(); close(log_con)
