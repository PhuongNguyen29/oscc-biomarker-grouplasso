#!/usr/bin/env Rscript
# ==============================================================================
# Phase K1.6.7: Do hub genes carry survival signal?
# ==============================================================================
#
# THE QUESTION
# ------------
# K1.6.4 showed hub selection transports where the lasso does not: Kuncheva 0.51
# vs 0.007 at 250 genes. But kME measures CONNECTIVITY, not survival association.
# A gene set can be perfectly reproducible and prognostically useless.
#
# THIS IS THE FIRST SUPERVISED STEP (K1.6.7.1)
# --------------------------------------------
# K1 and K1.6.4 were unsupervised throughout. The hub sets are therefore FIXED
# before any survival data is touched, and are NOT revised in light of anything
# computed here. That ordering is what makes this a validation rather than a
# selection.
#
# WHY A RANDOM-SET NULL IS MANDATORY (K1.6.7.3)
# ---------------------------------------------
# Venet, Dumont & Detours (2011) showed most random gene sets of a given size are
# prognostic in breast cancer. This data has real marginal signal -- 865 and 749
# genes reach nominal p < 0.05 in the two external cohorts against 235 expected.
# So a random 250-gene set WILL contain prognostic genes by construction, and
# "hub genes are prognostic" is uninterpretable without knowing where random sets
# of the same size fall. The comparison is a percentile against 1,000 random
# draws, not a p-value against zero.
#
# WHAT WOULD COUNT AS SUCCESS
# ---------------------------
# Hub genes beating LASSO on cross-cohort agreement is expected (K1.6.4 already
# implies it). The informative question is whether they beat RANDOM sets. A
# finding that they transport well but are no more prognostic than random sets of
# the same size is legitimate and reportable -- it would mean kME selects for
# connectivity, exactly as defined.
#
# References:
#   Venet D, Dumont JE, Detours V (2011) PLoS Comput Biol 7:e1002240 -- random signatures
#   Langfelder P, Horvath S (2008) BMC Bioinformatics 9:559 -- WGCNA, kME
#   Benjamini Y, Hochberg Y (1995) JRSS-B 57:289-300 -- FDR
#
# Usage:
#   Rscript scripts/phaseK1_6_7_hub_survival.R
# ==============================================================================

suppressPackageStartupMessages({
  library(survival); library(glmnet); library(dplyr)
})
options(stringsAsFactors = FALSE)

PROC_DIR <- "data/processed"
EXT_DIR  <- "data/external"
HUB_DIR  <- "results/phaseK1_6_hub_overlap_k30"
CPSS_DIR <- "results/phase4_cpss_RULEFIX"   # rev b run of record (plan D2b)
E8_DIR   <- "results/phase8_transportability"
OUT_DIR  <- "results/phaseK1_6_7_hub_survival_k30"

K_VALUES <- c(8, 16, 30, 40)  
N_RANDOM  <- 1000
SEED      <- 42
AGE_FLOOR <- "per cohort: TCGA 0.565, ORIEN 0.551, GSE41613 0.518, GSE42743 0.571"


dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create("logs", showWarnings = FALSE)
log_file <- file.path("logs",
  sprintf("phaseK1_6_7_hub_survival_%s.log", format(Sys.time(), "%Y%m%d_%H%M%S")))
log_con <- file(log_file, open = "wt")
sink(log_con, split = TRUE); sink(log_con, type = "message")

cat("Phase K1.6.7: hub-gene survival association\n")
cat("Started:", format(Sys.time()), "\n")
cat("Set sizes:", paste(SIZES, collapse = ", "), "| random draws:", N_RANDOM, "\n\n")
set.seed(SEED)

# ==============================================================================
# Marginal effects -- reuse E8 rather than refit
# ==============================================================================

cat("--- Loading four-cohort marginal effects (E8) ---\n")
b4 <- read.csv(file.path(E8_DIR, "four_cohort_betas.csv"))
cat("  genes:", nrow(b4), "| columns:", paste(head(names(b4), 12), collapse = ", "), "\n")
universe <- b4$gene_id
cat("  universe for random draws:", length(universe), "\n\n")

# ==============================================================================
# Gene sets
# ==============================================================================

cat("--- Gene sets ---\n")
sets <- list()

# 1. Hubs. K1.6.4 wrote the SHARED hubs; here we need each cohort's own hub set
#    (their union), since a set defined by cross-cohort agreement would already
#    be selected on the thing we are about to measure.
mods  <- read.csv("results/phaseK1_module_preservation/module_assignments.csv")
pairs <- read.csv(file.path(HUB_DIR, "module_pairs.csv"))
kme_file <- file.path(HUB_DIR, "kme_values.csv")

if (file.exists(kme_file)) {
  kme <- read.csv(kme_file, row.names = 1)
} else {
  cat("  kME table not saved by K1.6.4; recomputing\n")
  suppressPackageStartupMessages(library(WGCNA))
  load_expr <- function(p) as.matrix(read.csv(p, row.names = 1, check.names = FALSE))
  ex <- list(TCGA  = load_expr(file.path(PROC_DIR, "tcga_corrected_preprocessed_v4.csv")),
             ORIEN = load_expr(file.path(PROC_DIR, "orien_corrected_preprocessed_v4.csv")))
  g <- mods$gene_id
  kme <- list()
  for (co in c("TCGA", "ORIEN")) {
    dE <- t(ex[[co]][g, , drop = FALSE])
    ME <- moduleEigengenes(dE, mods[[paste0("module_", co)]],
                           align = "along average", excludeGrey = TRUE,
                           verbose = 0)$eigengenes
    kme[[co]] <- signedKME(dE, ME, outputColumnName = "")
  }
}

hub_set <- function(k) {
  out <- character()
  for (i in seq_len(nrow(pairs))) {
    for (co in c("TCGA", "ORIEN")) {
      m <- if (co == "TCGA") pairs$tcga_module[i] else pairs$orien_module[i]
      idx <- which(mods[[paste0("module_", co)]] == m)
      col <- if (m %in% colnames(kme[[co]])) m else paste0("ME", m)
      v <- kme[[co]][idx, col]
      gg <- mods$gene_id[idx]
      out <- c(out, gg[order(-abs(v))][seq_len(min(k, length(gg)))])
    }
  }
  unique(out)
}

# 2. Lasso: top-N by CPSS selection probability, both cohorts pooled
# Canonical selected-cell artifacts (repointed after audit); see K1.6.4 note.
ft <- file.path(CPSS_DIR, "selected_tcga_selection_probabilities.csv")
fo <- file.path(CPSS_DIR, "selected_orien_selection_probabilities.csv")
lasso_set <- function(n) {
  if (!file.exists(ft) || !file.exists(fo)) return(NULL)
  pt <- read.csv(ft, row.names = 1); po <- read.csv(fo, row.names = 1)
  half <- ceiling(n / 2)
  unique(c(rownames(pt)[order(-pt[, 1])][seq_len(half)],
           rownames(po)[order(-po[, 1])][seq_len(half)]))
}

for (k in K_VALUES) {
  h <- hub_set(k)
  l <- lasso_set(length(h))
  nm <- paste0("k", k)
  sets[[paste0("hub_",   nm)]] <- h
  if (!is.null(l)) sets[[paste0("lasso_", nm)]] <- l
  cat(sprintf("  k=%2d -> hub %3d genes | lasso %3d genes\n",
              k, length(h), if (is.null(l)) 0 else length(l)))
}

# ==============================================================================
# K1.6.7.2 Cross-cohort agreement of marginal effects
# ==============================================================================

agreement <- function(g) {
  d <- b4[b4$gene_id %in% g, ]
  if (nrow(d) < 10) return(NULL)
  c(n            = nrow(d),
    rho_dev_ext  = suppressWarnings(cor(d$beta_pooled, d$beta_ext, method = "spearman")),
    rho_tcga_ori = suppressWarnings(cor(d$beta_tcga,   d$beta_orien, method = "spearman")),
    rho_ext_pair = suppressWarnings(cor(d$beta_41613,  d$beta_42743, method = "spearman")),
    sign_dev_ext = mean((d$beta_pooled > 0) == (d$beta_ext > 0)),
    n_q10_both   = sum(d$q_41613 < 0.10 & d$q_42743 < 0.10),
    med_abs_dev  = median(abs(d$beta_pooled)))
}

cat(rep("=", 76), "\n", sep = "")
cat("K1.6.7.2  CROSS-COHORT AGREEMENT OF MARGINAL EFFECTS\n")
cat(rep("=", 76), "\n", sep = "")
cat("Genome-wide reference (E8): rho dev-ext = 0.591, sign concordance = 0.707\n\n")

obs <- list()
for (nm in names(sets)) {
  a <- agreement(sets[[nm]])
  if (is.null(a)) next
  obs[[nm]] <- data.frame(set = nm, t(a))
}
obs <- bind_rows(obs)
print(as.data.frame(obs), row.names = FALSE, digits = 3)

# ---- random-set null --------------------------------------------------------
cat("\n--- Random-set null (Venet et al. 2011):", N_RANDOM, "draws per size ---\n")
cat("Percentile = fraction of random sets the observed value EXCEEDS.\n\n")

null_rows <- list()
for (n in SIZES) {
  for (target in c("hub", "lasso")) {
    key <- paste0(target, "_", n)
    if (!key %in% names(sets)) next
    sz <- length(sets[[key]])
    set.seed(SEED + n)
    draws <- replicate(N_RANDOM, {
      a <- agreement(sample(universe, sz))
      if (is.null(a)) rep(NA, 7) else a
    })
    o <- obs[obs$set == key, ]
    for (metric in c("rho_dev_ext", "sign_dev_ext", "n_q10_both", "med_abs_dev")) {
      nulls <- draws[metric, ]
      pct <- mean(nulls < o[[metric]], na.rm = TRUE)
      null_rows[[paste(key, metric)]] <- data.frame(
        set = key, size = sz, metric = metric, observed = o[[metric]],
        null_median = median(nulls, na.rm = TRUE),
        null_q95 = quantile(nulls, 0.95, na.rm = TRUE),
        percentile = pct)
    }
  }
}
nulls_df <- bind_rows(null_rows)
cat(sprintf("  %-12s %5s %-14s %10s %11s %9s %10s\n",
            "set", "size", "metric", "observed", "null median", "null q95", "pctile"))
for (i in seq_len(nrow(nulls_df))) {
  r <- nulls_df[i, ]
  flag <- if (r$percentile >= 0.95) " *" else ""
  cat(sprintf("  %-12s %5d %-14s %10.3f %11.3f %9.3f %9.1f%%%s\n",
              r$set, r$size, r$metric, r$observed, r$null_median,
              r$null_q95, 100 * r$percentile, flag))
}
write.csv(nulls_df, file.path(OUT_DIR, "random_null_comparison.csv"), row.names = FALSE)
write.csv(obs, file.path(OUT_DIR, "marginal_agreement.csv"), row.names = FALSE)

# ==============================================================================
# K1.6.7.4 Prediction: ridge Cox fit in ORIEN, fixed coefficients elsewhere
# ==============================================================================

cat("\n", rep("=", 76), "\n", sep = "")
cat("K1.6.7.4  PREDICTION — ridge Cox fit in ORIEN, coefficients applied unchanged\n")
cat(rep("=", 76), "\n", sep = "")
cat("Same protocol as K1.5, so numbers are directly comparable to the\n")
cat("module-eigengene score (TCGA 0.550, GSE41613 0.638, GSE42743 0.653).\n")
cat("Age-alone floor:", AGE_FLOOR, "\n\n")

load_expr <- function(p) as.matrix(read.csv(p, row.names = 1, check.names = FALSE))
read_surv <- function(p) {
  s <- read.csv(p); if (!"sampleID" %in% names(s)) names(s)[1] <- "sampleID"
  s[, c("sampleID", "time", "event")]
}
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
}
rm(EX); gc()

cidx <- function(r, y) as.numeric(concordance(y ~ r, reverse = TRUE)$concordance)

pred_rows <- list()
for (nm in names(sets)) {
  g <- intersect(sets[[nm]], colnames(X$ORIEN))
  g <- intersect(g, colnames(X$GSE41613))
  if (length(g) < 10) next
  set.seed(SEED)
  cv <- cv.glmnet(X$ORIEN[, g, drop = FALSE], Y$ORIEN, family = "cox",
                  alpha = 0, nfolds = 10, standardize = FALSE,
                  cox.ties = "breslow")
  b <- as.numeric(coef(cv, s = "lambda.min"))
  for (co in names(X)) {
    gg <- intersect(g, colnames(X[[co]]))
    if (length(gg) < length(g)) next
    pred_rows[[paste(nm, co)]] <- data.frame(
      set = nm, n_genes = length(g), cohort = co,
      role = if (co == "ORIEN") "SOURCE (in-sample)" else "transfer",
      c_index = cidx(as.numeric(X[[co]][, g, drop = FALSE] %*% b), Y[[co]]))
  }
}
pred <- bind_rows(pred_rows)
write.csv(pred, file.path(OUT_DIR, "prediction_transfer.csv"), row.names = FALSE)

cat(sprintf("  %-12s %7s %-10s %-20s %8s\n",
            "set", "genes", "cohort", "role", "C-index"))
for (i in seq_len(nrow(pred))) {
  r <- pred[i, ]
  cat(sprintf("  %-12s %7d %-10s %-20s %8.3f\n",
              r$set, r$n_genes, r$cohort, r$role, r$c_index))
}

cat("\n", rep("=", 76), "\n", sep = "")
cat("READING THIS\n")
cat(rep("=", 76), "\n", sep = "")
cat("The lasso comparison is expected to favour hubs -- K1.6.4 already implies\n")
cat("it. The informative comparison is the RANDOM-SET percentile. Hub genes\n")
cat("transporting well but sitting inside the random distribution would mean\n")
cat("kME selects for connectivity, not survival -- which is what kME is\n")
cat("defined to do, and a legitimate result (K1.6.7.5).\n")

cat("\nFinished:", format(Sys.time()), "\n")
cat("--- sessionInfo ---\n"); print(sessionInfo())
sink(type = "message"); sink(); close(log_con)
