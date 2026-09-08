#!/usr/bin/env Rscript
# ==============================================================================
# Phase K1.6.8: What kind of biology does each selection method pick?
# ==============================================================================
# DEPENDENCY-FREE VERSION -- base R only, plus WGCNA (already installed).
#
# clusterProfiler would not build on IDAS: it pulls in `Deriv`, whose current
# version calls R_ClosureFormals, a C API function newer than the R 4.4.1
# headers here. Rather than fight the dependency chain, this computes
# over-representation directly. enrichGO/enricher are convenience wrappers
# around a hypergeometric test, and phyper() is that test:
#
#   p = P(X >= k),  X ~ Hypergeometric
#     N = background universe (4,702 genes common to all four cohorts)
#     K = genes in the term, RESTRICTED TO THE UNIVERSE
#     n = size of the selected set
#     k = overlap
#
# Gene sets come from a local GMT file. If none is present the annotation-free
# checks still run, and those are the more direct evidence anyway.
#
# THE CLAIM UNDER TEST -- AND ITS FIRST REFUTATION
# ------------------------------------------------
# K1.6.7 found the two methods fail in opposite ways:
#   lasso : 97th-100th percentile on effect size / cross-cohort concordance,
#           ZERO selection reproducibility (0/42 cells)
#   kME   : Kuncheva 0.51 reproducibility, but 0.1st-4.6th percentile on effect
#           size -- significantly WORSE than random sets of the same size
#
# The proposed explanation was that hub genes are housekeeping-like.
# ALREADY REFUTED by the annotation-free check: hub median IQR is 2.424 (94.5th
# percentile of the universe) vs lasso 2.032 (83rd). Hub genes are the MOST
# variable genes, not the least, and housekeeping genes are low-variance by
# definition.
#
# In hindsight it is wrong by construction: WGCNA builds correlation networks,
# and a gene needs variance to correlate with anything. Low-variance genes land
# in grey and cannot be hubs.
#
# THE REMAINING PUZZLE, which this addresses
# ------------------------------------------
# Hub genes are MORE variable than lasso genes yet have SMALLER marginal effects
# (median |beta| 0.045 vs 0.057). They vary strongly and coherently across
# patients, and that variation does not predict survival. What coherent
# high-variance programme is it? Proliferation, immune infiltration and stromal
# content are all plausible and all testable.
#
# Only 6 genes overlap between the 309 hub and 303 lasso sets -- near-disjoint
# biology from the same 4,702 genes.
#
# COMMITMENT: if the enrichment gives nothing interpretable, no mechanism is
# asserted, and K1.6.7 is reported as a bare reproducibility/signal trade-off.
#
# References:
#   Liberzon A et al. (2015) Cell Syst 1:417-425 -- Hallmark gene sets
#   Benjamini Y, Hochberg Y (1995) JRSS-B 57:289-300 -- FDR
#   Young MD et al. (2010) Genome Biol 11:R14 -- background choice matters
#
# GMT files (download once, no install):
#   https://www.gsea-msigdb.org/gsea/downloads.jsp
#     h.all.v2023.2.Hs.symbols.gmt      Hallmark, 50 sets -- start here
#     c5.go.bp.v2023.2.Hs.symbols.gmt   GO Biological Process
#   Place in data/genesets/
#
# Usage:
#   Rscript scripts/phaseK1_6_8_enrichment.R
# ==============================================================================

suppressPackageStartupMessages({ library(WGCNA); library(dplyr) })
options(stringsAsFactors = FALSE)

HUB_DIR  <- "results/phaseK1_6_hub_overlap"
E8_DIR   <- "results/phase8_transportability"
K1_DIR   <- "results/phaseK1_module_preservation"
CPSS_DIR <- "results/phase4_cpss_RULEFIX"   # rev b run of record (plan D2b)
GMT_DIR  <- "data/genesets"
OUT_DIR  <- "results/phaseK1_6_8_enrichment_k30"

MIN_TERM_SIZE <- 10     # smaller terms give unstable hypergeometric estimates
MAX_TERM_SIZE <- 500    # very large terms are uninformative
Q_CUTOFF      <- 0.10
SEED          <- 42

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create("logs", showWarnings = FALSE)
log_file <- file.path("logs",
  sprintf("phaseK1_6_8_enrichment_%s.log", format(Sys.time(), "%Y%m%d_%H%M%S")))
log_con <- file(log_file, open = "wt")
sink(log_con, split = TRUE); sink(log_con, type = "message")

cat("Phase K1.6.8: enrichment of hub vs lasso sets (dependency-free)\n")
cat("Started:", format(Sys.time()), "\n\n")
set.seed(SEED)

# ==============================================================================
# Universe and gene sets
# ==============================================================================

cat("--- Universe and gene sets ---\n")
b4 <- read.csv(file.path(E8_DIR, "four_cohort_betas.csv"))
universe <- b4$gene_id
sym <- setNames(b4$gene_symbol, b4$gene_id)
cat("  background universe:", length(universe), "genes\n")

mods  <- read.csv(file.path(K1_DIR, "module_assignments.csv"))
pairs <- read.csv(file.path(HUB_DIR, "module_pairs.csv"))

load_expr <- function(p) as.matrix(read.csv(p, row.names = 1, check.names = FALSE))
ex <- list(TCGA  = load_expr("data/processed/tcga_corrected_preprocessed_v4.csv"),
           ORIEN = load_expr("data/processed/orien_corrected_preprocessed_v4.csv"))
g_all <- mods$gene_id
kme <- list()
for (co in c("TCGA", "ORIEN")) {
  dE <- t(ex[[co]][g_all, , drop = FALSE])
  ME <- moduleEigengenes(dE, mods[[paste0("module_", co)]],
                         align = "along average", excludeGrey = TRUE,
                         verbose = 0)$eigengenes
  kme[[co]] <- signedKME(dE, ME, outputColumnName = "")
}
rm(ex); gc()

hub_set <- function(k) {
  out <- character()
  for (i in seq_len(nrow(pairs))) for (co in c("TCGA", "ORIEN")) {
    m <- if (co == "TCGA") pairs$tcga_module[i] else pairs$orien_module[i]
    idx <- which(mods[[paste0("module_", co)]] == m)
    col <- if (m %in% colnames(kme[[co]])) m else paste0("ME", m)
    gg <- mods$gene_id[idx]
    out <- c(out, gg[order(-abs(kme[[co]][idx, col]))][seq_len(min(k, length(gg)))])
  }
  unique(out)
}
# Canonical selected-cell artifacts (repointed after audit); see K1.6.4 note.
ft <- file.path(CPSS_DIR, "selected_tcga_selection_probabilities.csv")
fo <- file.path(CPSS_DIR, "selected_orien_selection_probabilities.csv")
lasso_set <- function(n) {
  stopifnot(file.exists(ft), file.exists(fo))
  pt <- read.csv(ft, row.names = 1); po <- read.csv(fo, row.names = 1)
  h <- ceiling(n / 2)
  unique(c(rownames(pt)[order(-pt[, 1])][seq_len(h)],
           rownames(po)[order(-po[, 1])][seq_len(h)]))
}

hub   <- hub_set(40)
lasso <- lasso_set(length(hub))
cat("  hub set:  ", length(hub),  "genes\n")
cat("  lasso set:", length(lasso), "genes\n")
cat("  overlap:  ", length(intersect(hub, lasso)), "genes\n\n")

writeLines(sort(unname(sym[hub])),   file.path(OUT_DIR, "hub_symbols.txt"))
writeLines(sort(unname(sym[lasso])), file.path(OUT_DIR, "lasso_symbols.txt"))

# ==============================================================================
# Hypergeometric over-representation
# ==============================================================================

read_gmt <- function(path) {
  ln <- strsplit(readLines(path), "\t")
  setNames(lapply(ln, function(x) unique(x[-c(1, 2)])), sapply(ln, `[`, 1))
}

enrich <- function(set_sym, bg_sym, gs, label) {
  cat("\n### ", label, " ###\n", sep = "")
  # Restricting each term to the background is what makes this a test about the
  # 4,702-gene universe rather than the genome. Skipping it would mostly test
  # whether the IQR filter enriched for something (Young et al. 2010).
  gs <- lapply(gs, intersect, bg_sym)
  sz <- sapply(gs, length)
  gs <- gs[sz >= MIN_TERM_SIZE & sz <= MAX_TERM_SIZE]
  if (!length(gs)) { cat("  no usable terms after size filtering\n"); return(NULL) }

  N <- length(bg_sym); n <- length(set_sym)
  res <- bind_rows(lapply(names(gs), function(t) {
    K <- length(gs[[t]]); k <- length(intersect(set_sym, gs[[t]]))
    data.frame(term = t, term_size = K, overlap = k, expected = n * K / N,
               fold = (k / n) / (K / N),
               p = phyper(k - 1, K, N - K, n, lower.tail = FALSE),
               genes = paste(sort(intersect(set_sym, gs[[t]])), collapse = ","))
  }))
  res$q <- p.adjust(res$p, method = "BH")
  res <- res[order(res$p), ]

  sig <- res[res$q < Q_CUTOFF, ]
  if (!nrow(sig)) {
    cat("  NO terms at q <", Q_CUTOFF, ". Top 5 by p:\n")
    for (i in seq_len(min(5, nrow(res))))
      cat(sprintf("    %-50s %2d/%3d fold %.2f p=%.3f q=%.2f\n",
                  substr(res$term[i], 1, 50), res$overlap[i], res$term_size[i],
                  res$fold[i], res$p[i], res$q[i]))
  } else {
    cat(sprintf("  %d terms at q < %.2f. Top 15:\n", nrow(sig), Q_CUTOFF))
    for (i in seq_len(min(15, nrow(sig))))
      cat(sprintf("    %-50s %2d/%3d fold %.2f q=%.2e\n",
                  substr(sig$term[i], 1, 50), sig$overlap[i], sig$term_size[i],
                  sig$fold[i], sig$q[i]))
  }
  res$set <- label
  res
}

gmts <- list.files(GMT_DIR, pattern = "\\.gmt$", full.names = TRUE)
all_res <- list()
if (length(gmts)) {
  bg_sym <- unname(sym[universe]); bg_sym <- bg_sym[nchar(bg_sym) > 0]
  for (g in gmts) {
    cat("\n", rep("=", 76), "\n", sep = "")
    cat("GMT:", basename(g), "\n")
    cat("BACKGROUND =", length(bg_sym), "gene analysis universe, NOT the genome\n")
    cat(rep("=", 76), "\n", sep = "")
    gs <- read_gmt(g)
    cat("  terms in file:", length(gs), "\n")
    all_res[[paste0(basename(g), "_hub")]] <-
      enrich(unname(sym[hub]), bg_sym, gs, paste0("HUB (kME) — ", basename(g)))
    all_res[[paste0(basename(g), "_lasso")]] <-
      enrich(unname(sym[lasso]), bg_sym, gs, paste0("LASSO (CPSS) — ", basename(g)))
  }
  out <- bind_rows(all_res[!sapply(all_res, is.null)])
  if (nrow(out)) {
    write.csv(out, file.path(OUT_DIR, "enrichment_results.csv"), row.names = FALSE)
    cat("\n  Full table:", file.path(OUT_DIR, "enrichment_results.csv"), "\n")
  }
} else {
  cat("\n", rep("=", 76), "\n", sep = "")
  cat("NO GMT FILES FOUND in", GMT_DIR, "\n")
  cat(rep("=", 76), "\n", sep = "")
  cat("Download from https://www.gsea-msigdb.org/gsea/downloads.jsp:\n")
  cat("  h.all.v2023.2.Hs.symbols.gmt      Hallmark, 50 sets — start here\n")
  cat("  c5.go.bp.v2023.2.Hs.symbols.gmt   GO Biological Process\n")
  cat("Place in", GMT_DIR, "and rerun. Symbols are written to", OUT_DIR, "\n")
  cat("so they can also be pasted into a web tool in the meantime.\n")
}

# ==============================================================================
# Annotation-free characterisation -- the more direct evidence
# ==============================================================================

cat("\n", rep("=", 76), "\n", sep = "")
cat("ANNOTATION-FREE CHARACTERISATION\n")
cat(rep("=", 76), "\n", sep = "")

pctile <- function(v, x) 100 * mean(v < x, na.rm = TRUE)

iqr_file <- "data/processed/iqr_filtered_v4_prez/iqr_filter_values_v4.csv"
if (file.exists(iqr_file)) {
  iq <- read.csv(iqr_file); gcol <- names(iq)[1]
  cols <- grep("iqr", names(iq), ignore.case = TRUE, value = TRUE)
  cat("\n-- expression variability (IQR) --\n")
  cat("  Housekeeping genes are LOW variance; this already refutes that reading.\n")
  cat(sprintf("  %-8s %6s %12s %10s\n", "set", "n", "median IQR", "pctile"))
  for (nm in c("hub", "lasso")) {
    g <- if (nm == "hub") hub else lasso
    v <- iq[[cols[1]]][match(g, iq[[gcol]])]; v <- v[!is.na(v)]
    cat(sprintf("  %-8s %6d %12.3f %9.1f%%\n", nm, length(v), median(v),
                pctile(iq[[cols[1]]], median(v))))
  }
}

cat("\n-- marginal effect size and cross-cohort agreement --\n")
cat(sprintf("  %-8s %6s %12s %14s %12s\n",
            "set", "n", "med |b_dev|", "sign dev-ext", "rho dev-ext"))
for (nm in c("hub", "lasso")) {
  g <- if (nm == "hub") hub else lasso
  d <- b4[b4$gene_id %in% g, ]
  cat(sprintf("  %-8s %6d %12.4f %14.3f %12.3f\n", nm, nrow(d),
              median(abs(d$beta_pooled)),
              mean((d$beta_pooled > 0) == (d$beta_ext > 0)),
              cor(d$beta_pooled, d$beta_ext, method = "spearman")))
}

# Hub genes are selected FOR co-expression, so they should be far more mutually
# correlated than lasso genes. Quantifying that says what "hub" means
# operationally, with no annotation involved.
cat("\n-- within-set co-expression coherence (ORIEN) --\n")
oe <- load_expr("data/processed/orien_corrected_preprocessed_v4.csv")
for (nm in c("hub", "lasso")) {
  g <- intersect(if (nm == "hub") hub else lasso, rownames(oe))
  cm <- cor(t(oe[g, , drop = FALSE])); v <- abs(cm[upper.tri(cm)])
  cat(sprintf("  %-8s n=%3d  mean |r| %.3f  median %.3f  frac>0.5 %.3f\n",
              nm, length(g), mean(v), median(v), mean(v > 0.5)))
}
set.seed(SEED)
rg <- sample(intersect(universe, rownames(oe)), 300)
cm <- cor(t(oe[rg, , drop = FALSE])); v <- abs(cm[upper.tri(cm)])
cat(sprintf("  %-8s n=%3d  mean |r| %.3f  median %.3f  frac>0.5 %.3f\n",
            "random", length(rg), mean(v), median(v), mean(v > 0.5)))

cat("\n", rep("=", 76), "\n", sep = "")
cat("HOW TO READ THIS\n")
cat(rep("=", 76), "\n", sep = "")
cat("The housekeeping interpretation is ALREADY REFUTED: hub genes are the most\n")
cat("variable genes in the universe, not the least. WGCNA needs variance to form\n")
cat("correlations, so kME selects high-variance genes by construction.\n\n")
cat("The open question is what coherent high-variance programme hubs represent,\n")
cat("given they vary strongly yet carry sub-random prognostic signal. If the\n")
cat("enrichment gives nothing interpretable, assert no mechanism: report K1.6.7\n")
cat("as a measured trade-off between reproducibility and prognostic content,\n")
cat("which stands on its own.\n")

cat("\nFinished:", format(Sys.time()), "\n")
cat("--- sessionInfo ---\n"); print(sessionInfo())
sink(type = "message"); sink(); close(log_con)
