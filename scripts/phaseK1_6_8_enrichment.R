#!/usr/bin/env Rscript
# ==============================================================================
# Phase K1.6.8: What kind of biology does each selection method pick?
# ==============================================================================
#
# THE CLAIM BEING TESTED
# ----------------------
# K1.6.7 found that the two selection methods fail in opposite ways:
#
#   lasso : 97th-100th percentile on effect size and cross-cohort beta
#           concordance, but ZERO selection reproducibility (0/42 cells)
#   kME   : Kuncheva 0.51 reproducibility, but 0.1st-3.3rd percentile on
#           cross-cohort beta concordance and 0th-4.6th on effect size --
#           SIGNIFICANTLY WORSE than random sets of the same size
#
# A plausible explanation is that hub genes -- the most connected members of a
# module -- are housekeeping-like: stably co-expressed, biologically central,
# prognostically inert. That is currently an INTERPRETATION, not a finding.
# This script tests it.
#
# PREDICTION, RECORDED BEFORE RUNNING
# -----------------------------------
#   hubs   enrich for translation / ribosome / oxidative phosphorylation /
#          RNA processing  (housekeeping)
#   lasso  enrich for ECM / hypoxia / EMT / immune  (tumour biology)
#
# If hubs enrich for nothing, the housekeeping claim is DROPPED rather than
# argued. That commitment is the point of running this.
#
# BACKGROUND UNIVERSE
# -------------------
# The background is the 4,702 genes common to all four cohorts -- NOT the whole
# genome. Using a genome background would test whether the IQR filter enriched
# for something, which is a different and uninteresting question, and would make
# almost everything look significant. This is the standard error in enrichment
# analysis of a pre-filtered feature space.
#
# References:
#   Subramanian A et al. (2005) PNAS 102:15545-15550 -- gene set enrichment
#   Liberzon A et al. (2015) Cell Syst 1:417-425 -- Hallmark gene sets
#   Benjamini Y, Hochberg Y (1995) JRSS-B 57:289-300 -- FDR
#   Young MD et al. (2010) Genome Biol 11:R14 -- selection bias in GO analysis
#
# Usage:
#   Rscript scripts/phaseK1_6_8_enrichment.R
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  ok_cp <- requireNamespace("clusterProfiler", quietly = TRUE) &&
           requireNamespace("org.Hs.eg.db", quietly = TRUE)
  if (ok_cp) { library(clusterProfiler); library(org.Hs.eg.db) }
  ok_msig <- requireNamespace("msigdbr", quietly = TRUE)
  if (ok_msig) library(msigdbr)
})
options(stringsAsFactors = FALSE)

HUB_DIR  <- "results/phaseK1_6_hub_overlap"
SURV_DIR <- "results/phaseK1_6_7_hub_survival"
E8_DIR   <- "results/phase8_transportability"
K1_DIR   <- "results/phaseK1_module_preservation"
CPSS_DIR <- "results/phase4_cpss_FINAL"
OUT_DIR  <- "results/phaseK1_6_8_enrichment"

Q_CUTOFF <- 0.10
SEED     <- 42

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create("logs", showWarnings = FALSE)
log_file <- file.path("logs",
  sprintf("phaseK1_6_8_enrichment_%s.log", format(Sys.time(), "%Y%m%d_%H%M%S")))
log_con <- file(log_file, open = "wt")
sink(log_con, split = TRUE); sink(log_con, type = "message")

cat("Phase K1.6.8: enrichment of hub vs lasso gene sets\n")
cat("Started:", format(Sys.time()), "\n")
if (!ok_cp) cat("\n*** clusterProfiler / org.Hs.eg.db not installed:\n",
                "*** BiocManager::install(c('clusterProfiler','org.Hs.eg.db','msigdbr'))\n\n")
set.seed(SEED)

# ==============================================================================
# Universe and gene sets -- rebuilt exactly as in K1.6.7
# ==============================================================================

cat("--- Universe and gene sets ---\n")
b4 <- read.csv(file.path(E8_DIR, "four_cohort_betas.csv"))
universe <- b4$gene_id
sym <- setNames(b4$gene_symbol, b4$gene_id)
cat("  background universe:", length(universe), "genes (4 cohorts common)\n")

mods  <- read.csv(file.path(K1_DIR, "module_assignments.csv"))
pairs <- read.csv(file.path(HUB_DIR, "module_pairs.csv"))

suppressPackageStartupMessages(library(WGCNA))
load_expr <- function(p) as.matrix(read.csv(p, row.names = 1, check.names = FALSE))
ex <- list(
  TCGA  = load_expr("data/processed/tcga_corrected_preprocessed_v4.csv"),
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
sp <- file.path(CPSS_DIR, "selection_probabilities")
ft <- list.files(sp, pattern = "^tcga.*mult0p80",  full.names = TRUE)
fo <- list.files(sp, pattern = "^orien.*mult0p80", full.names = TRUE)
lasso_set <- function(n) {
  pt <- read.csv(ft[1], row.names = 1); po <- read.csv(fo[1], row.names = 1)
  h <- ceiling(n / 2)
  unique(c(rownames(pt)[order(-pt[, 1])][seq_len(h)],
           rownames(po)[order(-po[, 1])][seq_len(h)]))
}

# k = 40 gives 309 hub genes, the largest set in K1.6.7 and the one with the
# clearest separation (hub 0.1st percentile, lasso 97.6th)
hub   <- hub_set(40)
lasso <- lasso_set(length(hub))
cat("  hub set:  ", length(hub),  "genes\n")
cat("  lasso set:", length(lasso), "genes\n")
cat("  overlap between the two:", length(intersect(hub, lasso)), "genes\n\n")

writeLines(sort(sym[hub]),   file.path(OUT_DIR, "hub_symbols.txt"))
writeLines(sort(sym[lasso]), file.path(OUT_DIR, "lasso_symbols.txt"))

# ==============================================================================
# Enrichment
# ==============================================================================

to_entrez <- function(ensg) {
  s <- suppressWarnings(
    AnnotationDbi::select(org.Hs.eg.db, keys = ensg, keytype = "ENSEMBL",
                          columns = "ENTREZID"))
  unique(na.omit(s$ENTREZID))
}

run_go <- function(genes, bg, label) {
  cat("\n### GO Biological Process —", label, "###\n")
  e <- enrichGO(gene = to_entrez(genes), universe = to_entrez(bg),
                OrgDb = org.Hs.eg.db, ont = "BP",
                pAdjustMethod = "BH", qvalueCutoff = Q_CUTOFF,
                readable = TRUE)
  if (is.null(e) || nrow(as.data.frame(e)) == 0) {
    cat("  NO significant terms at q <", Q_CUTOFF, "\n")
    return(NULL)
  }
  d <- as.data.frame(e)
  cat(sprintf("  %d terms at q < %.2f. Top 15:\n", nrow(d), Q_CUTOFF))
  for (i in seq_len(min(15, nrow(d)))) {
    cat(sprintf("    %-58s %8s q=%.2e\n",
                substr(d$Description[i], 1, 58), d$GeneRatio[i], d$qvalue[i]))
  }
  d$set <- label
  d
}

run_hallmark <- function(genes, bg, label) {
  if (!ok_msig) { cat("\n  (msigdbr not installed; skipping Hallmark)\n"); return(NULL) }
  cat("\n### MSigDB Hallmark —", label, "###\n")
  h <- msigdbr(species = "Homo sapiens", category = "H")
  t2g <- h[, c("gs_name", "human_gene_symbol")]
  names(t2g) <- c("term", "gene")
  e <- enricher(gene = unname(sym[genes]), universe = unname(sym[bg]),
                TERM2GENE = t2g, pAdjustMethod = "BH", qvalueCutoff = Q_CUTOFF)
  if (is.null(e) || nrow(as.data.frame(e)) == 0) {
    cat("  NO significant terms at q <", Q_CUTOFF, "\n")
    return(NULL)
  }
  d <- as.data.frame(e)
  cat(sprintf("  %d terms at q < %.2f. Top 15:\n", nrow(d), Q_CUTOFF))
  for (i in seq_len(min(15, nrow(d)))) {
    cat(sprintf("    %-52s %8s q=%.2e\n",
                sub("^HALLMARK_", "", substr(d$ID[i], 1, 52)),
                d$GeneRatio[i], d$qvalue[i]))
  }
  d$set <- label
  d
}

res <- list()
if (ok_cp) {
  cat(rep("=", 76), "\n", sep = "")
  cat("BACKGROUND = the 4,702-gene analysis universe, NOT the genome\n")
  cat(rep("=", 76), "\n", sep = "")
  res$go_hub     <- run_go(hub,   universe, "HUB (kME)")
  res$go_lasso   <- run_go(lasso, universe, "LASSO (CPSS)")
  res$hm_hub     <- run_hallmark(hub,   universe, "HUB (kME)")
  res$hm_lasso   <- run_hallmark(lasso, universe, "LASSO (CPSS)")
  out <- bind_rows(res[!sapply(res, is.null)])
  if (nrow(out)) write.csv(out, file.path(OUT_DIR, "enrichment_results.csv"),
                           row.names = FALSE)
}

# ==============================================================================
# Direct test of the housekeeping reading, independent of any annotation
# ==============================================================================

cat("\n", rep("=", 76), "\n", sep = "")
cat("ANNOTATION-FREE CHECK: are hub genes more highly and stably expressed?\n")
cat(rep("=", 76), "\n", sep = "")
cat("Housekeeping genes are typically high-expressing and low-variance. This\n")
cat("tests the interpretation without relying on GO terms at all.\n\n")

iqr_file <- "data/processed/iqr_filtered_v4_prez/iqr_filter_values_v4.csv"
if (file.exists(iqr_file)) {
  iq <- read.csv(iqr_file)
  gcol <- names(iq)[1]
  cols <- grep("iqr", names(iq), ignore.case = TRUE, value = TRUE)
  cat(sprintf("  %-16s %8s %10s %10s\n", "set", "n", "median IQR", "pctile"))
  for (nm in c("hub", "lasso")) {
    g <- if (nm == "hub") hub else lasso
    v <- iq[[cols[1]]][match(g, iq[[gcol]])]
    v <- v[!is.na(v)]
    all_v <- iq[[cols[1]]]
    cat(sprintf("  %-16s %8d %10.3f %9.1f%%\n", nm, length(v), median(v),
                100 * mean(all_v < median(v), na.rm = TRUE)))
  }
  cat("\n  Lower IQR for hubs would support the housekeeping reading: stable\n")
  cat("  expression is exactly what makes a gene highly connected and\n")
  cat("  prognostically uninformative.\n")
} else {
  cat("  iqr_filter_values_v4.csv not found; skipping.\n")
}

cat("\n", rep("=", 76), "\n", sep = "")
cat("HOW TO READ THIS\n")
cat(rep("=", 76), "\n", sep = "")
cat("SUPPORTS the housekeeping interpretation:\n")
cat("  hubs enrich for translation / ribosome / oxidative phosphorylation /\n")
cat("  RNA processing, AND lasso genes enrich for ECM / hypoxia / EMT / immune.\n\n")
cat("DOES NOT SUPPORT IT:\n")
cat("  hubs enrich for nothing, or for tumour-biology terms. In that case the\n")
cat("  claim is DROPPED and K1.6.7 is reported as a bare trade-off between\n")
cat("  reproducibility and prognostic content, with no mechanism asserted.\n")

cat("\nFinished:", format(Sys.time()), "\n")
cat("--- sessionInfo ---\n"); print(sessionInfo())
sink(type = "message"); sink(); close(log_con)
