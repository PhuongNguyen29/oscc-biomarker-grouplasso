#!/usr/bin/env Rscript
# ==============================================================================
# Phase K1: Module preservation across cohorts
# ==============================================================================
#
# Does coordinated co-expression structure transport across cohorts, even though
# individual gene selection does not?
#
# Context: marginal effects transport (dev <-> ext r_true = 0.860) while
# cross-cohort selection overlap is zero in all 42 estimable stability cells.
# One explanation is correlated-group structure -- the lasso keeps one arbitrary
# representative per group (Zhao & Yu 2006), and which one it keeps is unstable.
# If so, modules should preserve where individual genes do not.
#
# UNSUPERVISED THROUGHOUT (K1.1)
# ------------------------------
# Network construction, module detection and preservation assessment use
# expression only. No survival variable enters any of it. Eigengene-survival
# association runs in a SEPARATE script, only after preservation is assessed,
# and preservation results are not revised in light of it. Selecting modules on
# survival and then validating them against survival is the defect this
# separation prevents.
#
# References:
#   Langfelder P, Horvath S (2008) BMC Bioinformatics 9:559 -- WGCNA
#   Langfelder P et al. (2011) PLoS Comput Biol 7:e1001057 -- module preservation
#   Zhao P, Yu B (2006) JMLR 7:2541-2563 -- irrepresentable condition
#
# Usage:
#   Rscript scripts/phaseK1_module_preservation.R
#
# RUNTIME: modulePreservation with 200 permutations across 4 datasets takes
# HOURS. Run it as a batch job, not interactively.
# ==============================================================================

suppressPackageStartupMessages({
  library(WGCNA)
  library(dplyr)
})

options(stringsAsFactors = FALSE)
enableWGCNAThreads()

# ==============================================================================
# Configuration -- frozen per K1.2
# ==============================================================================

PROC_DIR   <- "data/processed"
EXT_DIR    <- "data/external"
GENE_LIST  <- file.path(PROC_DIR, "iqr_filtered_v4_prez", "retained_genes.txt")
OUT_DIR    <- "results/phaseK1_module_preservation"

NETWORK_TYPE     <- "signed"   # preserves direction of co-expression
MIN_MODULE_SIZE  <- 30
MERGE_CUT_HEIGHT <- 0.25
DEEP_SPLIT       <- 2
RSQ_TARGET       <- 0.80       # scale-free topology fit
POWER_VECTOR     <- c(1:10, seq(12, 30, 2))
N_PERMUTATIONS   <- 200
SEED             <- 42

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

LOG_DIR <- "logs"
dir.create(LOG_DIR, showWarnings = FALSE, recursive = TRUE)
log_file <- file.path(LOG_DIR,
  sprintf("phaseK1_module_preservation_%s.log", format(Sys.time(), "%Y%m%d_%H%M%S")))
log_con <- file(log_file, open = "wt")
sink(log_con, split = TRUE)
sink(log_con, type = "message")

cat("Phase K1: Module preservation\n")
cat("Log:", log_file, "\n")
cat("Started:", format(Sys.time()), "\n")
cat("Network type:", NETWORK_TYPE, "| minModuleSize:", MIN_MODULE_SIZE,
    "| mergeCutHeight:", MERGE_CUT_HEIGHT, "\n")
cat("Permutations:", N_PERMUTATIONS, "| seed:", SEED, "\n\n")

set.seed(SEED)

# ==============================================================================
# Load: genes x samples -> WGCNA wants samples x genes
# ==============================================================================

cat("--- Loading cohorts ---\n")
retained <- readLines(GENE_LIST)
retained <- retained[nchar(retained) > 0]
cat("Retained genes:", length(retained), "\n")

load_expr <- function(path, label) {
  m <- as.matrix(read.csv(path, row.names = 1, check.names = FALSE))
  cat(sprintf("  %-10s %d genes x %d samples\n", label, nrow(m), ncol(m)))
  m
}

tcga  <- load_expr(file.path(PROC_DIR, "tcga_preprocessed_v4.csv"),  "TCGA")
orien <- load_expr(file.path(PROC_DIR, "orien_preprocessed_v4.csv"), "ORIEN")
g1    <- load_expr(file.path(EXT_DIR,  "GSE41613_expression.csv"),   "GSE41613")
g2    <- load_expr(file.path(EXT_DIR,  "GSE42743_expression.csv"),   "GSE42743")

# Common gene set across ALL cohorts. Preservation statistics require the same
# genes everywhere, so the external platforms' coverage sets the ceiling.
common <- Reduce(intersect, list(rownames(tcga), rownames(orien),
                                 rownames(g1), rownames(g2)))
common <- intersect(common, retained)
cat("Genes common to all four cohorts:", length(common), "\n\n")

# t() gives samples x genes, which is WGCNA's expected orientation
multiExpr <- list(
  TCGA     = list(data = as.data.frame(t(tcga[common, ]))),
  ORIEN    = list(data = as.data.frame(t(orien[common, ]))),
  GSE41613 = list(data = as.data.frame(t(g1[common, ]))),
  GSE42743 = list(data = as.data.frame(t(g2[common, ])))
)

gsg <- goodSamplesGenesMS(multiExpr, verbose = 0)
if (!gsg$allOK) {
  cat("WARNING: goodSamplesGenesMS flagged problems; filtering\n")
  multiExpr <- multiExpr[gsg$goodSets]
  for (s in seq_along(multiExpr)) {
    multiExpr[[s]]$data <- multiExpr[[s]]$data[gsg$goodSamples[[s]],
                                               gsg$goodGenes]
  }
  common <- common[gsg$goodGenes]
  cat("  After filtering:", length(common), "genes\n")
}

# ==============================================================================
# Soft-threshold power, per cohort
# ==============================================================================

cat("--- Soft-threshold power selection (scale-free R2 >=", RSQ_TARGET, ") ---\n")

powers <- list()
for (nm in names(multiExpr)) {
  sft <- pickSoftThreshold(multiExpr[[nm]]$data, powerVector = POWER_VECTOR,
                           networkType = NETWORK_TYPE, verbose = 0)
  fit <- -sign(sft$fitIndices[, 3]) * sft$fitIndices[, 2]
  ok <- which(fit >= RSQ_TARGET)
  if (length(ok) > 0) {
    powers[[nm]] <- sft$fitIndices[min(ok), 1]
    cat(sprintf("  %-10s power = %2d (R2 = %.3f)\n", nm, powers[[nm]],
                fit[min(ok)]))
  } else {
    # WGCNA's documented fallback for signed networks by sample size
    n <- nrow(multiExpr[[nm]]$data)
    powers[[nm]] <- if (n < 20) 20 else if (n < 30) 18 else if (n < 40) 16 else 12
    cat(sprintf("  %-10s R2 never reached %.2f (max %.3f); using default %d\n",
                nm, RSQ_TARGET, max(fit, na.rm = TRUE), powers[[nm]]))
  }
  write.csv(sft$fitIndices,
            file.path(OUT_DIR, paste0("softthreshold_", nm, ".csv")),
            row.names = FALSE)
}
cat("\n")

# ==============================================================================
# Module detection in each REFERENCE cohort (TCGA, ORIEN)
# ==============================================================================

detect_modules <- function(datExpr, power, label) {
  cat("--- Modules in", label, "(power", power, ") ---\n")
  net <- blockwiseModules(
    datExpr,
    power             = power,
    networkType       = NETWORK_TYPE,
    TOMType           = NETWORK_TYPE,
    minModuleSize     = MIN_MODULE_SIZE,
    mergeCutHeight    = MERGE_CUT_HEIGHT,
    deepSplit         = DEEP_SPLIT,
    numericLabels     = FALSE,
    saveTOMs          = FALSE,
    maxBlockSize      = 6000,   # above the gene count, so a single block
    randomSeed        = SEED,
    verbose           = 0
  )
  tab <- sort(table(net$colors), decreasing = TRUE)
  cat("  Modules:", sum(names(tab) != "grey"), "| unassigned (grey):",
      ifelse("grey" %in% names(tab), tab[["grey"]], 0), "\n")
  print(tab)
  cat("\n")
  net
}

net_tcga  <- detect_modules(multiExpr$TCGA$data,  powers$TCGA,  "TCGA")
net_orien <- detect_modules(multiExpr$ORIEN$data, powers$ORIEN, "ORIEN")

module_table <- data.frame(
  gene_id       = common,
  module_TCGA   = net_tcga$colors,
  module_ORIEN  = net_orien$colors
)
write.csv(module_table, file.path(OUT_DIR, "module_assignments.csv"),
          row.names = FALSE)

# Cross-tabulation: do the two development cohorts find similar partitions?
cat("--- TCGA vs ORIEN module overlap (Fisher) ---\n")
ov <- overlapTable(net_tcga$colors, net_orien$colors)
write.csv(ov$countTable, file.path(OUT_DIR, "module_overlap_counts.csv"))
write.csv(ov$pTable,     file.path(OUT_DIR, "module_overlap_pvalues.csv"))
cat("  Written to module_overlap_*.csv\n\n")

# ==============================================================================
# Preservation (K1.3): each reference against every other cohort
# ==============================================================================

cat("--- Module preservation:", N_PERMUTATIONS, "permutations ---\n")
cat("    This takes hours. Progress is written to the log.\n\n")

multiColor <- list(TCGA = net_tcga$colors, ORIEN = net_orien$colors)

t0 <- Sys.time()
mp <- modulePreservation(
  multiExpr, multiColor,
  referenceNetworks = c(1, 2),      # TCGA and ORIEN as references
  nPermutations     = N_PERMUTATIONS,
  randomSeed        = SEED,
  networkType       = NETWORK_TYPE,
  verbose           = 3
)
cat("Preservation completed in", format(Sys.time() - t0), "\n\n")

saveRDS(mp, file.path(OUT_DIR, "modulePreservation.rds"))

# ==============================================================================
# Report
# ==============================================================================

ref_names <- names(multiColor)
all_rows <- list()

for (ref in ref_names) {
  for (test in names(multiExpr)) {
    if (test == ref) next
    z_key <- paste0("inColumnsAlsoPresentIn.", test)
    Z   <- mp$preservation$Z[[ref]][[z_key]]
    obs <- mp$preservation$observed[[ref]][[z_key]]
    if (is.null(Z)) next

    df <- data.frame(
      reference   = ref,
      test        = test,
      module      = rownames(Z),
      moduleSize  = obs$moduleSize,
      Zsummary    = Z$Zsummary.pres,
      Zdensity    = Z$Zdensity.pres,
      Zconnect    = Z$Zconnectivity.pres,
      medianRank  = mp$preservation$observed[[ref]][[z_key]]$medianRank.pres
    )
    all_rows[[paste(ref, test)]] <- df

    cat("=== reference", ref, "-> test", test, "===\n")
    d <- df[order(-df$Zsummary), ]
    cat(sprintf("  %-14s %6s %9s %11s\n",
                "module", "size", "Zsummary", "medianRank"))
    for (i in seq_len(nrow(d))) {
      flag <- if (is.na(d$Zsummary[i])) "" else
              if (d$Zsummary[i] > 10) "STRONG" else
              if (d$Zsummary[i] > 5)  "moderate" else
              if (d$Zsummary[i] > 2)  "weak" else "none"
      cat(sprintf("  %-14s %6d %9.2f %11.1f  %s\n",
                  d$module[i], d$moduleSize[i], d$Zsummary[i],
                  d$medianRank[i], flag))
    }
    cat("\n")
  }
}

res <- bind_rows(all_rows)
write.csv(res, file.path(OUT_DIR, "preservation_statistics.csv"), row.names = FALSE)

# ==============================================================================
# Summary against the K1.4 thresholds
# ==============================================================================

cat("=" , rep("=", 65), "\n", sep = "")
cat("SUMMARY (Langfelder et al. 2011 thresholds, fixed in K1.4)\n")
cat("=" , rep("=", 65), "\n", sep = "")

real <- res[!res$module %in% c("grey", "gold"), ]
cat("  Module-by-test-set combinations:", nrow(real), "\n")
cat("  Zsummary > 10 (strong):  ", sum(real$Zsummary > 10, na.rm = TRUE), "\n")
cat("  5 < Zsummary <= 10:      ", sum(real$Zsummary > 5 & real$Zsummary <= 10,
                                       na.rm = TRUE), "\n")
cat("  Zsummary <= 2 (none):    ", sum(real$Zsummary <= 2, na.rm = TRUE), "\n\n")

# grey = unassigned, gold = random sample of genes. Both are built-in negative
# controls and SHOULD show no preservation. If they do, something is wrong with
# the network or the permutation scheme.
ctrl <- res[res$module %in% c("grey", "gold"), ]
if (nrow(ctrl) > 0) {
  cat("  Negative controls (should NOT be preserved):\n")
  for (i in seq_len(nrow(ctrl))) {
    cat(sprintf("    %-6s %s->%s: Zsummary = %.2f\n", ctrl$module[i],
                ctrl$reference[i], ctrl$test[i], ctrl$Zsummary[i]))
  }
  if (any(ctrl$Zsummary > 10, na.rm = TRUE)) {
    cat("    *** WARNING: a control module appears preserved. Do not\n")
    cat("    *** interpret the other statistics until this is explained.\n")
  }
  cat("\n")
}

# Modules preserved in BOTH external cohorts are the strongest claim available:
# structure found in a development cohort that holds on two independent
# platforms.
for (ref in ref_names) {
  ext <- real[real$reference == ref & real$test %in% c("GSE41613", "GSE42743"), ]
  if (nrow(ext) == 0) next
  wide <- reshape(ext[, c("module", "test", "Zsummary")],
                  idvar = "module", timevar = "test", direction = "wide")
  both <- wide[rowSums(wide[, -1, drop = FALSE] > 10, na.rm = TRUE) ==
               (ncol(wide) - 1), ]
  cat("  ", ref, "modules with Zsummary > 10 in BOTH external cohorts:",
      nrow(both), "\n")
  if (nrow(both) > 0) print(both, row.names = FALSE)
}

cat("\nNEXT: eigengene-survival association (K1.5) is a SEPARATE script and\n")
cat("runs only on modules assessed here. Preservation is not revised in\n")
cat("light of survival association.\n")

cat("\nFinished:", format(Sys.time()), "\n")
cat("--- sessionInfo ---\n")
print(sessionInfo())

sink(type = "message")
sink()
close(log_con)
