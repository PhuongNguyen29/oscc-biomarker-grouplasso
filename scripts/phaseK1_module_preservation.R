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
# association is a SEPARATE script (K1.5) run afterwards, and preservation
# results are not revised in light of it.
#
# FIXES IN THIS REVISION
# ----------------------
# 1. RESULT EXTRACTION WAS BROKEN. modulePreservation names its reference
#    entries "ref.<name>", not "<name>". Indexing with the bare cohort name
#    returned NULL for every reference, so the reporting loop skipped silently
#    and printed "Module-by-test-set combinations: 0" despite a successful
#    1.4-hour computation. Keys are now validated against names() before use,
#    so a future mismatch fails loudly instead of reporting an empty result.
#
# 2. Zsummary > 10 IS NOT A USABLE THRESHOLD HERE. The built-in null modules
#    ("gold" = random genes, "grey" = unassigned) reached Zsummary 24.6 and
#    13.9 in this dataset -- far above the published cutoff. Zsummary increases
#    with module size, and both controls are set at 1000 genes, larger than any
#    real module. Every module is now calibrated against the observed
#    gold/grey values WITHIN its own comparison, with medianRank
#    (size-independent) alongside. The Langfelder label is still printed but
#    marked uncalibrated for this data.
#
# 3. The report can be regenerated from the saved RDS without repeating the
#    permutations (REUSE_RDS).
#
# References:
#   Langfelder P, Horvath S (2008) BMC Bioinformatics 9:559 -- WGCNA
#   Langfelder P et al. (2011) PLoS Comput Biol 7:e1001057 -- module preservation
#   Zhao P, Yu B (2006) JMLR 7:2541-2563 -- irrepresentable condition
#
# Usage:
#   Rscript scripts/phaseK1_module_preservation.R
#
# RUNTIME: ~1.4 hours at 200 permutations, 4 datasets, 4702 genes.
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

NETWORK_TYPE     <- "signed"
MIN_MODULE_SIZE  <- 30
MERGE_CUT_HEIGHT <- 0.25
DEEP_SPLIT       <- 2
RSQ_TARGET       <- 0.80
POWER_VECTOR     <- c(1:10, seq(12, 30, 2))
N_PERMUTATIONS   <- 200
SEED             <- 42
REUSE_RDS        <- TRUE     # skip permutations if modulePreservation.rds exists

REF_COHORTS     <- c("TCGA", "ORIEN")
ALL_COHORTS     <- c("TCGA", "ORIEN", "GSE41613", "GSE42743")
CONTROL_MODULES <- c("gold", "grey")

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
cat("Network:", NETWORK_TYPE, "| minModuleSize:", MIN_MODULE_SIZE,
    "| mergeCutHeight:", MERGE_CUT_HEIGHT, "| deepSplit:", DEEP_SPLIT, "\n")
cat("Permutations:", N_PERMUTATIONS, "| seed:", SEED, "\n\n")

set.seed(SEED)

# ==============================================================================
# Load: matrices are genes x samples; WGCNA wants samples x genes
# ==============================================================================

cat("--- Loading cohorts ---\n")
retained <- readLines(GENE_LIST)
retained <- retained[nchar(retained) > 0]
cat("Retained genes:", length(retained), "\n")

load_expr <- function(path, label) {
  m <- as.matrix(read.csv(path, row.names = 1, check.names = FALSE))
  if (any(duplicated(rownames(m)))) {
    stop(label, ": duplicate gene IDs in expression matrix")
  }
  cat(sprintf("  %-10s %d genes x %d samples\n", label, nrow(m), ncol(m)))
  m
}

expr_list <- list(
  TCGA     = load_expr(file.path(PROC_DIR, "tcga_corrected_preprocessed_v4.csv"),  "TCGA"),
  ORIEN    = load_expr(file.path(PROC_DIR, "orien_corrected_preprocessed_v4.csv"), "ORIEN"),
  GSE41613 = load_expr(file.path(EXT_DIR,  "GSE41613_expression.csv"),   "GSE41613"),
  GSE42743 = load_expr(file.path(EXT_DIR,  "GSE42743_expression.csv"),   "GSE42743")
)

# Preservation requires the same gene set in every cohort, so the external
# platforms' coverage sets the ceiling.
common <- Reduce(intersect, lapply(expr_list, rownames))
common <- intersect(common, retained)
cat("Genes common to all four cohorts:", length(common), "\n\n")

multiExpr <- lapply(expr_list, function(m)
  list(data = as.data.frame(t(m[common, ]))))
names(multiExpr) <- names(expr_list)

gsg <- goodSamplesGenesMS(multiExpr, verbose = 0)
if (!gsg$allOK) {
  cat("WARNING: goodSamplesGenesMS flagged problems; filtering\n")
  for (s in seq_along(multiExpr)) {
    multiExpr[[s]]$data <- multiExpr[[s]]$data[gsg$goodSamples[[s]], gsg$goodGenes]
  }
  common <- common[gsg$goodGenes]
  cat("  After filtering:", length(common), "genes\n")
}

# ==============================================================================
# Soft-threshold power, per cohort
# ==============================================================================

cat("--- Soft-threshold power (scale-free R2 >=", RSQ_TARGET, ") ---\n")

powers <- list()
for (nm in names(multiExpr)) {
  sft <- pickSoftThreshold(multiExpr[[nm]]$data, powerVector = POWER_VECTOR,
                           networkType = NETWORK_TYPE, verbose = 0)
  fit <- -sign(sft$fitIndices[, 3]) * sft$fitIndices[, 2]
  ok <- which(fit >= RSQ_TARGET)
  if (length(ok) > 0) {
    powers[[nm]] <- sft$fitIndices[min(ok), 1]
    cat(sprintf("  %-10s power = %2d (R2 = %.3f)\n", nm, powers[[nm]], fit[min(ok)]))
  } else {
    n <- nrow(multiExpr[[nm]]$data)
    powers[[nm]] <- if (n < 20) 20 else if (n < 30) 18 else if (n < 40) 16 else 12
    cat(sprintf("  %-10s R2 never reached %.2f (max %.3f); WGCNA default %d\n",
                nm, RSQ_TARGET, max(fit, na.rm = TRUE), powers[[nm]]))
  }
  write.csv(sft$fitIndices,
            file.path(OUT_DIR, paste0("softthreshold_", nm, ".csv")),
            row.names = FALSE)
}
cat("\n")

# ==============================================================================
# Module detection in each reference cohort
# ==============================================================================

detect_modules <- function(datExpr, power, label) {
  cat("--- Modules in", label, "(power", power, ") ---\n")
  net <- blockwiseModules(
    datExpr, power = power,
    networkType = NETWORK_TYPE, TOMType = NETWORK_TYPE,
    minModuleSize = MIN_MODULE_SIZE, mergeCutHeight = MERGE_CUT_HEIGHT,
    deepSplit = DEEP_SPLIT, numericLabels = FALSE, saveTOMs = FALSE,
    maxBlockSize = 6000,     # above the gene count, so a single block
    randomSeed = SEED, verbose = 0
  )
  tab <- sort(table(net$colors), decreasing = TRUE)
  cat("  Modules:", sum(names(tab) != "grey"),
      "| unassigned (grey):", ifelse("grey" %in% names(tab), tab[["grey"]], 0), "\n")
  print(tab); cat("\n")
  net
}

nets <- list()
for (r in REF_COHORTS) {
  nets[[r]] <- detect_modules(multiExpr[[r]]$data, powers[[r]], r)
}

module_table <- data.frame(gene_id = common)
for (r in REF_COHORTS) module_table[[paste0("module_", r)]] <- nets[[r]]$colors
write.csv(module_table, file.path(OUT_DIR, "module_assignments.csv"), row.names = FALSE)

# Do the reference cohorts find similar partitions? Fisher on gene membership.
# NOTE: anticonservative -- co-expressed genes are not exchangeable. Use for
# ranking evidence, not as calibrated p-values.
cat("--- TCGA vs ORIEN module overlap (Fisher) ---\n")
ov <- overlapTable(nets$TCGA$colors, nets$ORIEN$colors)
write.csv(ov$countTable, file.path(OUT_DIR, "module_overlap_counts.csv"))
write.csv(ov$pTable,     file.path(OUT_DIR, "module_overlap_pvalues.csv"))
cat("  Written to module_overlap_*.csv\n\n")

# ==============================================================================
# Preservation
# ==============================================================================

rds_path <- file.path(OUT_DIR, "modulePreservation.rds")
multiColor <- lapply(nets[REF_COHORTS], function(n) n$colors)
names(multiColor) <- REF_COHORTS

if (REUSE_RDS && file.exists(rds_path)) {
  cat("--- Reusing saved modulePreservation.rds (skipping permutations) ---\n")
  cat("    Delete it or set REUSE_RDS <- FALSE to recompute.\n\n")
  mp <- readRDS(rds_path)
} else {
  cat("--- Module preservation:", N_PERMUTATIONS, "permutations ---\n")
  cat("    ~1.4 hours. Progress below.\n\n")
  t0 <- Sys.time()
  mp <- modulePreservation(
    multiExpr, multiColor,
    referenceNetworks = match(REF_COHORTS, names(multiExpr)),
    nPermutations = N_PERMUTATIONS, randomSeed = SEED,
    networkType = NETWORK_TYPE, verbose = 3
  )
  cat("Completed in", format(Sys.time() - t0), "\n\n")
  saveRDS(mp, rds_path)
}

# ==============================================================================
# Extraction
#
# modulePreservation names reference entries "ref.<name>" and test entries
# "inColumnsAlsoPresentIn.<name>". Constructing these by hand is what broke the
# previous version, so both are validated against names() before use.
# ==============================================================================

stopifnot(all(paste0("ref.", REF_COHORTS) %in% names(mp$preservation$Z)))

rows <- list()
for (ref in REF_COHORTS) {
  rk <- paste0("ref.", ref)
  for (test in ALL_COHORTS) {
    if (test == ref) next
    tk <- paste0("inColumnsAlsoPresentIn.", test)
    if (!tk %in% names(mp$preservation$Z[[rk]])) {
      stop("Missing test key '", tk, "' under '", rk, "'. Available: ",
           paste(names(mp$preservation$Z[[rk]]), collapse = ", "))
    }
    Z   <- mp$preservation$Z[[rk]][[tk]]
    obs <- mp$preservation$observed[[rk]][[tk]]
    if (!is.data.frame(Z)) next          # self-comparison is a logical NA
    rows[[paste(ref, test)]] <- data.frame(
      reference = ref, test = test, module = rownames(Z),
      moduleSize = obs$moduleSize,
      Zsummary   = Z$Zsummary.pres,
      Zdensity   = Z$Zdensity.pres,
      Zconnect   = Z$Zconnectivity.pres,
      medianRank = obs$medianRank.pres
    )
  }
}
if (length(rows) == 0) stop("No preservation results extracted -- check key names.")

res <- bind_rows(rows)

# Calibrate against the null modules WITHIN each comparison. gold = random
# genes, grey = unassigned; both are set at 1000 genes, larger than any real
# module, and Zsummary grows with module size -- which is why the published
# Zsummary > 10 threshold does not hold in this dataset.
ctrl <- res %>%
  filter(module %in% CONTROL_MODULES) %>%
  group_by(reference, test) %>%
  summarise(gold_Z = max(Zsummary[module == "gold"], -Inf),
            grey_Z = max(Zsummary[module == "grey"], -Inf),
            null_Z = max(Zsummary), .groups = "drop")

res <- res %>%
  left_join(ctrl, by = c("reference", "test")) %>%
  mutate(
    is_control       = module %in% CONTROL_MODULES,
    above_null       = !is_control & Zsummary > null_Z,
    Z_over_null      = Zsummary - null_Z,
    langfelder_label = cut(Zsummary, c(-Inf, 2, 5, 10, Inf),
                           labels = c("none", "weak", "moderate", "STRONG"))
  )

write.csv(res, file.path(OUT_DIR, "preservation_statistics.csv"), row.names = FALSE)

# ==============================================================================
# Report
# ==============================================================================

for (ref in REF_COHORTS) for (test in ALL_COHORTS) {
  d <- res[res$reference == ref & res$test == test, ]
  if (!nrow(d)) next
  d <- d[order(-d$Zsummary), ]
  cat("=== reference", ref, "-> test", test, "===\n")
  cat(sprintf("  null ceiling (max of gold/grey) = %.2f\n", d$null_Z[1]))
  cat(sprintf("  %-11s %6s %9s %8s %11s %s\n",
              "module", "size", "Zsummary", "vs null", "medianRank", "verdict"))
  for (i in seq_len(nrow(d))) {
    verdict <- if (d$is_control[i]) "[NULL CONTROL]"
               else if (isTRUE(d$above_null[i])) "preserved" else "not above null"
    cat(sprintf("  %-11s %6d %9.2f %8.2f %11.1f %s\n",
                d$module[i], d$moduleSize[i], d$Zsummary[i],
                d$Z_over_null[i], d$medianRank[i], verdict))
  }
  cat("\n")
}

cat(rep("=", 66), "\n", sep = "")
cat("SUMMARY\n")
cat(rep("=", 66), "\n", sep = "")

real <- res[!res$is_control, ]
cat("  Module-by-test-set combinations:", nrow(real), "\n")
cat("  Preserved (Zsummary above the null ceiling):",
    sum(real$above_null, na.rm = TRUE), "\n")
cat("  Langfelder Zsummary > 10:", sum(real$Zsummary > 10, na.rm = TRUE),
    " <- UNCALIBRATED here; see note below\n\n")

cat("  Null-control Zsummary by comparison:\n")
for (i in seq_len(nrow(ctrl))) {
  cat(sprintf("    %-6s -> %-9s gold=%6.2f  grey=%6.2f\n",
              ctrl$reference[i], ctrl$test[i], ctrl$gold_Z[i], ctrl$grey_Z[i]))
}
if (any(ctrl$null_Z > 10)) {
  cat("\n  NOTE: null modules exceed Zsummary = 10 in at least one comparison.\n")
  cat("  Zsummary increases with module size and both controls are set at 1000\n")
  cat("  genes, larger than any real module here. Report preservation relative\n")
  cat("  to these controls and alongside medianRank, not against the published\n")
  cat("  threshold alone.\n")
}

# Modules preserved in BOTH external cohorts: structure found in a development
# cohort that holds on two independent microarray platforms.
cat("\n  Preserved in BOTH external cohorts (above null in each):\n")
for (ref in REF_COHORTS) {
  ext <- real %>%
    filter(reference == ref, test %in% c("GSE41613", "GSE42743")) %>%
    group_by(module) %>%
    summarise(n_ok = sum(above_null, na.rm = TRUE), size = first(moduleSize),
              minZ = min(Zsummary), maxRank = max(medianRank), .groups = "drop") %>%
    filter(n_ok == 2)
  cat("    ", ref, ":", nrow(ext), "modules\n")
  if (nrow(ext)) print(as.data.frame(ext), row.names = FALSE)
}

cat("\nNEXT: eigengene-survival association (K1.5) is a SEPARATE script, run\n")
cat("only on modules assessed here. Preservation is not revised in light of it.\n")

cat("\nFinished:", format(Sys.time()), "\n")
cat("--- sessionInfo ---\n")
print(sessionInfo())

sink(type = "message")
sink()
close(log_con)
