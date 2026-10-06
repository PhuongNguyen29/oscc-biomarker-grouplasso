# ==============================================================================
# ECMS matched-size null -- ONE CALL (run many in parallel as an array job)
#
# Each call randomly partitions the 4,702 analysis genes into "random modules"
# whose sizes are drawn from the REAL module sizes of each reference cohort
# (plus 1000, the size WGCNA caps grey and gold at). The random modules are then
# run through EXACTLY the same modulePreservation pipeline as K1: same cohorts,
# same genes, signed network, 200 permutations, WGCNA's default size caps.
# Their Zsummary values form the null against which the real modules are judged:
# "is this module more preserved than random genes of the same size, in the
# same comparison?" This generalises the ceiling rule, which compared each
# module with a single gold module of the wrong size.
#
# Labels are NUMERIC (0 = unassigned, 0.1 = WGCNA's gold), so random modules can
# never collide with WGCNA's reserved colour names.
#
# Usage:
#   Rscript scripts/ecms_null_run.R <call_index>
#   or as an array job: the index is read from SLURM_ARRAY_TASK_ID or
#   SGE_TASK_ID if no argument is given.
# Output: results/ecms_null/random_call_<index>.csv
#
# TIME ONE CALL FIRST. A call covers nearly all 4,702 genes (the real run
# covered fewer), so expect it to take somewhat longer than K1's ~1.4 h.
# ==============================================================================

suppressPackageStartupMessages({ library(WGCNA); library(dplyr) })
options(stringsAsFactors = FALSE)

args <- commandArgs(trailingOnly = TRUE)
CALL <- as.integer(if (length(args)) args[1] else
          Sys.getenv("SLURM_ARRAY_TASK_ID", Sys.getenv("SGE_TASK_ID", "1")))
stopifnot(!is.na(CALL), CALL >= 1)

NTHREADS <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", Sys.getenv("NSLOTS", "8")))
enableWGCNAThreads(nThreads = NTHREADS)

# --- Identical settings to phaseK1_module_preservation.R ---------------------
PROC_DIR  <- "data/processed"
EXT_DIR   <- "data/external"
GENE_LIST <- file.path(PROC_DIR, "iqr_filtered_v4_prez", "retained_genes.txt")
K1_DIR    <- "results/phaseK1_module_preservation"
OUT_DIR   <- "results/ecms_null"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

NETWORK_TYPE   <- "signed"
N_PERMUTATIONS <- 200          # must match K1 so null and observed Z are comparable
SEED           <- 42
REF_COHORTS    <- c("TCGA", "ORIEN")
ALL_COHORTS    <- c("TCGA", "ORIEN", "GSE41613", "GSE42743")

cat("ECMS null call", CALL, "| threads", NTHREADS, "| started", format(Sys.time()), "\n")
t0 <- Sys.time()

# --- Load exactly as K1 does --------------------------------------------------
retained <- readLines(GENE_LIST); retained <- retained[nchar(retained) > 0]
load_expr <- function(p) as.matrix(read.csv(p, row.names = 1, check.names = FALSE))
expr_list <- list(
  TCGA     = load_expr(file.path(PROC_DIR, "tcga_corrected_preprocessed_v4.csv")),
  ORIEN    = load_expr(file.path(PROC_DIR, "orien_corrected_preprocessed_v4.csv")),
  GSE41613 = load_expr(file.path(EXT_DIR,  "GSE41613_expression.csv")),
  GSE42743 = load_expr(file.path(EXT_DIR,  "GSE42743_expression.csv")))
common <- Reduce(intersect, lapply(expr_list, rownames))
common <- intersect(common, retained)
multiExpr <- lapply(expr_list, function(m) list(data = as.data.frame(t(m[common, ]))))
names(multiExpr) <- names(expr_list)
gsg <- goodSamplesGenesMS(multiExpr, verbose = 0)
if (!gsg$allOK) {
  for (s in seq_along(multiExpr))
    multiExpr[[s]]$data <- multiExpr[[s]]$data[gsg$goodSamples[[s]], gsg$goodGenes]
  common <- common[gsg$goodGenes]
}
rm(expr_list); gc()
cat("Genes:", length(common), "\n")

# --- Size sets: the real module sizes per reference, plus the 1000 cap --------
pres <- read.csv(file.path(K1_DIR, "preservation_statistics.csv"))
size_set <- function(ref) {
  s <- pres$moduleSize[pres$reference == ref & !pres$module %in% c("gold", "grey")]
  sort(unique(c(s, 1000)))
}

# Random partition of `genes` into chunks whose sizes are drawn from `sizes`.
# Labels are aligned to the input order of `genes`; leftover genes get 0.
random_partition <- function(genes, sizes, seed) {
  set.seed(seed)
  perm <- sample(genes)
  lab  <- integer(length(perm))
  i <- 1L; k <- 0L
  repeat {
    remaining <- length(perm) - i + 1L
    fits <- sizes[sizes <= remaining]
    if (!length(fits)) break
    s <- fits[sample.int(length(fits), 1L)]
    k <- k + 1L
    lab[i:(i + s - 1L)] <- k
    i <- i + s
    if (i > length(perm)) break
  }
  out <- integer(length(genes))
  out[match(perm, genes)] <- lab
  out
}

multiColor <- list(
  TCGA  = random_partition(common, size_set("TCGA"),  SEED * 10000 + CALL),
  ORIEN = random_partition(common, size_set("ORIEN"), SEED * 10000 + CALL + 5000))
for (r in REF_COHORTS)
  cat(sprintf("  %-5s: %d random modules, %d genes left unassigned\n",
              r, length(setdiff(unique(multiColor[[r]]), 0)), sum(multiColor[[r]] == 0)))

# --- Preservation, identical pipeline ------------------------------------------
mp <- modulePreservation(
  multiExpr, multiColor,
  referenceNetworks = match(REF_COHORTS, names(multiExpr)),
  nPermutations = N_PERMUTATIONS, randomSeed = SEED + CALL,
  networkType = NETWORK_TYPE,
  maxModuleSize = 1000, maxGoldModuleSize = 1000,   # WGCNA defaults, as in K1
  savePermutedStatistics = FALSE,                    # avoid file collisions in arrays
  verbose = 0)

# --- Extract ---------------------------------------------------------------------
rows <- list()
for (ref in REF_COHORTS) {
  rk <- paste0("ref.", ref)
  for (test in setdiff(ALL_COHORTS, ref)) {
    tk <- paste0("inColumnsAlsoPresentIn.", test)
    Z <- mp$preservation$Z[[rk]][[tk]]
    obs <- mp$preservation$observed[[rk]][[tk]]
    if (!is.data.frame(Z)) next
    rows[[paste(ref, test)]] <- data.frame(
      call = CALL, reference = ref, test = test, module = rownames(Z),
      moduleSize = obs$moduleSize, Zsummary = Z$Zsummary.pres,
      is_random = !rownames(Z) %in% c("0", "0.1"))
  }
}
res <- bind_rows(rows)
write.csv(res, file.path(OUT_DIR, sprintf("random_call_%03d.csv", CALL)), row.names = FALSE)

cat(sprintf("Call %d done: %d random-module rows | %.1f min\n", CALL,
            sum(res$is_random), as.numeric(difftime(Sys.time(), t0, units = "mins"))))
