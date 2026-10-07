# ==============================================================================
# ECMS relabel check (design §7b) -- ONE CALL = one shuffle draw for one reference
#
# Question (claim C0): for a module that is coherent in the discovery cohort but
# ABSENT in a test cohort, is the matched-size empirical p-value uniform or
# conservative?
#
# One call:
#   - reference = TCGA or ORIEN, with its REAL K1 modules (module_assignments.csv)
#   - every OTHER cohort (the three test cohorts) gets its gene identities
#     shuffled: same matrix, permuted column names. Each test cohort keeps its own
#     correlation structure; correspondence with the reference is destroyed, so
#     every real module is absent by construction.
#   - modulePreservation exactly as K1 / the null (signed, 200 permutations,
#     default size caps).
# The reference cohort itself is never shuffled (its labels must stay valid),
# which is why each call handles one reference.
#
# Usage (from the project root, in a terminal):
#   OPENBLAS_NUM_THREADS=1 Rscript scripts/ecms_relabel_run.R <draw> <TCGA|ORIEN>
# Normally launched by scripts/ecms_relabel_driver.R.
# Output: results/ecms_relabel/relabel_<ref>_draw<NN>.csv
# TIME ONE CALL FIRST.
# ==============================================================================

suppressPackageStartupMessages({ library(WGCNA); library(dplyr) })
options(stringsAsFactors = FALSE)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop(
  "Needs <draw> <TCGA|ORIEN>; do not source() it.\n",
  "  OPENBLAS_NUM_THREADS=1 Rscript scripts/ecms_relabel_run.R 1 TCGA")
DRAW <- as.integer(args[1]); REF <- args[2]
stopifnot(!is.na(DRAW), DRAW >= 1, REF %in% c("TCGA", "ORIEN"))
if (Sys.getenv("OPENBLAS_NUM_THREADS") != "1") stop("set OPENBLAS_NUM_THREADS=1")

NTHREADS <- max(2L, as.integer(Sys.getenv("ECMS_WGCNA_THREADS", "2")))
enableWGCNAThreads(nThreads = NTHREADS)
stamp <- function(...) cat(format(Sys.time(), "[%Y-%m-%d %H:%M:%S] "), ..., "\n", sep = "")

# --- Identical settings to phaseK1_module_preservation.R / ecms_null_run.R ----
PROC_DIR  <- "data/processed"
EXT_DIR   <- "data/external"
GENE_LIST <- file.path(PROC_DIR, "iqr_filtered_v4_prez", "retained_genes.txt")
K1_DIR    <- "results/phaseK1_module_preservation"
OUT_DIR   <- "results/ecms_relabel"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
NETWORK_TYPE   <- "signed"
N_PERMUTATIONS <- 200
ALL_COHORTS    <- c("TCGA", "ORIEN", "GSE41613", "GSE42743")
SHUFFLE_SEED   <- 70000L + DRAW * 10L + match(REF, c("TCGA", "ORIEN"))
MP_SEED        <- 4200L + DRAW

tag <- sprintf("%s_draw%02d", REF, DRAW)
stamp("START relabel ", tag, " | threads ", NTHREADS)
t0 <- Sys.time()

# --- Load exactly as K1 / the null does ------------------------------------------
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
rm(expr_list); invisible(gc())
stamp("genes: ", length(common))

# --- Real K1 module labels for the reference --------------------------------------
# module_assignments.csv layout (confirmed 7 Oct): gene_id, module_TCGA, module_ORIEN
ma <- read.csv(file.path(K1_DIR, "module_assignments.csv"), check.names = FALSE)
mod_col <- paste0("module_", REF)
if (!all(c("gene_id", mod_col) %in% names(ma)))
  stop("module_assignments.csv: expected columns gene_id and ", mod_col,
       ". Found: ", paste(names(ma), collapse = ", "))
if (anyDuplicated(ma$gene_id)) stop("duplicate gene_id in module_assignments.csv")
lab_raw <- setNames(as.character(ma[[mod_col]]), ma$gene_id)
labels <- unname(lab_raw[common])
if (anyNA(labels)) stop(sum(is.na(labels)), " of ", length(common),
                        " analysis genes have no ", REF, " module label")

# Tripwire: label counts must reproduce the K1 module sizes exactly
pres <- read.csv(file.path(K1_DIR, "preservation_statistics.csv"))
k1 <- unique(pres[pres$reference == REF & !pres$module %in% c("gold", "grey"),
                  c("module", "moduleSize")])
tab <- table(labels)
bad <- k1$module[as.integer(tab[k1$module]) != k1$moduleSize | is.na(tab[k1$module])]
if (length(bad)) stop("label counts do not match K1 module sizes for: ",
                      paste(bad, collapse = ", "))
stamp("labels verified against K1 sizes: ", nrow(k1), " modules")

# --- Shuffle gene identities in every test cohort (independently) ----------------
set.seed(SHUFFLE_SEED)
tests <- setdiff(ALL_COHORTS, REF)
for (tt in tests) {
  d <- multiExpr[[tt]]$data
  nm_keep <- colnames(d)
  d <- d[, sample(ncol(d))]
  colnames(d) <- nm_keep            # same gene names, permuted content
  multiExpr[[tt]]$data <- d
}
stamp("shuffled: ", paste(tests, collapse = ", "), " (seed ", SHUFFLE_SEED, ")")

# --- Preservation, identical pipeline ------------------------------------------
multiColor <- setNames(list(labels), REF)
mp <- modulePreservation(
  multiExpr, multiColor,
  referenceNetworks = match(REF, names(multiExpr)),
  nPermutations = N_PERMUTATIONS, randomSeed = MP_SEED,
  networkType = NETWORK_TYPE,
  maxModuleSize = 1000, maxGoldModuleSize = 1000,
  savePermutedStatistics = FALSE, verbose = 1)
stamp("modulePreservation done")

# --- Extract (real modules only; gold/grey kept but flagged) -------------------
rk <- paste0("ref.", REF)
if (!rk %in% names(mp$preservation$Z))
  stop("reference key '", rk, "' not found; names: ", paste(names(mp$preservation$Z), collapse = ", "))
rows <- list()
for (tt in tests) {
  tk  <- paste0("inColumnsAlsoPresentIn.", tt)
  Z   <- mp$preservation$Z[[rk]][[tk]]
  obs <- mp$preservation$observed[[rk]][[tk]]
  if (!is.data.frame(Z)) stop("no Z table for ", tk)
  rows[[tt]] <- data.frame(draw = DRAW, reference = REF, test = tt, module = rownames(Z),
                           moduleSize = obs$moduleSize, Zsummary = Z$Zsummary.pres,
                           is_control = rownames(Z) %in% c("gold", "grey"))
}
res <- bind_rows(rows)
if (sum(!res$is_control) != 3 * nrow(k1)) stop("expected ", 3 * nrow(k1),
                                               " real-module rows, got ", sum(!res$is_control))
write.csv(res, file.path(OUT_DIR, sprintf("relabel_%s.csv", tag)), row.names = FALSE)
stamp(sprintf("END %s | %d rows | %.1f min", tag, nrow(res),
              as.numeric(difftime(Sys.time(), t0, units = "mins"))))
