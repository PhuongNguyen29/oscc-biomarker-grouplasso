# ==============================================================================
# ECMS hard null (Scenario E): calibrate reorg_phi on RAW correlation structure
# (replaces the Zsummary-based calibration of 6 Oct, which is kept in the record
#  as superseded; design v2, decision of 7 Oct 2026)
#
# PRE-REGISTERED (7 Oct 2026, before this script ran):
#   Quantity: EXCESS within-module correlation in a test cohort =
#       mean pairwise Pearson r among the module's genes in that cohort
#     - mean pairwise r among ALL analysis genes in that cohort.
#   Real target: median excess r over the same 12 real non-preserved
#     module x test-cohort cells as before (TCGA black, magenta, yellow;
#     ORIEN turquoise; each in its three test cohorts). Modules = real K1 labels.
#   Simulation: base parameters with reorganized absence; phi grid 0.05 to 1.00
#     in steps of 0.05; 20 replicates per phi; statistic = median excess r over
#     all absent (reorganized) module x cohort cells (12 per replicate).
#   Rule: phi* = grid value whose simulated median is closest to the real median.
#     If the real median exceeds the simulated median at phi = 1, use phi* = 1
#     and report the model's limit. No other tuning. No Zsummary, no method
#     output, is computed anywhere in this script.
#
# Usage (project root, terminal):
#   nohup env OPENBLAS_NUM_THREADS=1 Rscript scripts/ecms_excess_r_calibration.R > results/ecms_excess_r.log 2>&1 &
# Output: results/ecms_excess_r/{real_cells.csv, sim_grid.csv, summary.txt}
# Runtime: about 1 min for real data + roughly 10 to 20 min for the grid.
# ==============================================================================

options(stringsAsFactors = FALSE)
OUT_DIR <- "results/ecms_excess_r"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
stamp <- function(...) { cat(format(Sys.time(), "[%H:%M:%S] "), ..., "\n", sep = ""); flush.console() }

# Exact mean off-diagonal Pearson r of the columns of X, without forming the
# p x p matrix: sum of all r_ij = sum_i (sum_g z_ig)^2 / (n - 1).
mean_r_all <- function(X) {
  Z <- scale(X); n <- nrow(Z); p <- ncol(Z)
  s <- sum(rowSums(Z)^2) / (n - 1)
  (s - p) / (p * (p - 1))
}
mean_r_set <- function(X) { R <- cor(X); mean(R[upper.tri(R)]) }

# ==============================================================================
# 1. Real data: the 12 non-preserved cells
# ==============================================================================
stamp("REAL: loading cohorts exactly as K1")
PROC_DIR <- "data/processed"; EXT_DIR <- "data/external"
K1_DIR   <- "results/phaseK1_module_preservation"
retained <- readLines(file.path(PROC_DIR, "iqr_filtered_v4_prez", "retained_genes.txt"))
retained <- retained[nchar(retained) > 0]
load_expr <- function(p) as.matrix(read.csv(p, row.names = 1, check.names = FALSE))
expr <- list(
  TCGA     = load_expr(file.path(PROC_DIR, "tcga_corrected_preprocessed_v4.csv")),
  ORIEN    = load_expr(file.path(PROC_DIR, "orien_corrected_preprocessed_v4.csv")),
  GSE41613 = load_expr(file.path(EXT_DIR,  "GSE41613_expression.csv")),
  GSE42743 = load_expr(file.path(EXT_DIR,  "GSE42743_expression.csv")))
common <- intersect(Reduce(intersect, lapply(expr, rownames)), retained)
X <- lapply(expr, function(m) t(m[common, ]))           # samples x genes
rm(expr); invisible(gc())
stamp("genes: ", length(common))
if (length(common) != 4702) stop("expected 4,702 analysis genes, got ", length(common))

ma <- read.csv(file.path(K1_DIR, "module_assignments.csv"), check.names = FALSE)
stopifnot(all(c("gene_id", "module_TCGA", "module_ORIEN") %in% names(ma)))
lab <- list(TCGA  = setNames(ma$module_TCGA,  ma$gene_id)[common],
            ORIEN = setNames(ma$module_ORIEN, ma$gene_id)[common])
if (anyNA(unlist(lab))) stop("some analysis genes have no K1 label")

cells <- data.frame(reference = c(rep("TCGA", 9), rep("ORIEN", 3)),
                    module = c(rep(c("black", "magenta", "yellow"), each = 3), rep("turquoise", 3)))
cells$test <- c(rep(c("ORIEN", "GSE41613", "GSE42743"), 3), c("TCGA", "GSE41613", "GSE42743"))

bg <- vapply(X, mean_r_all, numeric(1))
stamp("background mean r per cohort: ", paste(sprintf("%s %.3f", names(bg), bg), collapse = " | "))

cells$size <- NA_integer_; cells$mean_r <- NA_real_; cells$mean_r_ref <- NA_real_
for (i in seq_len(nrow(cells))) {
  g <- common[lab[[cells$reference[i]]] == cells$module[i]]
  cells$size[i]       <- length(g)
  cells$mean_r[i]     <- mean_r_set(X[[cells$test[i]]][, g])
  cells$mean_r_ref[i] <- mean_r_set(X[[cells$reference[i]]][, g])
}
cells$background <- bg[cells$test]
cells$excess_r   <- cells$mean_r - cells$background
write.csv(cells, file.path(OUT_DIR, "real_cells.csv"), row.names = FALSE)
TARGET <- median(cells$excess_r)
stamp(sprintf("REAL: median excess r = %.4f (range %.4f to %.4f, n = %d cells)",
              TARGET, min(cells$excess_r), max(cells$excess_r), nrow(cells)))

# ==============================================================================
# 2. Simulation grid (generator only; no WGCNA)
# ==============================================================================
source("scripts/ecms_sim_generator.R")
PHIS <- seq(0.05, 1.00, by = 0.05); REPS <- 1:20
rows <- list()
for (phi in PHIS) {
  par <- ecms_base_params(); par$reorg_phi <- phi
  for (rep in REPS) {
    sim <- ecms_generate(par, seed = 50000L + rep, verbose = FALSE)
    for (tt in colnames(sim$M)) {
      Xt <- as.matrix(sim$multiExpr[[tt]]$data)
      b  <- mean_r_all(Xt)
      for (k in which(sim$M[, tt] == 0)) {
        idx <- which(sim$multiColor$disc == k)
        rows[[length(rows) + 1]] <- data.frame(
          phi = phi, rep = rep, test = tt, module = k, size = length(idx),
          excess_r = mean_r_set(Xt[, idx]) - b)
      }
    }
  }
  stamp(sprintf("SIM: phi %.2f done", phi))
}
grid <- do.call(rbind, rows)
write.csv(grid, file.path(OUT_DIR, "sim_grid.csv"), row.names = FALSE)

s <- aggregate(excess_r ~ phi, data = grid, FUN = median)
names(s)[2] <- "median_excess_r"
s$abs_diff <- abs(s$median_excess_r - TARGET)

out <- c(
  "=== Hard-null calibration on raw correlation (comparator-independent) ===",
  sprintf("Real target: median excess r = %.4f over %d non-preserved cells (range %.4f to %.4f)",
          TARGET, nrow(cells), min(cells$excess_r), max(cells$excess_r)),
  "", capture.output(print(cells[, c("reference", "module", "test", "size",
                                     "mean_r_ref", "mean_r", "background", "excess_r")],
                           row.names = FALSE, digits = 3)),
  "", "Simulated median excess r by phi:",
  capture.output(print(s, row.names = FALSE, digits = 4)))

if (TARGET > max(s$median_excess_r)) {
  out <- c(out, "", sprintf(paste0("Real target exceeds the model's maximum (%.4f at phi = 1): ",
                                   "phi* = 1.00; report this model limit."), max(s$median_excess_r)))
} else {
  out <- c(out, "", sprintf("Pre-registered choice: phi* = %.2f", s$phi[which.min(s$abs_diff)]))
}
writeLines(out, file.path(OUT_DIR, "summary.txt"))
cat(out, sep = "\n")
stamp("done")
