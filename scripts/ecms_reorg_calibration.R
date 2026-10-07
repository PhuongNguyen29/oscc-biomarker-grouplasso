# ==============================================================================
# ECMS design S11: calibrate reorg_phi (reorganized absence) to real data
#
# PRE-REGISTERED (7 Oct 2026, before any run of this script):
#   Target: median Zsummary of real NON-preserved module x cohort cells = 2.80.
#     The 12 cells are the modules ECMS rejects on the real data, in all three
#     test cohorts each (K1 preservation_statistics.csv):
#       TCGA black 2.95 / 2.96 / 2.47;  TCGA magenta 2.66 / -0.69 / 0.69;
#       TCGA yellow 8.65 / 0.79 / 1.68; ORIEN turquoise 7.89 / 6.09 / 5.92.
#     Real range -0.69 to 8.65.
#   Grid: reorg_phi in {0.1, 0.2, 0.3, 0.4, 0.5}; 5 replicates each; base
#     parameters otherwise; nPermutations 200 (the chosen setting).
#   Statistic: median Zsummary over ALL absent (reorganized) module x cohort
#     cells, pooled over replicates (12 cells per replicate, as in real data).
#   Rule: phi* = the grid value whose median is closest to 2.80.
#     If every median > 2.80 (even at 0.1), extend ONCE with {0.05, 0.025};
#     if every median < 2.80 (even at 0.5), extend ONCE with {0.6, 0.7, 0.8}.
#     No other change.
#   Only absent-module Z is written. No heuristic or ECMS selection is computed
#   here, so the choice of phi cannot be steered by any method's FDR.
#
# Modes:
#   Rscript scripts/ecms_reorg_calibration.R run <phi> <rep>
#   Rscript scripts/ecms_reorg_calibration.R summarize
# Normally launched by scripts/ecms_reorg_calibration.sh.
# Output: results/ecms_reorg_calibration/
# ==============================================================================

suppressPackageStartupMessages(library(WGCNA))
options(stringsAsFactors = FALSE)

args <- commandArgs(trailingOnly = TRUE)
USAGE <- paste0(
  "Needs command-line arguments; do not source() it.\n",
  "  nohup bash scripts/ecms_reorg_calibration.sh > results/ecms_reorg_calibration_driver.log 2>&1 &\n",
  "  One call: OPENBLAS_NUM_THREADS=1 Rscript scripts/ecms_reorg_calibration.R run 0.3 1\n",
  "  Summary:  Rscript scripts/ecms_reorg_calibration.R summarize")
if (!length(args)) stop(USAGE)
MODE <- args[1]

OUT_DIR   <- "results/ecms_reorg_calibration"
TARGET    <- 2.80
REAL_Z    <- c(2.95, 2.96, 2.47, 2.66, -0.69, 0.69, 8.65, 0.79, 1.68, 7.89, 6.09, 5.92)
NPERM     <- 200L
NTHREADS  <- max(2L, as.integer(Sys.getenv("ECMS_WGCNA_THREADS", "2")))
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
stamp <- function(...) cat(format(Sys.time(), "[%Y-%m-%d %H:%M:%S] "), ..., "\n", sep = "")
stopifnot(abs(median(REAL_Z) - TARGET) < 0.01)   # tripwire: target = median of the 12 cells

# ==============================================================================
if (MODE == "run") {
# ==============================================================================
  if (length(args) < 3) stop("'run' needs <phi> <rep>\n", USAGE)
  PHI <- as.numeric(args[2]); REP <- as.integer(args[3])
  stopifnot(!is.na(PHI), PHI > 0, PHI <= 1, !is.na(REP), REP >= 1)
  if (Sys.getenv("OPENBLAS_NUM_THREADS") != "1") stop("set OPENBLAS_NUM_THREADS=1")
  tag <- sprintf("phi%.3f_rep%02d", PHI, REP)

  enableWGCNAThreads(nThreads = NTHREADS)
  source("scripts/ecms_sim_generator.R")
  stamp("START ", tag)

  par <- ecms_base_params()
  par$reorg_phi <- PHI
  # same data seed for a given replicate across phi (common random numbers up
  # to the first reorganized cohort)
  sim <- ecms_generate(par, seed = 30000L + REP, verbose = FALSE)
  stamp("data generated | reorganized genes: ", sum(sim$reorg$n_reorganized),
        " of ", sum(sim$reorg$size), " absent-module genes")

  t0 <- proc.time()[["elapsed"]]
  mp <- modulePreservation(
    sim$multiExpr, sim$multiColor, referenceNetworks = 1,
    nPermutations = NPERM, randomSeed = 42 + REP, networkType = "signed",
    maxModuleSize = 1000, maxGoldModuleSize = 1000,
    savePermutedStatistics = FALSE, verbose = 1)
  stamp("modulePreservation done in ", round(proc.time()[["elapsed"]] - t0), " s")

  rk <- "ref.disc"
  if (!rk %in% names(mp$preservation$Z))
    stop("reference key '", rk, "' not found; names: ",
         paste(names(mp$preservation$Z), collapse = ", "))
  rows <- list()
  for (tt in colnames(sim$M)) {
    tk  <- paste0("inColumnsAlsoPresentIn.", tt)
    Z   <- mp$preservation$Z[[rk]][[tk]]
    if (!is.data.frame(Z)) stop("no Z table for ", tk)
    absent <- which(sim$M[, tt] == 0)
    z <- Z[as.character(absent), "Zsummary.pres"]
    if (anyNA(z)) stop("absent module missing from Z table in ", tt)
    rg <- sim$reorg[sim$reorg$cohort == tt, ]
    rows[[tt]] <- data.frame(phi = PHI, rep = REP, test = tt, module = absent,
                             size = sim$params$sizes[absent], Zsummary = z,
                             n_reorganized = rg$n_reorganized[match(absent, rg$module)])
  }
  res <- do.call(rbind, rows)
  write.csv(res, file.path(OUT_DIR, sprintf("absentZ_%s.csv", tag)), row.names = FALSE)
  stamp("END ", tag, " | ", nrow(res), " absent cells | median Z ",
        round(median(res$Zsummary), 2))

# ==============================================================================
} else if (MODE == "summarize") {
# ==============================================================================
  f <- list.files(OUT_DIR, "^absentZ_.*\\.csv$", full.names = TRUE)
  if (!length(f)) stop("no result files in ", OUT_DIR)
  z <- do.call(rbind, lapply(f, read.csv))

  s <- do.call(rbind, lapply(split(z, z$phi), function(d) data.frame(
    phi = d$phi[1], reps = length(unique(d$rep)), cells = nrow(d),
    median_Z = median(d$Zsummary), q10 = quantile(d$Zsummary, 0.10),
    q90 = quantile(d$Zsummary, 0.90), min_Z = min(d$Zsummary), max_Z = max(d$Zsummary))))
  s <- s[order(s$phi), ]
  cat("=== Absent (reorganized) module Zsummary by phi ===\n")
  print(s, row.names = FALSE, digits = 3)
  cat(sprintf("\nReal non-preserved cells: median %.2f, range %.2f to %.2f (n = 12)\n",
              median(REAL_Z), min(REAL_Z), max(REAL_Z)))

  if (all(s$median_Z > TARGET)) {
    cat("\nEvery median is ABOVE the target: extend once with phi 0.05 and 0.025.\n")
  } else if (all(s$median_Z < TARGET)) {
    cat("\nEvery median is BELOW the target: extend once with phi 0.6, 0.7 and 0.8.\n")
  } else {
    best <- s$phi[which.min(abs(s$median_Z - TARGET))]
    cat(sprintf("\nPre-registered choice: phi* = %.3f (median closest to %.2f)\n", best, TARGET))
  }

  cat("\nDescriptive: median Z by module size at each phi\n")
  print(tapply(z$Zsummary, list(size = z$size, phi = z$phi), median), digits = 3)
} else {
  stop(USAGE)
}
