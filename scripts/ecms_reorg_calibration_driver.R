# ==============================================================================
# ECMS S11 calibration driver, pure R version (same job as the .sh driver).
# Run from the project root, in a terminal (not the RStudio console):
#
#   nohup Rscript scripts/ecms_reorg_calibration_driver.R > results/ecms_reorg_calibration_driver.log 2>&1 &
#   tail -f results/ecms_reorg_calibration_driver.log
#
# Pre-registered extension only, if the summary asks for it:
#   nohup Rscript scripts/ecms_reorg_calibration_driver.R 0.05 0.025 > results/ecms_reorg_calibration_driver_ext.log 2>&1 &
#
# Launches each (phi, replicate) as its own Rscript process, in parallel, then
# writes results/ecms_reorg_calibration/summary.txt.
# ==============================================================================

args <- commandArgs(trailingOnly = TRUE)
PHIS <- if (length(args)) as.numeric(args) else c(0.1, 0.2, 0.3, 0.4, 0.5)
stopifnot(!anyNA(PHIS), all(PHIS > 0), all(PHIS <= 1))
REPS <- 1:5
MAX_PARALLEL <- 30

# Children inherit these (the worker refuses to run without OPENBLAS = 1)
Sys.setenv(OPENBLAS_NUM_THREADS = "1", OMP_NUM_THREADS = "1", ECMS_WGCNA_THREADS = "2")

OUT <- "results/ecms_reorg_calibration"
LOG <- file.path(OUT, "logs")
dir.create(LOG, recursive = TRUE, showWarnings = FALSE)
stopifnot(file.exists("scripts/ecms_reorg_calibration.R"),
          file.exists("scripts/ecms_sim_generator.R"))

stamp <- function(...) {
  cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S "), ..., "\n", sep = "")
  flush.console()
}

jobs <- expand.grid(phi = PHIS, rep = REPS)
stamp("driver start | ", nrow(jobs), " calls | phi = ", paste(PHIS, collapse = ", "),
      " | parallel ", min(MAX_PARALLEL, nrow(jobs)))

run_one <- function(i) {
  phi <- jobs$phi[i]; rep <- jobs$rep[i]
  tag <- sprintf("phi%.3f_rep%02d", phi, rep)
  logf <- file.path(LOG, paste0(tag, ".log"))
  stamp("start  ", tag)
  rc <- system2("Rscript",
                c("scripts/ecms_reorg_calibration.R", "run", format(phi), rep),
                stdout = logf, stderr = logf)
  stamp("finish ", tag, " (exit ", rc, ")", if (rc != 0) paste0("  see ", logf) else "")
  rc
}

rc <- unlist(parallel::mclapply(seq_len(nrow(jobs)), run_one,
                                mc.cores = min(MAX_PARALLEL, nrow(jobs)),
                                mc.preschedule = FALSE))
stamp("all calls done | failed: ", sum(rc != 0), " of ", length(rc))

stamp("summarizing")
summ <- system2("Rscript", c("scripts/ecms_reorg_calibration.R", "summarize"),
                stdout = TRUE, stderr = TRUE)
writeLines(summ, file.path(OUT, "summary.txt"))
cat(summ, sep = "\n")
stamp("driver end | summary: ", file.path(OUT, "summary.txt"))
