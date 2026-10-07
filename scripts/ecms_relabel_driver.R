# ==============================================================================
# ECMS relabel check driver (pure R). From the project root, in a terminal:
#
#   nohup Rscript scripts/ecms_relabel_driver.R > results/ecms_relabel_driver.log 2>&1 &
#   tail -f results/ecms_relabel_driver.log
#
# Pre-registered size: 30 shuffle draws per reference (TCGA, ORIEN) = 60 calls,
# at most 30 running at once. Draws already finished are skipped, so the driver
# can be restarted safely. Ends by running ecms_relabel_aggregate.R.
# Optional: a smaller first batch, e.g. draws 1 to 2 only:
#   nohup Rscript scripts/ecms_relabel_driver.R 2 > results/ecms_relabel_driver_test.log 2>&1 &
# ==============================================================================

args  <- commandArgs(trailingOnly = TRUE)
NDRAW <- if (length(args)) as.integer(args[1]) else 30L
stopifnot(!is.na(NDRAW), NDRAW >= 1)
MAX_PARALLEL <- 30

Sys.setenv(OPENBLAS_NUM_THREADS = "1", OMP_NUM_THREADS = "1", ECMS_WGCNA_THREADS = "2")
OUT <- "results/ecms_relabel"; LOG <- file.path(OUT, "logs")
dir.create(LOG, recursive = TRUE, showWarnings = FALSE)
stopifnot(file.exists("scripts/ecms_relabel_run.R"),
          file.exists("scripts/ecms_relabel_aggregate.R"))

stamp <- function(...) { cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S "), ..., "\n", sep = ""); flush.console() }

jobs <- expand.grid(draw = seq_len(NDRAW), ref = c("TCGA", "ORIEN"), stringsAsFactors = FALSE)
done <- file.exists(file.path(OUT, sprintf("relabel_%s_draw%02d.csv", jobs$ref, jobs$draw)))
jobs <- jobs[!done, ]
stamp("driver start | ", nrow(jobs), " calls to run (", sum(done), " already done) | parallel ",
      min(MAX_PARALLEL, max(1, nrow(jobs))))

run_one <- function(i) {
  tag  <- sprintf("%s_draw%02d", jobs$ref[i], jobs$draw[i])
  logf <- file.path(LOG, paste0(tag, ".log"))
  stamp("start  ", tag)
  rc <- system2("Rscript", c("scripts/ecms_relabel_run.R", jobs$draw[i], jobs$ref[i]),
                stdout = logf, stderr = logf)
  stamp("finish ", tag, " (exit ", rc, ")", if (rc != 0) paste0("  see ", logf) else "")
  rc
}

if (nrow(jobs)) {
  rc <- unlist(parallel::mclapply(seq_len(nrow(jobs)), run_one,
                                  mc.cores = min(MAX_PARALLEL, nrow(jobs)),
                                  mc.preschedule = FALSE))
  stamp("calls done | failed: ", sum(rc != 0), " of ", length(rc))
}

stamp("aggregating")
out <- system2("Rscript", "scripts/ecms_relabel_aggregate.R", stdout = TRUE, stderr = TRUE)
writeLines(out, file.path(OUT, "summary.txt"))
cat(out, sep = "\n")
stamp("driver end | summary: ", file.path(OUT, "summary.txt"))
