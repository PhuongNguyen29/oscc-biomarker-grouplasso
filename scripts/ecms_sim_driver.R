# ==============================================================================
# ECMS simulation v2: driver (pure R). Runs calls 1..N for one scenario and one
# call type, at most 30 at once, skipping calls whose output already exists.
# Safe to restart. Run the null BEFORE scoring a scenario.
#
#   nohup Rscript scripts/ecms_sim_driver.R <scenario> <null|data> [N] > results/ecms_sim_<scenario>_<type>.log 2>&1 &
#
# N defaults to the pre-registered count (ecms_n_null / ecms_n_data).
# Several scenarios: list them comma-separated, e.g.  A,B,E null
# ==============================================================================

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript scripts/ecms_sim_driver.R <scenario[,scenario]> <null|data> [N]")
source("scripts/ecms_sim_generator.R"); source("scripts/ecms_sim_scenarios.R")
SCS <- strsplit(args[1], ",")[[1]]; TYPE <- args[2]
stopifnot(all(SCS %in% names(ecms_scenarios())), TYPE %in% c("null", "data"))
MAX_PARALLEL <- as.integer(Sys.getenv("ECMS_MAX_PARALLEL", "30"))

Sys.setenv(OPENBLAS_NUM_THREADS = "1", OMP_NUM_THREADS = "1", ECMS_WGCNA_THREADS = "2")
stamp <- function(...) { cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S "), ..., "\n", sep = ""); flush.console() }

jobs <- do.call(rbind, lapply(SCS, function(sc) {
  N <- if (length(args) >= 3) as.integer(args[3]) else
       (if (TYPE == "null") ecms_n_null[[sc]] else ecms_n_data[[sc]])
  data.frame(sc = sc, idx = seq_len(N))
}))
out_file <- function(sc, i) file.path("results/ecms_sim", sc, sprintf("%s_%04d.csv", TYPE, i))
done <- file.exists(mapply(out_file, jobs$sc, jobs$idx))
jobs <- jobs[!done, ]
stamp("driver start | ", paste(SCS, collapse = ","), " ", TYPE, " | to run ", nrow(jobs),
      " (", sum(done), " already done) | parallel ", MAX_PARALLEL)

run_one <- function(i) {
  sc <- jobs$sc[i]; idx <- jobs$idx[i]
  logd <- file.path("results/ecms_sim", sc, "logs"); dir.create(logd, recursive = TRUE, showWarnings = FALSE)
  logf <- file.path(logd, sprintf("%s_%04d.log", TYPE, idx))
  rc <- system2("Rscript", c("scripts/ecms_sim_call.R", sc, TYPE, idx), stdout = logf, stderr = logf)
  stamp("finish ", sc, " ", TYPE, " ", idx, " (exit ", rc, ")", if (rc != 0) paste0("  see ", logf) else "")
  rc
}
if (nrow(jobs)) {
  rc <- unlist(parallel::mclapply(seq_len(nrow(jobs)), run_one,
                                  mc.cores = min(MAX_PARALLEL, nrow(jobs)), mc.preschedule = FALSE))
  stamp("driver end | failed: ", sum(rc != 0), " of ", length(rc))
} else stamp("nothing to run")
