# ==============================================================================
# ECMS simulation v2: ONE modulePreservation call
#
#   data call: one replicate of the scenario, planted module labels.
#              Writes Zsummary, WGCNA log10 summary p, medianRank per module x
#              test cohort, plus gold and grey (for the ceiling rule) and truth.
#   null call: an independent replicate of the same scenario, labels replaced by
#              a random partition with the planted sizes (same algorithm as
#              ecms_null_run.R). Writes Zsummary of the random modules. All null
#              calls of a scenario form its ONE reused null.
#
# Usage (project root, terminal; normally via ecms_sim_driver.R):
#   OPENBLAS_NUM_THREADS=1 Rscript scripts/ecms_sim_call.R <scenario> <data|null> <index>
# Output: results/ecms_sim/<scenario>/<type>_<index>.csv
# ==============================================================================

suppressPackageStartupMessages(library(WGCNA))
options(stringsAsFactors = FALSE)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("Needs <scenario> <data|null> <index>; do not source() it.")
SC <- args[1]; TYPE <- args[2]; IDX <- as.integer(args[3])
if (Sys.getenv("OPENBLAS_NUM_THREADS") != "1") stop("set OPENBLAS_NUM_THREADS=1")

source("scripts/ecms_sim_generator.R")
source("scripts/ecms_sim_scenarios.R")
scs <- ecms_scenarios()
if (!SC %in% names(scs)) stop("unknown scenario ", SC, "; choose from ", paste(names(scs), collapse = ", "))
stopifnot(TYPE %in% c("data", "null"), !is.na(IDX), IDX >= 1)

enableWGCNAThreads(nThreads = max(2L, as.integer(Sys.getenv("ECMS_WGCNA_THREADS", "2"))))
stamp <- function(...) cat(format(Sys.time(), "[%Y-%m-%d %H:%M:%S] "), ..., "\n", sep = "")
OUT <- file.path("results/ecms_sim", SC); dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
tag <- sprintf("%s_%04d", TYPE, IDX)
stamp("START ", SC, " ", tag)
t0 <- proc.time()[["elapsed"]]

par  <- scs[[SC]]
seed <- ecms_seed(SC, TYPE, IDX)
sim  <- ecms_generate(par, seed = seed, verbose = FALSE)
colors <- if (TYPE == "data") sim$multiColor else
  ecms_null_labels(sim, call = IDX, seed_base = seed * 10L)
stamp("generated (seed ", seed, ") | ", length(setdiff(unique(colors$disc), 0)), " modules labelled")

mp <- modulePreservation(
  sim$multiExpr, colors, referenceNetworks = 1,
  nPermutations = ECMS_NPERM, randomSeed = seed %% 100000L,
  networkType = "signed", maxModuleSize = 1000, maxGoldModuleSize = 1000,
  savePermutedStatistics = FALSE, verbose = 1)
stamp("modulePreservation done")

rk <- "ref.disc"
if (!rk %in% names(mp$preservation$Z))
  stop("reference key '", rk, "' not found; names: ", paste(names(mp$preservation$Z), collapse = ", "))
rows <- list()
for (tt in colnames(sim$M)) {
  tk <- paste0("inColumnsAlsoPresentIn.", tt)
  Zt <- mp$preservation$Z[[rk]][[tk]]
  ob <- mp$preservation$observed[[rk]][[tk]]
  lp <- mp$preservation$log.p[[rk]][[tk]]
  if (!is.data.frame(Zt) || !is.data.frame(ob) || !is.data.frame(lp)) stop("missing tables for ", tk)
  if (!identical(rownames(Zt), rownames(ob)) || !identical(rownames(Zt), rownames(lp)))
    stop("row order differs between Z, observed and log.p tables for ", tk)
  rows[[tt]] <- data.frame(
    scenario = SC, type = TYPE, index = IDX, test = tt, module = rownames(Zt),
    moduleSize = ob$moduleSize, Zsummary = Zt$Zsummary.pres,
    log10_psummary = lp$log.psummary.pres,
    medianRank = if ("medianRank.pres" %in% names(ob)) ob$medianRank.pres else NA_real_)
}
res <- do.call(rbind, rows)
res$set <- ifelse(res$module == "0", "grey", ifelse(res$module == "0.1", "gold", "module"))

if (TYPE == "data") {
  pl <- res$set == "module"
  k  <- as.integer(res$module[pl])
  res$multiplier <- NA_real_
  res$multiplier[pl] <- sim$M[cbind(k, match(res$test[pl], colnames(sim$M)))]
  if (sum(pl) != 3 * length(par$sizes)) stop("expected ", 3 * length(par$sizes), " planted rows, got ", sum(pl))
} else {
  res <- res[res$set == "module", ]                 # null keeps random modules only
  res$multiplier <- NA_real_
}
write.csv(res, file.path(OUT, paste0(tag, ".csv")), row.names = FALSE)
stamp(sprintf("END %s %s | %d rows | %.1f min", SC, tag, nrow(res),
              (proc.time()[["elapsed"]] - t0) / 60))
