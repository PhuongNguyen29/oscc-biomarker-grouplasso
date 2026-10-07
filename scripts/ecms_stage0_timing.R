# ==============================================================================
# ECMS Stage 0: time ONE simulated modulePreservation call
#
# Same data in every run (base replicate, DATA_SEED fixed) so timing differences
# come from nPermutations and server load only, not from the data.
# Base = ecms_base_params(): p 4,702, 9 real TCGA sizes, n 300 x 4 cohorts,
# lambda 0.98, gamma 0.154. modulePreservation settings as in K1 / the null:
# signed, maxModuleSize 1000, maxGoldModuleSize 1000.
#
# Modes:
#   Rscript scripts/ecms_stage0_timing.R run <nperm> <rep> <alone|concurrent>
#   Rscript scripts/ecms_stage0_timing.R summarize
# Normally launched by scripts/ecms_stage0_timing.sh (sets OPENBLAS_NUM_THREADS=1,
# runs phase A alone and phase B at concurrency 30, then summarizes).
#
# Outputs (results/ecms_stage0_timing/):
#   timing_<tag>.csv  one row per call, written as soon as the call ends
#   z_<tag>.csv       Zsummary per planted module x test cohort (+ truth)
#   summary.txt       written by the driver
#
# PRE-REGISTERED DECISION RULE (fixed 6 Oct, before any timing):
#   budget 14 days x 30 slots; 12 scenario cells x 200 replicates; about 100
#   null calls per null; x2 safety factor. t = median total seconds per call at
#   concurrency 30, nPermutations 50.
#     t <= 75 s      per-replicate null feasible (method simulated as deployed)
#     t <= 5,040 s   reuse one null per scenario; reuse sub-study is critical path
#     t >  5,040 s   neither fits by 26 Oct: fallback p 2,000 / fewer replicates /
#                    Option 3 (fast statistic). Decide before 12 Oct.
# ==============================================================================

suppressPackageStartupMessages(library(WGCNA))
options(stringsAsFactors = FALSE)

args      <- commandArgs(trailingOnly = TRUE)
MODE      <- if (length(args)) args[1] else "run"
OUT_DIR   <- "results/ecms_stage0_timing"
DATA_SEED <- 20261007L
NTHREADS  <- max(2L, as.integer(Sys.getenv("ECMS_WGCNA_THREADS", "2")))  # WGCNA needs >= 2
VERBOSE   <- as.integer(Sys.getenv("ECMS_MP_VERBOSE", "1"))
T_PER_REP <- 75       # seconds, rule above
T_REUSE   <- 5040     # seconds (1.4 h), rule above
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

stamp <- function(...) cat(format(Sys.time(), "[%Y-%m-%d %H:%M:%S] "), ..., "\n", sep = "")

# ==============================================================================
if (MODE == "run") {
# ==============================================================================
  NPERM <- as.integer(args[2]); REP <- as.integer(args[3]); PHASE <- args[4]
  stopifnot(!is.na(NPERM), NPERM >= 1, !is.na(REP), REP >= 1,
            PHASE %in% c("alone", "concurrent"))
  if (Sys.getenv("OPENBLAS_NUM_THREADS") != "1")
    stop("set OPENBLAS_NUM_THREADS=1 before running (use the .sh driver)")
  tag <- sprintf("%s_nperm%03d_rep%02d", PHASE, NPERM, REP)

  enableWGCNAThreads(nThreads = NTHREADS)
  source("scripts/ecms_sim_generator.R")
  stamp("START ", tag, " | WGCNA threads ", NTHREADS, " | pid ", Sys.getpid())

  t0  <- proc.time()[["elapsed"]]
  par <- ecms_base_params()
  sim <- ecms_generate(par, seed = DATA_SEED, verbose = FALSE)
  t1  <- proc.time()[["elapsed"]]
  stamp("data generated in ", round(t1 - t0, 1), " s | p ", par$p, " | K ",
        length(par$sizes), " | starting modulePreservation (", NPERM, " permutations)")

  invisible(gc(reset = TRUE))
  mp <- modulePreservation(
    sim$multiExpr, sim$multiColor,
    referenceNetworks = 1,
    nPermutations = NPERM, randomSeed = 42 + REP,
    networkType = "signed",
    maxModuleSize = 1000, maxGoldModuleSize = 1000,
    savePermutedStatistics = FALSE,
    verbose = VERBOSE)
  t2 <- proc.time()[["elapsed"]]
  g  <- gc()
  r_max_mb <- sum(g[, ncol(g)])          # last column of gc(): "max used" (Mb)
  stamp("modulePreservation done in ", round(t2 - t1, 1), " s")

  # --- Extract Zsummary; fail loudly on a name mismatch (K1 lesson) ----------
  rk <- "ref.disc"
  if (!rk %in% names(mp$preservation$Z))
    stop("reference key '", rk, "' not found; names: ",
         paste(names(mp$preservation$Z), collapse = ", "))
  tests <- setdiff(names(sim$multiExpr), "disc")
  ztab <- do.call(rbind, lapply(tests, function(tt) {
    tk  <- paste0("inColumnsAlsoPresentIn.", tt)
    Z   <- mp$preservation$Z[[rk]][[tk]]
    obs <- mp$preservation$observed[[rk]][[tk]]
    if (!is.data.frame(Z)) stop("no Z table for ", tk, "; names: ",
                                paste(names(mp$preservation$Z[[rk]]), collapse = ", "))
    data.frame(test = tt, module = rownames(Z), moduleSize = obs$moduleSize,
               Zsummary = Z$Zsummary.pres)
  }))
  ztab$multiplier <- NA_real_
  planted <- !ztab$module %in% c("0", "0.1")
  ztab$multiplier[planted] <- sim$M[cbind(as.integer(ztab$module[planted]),
                                          match(ztab$test[planted], colnames(sim$M)))]
  ztab$set <- ifelse(ztab$module == "0", "grey",
              ifelse(ztab$module == "0.1", "gold",
              ifelse(ztab$multiplier > 0, "present", "absent")))
  write.csv(ztab, file.path(OUT_DIR, sprintf("z_%s.csv", tag)), row.names = FALSE)

  row <- data.frame(tag = tag, phase = PHASE, nperm = NPERM, rep = REP,
                    gen_sec = round(t1 - t0, 2), mp_sec = round(t2 - t1, 2),
                    total_sec = round(t2 - t0, 2), r_max_mb = round(r_max_mb, 1),
                    wgcna_threads = NTHREADS, p = par$p, K = length(par$sizes),
                    n = par$n[1], host = Sys.info()[["nodename"]],
                    finished = format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
  write.csv(row, file.path(OUT_DIR, sprintf("timing_%s.csv", tag)), row.names = FALSE)
  stamp("END ", tag, " | total ", round(t2 - t0, 1), " s | R max used ",
        round(r_max_mb), " Mb")

# ==============================================================================
} else if (MODE == "summarize") {
# ==============================================================================
  f <- list.files(OUT_DIR, "^timing_.*\\.csv$", full.names = TRUE)
  if (!length(f)) stop("no timing files in ", OUT_DIR)
  tm <- do.call(rbind, lapply(f, read.csv))

  # Peak resident memory from /usr/bin/time -v, if the driver recorded it
  tm$peak_rss_mb <- vapply(tm$tag, function(tg) {
    tf <- file.path(OUT_DIR, "logs", paste0(tg, ".time"))
    if (!file.exists(tf)) return(NA_real_)
    l <- grep("Maximum resident set size", readLines(tf), value = TRUE)
    if (!length(l)) NA_real_ else as.numeric(sub(".*:\\s*", "", l)) / 1024
  }, numeric(1))

  cat("=== Stage 0 timing ===\n")
  print(tm[order(tm$phase, tm$nperm, tm$rep),
           c("phase", "nperm", "rep", "gen_sec", "mp_sec", "total_sec", "peak_rss_mb")],
        row.names = FALSE)

  alone <- tm[tm$phase == "alone", ]
  if (length(unique(alone$nperm)) >= 2) {
    fit <- lm(mp_sec ~ nperm, data = alone)
    cat(sprintf("\nAlone: mp_sec = %.1f + %.2f x nPermutations\n",
                coef(fit)[1], coef(fit)[2]))
  }

  conc <- tm[tm$phase == "concurrent" & tm$nperm == 50, ]
  if (nrow(conc)) {
    t <- median(conc$total_sec)
    cat(sprintf("\nConcurrent (%d calls, nPerm 50): median %.1f s, max %.1f s\n",
                nrow(conc), t, max(conc$total_sec)))
    decision <- if (t <= T_PER_REP) {
      "PER-REPLICATE NULL is feasible"
    } else if (t <= T_REUSE) {
      "REUSE one null per scenario; the reuse sub-study is on the critical path"
    } else {
      "NEITHER fits by 26 Oct: fallback (p 2,000 / fewer replicates / Option 3)"
    }
    cat("Pre-registered decision:", decision, "\n")
  } else {
    cat("\nNo concurrent runs yet: decision not made.\n")
  }

  # Descriptive sanity check of the pipeline (no pass rule): present modules
  # should show higher Zsummary than absent ones of similar size.
  zf <- file.path(OUT_DIR, "z_alone_nperm050_rep01.csv")
  if (file.exists(zf)) {
    z <- read.csv(zf, colClasses = c(module = "character"))
    cat("\nZsummary, alone nPerm 50 (descriptive):\n")
    print(z[order(z$test, -z$moduleSize), c("test", "module", "moduleSize", "set", "Zsummary")],
          row.names = FALSE)
  }
} else {
  stop("MODE must be 'run' or 'summarize'")
}
