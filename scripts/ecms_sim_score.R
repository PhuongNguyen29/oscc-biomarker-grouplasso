# ==============================================================================
# ECMS simulation v2: SCORE one or more scenarios (design Part I, V2.2 to V2.6)
#
# Null (reused, one per scenario): location/scale of null Zsummary per
# (test cohort, module size); standardized residual r = (Z - mu) / sd;
# p_emp = (1 + #null residuals >= r) / (N + 1), pooled within test cohort;
# p_par = 1 - Phi(r). Identical to ecms_null_aggregate.R.
#
# Methods (all at r = 2 of 3, primary, and r = 3 of 3):
#   ECMS_BH     p_emp -> Bonferroni partial conjunction -> BH       (primary)
#   ECMS_BY     p_emp -> PC -> BY                                   (supplement)
#   ECMS_par    p_par -> PC -> BH                                   (supplement)
#   WGCNA_sumP  p = 10^log.psummary -> PC -> BH  (mean of log10 density and
#               connectivity p, each a median of components; verified 7 Oct)
#   Z10         Zsummary > 10 in >= r cohorts
#   Z2          Zsummary > 2  in >= r cohorts                       (supplement)
#   Ceiling     Zsummary > max(gold Z, grey Z) in >= r cohorts (K1 rule)
# Truth: module preserved in a cohort iff multiplier > 0; true for r iff
# preserved in >= r cohorts (design V2.1).
# Metrics: FDR = mean FDP, MC SE = sd(FDP)/sqrt(B); FWER; power; mean number
# selected; calibration of absent-cell p-values.
# GO check (V2.6): ECMS_BH, r = 2, alpha 0.05: FDR <= 0.05 + 2 MC SE in A, B, E.
#
# Usage: Rscript scripts/ecms_sim_score.R <scenario[,scenario]>   (e.g. A,B,E)
# Output: results/ecms_sim/<scenario>/score/{metrics.csv, calibration.csv, pvalues.csv}
# ==============================================================================

options(stringsAsFactors = FALSE)
args <- commandArgs(trailingOnly = TRUE)
if (!length(args)) stop("Usage: Rscript scripts/ecms_sim_score.R <scenario[,scenario]>")
SCS <- strsplit(args[1], ",")[[1]]
source("scripts/ecms_sim_generator.R"); source("scripts/ecms_sim_scenarios.R")
stopifnot(all(SCS %in% names(ecms_scenarios())))

pc_bonf <- function(p, r) { p <- sort(p); min(1, (length(p) - r + 1) * p[r]) }
read_all <- function(dir, pattern) {
  f <- list.files(dir, pattern = pattern, full.names = TRUE)
  if (!length(f)) return(NULL)
  do.call(rbind, lapply(f, read.csv, colClasses = c(module = "character")))
}

score_scenario <- function(SC) {
  dir <- file.path("results/ecms_sim", SC)
  nl <- read_all(dir, "^null_\\d+\\.csv$"); dt <- read_all(dir, "^data_\\d+\\.csv$")
  if (is.null(nl)) stop(SC, ": no null calls yet")
  if (is.null(dt)) stop(SC, ": no data calls yet")
  out_dir <- file.path(dir, "score"); dir.create(out_dir, showWarnings = FALSE)

  # --- reused null ------------------------------------------------------------------
  loc <- aggregate(Zsummary ~ test + moduleSize, data = nl,
                   FUN = function(z) c(mu = mean(z), sd = sd(z), n = length(z)))
  loc <- data.frame(test = loc$test, moduleSize = loc$moduleSize,
                    mu = loc$Zsummary[, "mu"], sd = loc$Zsummary[, "sd"], n_null = loc$Zsummary[, "n"])
  nl <- merge(nl, loc, by = c("test", "moduleSize"))
  nl$r <- (nl$Zsummary - nl$mu) / nl$sd
  null_sorted <- lapply(split(nl$r, nl$test), sort)

  # --- data: planted modules ---------------------------------------------------------
  pm <- merge(dt[dt$set == "module", ], loc, by = c("test", "moduleSize"), all.x = TRUE)
  if (anyNA(pm$mu)) stop(SC, ": module sizes missing from the null: ",
                         paste(unique(pm$moduleSize[is.na(pm$mu)]), collapse = ", "))
  pm$r <- (pm$Zsummary - pm$mu) / pm$sd
  pm$p_emp <- NA_real_
  for (tt in names(null_sorted)) {
    i <- pm$test == tt; v <- null_sorted[[tt]]; N <- length(v)
    pm$p_emp[i] <- (1 + N - findInterval(pm$r[i], v, left.open = TRUE)) / (N + 1)
  }
  pm$p_par   <- pnorm(pm$r, lower.tail = FALSE)
  pm$p_wgcna <- pmin(1, 10^pm$log10_psummary)
  pm$present <- pm$multiplier > 0

  # ceiling per (replicate, test) = max(gold Z, grey Z), the K1 rule
  ctl  <- dt[dt$set %in% c("gold", "grey"), ]
  ceil <- aggregate(Zsummary ~ index + test, data = ctl, FUN = max)
  names(ceil)[3] <- "ceiling"
  pm <- merge(pm, ceil, by = c("index", "test"), all.x = TRUE)
  write.csv(pm, file.path(out_dir, "pvalues.csv"), row.names = FALSE)

  # --- per replicate x module -----------------------------------------------------------
  by_mod <- split(pm, list(pm$index, pm$module), drop = TRUE)
  mods <- do.call(rbind, lapply(by_mod, function(d) {
    if (nrow(d) != 3) stop(SC, ": replicate ", d$index[1], " module ", d$module[1], " has ", nrow(d), " cohorts")
    data.frame(index = d$index[1], module = d$module[1], n_pres = sum(d$present),
               emp2 = pc_bonf(d$p_emp, 2), emp3 = pc_bonf(d$p_emp, 3),
               par2 = pc_bonf(d$p_par, 2), par3 = pc_bonf(d$p_par, 3),
               wg2 = pc_bonf(d$p_wgcna, 2), wg3 = pc_bonf(d$p_wgcna, 3),
               nZ10 = sum(d$Zsummary > 10), nZ2 = sum(d$Zsummary > 2),
               nCeil = sum(d$Zsummary > d$ceiling))
  }))

  metrics <- list()
  for (r in 2:3) {
    truth <- mods$n_pres >= r
    sel <- list()
    for (a in ECMS_ALPHAS) {
      adj <- function(col, method) ave(mods[[col]], mods$index, FUN = function(p) p.adjust(p, method))
      sel[[sprintf("ECMS_BH|%.2f", a)]]    <- adj(paste0("emp", r), "BH") <= a
      sel[[sprintf("ECMS_BY|%.2f", a)]]    <- adj(paste0("emp", r), "BY") <= a
      sel[[sprintf("ECMS_par|%.2f", a)]]   <- adj(paste0("par", r), "BH") <= a
      sel[[sprintf("WGCNA_sumP|%.2f", a)]] <- adj(paste0("wg", r),  "BH") <= a
    }
    sel[["Z10|NA"]]     <- mods$nZ10  >= r
    sel[["Z2|NA"]]      <- mods$nZ2   >= r
    sel[["Ceiling|NA"]] <- mods$nCeil >= r

    for (nm in names(sel)) {
      s <- sel[[nm]]
      per <- do.call(rbind, lapply(split(data.frame(s = s, t = truth), mods$index), function(d)
        c(R = sum(d$s), V = sum(d$s & !d$t), TP = sum(d$s & d$t), T = sum(d$t))))
      fdp <- per[, "V"] / pmax(per[, "R"], 1)
      pow <- ifelse(per[, "T"] > 0, per[, "TP"] / per[, "T"], NA)
      parts <- strsplit(nm, "|", fixed = TRUE)[[1]]
      metrics[[length(metrics) + 1]] <- data.frame(
        scenario = SC, r = r, method = parts[1],
        alpha = suppressWarnings(as.numeric(parts[2])), B = nrow(per),
        FDR = mean(fdp), FDR_mcse = sd(fdp) / sqrt(nrow(per)),
        FWER = mean(per[, "V"] > 0),
        power = if (all(is.na(pow))) NA_real_ else mean(pow, na.rm = TRUE),
        mean_selected = mean(per[, "R"]))
    }
  }
  metrics <- do.call(rbind, metrics)
  write.csv(metrics, file.path(out_dir, "metrics.csv"), row.names = FALSE)

  ab <- pm[!pm$present, ]
  cal <- data.frame(scenario = SC, n_absent_cells = nrow(ab),
                    emp_lt05 = mean(ab$p_emp < 0.05), emp_lt01 = mean(ab$p_emp < 0.01),
                    par_lt05 = mean(ab$p_par < 0.05), wgcna_lt05 = mean(ab$p_wgcna < 0.05),
                    Z10_absent = mean(ab$Zsummary > 10))
  write.csv(cal, file.path(out_dir, "calibration.csv"), row.names = FALSE)

  cat(sprintf("\n================ Scenario %s ================\n", SC))
  cat(sprintf("Null: %d calls | samples per (test, size): min %d, median %d%s\n",
              length(unique(nl$index)), min(loc$n_null), as.integer(median(loc$n_null)),
              if (min(loc$n_null) < 20) "  (WARNING: < 20, run more null calls)" else ""))
  cat(sprintf("Data: %d replicates\n", length(unique(pm$index))))
  cat("Calibration of ABSENT module x cohort cells (fraction p < 0.05):\n")
  print(cal[, -1], row.names = FALSE, digits = 3)
  cat("\nMetrics at alpha 0.05 (p-based) and heuristics:\n")
  show <- metrics[is.na(metrics$alpha) | metrics$alpha == 0.05, ]
  print(show[order(show$r, show$method), -1], row.names = FALSE, digits = 3)

  go <- metrics[metrics$method == "ECMS_BH" & metrics$r == 2 & metrics$alpha == 0.05, ]
  ok <- go$FDR <= 0.05 + 2 * go$FDR_mcse
  if (SC %in% c("A", "B", "E"))
    cat(sprintf("\nGO criterion (V2.6) for %s: ECMS_BH r=2 FDR %.4f, limit %.4f -> %s%s\n",
                SC, go$FDR, 0.05 + 2 * go$FDR_mcse, if (ok) "MET" else "NOT MET",
                if (go$B < ecms_n_data[[SC]]) sprintf("  (PROVISIONAL: %d of %d replicates)", go$B, ecms_n_data[[SC]]) else ""))
  invisible(metrics)
}

for (SC in SCS) score_scenario(SC)
