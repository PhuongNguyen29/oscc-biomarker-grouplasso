# ==============================================================================
# ECMS reuse sub-study (design Part II §7 option 2; V2.3 "one reused null")
#
# Question: does scoring with ONE reused null per scenario give the same answer
# as scoring each replicate against its OWN null (built on its own data, as the
# method does on real data)?
#
# Inputs (scenario B):
#   data_0001..data_0005.csv           the 5 data replicates
#   null_*.csv                         the reused null (100 calls)
#   ownnull_<rep>_<j>.csv, j = 1..30   replicate-specific nulls
#
# PRE-REGISTERED (7 Oct 2026, before any ownnull call ran):
#   (1) ECMS_BH decisions (r = 2, alpha 0.05) agree between own and reused null
#       for >= 95% of the 45 module decisions (5 replicates x 9 modules);
#   (2) null location: for each (replicate, test, size), z = difference in null
#       mean / its standard error; the fraction with |z| > 2 is <= 0.10
#       (about 0.05 expected if the null does not depend on the replicate).
#   Both hold -> reuse supported (stated as a 5-replicate check, not a proof).
#   Either fails -> reported; the paper states that the shortcut changes results.
#   Reported, not gated: per-cohort p_emp agreement (Spearman) and max |diff|.
#
# Usage: Rscript scripts/ecms_reuse_check.R
# Output: results/ecms_sim/B/score/reuse_check.txt
# ==============================================================================

options(stringsAsFactors = FALSE)
SC <- "B"; REPS <- 1:5; NJ <- 30; ALPHA <- 0.05
dir <- file.path("results/ecms_sim", SC)
pc_bonf <- function(p, r) { p <- sort(p); min(1, (length(p) - r + 1) * p[r]) }
rd <- function(f) read.csv(f, colClasses = c(module = "character"))

build_null <- function(d) {
  loc <- aggregate(Zsummary ~ test + moduleSize, data = d,
                   FUN = function(z) c(mu = mean(z), sd = sd(z), n = length(z)))
  loc <- data.frame(test = loc$test, moduleSize = loc$moduleSize, mu = loc$Zsummary[, "mu"],
                    sd = loc$Zsummary[, "sd"], n = loc$Zsummary[, "n"])
  d <- merge(d, loc, by = c("test", "moduleSize")); d$r <- (d$Zsummary - d$mu) / d$sd
  list(loc = loc, sorted = lapply(split(d$r, d$test), sort))
}
score <- function(pm, nul) {
  pm <- merge(pm, nul$loc[, c("test", "moduleSize", "mu", "sd")], by = c("test", "moduleSize"))
  pm$r <- (pm$Zsummary - pm$mu) / pm$sd
  pm$p <- NA_real_
  for (tt in names(nul$sorted)) {
    i <- pm$test == tt; v <- nul$sorted[[tt]]; N <- length(v)
    pm$p[i] <- (1 + N - findInterval(pm$r[i], v, left.open = TRUE)) / (N + 1)
  }
  pm
}
decide <- function(pm) {
  m <- do.call(rbind, lapply(split(pm, pm$module), function(d)
    data.frame(module = d$module[1], pc2 = pc_bonf(d$p, 2))))
  m$sel <- p.adjust(m$pc2, "BH") <= ALPHA
  m
}

nf <- list.files(dir, "^null_\\d+\\.csv$", full.names = TRUE)
if (length(nf) < 20) stop("reused null has only ", length(nf), " calls; run the B null first")
reused <- build_null(do.call(rbind, lapply(nf, function(f) rd(f)[, c("test", "moduleSize", "Zsummary")])))

dec <- list(); locz <- list(); pagree <- list()
for (rep in REPS) {
  df <- file.path(dir, sprintf("data_%04d.csv", rep))
  of <- list.files(dir, sprintf("^ownnull_%04d_\\d{3}\\.csv$", rep), full.names = TRUE)
  if (!file.exists(df)) stop("missing ", df)
  if (length(of) < NJ) stop("replicate ", rep, ": only ", length(of), " of ", NJ, " ownnull calls")
  pm  <- rd(df); pm <- pm[pm$set == "module", ]
  own <- build_null(do.call(rbind, lapply(of, function(f) rd(f)[, c("test", "moduleSize", "Zsummary")])))
  a <- score(pm, own); b <- score(pm, reused)
  a <- a[order(a$test, a$module), ]; b <- b[order(b$test, b$module), ]
  da <- decide(a); db <- decide(b)
  dec[[rep]] <- data.frame(rep = rep, module = da$module, own = da$sel, reused = db$sel,
                           truth_r2 = tapply(pm$multiplier > 0, pm$module, sum)[da$module] >= 2)
  pagree[[rep]] <- data.frame(rep = rep, p_own = a$p, p_reused = b$p)
  L <- merge(own$loc, reused$loc, by = c("test", "moduleSize"), suffixes = c("_own", "_re"))
  L$z <- (L$mu_own - L$mu_re) / sqrt(L$sd_own^2 / L$n_own + L$sd_re^2 / L$n_re)
  locz[[rep]] <- cbind(rep = rep, L)
}
dec <- do.call(rbind, dec); locz <- do.call(rbind, locz); pagree <- do.call(rbind, pagree)

agree <- mean(dec$own == dec$reused); fz <- mean(abs(locz$z) > 2)
ok1 <- agree >= 0.95; ok2 <- fz <= 0.10
out <- c(
  "=== Reuse sub-study (scenario B, 5 replicates, 30 own-null calls each) ===",
  sprintf("(1) ECMS_BH decisions agree: %d of %d = %.3f (gate >= 0.95: %s)",
          sum(dec$own == dec$reused), nrow(dec), agree, if (ok1) "ok" else "FAIL"),
  sprintf("    selections own %d | reused %d | truly preserved (r=2) %d",
          sum(dec$own), sum(dec$reused), sum(dec$truth_r2)),
  sprintf("(2) null mean differences |z| > 2: %d of %d = %.3f (gate <= 0.10: %s)",
          sum(abs(locz$z) > 2), nrow(locz), fz, if (ok2) "ok" else "FAIL"),
  sprintf("Reported: p_emp Spearman own vs reused = %.3f | max |diff| = %.3f",
          cor(pagree$p_own, pagree$p_reused, method = "spearman"),
          max(abs(pagree$p_own - pagree$p_reused))),
  sprintf("own-null samples per (test, size): min %d, median %d",
          min(locz$n_own), as.integer(median(locz$n_own))),
  sprintf("REUSE CHECK: %s", if (ok1 && ok2) "PASS (reuse supported; 5-replicate check)" else "FAIL (report: shortcut changes results)"),
  "", "Disagreements (if any):",
  capture.output(print(dec[dec$own != dec$reused, ], row.names = FALSE)))
dir.create(file.path(dir, "score"), showWarnings = FALSE)
writeLines(out, file.path(dir, "score", "reuse_check.txt")); cat(out, sep = "\n")
