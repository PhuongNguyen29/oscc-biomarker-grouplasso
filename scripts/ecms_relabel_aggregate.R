# ==============================================================================
# ECMS relabel check (design §7b) -- AGGREGATE
#
# Scores every relabelled real module against the EXISTING matched-size null
# (results/ecms_null), with exactly the scoring of ecms_null_aggregate.R:
# per (reference, test, size) location/scale from the TRAINING null calls
# (calls after the first 5), standardized residual, empirical p pooled within
# (reference, test). Then partial conjunction (Bonferroni form) and BH per draw.
#
# Every relabelled module is ABSENT in every test cohort by construction.
#
# PRE-REGISTERED (7 Oct 2026, before any relabel call ran):
#   PASS if, for EACH reference:
#     (1) fraction of module x cohort p_emp < 0.05 is <= 0.08 (conservative,
#         i.e. below 0.05, is acceptable: C0 asks "uniform or conservative");
#     (2) draws with ANY module selected (r = 2 of 3, BH q <= 0.05) <= 0.05 of
#         draws + 2 Monte Carlo SE.
#   Reported, not gated: KS vs uniform (p-values within a draw are dependent),
#   fraction p < 0.01, and per-module fraction p < 0.05 (a module that is
#   consistently small points to a discovery property the null misses).
#   FAIL -> match the null on discovery properties (density / connectivity),
#   then rerun. No other change.
#
# Usage: Rscript scripts/ecms_relabel_aggregate.R
# ==============================================================================

suppressPackageStartupMessages(library(dplyr))

NULL_DIR  <- "results/ecms_null"
REL_DIR   <- "results/ecms_relabel"
N_HOLDOUT <- 5
ALPHA     <- 0.05
CONTROLS  <- c("gold", "grey")
pc_bonf <- function(p, r) { p <- sort(p); C <- length(p); min(1, (C - r + 1) * p[r]) }

# --- Null, built exactly as in ecms_null_aggregate.R -----------------------------
files <- list.files(NULL_DIR, pattern = "^random_call_\\d+\\.csv$", full.names = TRUE)
stopifnot(length(files) > N_HOLDOUT + 5)
nl <- bind_rows(lapply(files, read.csv, colClasses = c(module = "character"))) %>% filter(is_random)
calls <- sort(unique(nl$call))
train <- nl %>% filter(!call %in% head(calls, N_HOLDOUT))
loc <- train %>% group_by(reference, test, moduleSize) %>%
  summarise(mu = mean(Zsummary), sd = sd(Zsummary), n_null = n(), .groups = "drop")
null_resid <- train %>% left_join(loc, by = c("reference", "test", "moduleSize")) %>%
  mutate(r = (Zsummary - mu) / sd) %>% select(reference, test, r)
cat(sprintf("Null: %d training calls | samples per (ref, test, size) min %d, median %d\n",
            length(calls) - N_HOLDOUT, min(loc$n_null), as.integer(median(loc$n_null))))

score <- function(df) {
  out <- df %>% left_join(loc, by = c("reference", "test", "moduleSize")) %>%
    mutate(r = (Zsummary - mu) / sd)
  if (anyNA(out$mu)) stop("module sizes missing from the null: ",
                          paste(unique(out$moduleSize[is.na(out$mu)]), collapse = ", "))
  out %>% rowwise() %>%
    mutate(p_emp = {
      nr <- null_resid$r[null_resid$reference == reference & null_resid$test == test]
      (1 + sum(nr >= r)) / (length(nr) + 1)
    }) %>% ungroup()
}

# --- Relabel results ---------------------------------------------------------------
rf <- list.files(REL_DIR, pattern = "^relabel_.*_draw\\d+\\.csv$", full.names = TRUE)
if (!length(rf)) stop("no relabel results in ", REL_DIR)
rl <- bind_rows(lapply(rf, read.csv, colClasses = c(module = "character"))) %>%
  filter(!is_control)
s <- score(rl)
write.csv(s, file.path(REL_DIR, "relabel_pvalues.csv"), row.names = FALSE)

cat("\n=== Relabel check: every module ABSENT in every test cohort by construction ===\n")
gate <- c()
for (ref in sort(unique(s$reference))) {
  d <- s[s$reference == ref, ]
  ndraw <- length(unique(d$draw))
  f05 <- mean(d$p_emp < 0.05); f01 <- mean(d$p_emp < 0.01)
  ks  <- suppressWarnings(ks.test(d$p_emp, "punif"))$p.value
  sel <- d %>% group_by(draw, module) %>% filter(n() == 3) %>%
    summarise(pc2 = pc_bonf(p_emp, 2), pc3 = pc_bonf(p_emp, 3), .groups = "drop") %>%
    group_by(draw) %>% mutate(q2 = p.adjust(pc2, "BH"), q3 = p.adjust(pc3, "BH")) %>% ungroup()
  any2 <- sel %>% group_by(draw) %>% summarise(a = any(q2 <= ALPHA), .groups = "drop")
  frac_any <- mean(any2$a)
  thr <- 0.05 + 2 * sqrt(0.05 * 0.95 / ndraw)
  ok1 <- f05 <= 0.08; ok2 <- frac_any <= thr
  gate[ref] <- ok1 && ok2
  cat(sprintf(paste0("\n%s: %d draws, %d module x cohort p-values\n",
                     "  frac p < 0.05 = %.3f (gate <= 0.08: %s) | frac p < 0.01 = %.3f | KS p = %.3g (reported only)\n",
                     "  draws with any module selected (2 of 3, BH 0.05) = %d of %d = %.3f (gate <= %.3f: %s)\n",
                     "  module selections, 2 of 3: %d of %d | 3 of 3: %d of %d\n"),
              ref, ndraw, nrow(d), f05, if (ok1) "ok" else "FAIL", f01, ks,
              sum(any2$a), nrow(any2), frac_any, thr, if (ok2) "ok" else "FAIL",
              sum(sel$q2 <= ALPHA), nrow(sel), sum(sel$q3 <= ALPHA), nrow(sel)))
  per <- d %>% group_by(module, moduleSize) %>%
    summarise(frac_p05 = mean(p_emp < 0.05), median_p = median(p_emp),
              median_Z = median(Zsummary), .groups = "drop") %>% arrange(desc(frac_p05))
  cat("  per module (descriptive):\n")
  print(as.data.frame(per), row.names = FALSE, digits = 3)
}

cat(sprintf("\nGATE (relabel, C0): %s\n", if (all(gate)) "PASS" else "FAIL"))
if (!all(gate)) cat("Pre-registered action: match the null on discovery properties, then rerun.\n")
