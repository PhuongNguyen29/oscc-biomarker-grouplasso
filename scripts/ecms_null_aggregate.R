# ==============================================================================
# ECMS matched-size null -- AGGREGATE (run once all calls have finished)
#
# 1. Build the null from random modules in the TRAINING calls. For each
#    reference x test x module size, estimate the location and scale of null
#    Zsummary; convert every Z to a standardized residual r = (Z - mu) / sd.
#    Two p-values per module x test cohort:
#      p_emp: the residual's rank among all pooled training-null residuals
#             (nonparametric in the tail)
#      p_par: 1 - Phi(r) (parametric, for comparison)
#
# 2. CALIBRATION CHECK -- the gate. HELD-OUT calls are random modules that did
#    not build the null; they are true negatives by construction.
#    PRE-REGISTERED PASS CRITERIA (both must hold to proceed):
#      (a) held-out p_emp approximately uniform: fraction < 0.05 within
#          [0.02, 0.08], and no gross departure on the KS test;
#      (b) WGCNA's gold and grey controls from the REAL run are NOT selected
#          (partial-conjunction p > 0.05 at r = 2 of 3).
#    If either fails, this null is not valid and the real-module comparison
#    below must not be interpreted.
#
# 3. Only then: partial conjunction (Bonferroni form) across the 3 test cohorts,
#    BH across real modules within each reference, and comparison with
#    Z > 10 (2 of 3) and the ceiling rule.
#
# Usage: Rscript scripts/ecms_null_aggregate.R
# Output: results/ecms_null/aggregate/
# ==============================================================================

suppressPackageStartupMessages(library(dplyr))

NULL_DIR <- "results/ecms_null"
K1_DIR   <- "results/phaseK1_module_preservation"
OUT_DIR  <- file.path(NULL_DIR, "aggregate")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

N_HOLDOUT <- 5        # first N_HOLDOUT calls are held out for calibration
ALPHA     <- 0.05
CONTROLS  <- c("gold", "grey")

pc_bonf <- function(p, r) { p <- sort(p); C <- length(p); min(1, (C - r + 1) * p[r]) }

# --- Load null --------------------------------------------------------------------
files <- list.files(NULL_DIR, pattern = "^random_call_\\d+\\.csv$", full.names = TRUE)
stopifnot(length(files) > N_HOLDOUT + 5)
nl <- bind_rows(lapply(files, read.csv, colClasses = c(module = "character"))) %>%
  filter(is_random)
calls <- sort(unique(nl$call))
hold_calls <- head(calls, N_HOLDOUT)
cat(sprintf("Calls: %d (%d held out) | random-module rows: %d\n",
            length(calls), N_HOLDOUT, nrow(nl)))

train <- nl %>% filter(!call %in% hold_calls)
hold  <- nl %>% filter(call %in% hold_calls)

loc <- train %>% group_by(reference, test, moduleSize) %>%
  summarise(mu = mean(Zsummary), sd = sd(Zsummary), n_null = n(), .groups = "drop")
cat("\nNull samples per (reference, test, size): min",
    min(loc$n_null), "| median", median(loc$n_null), "\n")
if (min(loc$n_null) < 20)
  cat("WARNING: fewer than 20 null modules for some sizes; run more calls.\n")

null_resid <- train %>% left_join(loc, by = c("reference", "test", "moduleSize")) %>%
  mutate(r = (Zsummary - mu) / sd) %>% select(reference, test, r)

score <- function(df) {
  df %>% left_join(loc, by = c("reference", "test", "moduleSize")) %>%
    mutate(r = (Zsummary - mu) / sd) %>%
    rowwise() %>%
    mutate(
      p_emp = {
        nr <- null_resid$r[null_resid$reference == reference & null_resid$test == test]
        (1 + sum(nr >= r)) / (length(nr) + 1)
      },
      p_par = pnorm(r, lower.tail = FALSE)) %>%
    ungroup()
}

# ==============================================================================
# 2. Calibration gate
# ==============================================================================

hs <- score(hold)
cat("\n=== CALIBRATION: held-out random modules (true negatives) ===\n")
for (ref in unique(hs$reference)) {
  d <- hs[hs$reference == ref, ]
  ks <- suppressWarnings(ks.test(d$p_emp, "punif"))
  cat(sprintf("  %-5s n=%4d | frac p_emp<0.05 = %.3f | frac p_emp<0.01 = %.3f | KS p = %.3g\n",
              ref, nrow(d), mean(d$p_emp < 0.05), mean(d$p_emp < 0.01), ks$p.value))
}
# Selection rate among held-out random modules, through the full ECMS procedure
hold_sel <- hs %>% group_by(call, reference, module) %>%
  filter(n() == 3) %>%
  summarise(pc2 = pc_bonf(p_emp, 2), .groups = "drop") %>%
  group_by(call, reference) %>% mutate(q2 = p.adjust(pc2, "BH")) %>% ungroup()
cat(sprintf("  Held-out random modules selected at q <= %.2f: %d of %d (should be ~0)\n",
            ALPHA, sum(hold_sel$q2 <= ALPHA), nrow(hold_sel)))
gate_a <- all(sapply(unique(hs$reference), function(ref) {
  f <- mean(hs$p_emp[hs$reference == ref] < 0.05); f >= 0.02 && f <= 0.08 }))

# ==============================================================================
# 3. Real modules and WGCNA controls
# ==============================================================================

pres <- read.csv(file.path(K1_DIR, "preservation_statistics.csv"))
real <- score(pres %>% select(reference, test, module, moduleSize, Zsummary, above_null))
write.csv(real, file.path(OUT_DIR, "real_module_pvalues.csv"), row.names = FALSE)

rules <- real %>% group_by(reference, module) %>% filter(n() == 3) %>%
  summarise(Z_min = min(Zsummary),
            rule_Z10_2of3   = sum(Zsummary > 10) >= 2,
            rule_ceiling_all = all(above_null %in% TRUE),
            pc2_emp = pc_bonf(p_emp, 2), pc3_emp = pc_bonf(p_emp, 3),
            pc2_par = pc_bonf(p_par, 2), pc3_par = pc_bonf(p_par, 3),
            .groups = "drop") %>%
  mutate(is_control = module %in% CONTROLS) %>%
  group_by(reference) %>%
  mutate(q2_emp = replace(rep(NA_real_, n()), !is_control, p.adjust(pc2_emp[!is_control], "BH")),
         q3_emp = replace(rep(NA_real_, n()), !is_control, p.adjust(pc3_emp[!is_control], "BH")),
         q2_par = replace(rep(NA_real_, n()), !is_control, p.adjust(pc2_par[!is_control], "BH"))) %>%
  ungroup()
write.csv(rules, file.path(OUT_DIR, "rule_comparison.csv"), row.names = FALSE)

ctrl <- rules %>% filter(is_control)
gate_b <- all(ctrl$pc2_emp > ALPHA)
cat("\n=== CALIBRATION: WGCNA controls from the real run (must NOT be selected) ===\n")
for (i in seq_len(nrow(ctrl)))
  cat(sprintf("  %-5s %-4s  pc2_emp = %.3g  pc3_emp = %.3g  pc2_par = %.3g\n",
              ctrl$reference[i], ctrl$module[i], ctrl$pc2_emp[i], ctrl$pc3_emp[i], ctrl$pc2_par[i]))

cat(sprintf("\nGATE (a) held-out calibration: %s\n", if (gate_a) "PASS" else "FAIL"))
cat(sprintf("GATE (b) controls not selected: %s\n", if (gate_b) "PASS" else "FAIL"))
if (!(gate_a && gate_b)) {
  cat("\nThe null is NOT valid. Do not interpret the real-module table below.\n")
}

yn <- function(x) ifelse(is.na(x), "  -", ifelse(x, "YES", " no"))
for (ref in unique(rules$reference)) {
  d <- rules %>% filter(reference == ref, !is_control) %>% arrange(desc(Z_min))
  cat(sprintf("\n=== Reference %s ===\n", ref))
  cat(sprintf("  %-10s %6s | %6s %7s | %8s %8s | %8s\n", "module", "minZ",
              "Z10_2", "ceiling", "ECMS 2/3", "ECMS 3/3", "param 2/3"))
  for (i in seq_len(nrow(d))) {
    r <- d[i, ]
    v <- c(r$rule_Z10_2of3, r$rule_ceiling_all, r$q2_emp <= ALPHA, r$q3_emp <= ALPHA)
    flag <- if (length(unique(v)) > 1) "  <- disagree" else ""
    cat(sprintf("  %-10s %6.1f | %6s %7s | %8s %8s | %8s%s\n",
                r$module, r$Z_min, yn(r$rule_Z10_2of3), yn(r$rule_ceiling_all),
                yn(r$q2_emp <= ALPHA), yn(r$q3_emp <= ALPHA), yn(r$q2_par <= ALPHA), flag))
  }
}
cat("\nWritten:", OUT_DIR, "\n")
