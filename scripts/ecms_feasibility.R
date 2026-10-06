# ==============================================================================
# ECMS feasibility check — does error-controlled module selection disagree with
# the heuristic rules on real data?
#
# QUESTION. For each module defined in a reference cohort (TCGA or ORIEN),
# compute a preservation p-value in each of its three test cohorts, combine them
# with a partial-conjunction test ("preserved in at least r of 3 cohorts"),
# control FDR across modules with BH, and compare the selected modules with
#   (i)   Zsummary > 10 in all three cohorts,
#   (ii)  Zsummary > 10 in at least two of three,
#   (iii) the current rule: above the empirical (control-module) ceiling in all three.
#
# READINGS, fixed before running:
#   * Rules DISAGREE on some modules -> the real-data application has its
#     interesting cases; the ISMB paper has a demonstration.
#   * Rules AGREE everywhere -> the method may still be valid, but simulations
#     must carry the paper; the OSCC application is confirmatory only.
#   * Grey (unassigned genes) is SELECTED by an ECMS arm -> that arm's null is
#     too lenient; the p-values are not usable as they stand.
#
# TWO SOURCES OF P-VALUES (reported side by side, neither assumed correct):
#   Arm W: WGCNA's own p-values from modulePreservation.rds. The script prints
#     which columns exist. If a summary p-value ('log.psummary') is present it
#     is used; otherwise p is derived from Zsummary under a normal approximation
#     and flagged as such. Whether WGCNA reports an empirical summary p-value at
#     all is itself one of the claims the paper would rest on -- check the
#     printed column list.
#   Arm E: an empirical permutation p-value computed here. Statistic: mean
#     within-module correlation (density) of the module's genes in the test
#     cohort. Null: B random gene sets of the same size drawn from the
#     4,702-gene universe. p = (1 + #{null >= obs}) / (B + 1).
#     EXPECTATION: this null is lenient (random sets have density ~0), so many
#     modules may sit at the floor p = 1/(B+1). If so, that is a finding about
#     the null, not a bug -- it is the 'valid null' problem the method must solve.
#
# PARTIAL CONJUNCTION (Bonferroni form, Benjamini & Heller 2008): with C test
# cohorts and sorted p_(1) <= ... <= p_(C), the p-value for "preserved in at
# least r cohorts" is min(1, (C - r + 1) * p_(r)). Valid under any dependence
# between cohorts.
#
# Inputs read-only. Output: results/ecms_feasibility/
# Usage: Rscript scripts/ecms_feasibility.R
# ==============================================================================

suppressPackageStartupMessages(library(dplyr))

K1_DIR   <- "results/phaseK1_module_preservation"
PROC_DIR <- "data/processed"
EXT_DIR  <- "data/external"
OUT_DIR  <- "results/ecms_feasibility"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

REF_COHORTS <- c("TCGA", "ORIEN")
ALL_COHORTS <- c("TCGA", "ORIEN", "GSE41613", "GSE42743")
CONTROLS    <- c("gold", "grey")
R_VALUES    <- c(2, 3)      # partial-conjunction thresholds: >= 2 of 3, all 3
ALPHA       <- 0.05
B           <- 1000         # empirical permutations (Arm E)
SEED        <- 42

pc_bonf <- function(p, r) {
  p <- sort(p); C <- length(p)
  min(1, (C - r + 1) * p[r])
}

# ==============================================================================
# Inputs
# ==============================================================================

mp   <- readRDS(file.path(K1_DIR, "modulePreservation.rds"))
pres <- read.csv(file.path(K1_DIR, "preservation_statistics.csv"))
mods <- read.csv(file.path(K1_DIR, "module_assignments.csv"), colClasses = "character")
universe <- mods$gene_id
cat("Universe:", length(universe), "genes\n\n")

# ==============================================================================
# Arm W -- WGCNA p-values
# ==============================================================================

cat("=== Arm W: what p-values does modulePreservation provide? ===\n")
rk0 <- paste0("ref.", REF_COHORTS[1])
tk0 <- names(mp$preservation$log.p[[rk0]])
tk0 <- tk0[sapply(mp$preservation$log.p[[rk0]][tk0], is.data.frame)][1]
lp0 <- mp$preservation$log.p[[rk0]][[tk0]]
cat("Columns in mp$preservation$log.p:\n  ", paste(colnames(lp0), collapse = "\n   "), "\n")
summary_col <- grep("summary", colnames(lp0), value = TRUE)
if (length(summary_col)) {
  cat("\n-> Using WGCNA summary p-value column:", summary_col[1], "\n")
  cat("   CHECK the WGCNA documentation for how it is computed (empirical or\n")
  cat("   normal approximation) before citing it in the paper.\n\n")
} else {
  cat("\n-> No summary p-value column. Arm W uses p = 1 - Phi(Zsummary),\n")
  cat("   a normal approximation, flagged as such in the output.\n\n")
}

w_rows <- list()
for (ref in REF_COHORTS) {
  rk <- paste0("ref.", ref)
  for (test in setdiff(ALL_COHORTS, ref)) {
    tk <- paste0("inColumnsAlsoPresentIn.", test)
    Z  <- mp$preservation$Z[[rk]][[tk]]
    lp <- mp$preservation$log.p[[rk]][[tk]]
    if (!is.data.frame(Z)) next
    p <- if (length(summary_col)) 10^lp[[summary_col[1]]] else
         pnorm(Z$Zsummary.pres, lower.tail = FALSE)
    w_rows[[paste(ref, test)]] <- data.frame(
      reference = ref, test = test, module = rownames(Z),
      p_W = pmin(1, pmax(p, .Machine$double.xmin)),
      p_W_source = if (length(summary_col)) summary_col[1] else "normal_from_Zsummary")
  }
}
pw <- bind_rows(w_rows)

# ==============================================================================
# Arm E -- empirical size-matched permutation p-values (density in test cohort)
# ==============================================================================

cat("=== Arm E: empirical permutation p-values (B =", B, ") ===\n")
load_expr <- function(p) as.matrix(read.csv(p, row.names = 1, check.names = FALSE))
EXPR <- list(
  TCGA     = file.path(PROC_DIR, "tcga_corrected_preprocessed_v4.csv"),
  ORIEN    = file.path(PROC_DIR, "orien_corrected_preprocessed_v4.csv"),
  GSE41613 = file.path(EXT_DIR,  "GSE41613_expression.csv"),
  GSE42743 = file.path(EXT_DIR,  "GSE42743_expression.csv"))

e_rows <- list()
for (test in ALL_COHORTS) {
  m <- load_expr(EXPR[[test]])
  g <- intersect(universe, rownames(m))
  cat(sprintf("  %-9s %d genes; computing gene-gene correlation...\n", test, length(g)))
  corm <- cor(t(m[g, , drop = FALSE]))
  rm(m); gc()
  dens <- function(genes) {
    s <- corm[genes, genes, drop = FALSE]
    mean(s[upper.tri(s)])
  }

  for (ref in setdiff(REF_COHORTS, test)) {
    labs <- mods[[paste0("module_", ref)]]
    for (mod in setdiff(unique(labs), "gold")) {   # grey kept as a negative control
      genes <- intersect(mods$gene_id[labs == mod], g)
      if (length(genes) < 10) next
      obs <- dens(genes)
      set.seed(SEED + match(mod, unique(labs)) + 100 * match(test, ALL_COHORTS))
      null <- replicate(B, dens(sample(g, length(genes))))
      e_rows[[paste(ref, test, mod)]] <- data.frame(
        reference = ref, test = test, module = mod, size = length(genes),
        density_obs = obs, density_null_med = median(null),
        density_null_q99 = quantile(null, 0.99),
        p_E = (1 + sum(null >= obs)) / (B + 1))
    }
  }
  rm(corm); gc()
}
pe <- bind_rows(e_rows)

# ==============================================================================
# Combine: partial conjunction per module, BH across modules within reference
# ==============================================================================

pv <- full_join(pw, pe, by = c("reference", "test", "module")) %>%
  left_join(pres %>% select(reference, test, module, Zsummary, above_null),
            by = c("reference", "test", "module"))
write.csv(pv, file.path(OUT_DIR, "ecms_pvalues_by_cohort.csv"), row.names = FALSE)

rules <- pv %>%
  group_by(reference, module) %>%
  summarise(
    n_tests     = n(),
    Z_min       = min(Zsummary, na.rm = TRUE),
    rule_Z10_all  = all(Zsummary > 10, na.rm = TRUE),
    rule_Z10_2of3 = sum(Zsummary > 10, na.rm = TRUE) >= 2,
    rule_ceiling_all = all(above_null %in% TRUE),
    pc2_W = pc_bonf(p_W, 2), pc3_W = pc_bonf(p_W, 3),
    pc2_E = if (all(!is.na(p_E))) pc_bonf(p_E, 2) else NA_real_,
    pc3_E = if (all(!is.na(p_E))) pc_bonf(p_E, 3) else NA_real_,
    .groups = "drop") %>%
  filter(n_tests == 3)

# BH within each reference, over REAL modules only; controls reported separately
rules <- rules %>% group_by(reference) %>%
  mutate(is_control = module %in% CONTROLS,
         q2_W = ifelse(is_control, NA, p.adjust(ifelse(is_control, NA, pc2_W), "BH")),
         q3_W = ifelse(is_control, NA, p.adjust(ifelse(is_control, NA, pc3_W), "BH")),
         q2_E = ifelse(is_control, NA, p.adjust(ifelse(is_control, NA, pc2_E), "BH")),
         q3_E = ifelse(is_control, NA, p.adjust(ifelse(is_control, NA, pc3_E), "BH"))) %>%
  ungroup() %>%
  mutate(ECMS_W_2of3 = q2_W <= ALPHA, ECMS_W_3of3 = q3_W <= ALPHA,
         ECMS_E_2of3 = q2_E <= ALPHA, ECMS_E_3of3 = q3_E <= ALPHA)

write.csv(rules, file.path(OUT_DIR, "ecms_rule_comparison.csv"), row.names = FALSE)

# ==============================================================================
# Report
# ==============================================================================

yn <- function(x) ifelse(is.na(x), "  -", ifelse(x, "YES", " no"))
for (ref in REF_COHORTS) {
  d <- rules %>% filter(reference == ref) %>% arrange(is_control, desc(Z_min))
  cat(sprintf("\n=== Reference %s (modules tested in the other three cohorts) ===\n", ref))
  cat(sprintf("  %-10s %6s | %6s %6s %7s | %6s %6s | %6s %6s\n", "module", "minZ",
              "Z10all", "Z10_2", "ceiling", "W_2/3", "W_3/3", "E_2/3", "E_3/3"))
  for (i in seq_len(nrow(d))) {
    r <- d[i, ]
    rule_vec <- c(r$rule_Z10_all, r$rule_Z10_2of3, r$rule_ceiling_all,
                  r$ECMS_W_2of3, r$ECMS_W_3of3, r$ECMS_E_2of3, r$ECMS_E_3of3)
    flag <- if (!r$is_control && length(unique(na.omit(rule_vec))) > 1) "  <- rules disagree" else ""
    cat(sprintf("  %-10s %6.1f | %6s %6s %7s | %6s %6s | %6s %6s%s\n",
                ifelse(r$is_control, paste0(r$module, "*"), r$module), r$Z_min,
                yn(r$rule_Z10_all), yn(r$rule_Z10_2of3), yn(r$rule_ceiling_all),
                yn(r$ECMS_W_2of3), yn(r$ECMS_W_3of3),
                yn(r$ECMS_E_2of3), yn(r$ECMS_E_3of3), flag))
  }
  ctrl <- d %>% filter(is_control)
  if (nrow(ctrl)) {
    cat("  * negative control. Unadjusted partial-conjunction p-values (should be LARGE):\n")
    for (i in seq_len(nrow(ctrl)))
      cat(sprintf("    %-5s pc2_W = %.3g  pc3_W = %.3g  pc2_E = %.3g  pc3_E = %.3g\n",
                  ctrl$module[i], ctrl$pc2_W[i], ctrl$pc3_W[i], ctrl$pc2_E[i], ctrl$pc3_E[i]))
  }
}

floor_frac <- mean(pe$p_E[!pe$module %in% CONTROLS] <= 1 / (B + 1) + 1e-12)
cat(sprintf("\nArm E: %.0f%% of real module x cohort p-values sit at the floor 1/(B+1).\n",
            100 * floor_frac))
cat("If this is near 100%, the random-gene-set null is too lenient to discriminate\n")
cat("modules -- report it as the 'valid null' problem the method has to solve.\n")
cat("\nWritten:", OUT_DIR, "\n")
