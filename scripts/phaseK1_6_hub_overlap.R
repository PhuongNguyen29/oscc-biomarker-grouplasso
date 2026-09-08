#!/usr/bin/env Rscript
# ==============================================================================
# Phase K1.6.4: Hub-gene selection — does kME transport where the lasso does not?
# ==============================================================================
#
# THE QUESTION
# ------------
# Gene-level lasso selection does not transport: zero cross-cohort overlap in
# 56 of 56 estimable CPSS cells (RULEFIX run, plan D2b), and within-cohort
# Kuncheva never above 0.29 even at subsample fraction 0.9 (D5 rev b). But co-expression MODULES do transport
# (K1: Zsummary above the null-control ceiling in every test cohort, Fisher
# one-to-one mapping at p = 5.7e-280).
#
# The lasso's instability has a known mechanism: with correlated predictors the
# irrepresentable condition fails (Zhao & Yu 2006) and the lasso keeps one
# arbitrary representative per correlated group. Modules ARE those groups. So:
# select genes by MODULE MEMBERSHIP instead, and see whether agreement returns.
#
# kME = correlation between a gene's expression and its module eigengene. It is
# a smooth function of the correlation matrix -- the very object that preserves
# -- rather than the argmax of a penalised regression.
#
# WHAT THIS CAN AND CANNOT SHOW
# -----------------------------
# K1 was UNSUPERVISED throughout. It establishes shared CO-EXPRESSION structure,
# NOT shared prognostic structure. K1.5 tested prognosis separately and found it
# thin: no module reached q < 0.05 in more than one cohort. So a high hub overlap
# here would be a finding about SELECTION STABILITY, not evidence that hub genes
# are prognostic. Hub genes are the most connected members of a module, not the
# most survival-associated. That distinction is preserved in the write-up.
#
# THE TRAP THIS SCRIPT AVOIDS (K1.6.2)
# ------------------------------------
# Modules were detected separately in TCGA and ORIEN, so colour labels are
# arbitrary: TCGA "turquoise" and ORIEN "turquoise" are unrelated. Intersecting
# hub sets by colour would compare different modules and return near-zero overlap
# for a trivial reason, manufacturing a negative result.
#
# Modules are matched by the K1 Fisher overlap table, and ONLY reciprocal best
# matches are used -- each must be the other's best partner. Non-reciprocal
# modules are excluded and reported.
#
# References:
#   Langfelder P, Horvath S (2008) BMC Bioinformatics 9:559 -- WGCNA, kME
#   Zhao P, Yu B (2006) JMLR 7:2541-2563 -- irrepresentable condition
#   Kuncheva LI (2007) Proc IASTED AIA 390-395 -- chance-corrected stability
#
# Usage:
#   Rscript scripts/phaseK1_6_hub_overlap.R
# ==============================================================================

suppressPackageStartupMessages({ library(WGCNA); library(dplyr) })
options(stringsAsFactors = FALSE)
enableWGCNAThreads()

PROC_DIR <- "data/processed"
K1_DIR   <- "results/phaseK1_module_preservation"
CPSS_DIR <- "results/phase4_cpss_RULEFIX"   # rev b run of record (plan D2b)
OUT_DIR  <- "results/phaseK1_6_hub_overlap_k30"

K_GRID   <- c(10, 20, 30, 50)     # report the curve, not a chosen k
SEED     <- 42

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create("logs", showWarnings = FALSE)
log_file <- file.path("logs",
  sprintf("phaseK1_6_hub_overlap_%s.log", format(Sys.time(), "%Y%m%d_%H%M%S")))
log_con <- file(log_file, open = "wt")
sink(log_con, split = TRUE); sink(log_con, type = "message")

cat("Phase K1.6.4: hub-gene cross-cohort overlap\n")
cat("Started:", format(Sys.time()), "\n")
cat("k grid:", paste(K_GRID, collapse = ", "), "\n\n")
set.seed(SEED)

# ==============================================================================
# Stability statistics -- the SAME ones used for lasso selection in D5, so the
# comparison is like-for-like
# ==============================================================================

kuncheva <- function(s1, s2, m) {
  # Kuncheva (2007): corrected for the overlap expected by chance given subset
  # size. Requires |s1| == |s2|, which top-k selection guarantees.
  c <- length(s1)
  if (length(s2) != c || c == 0 || c == m) return(NA_real_)
  (length(intersect(s1, s2)) * m - c * c) / (c * (m - c))
}
jaccard <- function(s1, s2) {
  u <- length(union(s1, s2)); if (!u) NA_real_ else length(intersect(s1, s2)) / u
}

# ==============================================================================
# Load
# ==============================================================================

cat("--- Loading ---\n")
mods <- read.csv(file.path(K1_DIR, "module_assignments.csv"))
pres <- read.csv(file.path(K1_DIR, "preservation_statistics.csv"))
cat("  genes with module assignments:", nrow(mods), "\n")

load_expr <- function(p) {
  m <- as.matrix(read.csv(p, row.names = 1, check.names = FALSE))
  cat(sprintf("  %-42s %5d x %4d | mean gene sd %.3f\n",
              basename(p), nrow(m), ncol(m), mean(apply(m, 1, sd))))
  m
}
expr <- list(
  TCGA  = load_expr(file.path(PROC_DIR, "tcga_corrected_preprocessed_v4.csv")),
  ORIEN = load_expr(file.path(PROC_DIR, "orien_corrected_preprocessed_v4.csv"))
)
genes <- mods$gene_id
stopifnot(all(genes %in% rownames(expr$TCGA)),
          all(genes %in% rownames(expr$ORIEN)))
p_total <- length(genes)
cat("  feature space:", p_total, "genes\n\n")

# Preserved modules only: above the null-control ceiling in ALL test cohorts
preserved <- pres %>% filter(!is_control) %>%
  group_by(reference, module) %>%
  summarise(n_tests = n(), n_above = sum(above_null, na.rm = TRUE),
            size = first(moduleSize), .groups = "drop") %>%
  filter(n_above == n_tests)

cat("--- Preserved modules (above null in every test cohort) ---\n")
for (r in c("TCGA", "ORIEN")) {
  q <- preserved[preserved$reference == r, ]
  cat("  ", r, ":", paste(q$module, collapse = ", "), "\n")
}
cat("\n")

# ==============================================================================
# K1.6.2 Reciprocal module matching via the Fisher overlap table
# ==============================================================================

cat("--- Matching TCGA modules to ORIEN modules (reciprocal best Fisher) ---\n")
pt <- as.matrix(read.csv(file.path(K1_DIR, "module_overlap_pvalues.csv"),
                         row.names = 1))
# rows = TCGA modules, cols = ORIEN modules
tk <- preserved$module[preserved$reference == "TCGA"]
ok <- preserved$module[preserved$reference == "ORIEN"]
tk <- intersect(tk, rownames(pt)); ok <- intersect(ok, colnames(pt))
sub <- pt[tk, ok, drop = FALSE]

pairs <- list()
for (i in rownames(sub)) {
  j <- colnames(sub)[which.min(sub[i, ])]                 # TCGA i -> best ORIEN
  back <- rownames(sub)[which.min(sub[, j])]              # ORIEN j -> best TCGA
  recip <- identical(back, i)
  cat(sprintf("  TCGA %-10s -> ORIEN %-10s  p = %.3e  %s\n",
              i, j, sub[i, j],
              if (recip) "reciprocal" else paste0("NOT reciprocal (", j,
                                                  " prefers ", back, ") -- EXCLUDED")))
  if (recip) pairs[[length(pairs) + 1]] <-
    data.frame(tcga_module = i, orien_module = j, fisher_p = sub[i, j])
}
pairs <- bind_rows(pairs)
cat("\n  Reciprocally matched module pairs:", nrow(pairs), "\n\n")
if (nrow(pairs) == 0) stop("No reciprocal pairs; hub comparison is not defined.")
write.csv(pairs, file.path(OUT_DIR, "module_pairs.csv"), row.names = FALSE)

# ==============================================================================
# kME and hub selection
# ==============================================================================

cat("--- Computing kME ---\n")
# kME is computed WITHIN each cohort against that cohort's OWN eigengene, so the
# two selections are genuinely independent. Nothing crosses between cohorts.
kme <- list()
for (co in c("TCGA", "ORIEN")) {
  datExpr <- t(expr[[co]][genes, , drop = FALSE])          # samples x genes
  colors  <- mods[[paste0("module_", co)]]
  ME <- moduleEigengenes(datExpr, colors, align = "along average",
                         excludeGrey = TRUE, verbose = 0)$eigengenes
  k <- signedKME(datExpr, ME, outputColumnName = "")
  kme[[co]] <- k
  cat(sprintf("  %-6s %d genes x %d eigengenes\n", co, nrow(k), ncol(k)))
}
cat("\n")

hubs <- function(co, module, k) {
  colors <- mods[[paste0("module_", co)]]
  idx <- which(colors == module)
  col <- if (module %in% colnames(kme[[co]])) module else paste0("ME", module)
  v <- kme[[co]][idx, col]
  g <- genes[idx]
  g[order(-abs(v))][seq_len(min(k, length(g)))]
}

# ==============================================================================
# K1.6.4 Cross-cohort overlap: hubs vs lasso
# ==============================================================================

cat(rep("=", 74), "\n", sep = "")
cat("HUB-GENE CROSS-COHORT OVERLAP\n")
cat(rep("=", 74), "\n", sep = "")

rows <- list()
for (k in K_GRID) {
  cat("\n### top-", k, " hubs per module ###\n", sep = "")
  cat(sprintf("  %-12s %-12s %6s %6s %9s %8s\n",
              "TCGA mod", "ORIEN mod", "n_T", "n_O", "Kuncheva", "Jaccard"))
  allT <- character(); allO <- character()
  for (i in seq_len(nrow(pairs))) {
    hT <- hubs("TCGA",  pairs$tcga_module[i],  k)
    hO <- hubs("ORIEN", pairs$orien_module[i], k)
    allT <- c(allT, hT); allO <- c(allO, hO)
    kk <- kuncheva(hT, hO, p_total); jj <- jaccard(hT, hO)
    cat(sprintf("  %-12s %-12s %6d %6d %9.3f %8.3f\n",
                pairs$tcga_module[i], pairs$orien_module[i],
                length(hT), length(hO), kk, jj))
    rows[[paste(k, i)]] <- data.frame(
      k = k, tcga_module = pairs$tcga_module[i],
      orien_module = pairs$orien_module[i], n_overlap = length(intersect(hT, hO)),
      kuncheva = kk, jaccard = jj, level = "per_module")
  }
  allT <- unique(allT); allO <- unique(allO)
  cat(sprintf("  %-12s %-12s %6d %6d %9.3f %8.3f   <- UNION\n", "union", "union",
              length(allT), length(allO),
              kuncheva(allT, allO, p_total), jaccard(allT, allO)))
  rows[[paste(k, "union")]] <- data.frame(
    k = k, tcga_module = "union", orien_module = "union",
    n_overlap = length(intersect(allT, allO)),
    kuncheva = kuncheva(allT, allO, p_total),
    jaccard = jaccard(allT, allO), level = "union")
  writeLines(sort(intersect(allT, allO)),
             file.path(OUT_DIR, sprintf("shared_hubs_k%d.txt", k)))
}
res <- bind_rows(rows)
write.csv(res, file.path(OUT_DIR, "hub_overlap.csv"), row.names = FALSE)

# ==============================================================================
# The comparator: lasso selection at matched size
# ==============================================================================

cat("\n", rep("=", 74), "\n", sep = "")
cat("COMPARATOR — lasso (CPSS) selection at matched set size\n")
cat(rep("=", 74), "\n", sep = "")

# CANONICAL ARTIFACTS (repointed after audit): Phase 4 writes the SELECTED
# cell's selection probabilities under fixed names, so this script never
# reconstructs the operating point from a filename pattern. The old
# "^tcga.*mult0p80" pattern hard-coded a cell the A4.5 rule is supposed to
# choose. Ties in the top-N ranking are broken by file (gene-universe) order,
# deterministically but arbitrarily -- same as before, now stated.
f_t <- file.path(CPSS_DIR, "selected_tcga_selection_probabilities.csv")
f_o <- file.path(CPSS_DIR, "selected_orien_selection_probabilities.csv")

if (file.exists(f_t) && file.exists(f_o)) {
  pt2 <- read.csv(f_t, row.names = 1); po2 <- read.csv(f_o, row.names = 1)
  cat("  selection probabilities: canonical selected-cell artifacts\n")
  cat("  selected cell:",
      paste(readLines(file.path(CPSS_DIR, "selected_cell.json"), warn = FALSE),
            collapse = " "), "\n")
  cat(sprintf("  %8s %9s %8s\n", "set size", "Kuncheva", "Jaccard"))
  for (k in K_GRID) {
    for (m in c(1, nrow(pairs))) {  # per-module-equivalent and union-equivalent
      n <- k * m
      gt <- rownames(pt2)[order(-pt2[, 1])][seq_len(min(n, nrow(pt2)))]
      go <- rownames(po2)[order(-po2[, 1])][seq_len(min(n, nrow(po2)))]
      cat(sprintf("  %8d %9.3f %8.3f\n", n,
                  kuncheva(gt, go, p_total), jaccard(gt, go)))
      rows[[paste("lasso", n)]] <- data.frame(
        k = k, tcga_module = "lasso_topN", orien_module = "lasso_topN",
        n_overlap = length(intersect(gt, go)),
        kuncheva = kuncheva(gt, go, p_total),
        jaccard = jaccard(gt, go), level = paste0("lasso_n", n))
    }
  }
  write.csv(bind_rows(rows), file.path(OUT_DIR, "lasso_comparator_overlap.csv"),
            row.names = FALSE)
  write.csv(bind_rows(rows), file.path(OUT_DIR, "hub_overlap.csv"),
            row.names = FALSE)
  cat("\n  Lasso selection probabilities are ranked highest-first; the top-N\n")
  cat("  genes are the closest like-for-like comparator to a top-k hub set.\n")
} else {
  cat("  Canonical selected-cell artifacts not found under", CPSS_DIR, "\n")
  cat("  Compare against the D2b record instead: cross-cohort overlap was ZERO\n")
  cat("  in 56/56 estimable CPSS cells.\n")
}

cat("\n", rep("=", 74), "\n", sep = "")
cat("INTERPRETATION\n")
cat(rep("=", 74), "\n", sep = "")
cat("Kuncheva is chance-corrected: 0 = agreement no better than chance given\n")
cat("set size, 1 = identical. The lasso benchmark from D5/D2 is ~0.\n\n")
cat("A high hub overlap is a finding about SELECTION STABILITY. It does NOT\n")
cat("show hub genes are prognostic -- kME measures connectivity, not survival\n")
cat("association, and K1 was unsupervised throughout. Whether hub genes carry\n")
cat("survival signal is K1.6.7, a separate analysis.\n")

cat("\nFinished:", format(Sys.time()), "\n")
cat("--- sessionInfo ---\n"); print(sessionInfo())
sink(type = "message"); sink(); close(log_con)
