# ==============================================================================
# Chapter 4/5: EXACT module-level decomposition of the 309-gene hub score
#
# The hub_250 score is linear (ridge Cox, coefficients fit once in ORIEN and
# applied fixed). Each patient's score therefore decomposes EXACTLY:
#
#     score_i = sum_g beta_g * x_ig = sum_pairs [ sum_{g in pair} beta_g * x_ig ]
#
# giving, per patient, the contribution of each matched module pair (the
# interferon/immune, stromal, myogenesis, epithelial programmes). No path
# integral, no baseline choice, no completeness caveat -- this is the linear
# model's own arithmetic, and it REPLACES IG for the corrected signature
# (plan decision: IG is retained only for the legacy DeepSurv chapter).
#
# NATURE OF THE ANALYSIS: descriptive interpretation, not hypothesis testing.
# The only pre-registered check is mechanical: reconstructed scores must equal
# the canonical K3 scores (provenance gate, max |diff| < 1e-6), and per-patient
# module contributions must sum to the total (exactness gate).
#
# INPUTS (all read-only):
#   - results/phaseK3_bootstrap_variance/coefs_hub_250.csv   (gene, beta)
#       -> save from K3 right after the hub_250 ridge fit, e.g.:
#          write.csv(data.frame(gene = colnames(Xfit), beta = as.numeric(beta_vec)),
#                    file.path(OUT_DIR, "coefs_hub_250.csv"), row.names = FALSE)
#          then rerun K3 (bootstrap must reproduce identically).
#   - results/phaseK3_bootstrap_variance/scores_hub_250_{cohort}.csv (gate)
#   - results/phaseK1_6_hub_overlap/module_pairs.csv  (TCGA<->ORIEN matching)
#   - module_assignments.csv (gene_id, module_TCGA, module_ORIEN)
#   - expression matrices: MUST be the same files phaseK3 read (edit EXPR_FILES
#     below to match phaseK3 exactly; any mismatch is caught by the score gate)
#
# OUTPUTS (new directory results/module_decomposition/):
#   contributions_{cohort}.csv       patients x module-pairs (+ total, risk group)
#   group_difference_summary.csv     mean high-low contribution per pair per cohort
#   decomposition_{cohort}.png/.pdf  tile figure: pairs x patients (score-ordered)
#
# Usage:  Rscript scripts/module_decomposition_hub250.R
# ==============================================================================

suppressPackageStartupMessages(library(ggplot2))

K3_DIR  <- "results/phaseK3_bootstrap_variance"
HUB_DIR <- "results/phaseK1_6_hub_overlap"
OUT_DIR <- "results/module_decomposition"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# EDIT to match phaseK3's expression file paths EXACTLY (score gate verifies).
EXPR_FILES <- list(
  TCGA     = "data/processed/tcga_corrected_preprocessed_v4.csv",
  ORIEN    = "data/processed/orien_corrected_preprocessed_v4.csv",
  GSE41613 = "data/external/GSE41613_expression.csv",
  GSE42743 = "data/external/GSE42743_expression.csv")

MODULE_ASSIGN <- "results/phaseK1_module_preservation/module_assignments.csv"

# Programme annotations (from K1.6.8 enrichment; extend as the chapter names them)
PROGRAMS <- c("pink_black" = "interferon/immune",
              "green_red"  = "myogenesis/stromal")

TOL <- 1e-6

# --- Load coefficients and module structure -----------------------------------
cf <- read.csv(file.path(K3_DIR, "coefs_hub_250.csv"),
               colClasses = c(gene = "character"))
stopifnot(nrow(cf) == 309, !any(duplicated(cf$gene)))
beta <- setNames(cf$beta, cf$gene)

pairs <- read.csv(file.path(HUB_DIR, "module_pairs.csv"))
# Expect columns identifying the TCGA module and its matched ORIEN module
tc_col <- grep("tcga", names(pairs), ignore.case = TRUE, value = TRUE)[1]
or_col <- grep("orien", names(pairs), ignore.case = TRUE, value = TRUE)[1]
stopifnot(!is.na(tc_col), !is.na(or_col))
pair_label <- setNames(paste(pairs[[tc_col]], pairs[[or_col]], sep = "_"),
                       pairs[[or_col]])          # keyed by ORIEN colour
pair_label_tcga <- setNames(paste(pairs[[tc_col]], pairs[[or_col]], sep = "_"),
                            pairs[[tc_col]])      # keyed by TCGA colour

ma <- read.csv(MODULE_ASSIGN, colClasses = "character")
stopifnot(all(c("gene_id", "module_TCGA", "module_ORIEN") %in% names(ma)))
rownames(ma) <- ma$gene_id
stopifnot(all(cf$gene %in% ma$gene_id))

# Assign each hub gene to a matched pair: ORIEN membership first (the score's
# source cohort), TCGA membership translated through the pair mapping second.
assign_pair <- function(g) {
  mo <- ma[g, "module_ORIEN"]; mt <- ma[g, "module_TCGA"]
  if (mo %in% names(pair_label))       return(pair_label[[mo]])
  if (mt %in% names(pair_label_tcga))  return(pair_label_tcga[[mt]])
  "unassigned"
}
cf$pair <- vapply(cf$gene, assign_pair, character(1))
n_un <- sum(cf$pair == "unassigned")
if (n_un > 0)
  warning(sprintf("%d hub genes fall outside the matched pairs -- grouped as 'unassigned'; investigate.", n_un))
cat("Genes per module pair:\n"); print(table(cf$pair))

pair_levels <- names(sort(table(cf$pair), decreasing = TRUE))

# --- Per-cohort decomposition -------------------------------------------------
diff_rows <- list()

for (co in names(EXPR_FILES)) {
  ex <- read.csv(EXPR_FILES[[co]], row.names = 1, check.names = FALSE)
  miss <- setdiff(cf$gene, rownames(ex))
  stopifnot(length(miss) == 0)
  X <- t(as.matrix(ex[cf$gene, ]))                 # patients x 309, cf order

  contrib_gene <- sweep(X, 2, beta[cf$gene], `*`)  # beta_g * x_ig

  # Sum within pairs -> patients x pairs
  contrib <- sapply(pair_levels, function(pl)
    rowSums(contrib_gene[, cf$pair == pl, drop = FALSE]))
  total <- rowSums(contrib)

  # GATE 1 (provenance): reconstruct the canonical K3 score exactly
  sc <- read.csv(file.path(K3_DIR, sprintf("scores_hub_250_%s.csv", co)),
                 colClasses = c(patient_id = "character"))
  common <- intersect(rownames(X), sc$patient_id)
  stopifnot(length(common) == nrow(sc))
  dmax <- max(abs(total[sc$patient_id] - sc$score))
  cat(sprintf("[%s] score reconstruction max |diff| = %.2e\n", co, dmax))
  if (dmax > TOL)
    stop(co, ": reconstructed scores do not match canonical K3 scores. ",
         "Coefficients or expression files differ from the K3 run -- ",
         "do not interpret this decomposition.")

  # GATE 2 (exactness) is rowSums by construction; assert anyway
  stopifnot(max(abs(rowSums(contrib) - total)) < 1e-12)

  med <- median(sc$score)
  rg <- setNames(ifelse(sc$score >= med, "High Risk", "Low Risk"), sc$patient_id)

  out <- data.frame(patient_id = sc$patient_id,
                    contrib[sc$patient_id, , drop = FALSE],
                    total = total[sc$patient_id],
                    risk_group = rg[sc$patient_id],
                    check.names = FALSE)
  write.csv(out, file.path(OUT_DIR, sprintf("contributions_%s.csv", co)),
            row.names = FALSE)

  # Group-difference summary: which programmes carry the risk separation
  for (pl in pair_levels) {
    hi <- out[[pl]][out$risk_group == "High Risk"]
    lo <- out[[pl]][out$risk_group == "Low Risk"]
    diff_rows[[paste(co, pl)]] <- data.frame(
      cohort = co, pair = pl,
      program = ifelse(pl %in% names(PROGRAMS), PROGRAMS[[pl]], ""),
      n_genes = sum(cf$pair == pl),
      mean_high = mean(hi), mean_low = mean(lo),
      mean_diff_high_minus_low = mean(hi) - mean(lo),
      share_of_total_gap = (mean(hi) - mean(lo)) /
        (mean(out$total[out$risk_group == "High Risk"]) -
         mean(out$total[out$risk_group == "Low Risk"])))
  }

  # Figure: pairs x patients (ordered by total score), diverging fill
  ord <- order(out$total)
  m <- as.matrix(out[ord, pair_levels, drop = FALSE])
  df <- expand.grid(patient = seq_len(nrow(m)), pair = pair_levels)
  df$value <- as.vector(m)
  df$pair <- factor(df$pair, levels = rev(pair_levels))
  lim <- quantile(abs(df$value), 0.99)
  gg <- ggplot(df, aes(patient, pair, fill = pmax(pmin(value, lim), -lim))) +
    geom_tile() +
    scale_fill_gradient2(low = "#2E86DE", mid = "white", high = "#E74C3C",
                         limits = c(-lim, lim), name = "Contribution\n(beta x expr)") +
    labs(title = sprintf("%s \u2014 module-level decomposition of the hub score", co),
         subtitle = "Patients ordered by total score (left = lowest risk); exact linear decomposition",
         x = "Patients (score-ordered)", y = NULL) +
    theme_classic(base_size = 11) +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
  ggsave(file.path(OUT_DIR, sprintf("decomposition_%s.png", co)), gg,
         width = 10, height = 4, dpi = 300)
  ggsave(file.path(OUT_DIR, sprintf("decomposition_%s.pdf", co)), gg,
         width = 10, height = 4)
}

ds <- do.call(rbind, diff_rows)
write.csv(ds, file.path(OUT_DIR, "group_difference_summary.csv"), row.names = FALSE)

cat("\n=== Which programmes carry the high-low risk separation ===\n")
for (co in names(EXPR_FILES)) {
  cat(sprintf("\n[%s]\n", co))
  sub <- ds[ds$cohort == co, ]
  sub <- sub[order(-abs(sub$mean_diff_high_minus_low)), ]
  for (i in seq_len(nrow(sub)))
    cat(sprintf("  %-18s %-20s genes=%3d  mean diff (H-L) = %+.3f  share of gap = %5.1f%%\n",
                sub$pair[i], sub$program[i], sub$n_genes[i],
                sub$mean_diff_high_minus_low[i],
                100 * sub$share_of_total_gap[i]))
}
cat(sprintf("\nWritten: %s/ (contributions_{cohort}.csv, group_difference_summary.csv, decomposition_{cohort}.png/.pdf)\n", OUT_DIR))
