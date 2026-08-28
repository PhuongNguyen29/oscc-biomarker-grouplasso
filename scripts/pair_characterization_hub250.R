# ==============================================================================
# Chapter 4/5: characterization of the five matched module pairs
#
# Follow-up to module_decomposition_hub250.R, which showed the high-low risk
# gap is DISTRIBUTED: three unannotated pairs (brown_yellow, turquoise_blue,
# blue_green) carry ~75-85% of the separation in every cohort. Two analyses:
#
#   A. PER-PAIR ENRICHMENT -- name the programmes. Same machinery and
#      conventions as K1.6.8: hypergeometric test, BACKGROUND = the module-
#      assigned analysis universe (NOT the genome), symbols via the project
#      annotation, BH within collection (h.all, c5.go.bp) per pair.
#
#   B. DIRECTION CHECK -- per pair: sign composition of beta, and whether the
#      pair's per-patient contribution tracks HIGH or LOW expression of its
#      genes. Settles "hazardous-high-expression vs protective-low-expression"
#      per programme -- required before any biological sentence (e.g. the
#      immune-hot question) is written. Descriptive; the only inferential
#      content is the enrichment q-values.
#
# Inputs read-only; outputs appended to results/module_decomposition/.
#
# Usage:  Rscript scripts/pair_characterization_hub250.R
# ==============================================================================

K3_DIR   <- "results/phaseK3_bootstrap_variance"
K1_DIR   <- "results/phaseK1_module_preservation"
HUB_DIR  <- "results/phaseK1_6_hub_overlap"
GMT_DIR  <- "data/genesets"          # h.all + c5.go.bp; exploratory/ excluded
ANNOT    <- "data/raw/protein_coding_genes_annotation.csv"
OUT_DIR  <- "results/module_decomposition"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Match phaseK3 / module_decomposition_hub250.R EXACTLY (edit if yours differ)
EXPR_FILES <- list(
  TCGA     = "data/processed/tcga_corrected_preprocessed_v4.csv",
  ORIEN    = "data/processed/orien_corrected_preprocessed_v4.csv",
  GSE41613 = "data/external/GSE41613_expression.csv",
  GSE42743 = "data/external/GSE42743_expression.csv")

Q_CUT <- 0.10
TOL <- 1e-6

# --- Pair assignment: identical logic to module_decomposition_hub250.R -------
cf <- read.csv(file.path(K3_DIR, "coefs_hub_250.csv"),
               colClasses = c(gene = "character"))
stopifnot(nrow(cf) == 309)
beta <- setNames(cf$beta, cf$gene)

pairs <- read.csv(file.path(HUB_DIR, "module_pairs.csv"))
tc_col <- grep("tcga", names(pairs), ignore.case = TRUE, value = TRUE)[1]
or_col <- grep("orien", names(pairs), ignore.case = TRUE, value = TRUE)[1]
pair_label      <- setNames(paste(pairs[[tc_col]], pairs[[or_col]], sep = "_"),
                            pairs[[or_col]])
pair_label_tcga <- setNames(paste(pairs[[tc_col]], pairs[[or_col]], sep = "_"),
                            pairs[[tc_col]])

ma <- read.csv(file.path(K1_DIR, "module_assignments.csv"),
               colClasses = "character")
rownames(ma) <- ma$gene_id
assign_pair <- function(g) {
  mo <- ma[g, "module_ORIEN"]; mt <- ma[g, "module_TCGA"]
  if (mo %in% names(pair_label))      return(pair_label[[mo]])
  if (mt %in% names(pair_label_tcga)) return(pair_label_tcga[[mt]])
  "unassigned"
}
cf$pair <- vapply(cf$gene, assign_pair, character(1))
pair_levels <- setdiff(unique(cf$pair), "unassigned")

# --- Symbol mapping and background (K1.6.8 conventions) -----------------------
an <- read.csv(ANNOT)
id2sym <- setNames(an$gene_symbol, an$gene_id)
universe_ids <- ma$gene_id                       # the 4,702 analysis universe
universe_sym <- unique(na.omit(id2sym[universe_ids]))
universe_sym <- universe_sym[universe_sym != ""]
cf$symbol <- id2sym[cf$gene]
cat(sprintf("Background: %d symbols (from %d module-assigned genes)\n",
            length(universe_sym), length(universe_ids)))

# ==============================================================================
# A. Per-pair enrichment
# ==============================================================================

read_gmt <- function(path) {
  lines <- readLines(path, warn = FALSE)
  sets <- lapply(strsplit(lines, "\t"), function(x) unique(x[-(1:2)]))
  names(sets) <- vapply(strsplit(lines, "\t"), `[`, character(1), 1)
  sets
}

gmt_files <- list.files(GMT_DIR, pattern = "\\.gmt$", full.names = TRUE)
stopifnot(length(gmt_files) >= 1)

enr_rows <- list()
for (gf in gmt_files) {
  collection <- basename(gf)
  sets <- read_gmt(gf)
  sets <- lapply(sets, function(s) intersect(s, universe_sym))  # background-restricted
  sets <- sets[vapply(sets, length, 1L) >= 5]
  N <- length(universe_sym)

  for (pl in pair_levels) {
    genes <- unique(na.omit(cf$symbol[cf$pair == pl]))
    genes <- intersect(genes, universe_sym)
    n <- length(genes)
    p_vals <- vapply(sets, function(s) {
      k <- length(intersect(genes, s))
      phyper(k - 1, length(s), N - length(s), n, lower.tail = FALSE)
    }, numeric(1))
    k_vals <- vapply(sets, function(s) length(intersect(genes, s)), integer(1))
    q <- p.adjust(p_vals, method = "BH")          # BH within collection, per pair
    ord <- order(p_vals)
    for (i in ord[k_vals[ord] > 0]) {
      enr_rows[[paste(collection, pl, names(sets)[i])]] <- data.frame(
        collection = collection, pair = pl, term = names(sets)[i],
        overlap = k_vals[i], term_size = length(sets[[i]]),
        set_size = n,
        fold = (k_vals[i] / n) / (length(sets[[i]]) / N),
        p = p_vals[i], q = q[i])
    }
  }
}
enr <- do.call(rbind, enr_rows)
write.csv(enr, file.path(OUT_DIR, "pair_enrichment.csv"), row.names = FALSE)

cat("\n=== A. Per-pair enrichment (top 5 per collection; * = q <", Q_CUT, ") ===\n")
for (pl in pair_levels) {
  cat(sprintf("\n--- %s ---\n", pl))
  for (gf in gmt_files) {
    sub <- enr[enr$pair == pl & enr$collection == basename(gf), ]
    sub <- sub[order(sub$p), ][seq_len(min(5, nrow(sub))), ]
    n_sig <- sum(enr$pair == pl & enr$collection == basename(gf) & enr$q < Q_CUT)
    cat(sprintf("  [%s] %d terms q<%.2f\n", basename(gf), n_sig, Q_CUT))
    for (i in seq_len(nrow(sub)))
      cat(sprintf("    %s%-55s %3d/%-4d fold %4.1f q=%.2e\n",
                  ifelse(sub$q[i] < Q_CUT, "*", " "),
                  substr(sub$term[i], 1, 55), sub$overlap[i],
                  sub$term_size[i], sub$fold[i], sub$q[i]))
  }
}

# ==============================================================================
# B. Direction check
# ==============================================================================

cat("\n=== B. Direction check per pair ===\n")
dir_rows <- list()
for (co in names(EXPR_FILES)) {
  ex <- read.csv(EXPR_FILES[[co]], row.names = 1, check.names = FALSE)
  stopifnot(all(cf$gene %in% rownames(ex)))
  X <- t(as.matrix(ex[cf$gene, ]))

  # Provenance gate (same as decomposition): scores must reconstruct
  sc <- read.csv(file.path(K3_DIR, sprintf("scores_hub_250_%s.csv", co)),
                 colClasses = c(patient_id = "character"))
  total <- as.numeric(X %*% beta[cf$gene]); names(total) <- rownames(X)
  stopifnot(max(abs(total[sc$patient_id] - sc$score)) < TOL)

  med <- median(sc$score)
  hi_ids <- sc$patient_id[sc$score >= med]
  lo_ids <- sc$patient_id[sc$score < med]

  for (pl in pair_levels) {
    idx <- cf$pair == pl
    g <- cf$gene[idx]; b <- beta[g]
    contrib <- as.numeric(X[sc$patient_id, g, drop = FALSE] %*% b)
    names(contrib) <- sc$patient_id
    mean_expr <- rowMeans(X[sc$patient_id, g, drop = FALSE])
    r <- cor(contrib, mean_expr)
    expr_diff <- mean(mean_expr[hi_ids]) - mean(mean_expr[lo_ids])
    dir_rows[[paste(co, pl)]] <- data.frame(
      cohort = co, pair = pl, n_genes = length(g),
      frac_beta_pos = mean(b > 0), sum_beta = sum(b), mean_beta = mean(b),
      cor_contrib_meanexpr = r,
      mean_expr_diff_high_minus_low = expr_diff,
      reading = ifelse(r > 0.5 & expr_diff > 0,
                       "hazardous-high-expression",
                ifelse(r < -0.5 & expr_diff < 0,
                       "protective-low-expression (absence is hazardous)",
                       "mixed -- inspect gene-level signs")))
  }
}
dirs <- do.call(rbind, dir_rows)
write.csv(dirs, file.path(OUT_DIR, "pair_direction.csv"), row.names = FALSE)

for (pl in pair_levels) {
  sub <- dirs[dirs$pair == pl, ]
  cat(sprintf("\n--- %s: frac beta>0 = %.2f, sum beta = %+.3f ---\n",
              pl, sub$frac_beta_pos[1], sub$sum_beta[1]))
  for (i in seq_len(nrow(sub)))
    cat(sprintf("  [%s] cor(contrib, mean expr) = %+.2f | mean expr H-L = %+.3f | %s\n",
                sub$cohort[i], sub$cor_contrib_meanexpr[i],
                sub$mean_expr_diff_high_minus_low[i], sub$reading[i]))
}

cat(sprintf("\nWritten: %s/pair_enrichment.csv, %s/pair_direction.csv\n",
            OUT_DIR, OUT_DIR))
