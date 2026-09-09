# ==============================================================================
# Analysis A -- Discordant gene selections, concordant programmes?
#
# QUESTION (pre-registered). Penalized selection in TCGA and ORIEN yields gene
# sets that share essentially nothing (Kuncheva ~ 0 at every matched size,
# K1.6.4). Is that disagreement about BIOLOGY, or only about WHICH GENE NAMES
# IT? Test: map each cohort's selected genes to the reciprocally matched module
# pairs -- each via its OWN cohort's network -- and ask whether the two
# selections concentrate in the SAME pairs more than random gene sets of the
# same size would.
#
# READINGS, fixed before running:
#   Positive (pair-level agreement far above the permutation null):
#     "The cohorts do not disagree about biology; they disagree about which
#      member of a programme to name." Gene-level non-replication is an artifact
#     of representation. This is the mechanistic centrepiece of Ch3 §3.3.8.
#   Null (pair-level agreement at chance):
#     Gene selections are not merely unstable but uninformative about shared
#     programmes -- the negative result of Ch3 sharpens.
#
# CONSTRUCTION.
#   Selected genes = top-N per cohort by CPSS selection frequency at the
#   canonical selected cell (the same construction as the K1.6.4 comparator,
#   which is exactly "what penalized selection picks"). N in {50, 100, 250}.
#   DISCORDANT = selected in one cohort and not the other (the overlap, if any,
#   is removed, so agreement cannot come from shared genes).
#   Each discordant gene is assigned to a matched pair via its cohort's own
#   module membership (TCGA gene -> module_TCGA -> pair; ORIEN gene ->
#   module_ORIEN -> pair); genes outside any matched pair go to "unmatched".
#   Pair-level agreement = histogram overlap  sum_p min(t_p, o_p)  over the
#   six bins (5 pairs + unmatched), where t_p / o_p are the fractions of each
#   cohort's discordant genes landing in bin p. 0 = disjoint programme
#   profiles, 1 = identical. Also reported: the fraction of each set inside
#   ANY matched pair, and Spearman across the five pair bins.
#   Null: N_PERM random gene sets of the same sizes drawn from the 4,702-gene
#   universe, same statistic. Random sets land in pairs in proportion to module
#   size, so the null already accounts for size.
#
# The gene-level overlap of the same two sets is reported alongside, so the
# contrast -- near-zero at the gene level, X at the programme level -- is on
# one line.
#
# HOW TO RUN. Standalone. Reads the canonical CPSS artifacts and the K1 module
# files read-only; writes to a new directory.
#   Rscript scripts/analysisA_discordant_modules.R
# ==============================================================================

suppressPackageStartupMessages(library(dplyr))

CPSS_DIR <- "results/phase4_cpss_RULEFIX"
K1_DIR   <- "results/phaseK1_module_preservation"
HUB_DIR  <- "results/phaseK1_6_hub_overlap_k30"
OUT_DIR  <- "results/analysisA_discordant_modules"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

SIZES  <- c(50, 100, 250)   # per cohort, matched to the K1.6.4 comparator
N_PERM <- 2000
SEED   <- 42

# --- Inputs -------------------------------------------------------------------
pt <- read.csv(file.path(CPSS_DIR, "selected_tcga_selection_probabilities.csv"),
               row.names = 1)
po <- read.csv(file.path(CPSS_DIR, "selected_orien_selection_probabilities.csv"),
               row.names = 1)
mods  <- read.csv(file.path(K1_DIR, "module_assignments.csv"),
                  colClasses = "character")
pairs <- read.csv(file.path(HUB_DIR, "module_pairs.csv"))
stopifnot(all(c("gene_id", "module_TCGA", "module_ORIEN") %in% names(mods)))

universe <- mods$gene_id                       # 4,702 module-assigned genes
stopifnot(length(universe) > 4000)
rownames(mods) <- mods$gene_id

tc_col <- grep("tcga",  names(pairs), ignore.case = TRUE, value = TRUE)[1]
or_col <- grep("orien", names(pairs), ignore.case = TRUE, value = TRUE)[1]
pair_lab <- paste(pairs[[tc_col]], pairs[[or_col]], sep = "_")
map_T <- setNames(pair_lab, pairs[[tc_col]])   # TCGA colour  -> pair
map_O <- setNames(pair_lab, pairs[[or_col]])   # ORIEN colour -> pair
BINS  <- c(pair_lab, "unmatched")
cat("Matched pairs:", paste(pair_lab, collapse = ", "), "\n")

# Restrict the ranking to the universe (genes with module assignments)
pt <- pt[rownames(pt) %in% universe, , drop = FALSE]
po <- po[rownames(po) %in% universe, , drop = FALSE]
rank_T <- rownames(pt)[order(-pt[, 1])]
rank_O <- rownames(po)[order(-po[, 1])]

# --- Assignment and statistic -------------------------------------------------
to_bins <- function(genes, cohort) {
  m <- if (cohort == "TCGA") mods[genes, "module_TCGA"] else mods[genes, "module_ORIEN"]
  mp <- if (cohort == "TCGA") map_T else map_O
  b <- ifelse(m %in% names(mp), mp[m], "unmatched")
  b[is.na(b)] <- "unmatched"
  factor(b, levels = BINS)
}

profile <- function(genes, cohort) {
  as.numeric(table(to_bins(genes, cohort))) / length(genes)
}

agreement <- function(gT, gO) {
  t <- profile(gT, "TCGA"); o <- profile(gO, "ORIEN")
  c(overlap      = sum(pmin(t, o)),
    in_pairs_T   = 1 - t[length(BINS)],
    in_pairs_O   = 1 - o[length(BINS)],
    rho_pairs    = suppressWarnings(cor(t[-length(BINS)], o[-length(BINS)],
                                        method = "spearman")))
}

# --- Main ---------------------------------------------------------------------
rows <- list(); prof_rows <- list()
for (N in SIZES) {
  selT <- head(rank_T, N); selO <- head(rank_O, N)
  shared <- intersect(selT, selO)
  gene_jaccard <- length(shared) / length(union(selT, selO))
  dT <- setdiff(selT, shared); dO <- setdiff(selO, shared)   # DISCORDANT only

  obs <- agreement(dT, dO)

  set.seed(SEED + N)
  null <- replicate(N_PERM, {
    rT <- sample(universe, length(dT)); rO <- sample(universe, length(dO))
    agreement(rT, rO)
  })
  pct <- sapply(names(obs), function(k) mean(null[k, ] < obs[[k]], na.rm = TRUE))
  p_emp <- sapply(names(obs), function(k) mean(null[k, ] >= obs[[k]], na.rm = TRUE))

  cat(sprintf("\n=== N = %d per cohort ===\n", N))
  cat(sprintf("  gene-level: %d shared of %d selected -> Jaccard %.3f\n",
              length(shared), N, gene_jaccard))
  cat(sprintf("  discordant sets: TCGA %d, ORIEN %d\n", length(dT), length(dO)))
  cat(sprintf("  %-12s %8s %11s %9s %8s %8s\n",
              "statistic", "observed", "null median", "null q95", "pctile", "p_emp"))
  for (k in names(obs))
    cat(sprintf("  %-12s %8.3f %11.3f %9.3f %7.1f%% %8.4f\n",
                k, obs[[k]], median(null[k, ], na.rm = TRUE),
                quantile(null[k, ], 0.95, na.rm = TRUE), 100 * pct[[k]], p_emp[[k]]))

  # Programme profiles: where do the discordant genes land?
  tb <- table(to_bins(dT, "TCGA")); ob <- table(to_bins(dO, "ORIEN"))
  cat("  programme profile (counts):\n")
  cat(sprintf("    %-16s %6s %6s\n", "bin", "TCGA", "ORIEN"))
  for (b in BINS) {
    cat(sprintf("    %-16s %6d %6d\n", b, tb[[b]], ob[[b]]))
    prof_rows[[paste(N, b)]] <- data.frame(N = N, bin = b,
                                           n_tcga = tb[[b]], n_orien = ob[[b]])
  }

  rows[[as.character(N)]] <- data.frame(
    N = N, n_shared = length(shared), gene_jaccard = gene_jaccard,
    n_discordant_T = length(dT), n_discordant_O = length(dO),
    t(obs),
    null_median_overlap = median(null["overlap", ]),
    null_q95_overlap = quantile(null["overlap", ], 0.95),
    pctile_overlap = pct[["overlap"]], p_emp_overlap = p_emp[["overlap"]])
}

res <- bind_rows(rows)
write.csv(res, file.path(OUT_DIR, "discordant_module_agreement.csv"), row.names = FALSE)
write.csv(bind_rows(prof_rows), file.path(OUT_DIR, "programme_profiles.csv"),
          row.names = FALSE)

cat("\n=== THE CONTRAST (one line per size) ===\n")
cat(sprintf("  %5s %14s %18s %11s %9s\n",
            "N", "gene Jaccard", "programme overlap", "null median", "p_emp"))
for (i in seq_len(nrow(res)))
  cat(sprintf("  %5d %14.3f %18.3f %11.3f %9.4f\n",
              res$N[i], res$gene_jaccard[i], res$overlap[i],
              res$null_median_overlap[i], res$p_emp_overlap[i]))

cat("\nREADING. 'overlap' is the share of selected genes whose programme is\n")
cat("also implicated by the other cohort. If it sits far above the null, the\n")
cat("cohorts agree on programmes while disagreeing on genes. If it sits at the\n")
cat("null, gene selections carry no shared programme signal. 'in_pairs_*' shows\n")
cat("whether selections concentrate inside matched modules at all.\n")
cat("\nWritten:", OUT_DIR, "\n")
