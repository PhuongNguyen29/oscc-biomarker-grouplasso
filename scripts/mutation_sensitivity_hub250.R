# ==============================================================================
# Chapter 5 sensitivity checks for the hub-score mutation analysis
#
# The main analysis (mutation_analysis_hub250.R) carries the legacy convention
# that patients with NO mutation record are wildtype. Two checks quantify what
# that convention could distort:
#
#   CHECK 1 -- BALANCE: is having >=1 non-silent mutation record associated
#     with risk group? (Fisher per cohort.) If record availability is
#     imbalanced, the wildtype convention could bias ORs differentially.
#     PRE-REGISTERED expectation: non-significant.
#
#   CHECK 2 -- RESTRICTED, ALL DRIVERS: repeat the Fisher + MH pooling per gene
#     restricted to patients WITH >=1 mutation record (i.e., wildtype now
#     means sequenced-wildtype). PRE-REGISTERED expectation: ORs move slightly
#     UP (removing patients misclassified as wildtype), conclusion unchanged.
#
# Statistics identical to the main script (sample OR + Woolf CI + exact p;
# hand-rolled MH with RBG variance). Reads the same inputs read-only; appends
# outputs to the same results directory as new files.
#
# Usage:  Rscript scripts/mutation_sensitivity_hub250.R
# ==============================================================================

DATA_DIR <- "data/processed"
K3_DIR   <- "results/phaseK3_bootstrap_variance_k30"
OUT_DIR  <- "results/mutation_analysis_hub250_k30"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

PRIMARY_GENE <- "TP53"
# All drivers share the wildtype-by-default convention, so the restricted
# analysis must be run for every gene that will be reported -- not TP53 alone.
# At k = 30 the ORIEN balance test is significant, and CDKN2A (q = 9e-4) and
# PIK3CA (q = 0.045) both clear BH correction, so all three need it.
DRIVER_GENES <- c("TP53", "CDKN2A", "PIK3CA", "NOTCH1", "FAT1", "CASP8")

LOF_TCGA <- c("Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", "Splice_Site")
MIS_TCGA <- c("Missense_Mutation", "In_Frame_Del", "In_Frame_Ins")
NONSILENT_TCGA <- c(LOF_TCGA, MIS_TCGA, "Nonstop_Mutation", "Translation_Start_Site")
LOF_ORIEN <- c("FRAME_SHIFT_DEL", "FRAME_SHIFT_INS", "NONSENSE", "SPLICE_SITE")
MIS_ORIEN <- c("MISSENSE", "IN_FRAME_DEL", "IN_FRAME_INS")
NONSILENT_ORIEN <- c(LOF_ORIEN, MIS_ORIEN, "NONSTOP")

load_risk_groups <- function(cohort) {
  sc <- read.csv(file.path(K3_DIR, sprintf("scores_hub_250_%s.csv", cohort)),
                 colClasses = c(patient_id = "character"))
  med <- median(sc$score)
  setNames(ifelse(sc$score >= med, "High Risk", "Low Risk"), sc$patient_id)
}

load_mut <- function(cohort) {
  if (cohort == "TCGA") {
    m <- read.csv(file.path(DATA_DIR, "TCGA_somatic_mutation.csv"), row.names = 1)
    m$sid <- gsub("-", ".", substr(m$Tumor_Sample_Barcode, 1, 12), fixed = TRUE)
    m <- m[m$Variant_Classification %in% NONSILENT_TCGA, ]
    m$gene <- m$Hugo_Symbol
  } else {
    m <- read.delim(file.path(DATA_DIR, "all_mutations_combined.tsv"))
    m <- m[m$variant_class %in% NONSILENT_ORIEN, ]
    m$sid <- m$sampleID
  }
  m
}

fisher_with_woolf <- function(hm, ht, lm, lt) {
  tbl <- matrix(c(hm, ht - hm, lm, lt - lm), nrow = 2, byrow = TRUE)
  or_sample <- (tbl[1, 1] * tbl[2, 2]) / (tbl[1, 2] * tbl[2, 1])
  p <- fisher.test(tbl)$p.value
  a <- hm + 0.5; b <- ht - hm + 0.5; cc <- lm + 0.5; d <- lt - lm + 0.5
  log_or <- log(a * d / (b * cc)); se <- sqrt(1/a + 1/b + 1/cc + 1/d)
  list(or = or_sample, lo = exp(log_or - 1.96 * se),
       hi = exp(log_or + 1.96 * se), p = p)
}

mantel_haenszel <- function(tables) {
  num <- 0; den <- 0
  for (tbl in tables) { a <- tbl[1,1]; b <- tbl[1,2]; cc <- tbl[2,1]; d <- tbl[2,2]
    n <- a + b + cc + d; num <- num + a * d / n; den <- den + b * cc / n }
  or_mh <- num / den; P <- 0; Q <- 0; R <- 0; S <- num; T_sum <- den
  for (tbl in tables) { a <- tbl[1,1]; b <- tbl[1,2]; cc <- tbl[2,1]; d <- tbl[2,2]
    n <- a + b + cc + d
    P <- P + (a + d) * a * d / n^2
    Q <- Q + ((a + d) * b * cc + (b + cc) * a * d) / n^2
    R <- R + (b + cc) * b * cc / n^2 }
  ln <- log(or_mh); v <- P / (2 * S^2) + Q / (2 * S * T_sum) + R / (2 * T_sum^2)
  list(or = or_mh, lo = exp(ln - 1.96 * sqrt(v)), hi = exp(ln + 1.96 * sqrt(v)),
       p = 1 - pchisq(ln^2 / v, 1))
}

out <- list()
gene_tables <- list()   # gene_tables[[gene]][[cohort]] = 2x2 restricted table

for (co in c("TCGA", "ORIEN")) {
  rg <- load_risk_groups(co)
  mut <- load_mut(co)
  samples <- names(rg)
  has_rec <- samples %in% unique(mut$sid)

  # ---- CHECK 1: record availability vs risk group (gene-independent) -------
  ct <- table(has_record = has_rec, risk = rg[samples])
  bal <- fisher.test(ct)
  cat(sprintf("[%s] CHECK 1 -- record availability by risk group:\n", co))
  print(ct)
  cat(sprintf("  no-record: high %d/%d (%.1f%%) vs low %d/%d (%.1f%%)  Fisher p = %.4f\n",
      sum(!has_rec & rg[samples] == "High Risk"), sum(rg == "High Risk"),
      100 * mean(!has_rec[rg[samples] == "High Risk"]),
      sum(!has_rec & rg[samples] == "Low Risk"), sum(rg == "Low Risk"),
      100 * mean(!has_rec[rg[samples] == "Low Risk"]), bal$p.value))
  cat(sprintf("  -> %s\n\n",
      if (bal$p.value < 0.05)
        "IMBALANCED: the restricted analysis is PRIMARY for this cohort" else
        "balanced: all-patients analysis remains primary"))
  out[[paste(co, "balance")]] <- data.frame(
    cohort = co, gene = NA_character_, check = "record_availability_balance",
    n_no_record = sum(!has_rec), fisher_p = bal$p.value,
    or = NA, ci_lower = NA, ci_upper = NA,
    high_mut = NA, high_total = NA, low_mut = NA, low_total = NA)

  # ---- CHECK 2: every driver, restricted to patients with >=1 record -------
  seq_samples <- samples[has_rec]
  hi <- seq_samples[rg[seq_samples] == "High Risk"]
  lo <- seq_samples[rg[seq_samples] == "Low Risk"]
  cat(sprintf("[%s] CHECK 2 -- restricted to sequenced (n=%d: %d high / %d low)\n",
              co, length(seq_samples), length(hi), length(lo)))

  for (gene in DRIVER_GENES) {
    mut_pat <- unique(mut$sid[mut$gene == gene & mut$sid %in% seq_samples])
    hm <- sum(hi %in% mut_pat); lm <- sum(lo %in% mut_pat)
    f <- fisher_with_woolf(hm, length(hi), lm, length(lo))
    cat(sprintf("  %-8s %4d/%-4d vs %4d/%-4d  OR %5.2f (%4.2f-%5.2f)  p=%.4g\n",
                gene, hm, length(hi), lm, length(lo), f$or, f$lo, f$hi, f$p))
    out[[paste(co, gene)]] <- data.frame(
      cohort = co, gene = gene, check = "restricted_to_sequenced",
      n_no_record = sum(!has_rec), fisher_p = f$p,
      or = f$or, ci_lower = f$lo, ci_upper = f$hi,
      high_mut = hm, high_total = length(hi),
      low_mut = lm, low_total = length(lo))
    gene_tables[[gene]][[co]] <- matrix(
      c(hm, length(hi) - hm, lm, length(lo) - lm), nrow = 2, byrow = TRUE)
  }
  cat("\n")
}

# ---- MH pooling per gene, with BH across the six -----------------------------
cat("POOLED (Mantel-Haenszel), restricted to sequenced:\n")
pooled <- list()
for (gene in DRIVER_GENES) {
  m <- mantel_haenszel(gene_tables[[gene]])
  pooled[[gene]] <- data.frame(gene = gene, or_mh = m$or,
                               ci_lower = m$lo, ci_upper = m$hi, mh_p = m$p)
}
pooled <- do.call(rbind, pooled)
pooled$q_bh <- p.adjust(pooled$mh_p, method = "BH")
pooled <- pooled[order(pooled$mh_p), ]
for (i in seq_len(nrow(pooled))) {
  r <- pooled[i, ]
  cat(sprintf("  %-8s OR %5.2f (%4.2f-%5.2f)  p=%.4g  q=%.4g%s\n",
              r$gene, r$or_mh, r$ci_lower, r$ci_upper, r$mh_p, r$q_bh,
              if (r$q_bh < 0.05) " *" else ""))
}
write.csv(pooled, file.path(OUT_DIR, "sensitivity_pooled_by_gene.csv"),
          row.names = FALSE)

for (gene in DRIVER_GENES) {
  r <- pooled[pooled$gene == gene, ]
  out[[paste("pooled", gene)]] <- data.frame(
    cohort = "POOLED_MH", gene = gene, check = "restricted_to_sequenced",
    n_no_record = NA, fisher_p = r$mh_p, or = r$or_mh,
    ci_lower = r$ci_lower, ci_upper = r$ci_upper,
    high_mut = NA, high_total = NA, low_mut = NA, low_total = NA)
}

write.csv(do.call(rbind, out),
          file.path(OUT_DIR, "sensitivity_checks.csv"), row.names = FALSE)
cat(sprintf("\nWritten: %s/sensitivity_checks.csv\n", OUT_DIR))
