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
#   CHECK 2 -- RESTRICTED TP53: repeat the primary TP53 Fisher + MH pooling
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

out <- list(); tp53_tables <- list()

for (co in c("TCGA", "ORIEN")) {
  rg <- load_risk_groups(co)
  mut <- load_mut(co)
  samples <- names(rg)
  has_rec <- samples %in% unique(mut$sid)

  # ---- CHECK 1: record availability vs risk group ---------------------------
  ct <- table(has_record = has_rec, risk = rg[samples])
  bal <- fisher.test(ct)
  cat(sprintf("[%s] CHECK 1 -- record availability by risk group:\n", co))
  print(ct)
  cat(sprintf("  no-record: high %d/%d (%.1f%%) vs low %d/%d (%.1f%%)  Fisher p = %.4f\n\n",
      sum(!has_rec & rg[samples] == "High Risk"), sum(rg == "High Risk"),
      100 * mean(!has_rec[rg[samples] == "High Risk"]),
      sum(!has_rec & rg[samples] == "Low Risk"), sum(rg == "Low Risk"),
      100 * mean(!has_rec[rg[samples] == "Low Risk"]), bal$p.value))
  out[[paste(co, "balance")]] <- data.frame(
    cohort = co, check = "record_availability_balance",
    n_no_record = sum(!has_rec), fisher_p = bal$p.value,
    or = NA, ci_lower = NA, ci_upper = NA,
    high_mut = NA, high_total = NA, low_mut = NA, low_total = NA)

  # ---- CHECK 2: TP53 restricted to patients with >=1 record -----------------
  seq_samples <- samples[has_rec]
  tp53_pat <- unique(mut$sid[mut$gene == PRIMARY_GENE & mut$sid %in% seq_samples])
  hi <- seq_samples[rg[seq_samples] == "High Risk"]
  lo <- seq_samples[rg[seq_samples] == "Low Risk"]
  hm <- sum(hi %in% tp53_pat); lm <- sum(lo %in% tp53_pat)
  f <- fisher_with_woolf(hm, length(hi), lm, length(lo))
  cat(sprintf("[%s] CHECK 2 -- TP53 restricted to sequenced (n=%d): %d/%d vs %d/%d  OR %.2f (%.2f-%.2f) p=%.4g\n\n",
      co, length(seq_samples), hm, length(hi), lm, length(lo),
      f$or, f$lo, f$hi, f$p))
  out[[paste(co, "tp53")]] <- data.frame(
    cohort = co, check = "tp53_restricted_to_sequenced",
    n_no_record = sum(!has_rec), fisher_p = f$p,
    or = f$or, ci_lower = f$lo, ci_upper = f$hi,
    high_mut = hm, high_total = length(hi),
    low_mut = lm, low_total = length(lo))
  tp53_tables[[co]] <- matrix(c(hm, length(hi) - hm, lm, length(lo) - lm),
                              nrow = 2, byrow = TRUE)
}

m <- mantel_haenszel(tp53_tables)
cat(sprintf("POOLED  CHECK 2 -- TP53 restricted (MH): OR %.2f (%.2f-%.2f) p=%.4g\n",
            m$or, m$lo, m$hi, m$p))
out[["pooled tp53"]] <- data.frame(
  cohort = "POOLED_MH", check = "tp53_restricted_to_sequenced",
  n_no_record = NA, fisher_p = m$p, or = m$or,
  ci_lower = m$lo, ci_upper = m$hi,
  high_mut = NA, high_total = NA, low_mut = NA, low_total = NA)

write.csv(do.call(rbind, out),
          file.path(OUT_DIR, "sensitivity_checks.csv"), row.names = FALSE)
cat(sprintf("\nWritten: %s/sensitivity_checks.csv\n", OUT_DIR))
