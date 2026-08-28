# ==============================================================================
# Chapter 5: Somatic-mutation characterization of HUB-SCORE risk groups (R)
#
# R translation of mutation_analysis_hub250.py -- MINIMAL-CHANGE REPLICATION of
# the legacy notebook protocol (Survival_analysis.ipynb cells 6-10): identical
# mutation loading, identical non-silent/LoF/Missense variant classes,
# identical binary-matrix construction, identical statistics. The ONLY protocol
# change from legacy: risk groups come from the 309-gene hub score (K3
# canonical scores_hub_250_{cohort}.csv, median split within cohort).
#
# EXACT-EQUIVALENCE NOTES (so R numbers match the legacy Python outputs):
#   * Odds ratio = sample OR (a*d)/(b*c), as scipy.stats.fisher_exact reports.
#     R's fisher.test() returns the conditional-MLE OR, which DIFFERS -- so the
#     OR is computed by hand and fisher.test() supplies only the exact p-value.
#   * CI: Woolf's method with the 0.5 (Haldane) correction, as in the notebook.
#   * Mantel-Haenszel: the notebook's function translated verbatim -- MH pooled
#     OR with Robins-Breslow-Greenland variance and a WALD test on ln(OR_MH)
#     (NOT the classical MH chi-square, and NOT stats::mantelhaen.test, whose
#     p-value would differ). Reference: Mantel & Haenszel (1959) JNCI.
#   * Wilcoxon: wilcox.test normal approximation; may differ from scipy in the
#     3rd decimal under heavy ties -- immaterial, noted for the record.
#
# PRE-REGISTERED (in plan before running):
#   1. Primary: TP53 non-silent mutation, High vs Low hub-score group; Fisher
#      per cohort + MH pooled. Expectation: replicates in direction (OR > 1).
#   2. Secondary aggregates: TMB (all non-silent per patient) and driver-set
#      burden (TP53, CDKN2A, PIK3CA, NOTCH1, FAT1, CASP8), Wilcoxon per cohort.
#   3. TP53 LoF/Missense/Wildtype composition by risk group (chi-square).
#   Interpretation rule fixed in advance: both scores replicate ->
#   "score-robust"; legacy-only -> reported honestly (hub score captures
#   microenvironment programmes; weaker TP53 link plausible).
#
# LIMITATION carried from legacy for comparability (state in methods):
# no mutation record = wildtype; not-sequenced vs sequenced-WT undistinguished.
#
# Add-on safety: reads artifacts read-only; writes to a NEW directory.
#
# Usage:  Rscript scripts/mutation_analysis_hub250.R
# ==============================================================================

DATA_DIR <- "data/processed"
K3_DIR   <- "results/phaseK3_bootstrap_variance"
OUT_DIR  <- "results/mutation_analysis_hub250"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

PRIMARY_GENE <- "TP53"
DRIVER_GENES <- c("TP53", "CDKN2A", "PIK3CA", "NOTCH1", "FAT1", "CASP8")

# --- Variant classes: byte-identical to the legacy notebook -------------------
LOF_TCGA <- c("Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", "Splice_Site")
MIS_TCGA <- c("Missense_Mutation", "In_Frame_Del", "In_Frame_Ins")
NONSILENT_TCGA <- c(LOF_TCGA, MIS_TCGA, "Nonstop_Mutation", "Translation_Start_Site")

LOF_ORIEN <- c("FRAME_SHIFT_DEL", "FRAME_SHIFT_INS", "NONSENSE", "SPLICE_SITE")
MIS_ORIEN <- c("MISSENSE", "IN_FRAME_DEL", "IN_FRAME_INS")
NONSILENT_ORIEN <- c(LOF_ORIEN, MIS_ORIEN, "NONSTOP")

# ==============================================================================
# Risk groups from the K3 canonical hub scores (the ONE protocol change)
# ==============================================================================

load_risk_groups <- function(cohort) {
  sf <- file.path(K3_DIR, sprintf("scores_hub_250_%s.csv", cohort))
  if (!file.exists(sf))
    stop(sf, " not found -- rerun phaseK3 with the per-patient score save block.")
  sc <- read.csv(sf, colClasses = c(patient_id = "character"))
  med <- median(sc$score)
  setNames(ifelse(sc$score >= med, "High Risk", "Low Risk"), sc$patient_id)
}

# ==============================================================================
# Mutation loading: identical to legacy cells 6-7
# ==============================================================================

load_mut_tcga <- function() {
  m <- read.csv(file.path(DATA_DIR, "TCGA_somatic_mutation.csv"), row.names = 1)
  m$Processed_SampleID <- gsub("-", ".", substr(m$Tumor_Sample_Barcode, 1, 12),
                               fixed = TRUE)
  m <- m[m$Variant_Classification %in% NONSILENT_TCGA, ]
  m$mut_category <- ifelse(m$Variant_Classification %in% LOF_TCGA, "LoF",
                    ifelse(m$Variant_Classification %in% MIS_TCGA, "Missense",
                           "Other"))
  m
}

load_mut_orien <- function() {
  m <- read.delim(file.path(DATA_DIR, "all_mutations_combined.tsv"))
  m <- m[m$variant_class %in% NONSILENT_ORIEN, ]
  m$mut_category <- ifelse(m$variant_class %in% LOF_ORIEN, "LoF",
                    ifelse(m$variant_class %in% MIS_ORIEN, "Missense", "Other"))
  m
}

binary_matrix <- function(mut_df, gene_col, sample_col, all_samples, genes) {
  # Legacy cell 9: one row per patient-gene; absence = 0 (= wildtype)
  dedup <- mut_df[!duplicated(mut_df[, c(gene_col, sample_col)]), ]
  mat <- matrix(0L, nrow = length(all_samples), ncol = length(genes),
                dimnames = list(all_samples, genes))
  dedup <- dedup[dedup[[gene_col]] %in% genes &
                 dedup[[sample_col]] %in% all_samples, ]
  if (nrow(dedup) > 0)
    mat[cbind(dedup[[sample_col]], dedup[[gene_col]])] <- 1L
  mat
}

# ==============================================================================
# Statistics: legacy cells 9-10, translated verbatim
# ==============================================================================

fisher_with_woolf <- function(high_mut, high_total, low_mut, low_total) {
  tbl <- matrix(c(high_mut, high_total - high_mut,
                  low_mut,  low_total - low_mut), nrow = 2, byrow = TRUE)
  # Sample OR (scipy convention), NOT fisher.test's conditional MLE
  or_sample <- (tbl[1, 1] * tbl[2, 2]) / (tbl[1, 2] * tbl[2, 1])
  p <- fisher.test(tbl)$p.value
  a <- high_mut + 0.5; b <- high_total - high_mut + 0.5
  cc <- low_mut + 0.5; d <- low_total - low_mut + 0.5
  log_or <- log(a * d / (b * cc))
  se <- sqrt(1/a + 1/b + 1/cc + 1/d)
  list(or = or_sample, lo = exp(log_or - 1.96 * se),
       hi = exp(log_or + 1.96 * se), p = p)
}

mantel_haenszel <- function(tables) {
  numerator <- 0; denominator <- 0
  for (tbl in tables) {
    a <- tbl[1, 1]; b <- tbl[1, 2]; cc <- tbl[2, 1]; d <- tbl[2, 2]
    n <- a + b + cc + d
    numerator <- numerator + (a * d) / n
    denominator <- denominator + (b * cc) / n
  }
  if (denominator == 0) return(list(or = NA, lo = NA, hi = NA, p = NA))
  or_mh <- numerator / denominator
  P <- 0; Q <- 0; R <- 0; S <- numerator; T_sum <- denominator
  for (tbl in tables) {
    a <- tbl[1, 1]; b <- tbl[1, 2]; cc <- tbl[2, 1]; d <- tbl[2, 2]
    n <- a + b + cc + d
    P <- P + (a + d) * a * d / n^2
    Q <- Q + ((a + d) * b * cc + (b + cc) * a * d) / n^2
    R <- R + (b + cc) * b * cc / n^2
  }
  ln_or <- log(or_mh)
  var_ln <- P / (2 * S^2) + Q / (2 * S * T_sum) + R / (2 * T_sum^2)
  se_ln <- sqrt(var_ln)
  list(or = or_mh,
       lo = exp(ln_or - 1.96 * se_ln), hi = exp(ln_or + 1.96 * se_ln),
       p = 1 - pchisq(ln_or^2 / var_ln, df = 1))
}

# ==============================================================================
# Main
# ==============================================================================

specs <- list(
  TCGA  = list(load = load_mut_tcga,  gcol = "Hugo_Symbol", scol = "Processed_SampleID"),
  ORIEN = list(load = load_mut_orien, gcol = "gene",        scol = "sampleID"))

cohorts <- list()
for (nm in names(specs)) {
  sp <- specs[[nm]]
  rg <- load_risk_groups(nm)
  mut <- sp$load()
  samples <- names(rg)
  n_with_mut <- length(unique(mut[[sp$scol]][mut[[sp$scol]] %in% samples]))
  cat(sprintf("[%s] cohort n=%d (%d high / %d low); patients with >=1 non-silent mutation record: %d\n",
              nm, length(samples), sum(rg == "High Risk"),
              sum(rg == "Low Risk"), n_with_mut))
  cohorts[[nm]] <- list(rg = rg, mut = mut, gcol = sp$gcol, scol = sp$scol,
                        samples = samples,
                        binary = binary_matrix(mut, sp$gcol, sp$scol,
                                               samples, DRIVER_GENES))
}

# ---- (1) Per-gene Fisher + MH pooling: driver genes, TP53 primary ------------
rows <- list(); mh_rows <- list()
cat("\n=== Driver-gene mutation vs hub-score risk group ===\n")
cat(sprintf("%-8s %-7s %10s %10s %7s %15s %9s\n",
            "Gene", "Cohort", "High", "Low", "OR", "95% CI", "p"))
for (gene in DRIVER_GENES) {
  tables <- list()
  for (nm in names(cohorts)) {
    cd <- cohorts[[nm]]
    hi_ids <- names(cd$rg)[cd$rg == "High Risk"]
    lo_ids <- names(cd$rg)[cd$rg == "Low Risk"]
    hm <- sum(cd$binary[hi_ids, gene]); ht <- length(hi_ids)
    lm <- sum(cd$binary[lo_ids, gene]); lt <- length(lo_ids)
    f <- fisher_with_woolf(hm, ht, lm, lt)
    cat(sprintf("%-8s %-7s %6d/%-4d %6d/%-4d %7.2f (%5.2f-%5.2f) %9.4f\n",
                gene, nm, hm, ht, lm, lt, f$or, f$lo, f$hi, f$p))
    rows[[paste(gene, nm)]] <- data.frame(
      gene = gene, cohort = nm, high_mut = hm, high_total = ht,
      low_mut = lm, low_total = lt, odds_ratio = f$or,
      ci_lower = f$lo, ci_upper = f$hi, fisher_p = f$p)
    tables[[nm]] <- matrix(c(hm, ht - hm, lm, lt - lm), nrow = 2, byrow = TRUE)
  }
  m <- mantel_haenszel(tables)
  cat(sprintf("%-8s %-7s %10s %10s %7.2f (%5.2f-%5.2f) %9.4f  [MH]\n",
              gene, "POOLED", "", "", m$or, m$lo, m$hi, m$p))
  mh_rows[[gene]] <- data.frame(gene = gene, or_mh = m$or,
                                ci_lower = m$lo, ci_upper = m$hi, mh_p = m$p)
}
write.csv(do.call(rbind, rows),
          file.path(OUT_DIR, "driver_fisher_by_cohort.csv"), row.names = FALSE)
write.csv(do.call(rbind, mh_rows),
          file.path(OUT_DIR, "driver_mantel_haenszel.csv"), row.names = FALSE)

# ---- (2) Aggregates: TMB and driver-set burden (Wilcoxon) --------------------
agg <- list()
cat("\n=== Aggregate burden by hub-score risk group (Wilcoxon) ===\n")
for (nm in names(cohorts)) {
  cd <- cohorts[[nm]]
  in_cohort <- cd$mut[cd$mut[[cd$scol]] %in% cd$samples, ]
  tmb_tab <- table(factor(in_cohort[[cd$scol]], levels = cd$samples))
  tmb <- as.numeric(tmb_tab)                      # 0 for no-record patients
  names(tmb) <- cd$samples
  drv <- rowSums(cd$binary)
  for (meas in c("TMB_nonsilent", "driver_set_count")) {
    v <- if (meas == "TMB_nonsilent") tmb else drv
    hi <- v[cd$rg == "High Risk"]; lo <- v[cd$rg == "Low Risk"]
    p <- wilcox.test(hi, lo)$p.value
    cat(sprintf("[%s] %-17s median high=%.1f low=%.1f  Wilcoxon p=%.4f\n",
                nm, meas, median(hi), median(lo), p))
    agg[[paste(nm, meas)]] <- data.frame(
      cohort = nm, measure = meas,
      median_high = median(hi), median_low = median(lo),
      mean_high = mean(hi), mean_low = mean(lo), wilcoxon_p = p)
  }
}
write.csv(do.call(rbind, agg),
          file.path(OUT_DIR, "aggregate_burden.csv"), row.names = FALSE)

# ---- (3) TP53 LoF / Missense / Wildtype composition by risk group ------------
comp <- list()
cat("\n=== TP53 severity composition by hub-score risk group ===\n")
for (nm in names(cohorts)) {
  cd <- cohorts[[nm]]
  tp53 <- cd$mut[cd$mut[[cd$gcol]] == PRIMARY_GENE &
                 cd$mut[[cd$scol]] %in% cd$samples, ]
  lof <- unique(tp53[[cd$scol]][tp53$mut_category == "LoF"])
  mis <- setdiff(unique(tp53[[cd$scol]][tp53$mut_category == "Missense"]), lof)
  status <- setNames(rep("Wildtype", length(cd$samples)), cd$samples)
  status[mis] <- "Missense"; status[lof] <- "LoF"
  ct <- table(factor(status, levels = c("LoF", "Missense", "Wildtype")),
              factor(cd$rg[cd$samples],
                     levels = c("High Risk", "Low Risk")))
  p <- chisq.test(ct)$p.value   # 3x2: no continuity correction, as in scipy
  cat(sprintf("[%s] chi2 p=%.4f\n", nm, p)); print(ct); cat("\n")
  for (g in rownames(ct)) for (r in colnames(ct))
    comp[[paste(nm, g, r)]] <- data.frame(
      cohort = nm, tp53_status = g, risk_group = r,
      n = as.integer(ct[g, r]), chi2_p = p)
}
write.csv(do.call(rbind, comp),
          file.path(OUT_DIR, "tp53_severity_composition.csv"), row.names = FALSE)

cat(sprintf("\nWritten: %s/ (driver_fisher_by_cohort.csv, driver_mantel_haenszel.csv, aggregate_burden.csv, tp53_severity_composition.csv)\n",
            OUT_DIR))
