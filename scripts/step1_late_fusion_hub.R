# ==============================================================================
# Chapter 4 (Section 4.3.3 addendum): incremental value of the hub score over
# clinical variables in the external cohorts.  [R version]
#
# MINIMAL-CHANGE REPLICATION of the legacy late-fusion second stage
# (step1_late_fusion.py): identical design -- unpenalized Cox on standardized
# scores, likelihood-ratio test of combined vs single-score models -- with ONE
# change: the gene score is the 309-gene hub score (K3 canonical, coefficients
# fixed from ORIEN; a deterministic transfer, so no first-stage OOF machinery
# or gene-side seeds are needed).
#
# The CLINICAL side is byte-identical to the legacy run: it reuses the Step 0B
# out-of-fold clinical ridge scores from results/chapter3/step0b/
# ({dataset}_clinical_oof_scores.csv), whose encoded covariates are
#   GSE42743 (5): gender, age_zscore, smoking, T_stage, N_stage
#   GSE41613 (3): gender, tumor_stage, age_binary
# Seeds: 42 primary (as in legacy), 123 and 456 as sensitivity.
#
# EQUIVALENCE NOTES (R vs the legacy lifelines run):
#   * survival::coxph uses Efron tie handling by default; lifelines' CoxPHFitter
#     uses Breslow. With few tied event times the difference is negligible, but
#     it is a real difference -- set ties = "breslow" below to reproduce the
#     legacy numbers exactly. Default here is "efron" (the better estimator);
#     TIES is a single switch so the choice is explicit and documented.
#   * LRT is computed by hand from the partial log-likelihoods (2*(llf - llr),
#     1 df), identical arithmetic to the legacy script.
#   * concordance: summary(fit)$concordance[1] matches lifelines'
#     concordance_index_ for a fitted Cox model.
#
# PRE-REGISTERED (record in plan before running): the hub score is expected to
# add significant value over the clinical model in both GEO cohorts, mirroring
# the legacy 25-gene result (LRT p = 0.0004 / 0.013); either outcome reported.
# Both directions are tested (does clinical add to the hub score?) because a
# two-sided complementarity finding is the legacy conclusion.
#
# Scope: GEO-primary by design (as in legacy). No development-cohort
# incremental-value claim; ORIEN clinical data not used.
#
# Usage:  Rscript scripts/step1_late_fusion_hub.R
# ==============================================================================

suppressPackageStartupMessages(library(survival))

OOF_DIR <- "results.1/chapter3/step0b"          # legacy Step 0B outputs
K3_DIR  <- "results/phaseK3_bootstrap_variance"
OUT_DIR <- "results/late_fusion_hub"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

DATASETS <- c("GSE41613", "GSE42743")
CLINICAL_SEEDS <- c(42, 123, 456)
PRIMARY_SEED <- 42
TIES <- "efron"    # set to "breslow" to reproduce the legacy lifelines fits

load_clinical_oof <- function(dataset, seed) {
  f <- file.path(OOF_DIR, sprintf("%s_clinical_oof_scores.csv", dataset))
  if (!file.exists(f))
    stop(f, " not found -- point OOF_DIR at the legacy Step 0B outputs.")
  d <- read.csv(f, colClasses = c(sampleID = "character"))
  col <- sprintf("risk_score_seed%d", seed)
  if (!col %in% names(d))
    stop(f, ": column ", col, " not found; have: ",
         paste(names(d), collapse = ", "))
  data.frame(sampleID = d$sampleID, clinical_score = d[[col]],
             event = d$event, time = d$time)
}

load_hub_score <- function(dataset) {
  f <- file.path(K3_DIR, sprintf("scores_hub_250_%s.csv", dataset))
  if (!file.exists(f))
    stop(f, " not found -- rerun phaseK3 with the per-patient score save block.")
  d <- read.csv(f, colClasses = c(patient_id = "character"))
  data.frame(sampleID = d$patient_id, hub_score = d$score)
}

lrt <- function(ll_full, ll_reduced, df_diff = 1) {
  stat <- 2 * (ll_full - ll_reduced)
  c(stat = stat, p = pchisq(stat, df = df_diff, lower.tail = FALSE))
}

rows <- list()
for (ds in DATASETS) {
  hub <- load_hub_score(ds)
  for (seed in CLINICAL_SEEDS) {
    clin <- load_clinical_oof(ds, seed)
    d <- merge(clin, hub, by = "sampleID")
    stopifnot(nrow(d) == nrow(clin), nrow(d) == nrow(hub))

    # Standardize within cohort (comparable betas; the LRT is scale-invariant)
    d$clinical_score <- as.numeric(scale(d$clinical_score))
    d$hub_score      <- as.numeric(scale(d$hub_score))

    S <- Surv(d$time, d$event)
    m_clin <- coxph(S ~ clinical_score, data = d, ties = TIES)
    m_hub  <- coxph(S ~ hub_score,      data = d, ties = TIES)
    m_both <- coxph(S ~ clinical_score + hub_score, data = d, ties = TIES)

    a <- lrt(m_both$loglik[2], m_clin$loglik[2])   # does hub add to clinical?
    b <- lrt(m_both$loglik[2], m_hub$loglik[2])    # does clinical add to hub?
    cf <- summary(m_both)$coefficients

    rows[[paste(ds, seed)]] <- data.frame(
      dataset = ds, clinical_seed = seed,
      n = nrow(d), events = sum(d$event),
      c_clinical = summary(m_clin)$concordance[1],
      c_hub      = summary(m_hub)$concordance[1],
      c_combined = summary(m_both)$concordance[1],
      beta_hub = cf["hub_score", "coef"],
      se_hub   = cf["hub_score", "se(coef)"],
      beta_clinical = cf["clinical_score", "coef"],
      lrt_hub_over_clinical = a[["stat"]], p_hub_over_clinical = a[["p"]],
      lrt_clinical_over_hub = b[["stat"]], p_clinical_over_hub = b[["p"]],
      ties = TIES)

    cat(sprintf("[%s | clinical seed %d%s] n=%d, ev=%d\n", ds, seed,
                if (seed == PRIMARY_SEED) " (PRIMARY)" else "",
                nrow(d), sum(d$event)))
    cat(sprintf("  C: clinical=%.3f  hub=%.3f  combined=%.3f\n",
                summary(m_clin)$concordance[1], summary(m_hub)$concordance[1],
                summary(m_both)$concordance[1]))
    cat(sprintf("  Does the HUB score add to clinical?   LRT=%6.2f  p=%.4g\n",
                a[["stat"]], a[["p"]]))
    cat(sprintf("  Does CLINICAL add to the hub score?   LRT=%6.2f  p=%.4g\n\n",
                b[["stat"]], b[["p"]]))
  }
}

out <- do.call(rbind, rows)
write.csv(out, file.path(OUT_DIR, "late_fusion_hub_summary.csv"),
          row.names = FALSE)
cat(sprintf("Written: %s/late_fusion_hub_summary.csv\n", OUT_DIR))
