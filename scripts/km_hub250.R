# ==============================================================================
# Chapter 5: Kaplan-Meier stratification by the 309-gene hub score (hub_250)
#
# Reads the CANONICAL per-patient scores saved by phaseK3_bootstrap_variance.R
# (scores_hub_250_{cohort}.csv), so this figure describes the IDENTICAL score
# whose bootstrap C-index CIs are reported in K3. No score is recomputed.
# Survival is read from the SAME files, with the SAME sampleID matching and
# time > 0 filter, as K3 -- so n and events per cohort must match the K3 log
# exactly (regression check: TCGA 339/153, ORIEN 1112/450, GSE41613 97/51,
# GSE42743 71/40).
#
# Design (mirrors legacy km_stratification.py where applicable):
#   - Median split WITHIN each cohort; score >= median -> High Risk.
#   - Log-rank test High vs Low; 2x2 panel.
#
# Protocol difference from the legacy 25-gene KM, disclosed: the hub score is
# fit ONCE in ORIEN (ridge, fixed coefficients), so the ORIEN panel is
# IN-SAMPLE and labeled as such; TCGA and both GEO panels are transfer.
#
# Pre-registered expectations (in plan before running): significant
# stratification in GEO cohorts (hub_250 CIs clear the age floor); weak or
# absent in TCGA (floor-level C). Either outcome is reported.
#
# Add-on safety: reads existing artifacts read-only; writes to a NEW directory.
#
# Usage:  Rscript scripts/km_hub250.R
# ==============================================================================

suppressPackageStartupMessages({
  library(survival)
  if (!requireNamespace("survminer", quietly = TRUE))
    stop("survminer is required: install.packages('survminer')")
  library(survminer)
})

K3_DIR   <- "results/phaseK3_bootstrap_variance"
PROC_DIR <- "data/processed"
EXT_DIR  <- "data/external"
OUT_DIR  <- "results/km_hub250"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Same reader as phaseK3 -- byte-identical conventions
read_surv <- function(p) {
  s <- read.csv(p); if (!"sampleID" %in% names(s)) names(s)[1] <- "sampleID"
  s[, c("sampleID", "time", "event")]
}

SURV_FILES <- list(
  TCGA     = file.path(PROC_DIR, "surv_tcga_aligned_v4.csv"),
  ORIEN    = file.path(PROC_DIR, "surv_orien_aligned_v4.csv"),
  GSE41613 = file.path(EXT_DIR,  "GSE41613_survival.csv"),
  GSE42743 = file.path(EXT_DIR,  "GSE42743_survival.csv"))
IN_SAMPLE <- c(TCGA = FALSE, ORIEN = TRUE, GSE41613 = FALSE, GSE42743 = FALSE)

# Expected n/events from the K3 log -- hard regression gate
EXPECTED <- list(TCGA = c(339, 153), ORIEN = c(1112, 450),
                 GSE41613 = c(97, 51), GSE42743 = c(71, 40))

to_months <- function(t) if (max(t, na.rm = TRUE) > 300) t / 30.44 else t

plots <- list(); rows <- list()

for (co in names(SURV_FILES)) {
  sf <- file.path(K3_DIR, sprintf("scores_hub_250_%s.csv", co))
  if (!file.exists(sf))
    stop(sf, " not found -- rerun phaseK3 with the per-patient score save ",
         "block first.")
  sc <- read.csv(sf, colClasses = c(patient_id = "character"))
  sv <- read_surv(SURV_FILES[[co]])

  d <- merge(sc, sv, by.x = "patient_id", by.y = "sampleID")
  d <- d[d$time > 0 & !is.na(d$time) & !is.na(d$event), ]

  n_lost <- nrow(sc) - nrow(d)
  if (n_lost > 0)
    warning(sprintf("[%s] %d scored patients dropped in survival join -- %s",
                    co, n_lost, "investigate before publishing this panel."))

  # Regression gate: must reproduce K3's cohort composition exactly
  stopifnot(nrow(d) == EXPECTED[[co]][1], sum(d$event) == EXPECTED[[co]][2])

  d$time_months <- to_months(d$time)
  med <- median(d$score)
  d$risk_group <- factor(ifelse(d$score >= med, "High Risk", "Low Risk"),
                         levels = c("Low Risk", "High Risk"))

  fit <- survfit(Surv(time_months, event) ~ risk_group, data = d)
  lr  <- survdiff(Surv(time_months, event) ~ risk_group, data = d)
  p   <- pchisq(lr$chisq, df = 1, lower.tail = FALSE)

  ttl <- sprintf("%s \u2014 309-gene hub score%s", co,
                 if (IN_SAMPLE[co]) "  (source cohort, in-sample)" else "")

  plots[[co]] <- ggsurvplot(
    fit, data = d,
    palette = c("#2E86DE", "#E74C3C"),
    conf.int = TRUE, risk.table = FALSE, censor.size = 3,
    legend.labs = sprintf("%s (n=%d, %d ev)",
                          levels(d$risk_group),
                          table(d$risk_group),
                          tapply(d$event, d$risk_group, sum)),
    legend.title = "", title = ttl,
    xlab = "Months", ylab = "Survival probability",
    pval = sprintf("log-rank p = %.4g", p), pval.coord = c(0, 0.05),
    ggtheme = theme_classic(base_size = 11))

  rows[[co]] <- data.frame(
    cohort = co, n = nrow(d), events = sum(d$event),
    n_high = sum(d$risk_group == "High Risk"),
    n_low  = sum(d$risk_group == "Low Risk"),
    logrank_p = p, median_score = med, in_sample = IN_SAMPLE[co],
    median_surv_low  = summary(fit)$table["risk_group=Low Risk",  "median"],
    median_surv_high = summary(fit)$table["risk_group=High Risk", "median"])

  cat(sprintf("[%s] n=%d, events=%d, log-rank p = %.4g%s\n",
              co, nrow(d), sum(d$event), p,
              if (IN_SAMPLE[co]) "  [in-sample]" else ""))
}

panel <- arrange_ggsurvplots(plots, print = FALSE, ncol = 2, nrow = 2)
ggsave(file.path(OUT_DIR, "km_hub250_2x2.png"), panel,
       width = 13, height = 10, dpi = 300)
ggsave(file.path(OUT_DIR, "km_hub250_2x2.pdf"), panel, width = 13, height = 10)

write.csv(do.call(rbind, rows),
          file.path(OUT_DIR, "km_hub250_summary.csv"), row.names = FALSE)
cat("\nWritten:", OUT_DIR, "-> km_hub250_2x2.png/.pdf, km_hub250_summary.csv\n")
