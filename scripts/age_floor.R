# ==============================================================================
# Age-alone clinical floor: per-cohort computation with bootstrap intervals
#
# WHY THIS EXISTS. The floor 0.551-0.565 is hard-coded in phaseK3
# (AGE_LO/AGE_HI), phaseK1_6_7 (as a STRING) and phaseG, all citing "Rung 1"
# with no reproducible script. It is the benchmark for the chapter's headline
# claim ("only the hub score clears the clinical floor in both external
# cohorts"), so it must be computable from a named script and stored as a
# canonical artifact. This script recomputes it per cohort.
#
# DEFINITION (state verbatim in Methods 4.2.6): the age floor for a cohort is
# the concordance of a UNIVARIATE Cox model on age alone, fitted and evaluated
# within that cohort. It is an in-sample benchmark by construction -- that is
# deliberate and conservative: it gives age every advantage, so a genomic score
# clearing it is not clearing a handicapped comparator.
#
# COHORT CONSTRUCTION is identical to phaseK3 (same survival files, same
# sampleID matching, same time > 0 filter), so the floor applies to exactly the
# patients whose C-indices appear in the forest plot. A hard gate asserts the
# K3 composition (339/153, 1112/450, 97/51, 71/40) before any patient is
# dropped for missing age; any further loss is reported explicitly, since a
# floor computed on a subset is not a floor for the full cohort.
#
# AGE CODING DIFFERS BY COHORT and this is a real limitation to state:
#   TCGA, ORIEN   -- continuous age at diagnosis
#   GSE42743      -- continuous age (legacy Step 0A used it z-scored)
#   GSE41613      -- BINNED age only; the legacy encoding dichotomised at 60.
#                    Its floor is therefore a DICHOTOMISED-age floor and is
#                    expected to be lower than a continuous-age floor would be.
# The script reports the coding used per cohort so the table is self-documenting.
#
# OUTPUT: results/age_floor/age_floor.csv  (canonical artifact)
#   cohort, n, events, age_coding, c_index, ci_lo, ci_hi, n_missing_age
# Downstream: replace hard-coded AGE_LO/AGE_HI with a read of this file, and
# draw PER-FACET floor lines in Figure 4.3 instead of one shared band.
#
# Usage:  Rscript scripts/age_floor.R
# ==============================================================================

suppressPackageStartupMessages(library(survival))

PROC_DIR <- "data/processed"
EXT_DIR  <- "data/external"
OUT_DIR  <- "results/age_floor"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

N_BOOT <- 2000      # matches phaseK3
SEED   <- 42        # matches phaseK3
set.seed(SEED)

# Same reader as phaseK3
read_surv <- function(p) {
  s <- read.csv(p); if (!"sampleID" %in% names(s)) names(s)[1] <- "sampleID"
  s[, c("sampleID", "time", "event")]
}

SURV_FILES <- list(
  TCGA     = file.path(PROC_DIR, "surv_tcga_aligned_v4.csv"),
  ORIEN    = file.path(PROC_DIR, "surv_orien_aligned_v4.csv"),
  GSE41613 = file.path(EXT_DIR,  "GSE41613_survival.csv"),
  GSE42743 = file.path(EXT_DIR,  "GSE42743_survival.csv"))

# EDIT IF NEEDED: files holding age per cohort, and the age column name.
# TCGA/ORIEN: the aligned clinical tables (memory: clinical_{cohort}_aligned_v4.csv).
# GEO: the legacy Step 0A ENCODED files already carry the coding used in Ch3.
AGE_SOURCES <- list(
  TCGA     = list(file = file.path(PROC_DIR, "clinical_tcga_aligned_v4.csv"),
                  candidates = c("age"),
                  coding = "continuous"),
  ORIEN    = list(file = file.path(PROC_DIR, "clinical_orien_aligned_v4.csv"),
                  candidates = c("age"),
                  coding = "continuous"),
  GSE41613 = list(file = file.path(EXT_DIR, "GSE41613_clinical_encoded.csv"),
                  candidates = c("age_binary"),
                  coding = "binary (<60 vs >=60; binned source)"),
  GSE42743 = list(file = file.path(EXT_DIR, "GSE42743_clinical_encoded.csv"),
                  candidates = c("age_zscore", "age"),
                  coding = "continuous (z-scored)"))

# phaseK3 cohort composition -- hard gate before any age-related filtering
EXPECTED <- list(TCGA = c(339, 153), ORIEN = c(1112, 450),
                 GSE41613 = c(97, 51), GSE42743 = c(71, 40))

cidx <- function(risk, y) {
  if (sd(risk) < 1e-12 || sum(y[, "status"]) < 2) return(NA_real_)
  as.numeric(concordance(y ~ risk, reverse = TRUE)$concordance)
}

rows <- list()
for (co in names(SURV_FILES)) {
  sv <- read_surv(SURV_FILES[[co]])
  sv <- sv[sv$time > 0 & !is.na(sv$time) & !is.na(sv$event), ]
  sv$event <- if (is.character(sv$event) || is.factor(sv$event))
    as.integer(as.character(sv$event) %in% c("TRUE", "True", "true", "1")) else
    as.integer(sv$event)

  # Gate: must reproduce the phaseK3 cohort BEFORE age filtering
  stopifnot(nrow(sv) == EXPECTED[[co]][1], sum(sv$event) == EXPECTED[[co]][2])

  src <- AGE_SOURCES[[co]]
  if (!file.exists(src$file))
    stop(co, ": age source not found: ", src$file,
         " -- edit AGE_SOURCES to the correct clinical table.")
  cl <- read.csv(src$file)
  if (!"sampleID" %in% names(cl)) names(cl)[1] <- "sampleID"
  acol <- intersect(src$candidates, names(cl))
  if (!length(acol))
    stop(co, ": no age column among (", paste(src$candidates, collapse = ", "),
         "); have: ", paste(names(cl), collapse = ", "))
  acol <- acol[1]

  d <- merge(sv, cl[, c("sampleID", acol)], by = "sampleID")
  names(d)[names(d) == acol] <- "age"
  n_before <- nrow(d)
  d <- d[!is.na(d$age), ]
  n_missing <- n_before - nrow(d)
  if (n_missing > 0)
    warning(sprintf("[%s] %d patients dropped for missing age -- the floor is computed on a subset.",
                    co, n_missing))

  y <- Surv(d$time, d$event)
  fit <- coxph(y ~ age, data = d)
  risk <- as.numeric(predict(fit, type = "lp"))
  point <- cidx(risk, y)

  # Bootstrap over patients, model FIXED (same convention as phaseK3)
  bs <- replicate(N_BOOT, {
    i <- sample(nrow(d), replace = TRUE)
    cidx(risk[i], Surv(d$time[i], d$event[i]))
  })
  ci <- quantile(bs, c(0.025, 0.975), na.rm = TRUE)

  cat(sprintf("[%-8s] n=%4d ev=%3d  age: %-28s  C = %.3f [%.3f, %.3f]%s\n",
              co, nrow(d), sum(d$event), src$coding, point, ci[1], ci[2],
              if (n_missing > 0) sprintf("  (%d missing age)", n_missing) else ""))

  rows[[co]] <- data.frame(
    cohort = co, n = nrow(d), events = sum(d$event),
    age_column = acol, age_coding = src$coding,
    c_index = point, ci_lo = as.numeric(ci[1]), ci_hi = as.numeric(ci[2]),
    n_missing_age = n_missing, n_boot = N_BOOT, seed = SEED)
}

out <- do.call(rbind, rows)
write.csv(out, file.path(OUT_DIR, "age_floor.csv"), row.names = FALSE)

cat(sprintf("\nRange of per-cohort age floors: %.3f - %.3f\n",
            min(out$c_index), max(out$c_index)))
cat("Compare with the inherited hard-coded constants 0.551-0.565 (Rung 1).\n")
cat("If these differ materially, the recomputed values supersede them and every\n")
cat("script with hard-coded AGE_LO/AGE_HI must be repointed at age_floor.csv.\n")
cat(sprintf("\nWritten: %s/age_floor.csv\n", OUT_DIR))
