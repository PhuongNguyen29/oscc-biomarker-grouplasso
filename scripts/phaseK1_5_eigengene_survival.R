#!/usr/bin/env Rscript
# ==============================================================================
# Phase K1.5: Eigengene-survival association on preserved modules
# ==============================================================================
#
# K1 established that co-expression modules transport across all four cohorts
# while gene-level selection does not. This asks the next question: does the
# transported structure carry prognostic signal?
#
# ORDER MATTERS (K1.1). This runs AFTER preservation is complete. Preservation
# is unsupervised and is not revised in light of anything here. Filtering to
# preserved modules introduces no outcome leakage because the preservation
# criterion never saw survival.
#
# THE SIGN TRAP (K1.5.3)
# ----------------------
# moduleEigengenes returns the first principal component, whose SIGN IS
# ARBITRARY. The same module can produce HR 1.5 in one cohort and 0.67 in
# another purely from a sign flip, with no biological difference. Since
# cross-cohort direction agreement is the quantity of interest, this would
# invalidate the entire analysis.
#
# align = "along average" orients each eigengene to correlate positively with
# the mean expression of its member genes -- a definition that is well-posed in
# every cohort independently, so the orientation is comparable across cohorts
# without ever comparing samples.
#
# WHAT TRANSPORTS
# ---------------
# The MODULE (a gene list from the reference cohort) is the transportable
# object. The eigengene is recomputed from that gene list within each cohort.
# We do not transport eigengene values or re-detect modules per cohort.
#
# TIME UNITS
# ----------
# GSE42743 is in DAYS; the others in MONTHS. Cox HRs and the concordance index
# are invariant to monotone rescaling of time, so this analysis is unaffected.
# Anything at a fixed horizon would not be.
#
# References:
#   Langfelder P, Horvath S (2008) BMC Bioinformatics 9:559 -- WGCNA, eigengenes
#   Cox DR (1972) JRSS-B 34:187-220
#   Benjamini Y, Hochberg Y (1995) JRSS-B 57:289-300
#
# Usage:
#   Rscript scripts/phaseK1_5_eigengene_survival.R
# ==============================================================================

suppressPackageStartupMessages({
  library(WGCNA)
  library(survival)
  library(dplyr)
})
options(stringsAsFactors = FALSE)

PROC_DIR <- "data/processed"
EXT_DIR  <- "data/external"
K1_DIR   <- "results/phaseK1_module_preservation"
OUT_DIR  <- "results/phaseK1_5_eigengene_survival"

REF_COHORTS <- c("TCGA", "ORIEN")
ALL_COHORTS <- c("TCGA", "ORIEN", "GSE41613", "GSE42743")
SOURCE_COHORT <- "ORIEN"   # K1.5.5: n=1112, 450 events, marginal SNR 2.14
AGE_FLOOR <- "0.551-0.565" # Rung 1 age-alone C-index

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
LOG_DIR <- "logs"; dir.create(LOG_DIR, showWarnings = FALSE, recursive = TRUE)
log_file <- file.path(LOG_DIR,
  sprintf("phaseK1_5_eigengene_survival_%s.log", format(Sys.time(), "%Y%m%d_%H%M%S")))
log_con <- file(log_file, open = "wt")
sink(log_con, split = TRUE); sink(log_con, type = "message")

cat("Phase K1.5: Eigengene-survival on preserved modules\n")
cat("Log:", log_file, "\n")
cat("Started:", format(Sys.time()), "\n\n")

# ==============================================================================
# Which modules qualify (K1.5.1)
# ==============================================================================

pres <- read.csv(file.path(K1_DIR, "preservation_statistics.csv"))
mods <- read.csv(file.path(K1_DIR, "module_assignments.csv"))

cat("--- Qualifying modules (above null control in ALL test cohorts) ---\n")
qualifying <- pres %>%
  filter(!is_control) %>%
  group_by(reference, module) %>%
  summarise(n_tests = n(), n_above = sum(above_null, na.rm = TRUE),
            size = first(moduleSize), minZ = min(Zsummary),
            maxRank = max(medianRank), .groups = "drop") %>%
  filter(n_above == n_tests)

for (r in REF_COHORTS) {
  q <- qualifying[qualifying$reference == r, ]
  cat("  ", r, ":", nrow(q), "modules\n")
  if (nrow(q)) print(as.data.frame(q[, c("module","size","minZ","maxRank")]),
                     row.names = FALSE)
}
cat("\n")
if (nrow(qualifying) == 0) {
  stop("No modules preserved in all test cohorts. Report this as the K1.5 result.")
}

# ==============================================================================
# Load expression and survival
# ==============================================================================

cat("--- Loading data ---\n")
load_expr <- function(p) as.matrix(read.csv(p, row.names = 1, check.names = FALSE))

expr <- list(
  TCGA     = load_expr(file.path(PROC_DIR, "tcga_preprocessed_v4.csv")),
  ORIEN    = load_expr(file.path(PROC_DIR, "orien_preprocessed_v4.csv")),
  GSE41613 = load_expr(file.path(EXT_DIR,  "GSE41613_expression.csv")),
  GSE42743 = load_expr(file.path(EXT_DIR,  "GSE42743_expression.csv"))
)

read_surv <- function(p) {
  s <- read.csv(p)
  if (!"sampleID" %in% names(s)) names(s)[1] <- "sampleID"
  s[, c("sampleID", "time", "event")]
}
surv <- list(
  TCGA     = read_surv(file.path(PROC_DIR, "surv_tcga_aligned_v4.csv")),
  ORIEN    = read_surv(file.path(PROC_DIR, "surv_orien_aligned_v4.csv")),
  GSE41613 = read_surv(file.path(EXT_DIR,  "GSE41613_survival.csv")),
  GSE42743 = read_surv(file.path(EXT_DIR,  "GSE42743_survival.csv"))
)

for (nm in ALL_COHORTS) {
  keep <- intersect(colnames(expr[[nm]]), surv[[nm]]$sampleID)
  expr[[nm]] <- expr[[nm]][, keep, drop = FALSE]
  surv[[nm]] <- surv[[nm]][match(keep, surv[[nm]]$sampleID), ]
  stopifnot(identical(colnames(expr[[nm]]), surv[[nm]]$sampleID))
  cat(sprintf("  %-10s %d samples, %d events\n", nm, length(keep),
              sum(surv[[nm]]$event)))
}
cat("\n")

# ==============================================================================
# Eigengenes (K1.5.2, K1.5.3)
# ==============================================================================

cat("--- Computing eigengenes (align = 'along average') ---\n")

compute_MEs <- function(ref) {
  colors_all <- mods[[paste0("module_", ref)]]
  genes_all  <- mods$gene_id
  keep_mod   <- qualifying$module[qualifying$reference == ref]
  sel        <- colors_all %in% keep_mod
  genes      <- genes_all[sel]
  colors     <- colors_all[sel]

  out <- list()
  for (nm in ALL_COHORTS) {
    g <- intersect(genes, rownames(expr[[nm]]))
    idx <- match(g, genes)
    # t() -> samples x genes, which moduleEigengenes expects
    me <- moduleEigengenes(t(expr[[nm]][g, , drop = FALSE]), colors[idx],
                           align = "along average", verbose = 0)
    E <- me$eigengenes
    rownames(E) <- colnames(expr[[nm]])
    out[[nm]] <- E
    cat(sprintf("  %-6s -> %-10s %d genes, %d eigengenes, varExpl %.2f-%.2f\n",
                ref, nm, length(g), ncol(E),
                min(me$varExplained), max(me$varExplained)))
  }
  out
}

MEs <- lapply(setNames(REF_COHORTS, REF_COHORTS), compute_MEs)
cat("\n")

for (ref in REF_COHORTS) for (nm in ALL_COHORTS) {
  write.csv(MEs[[ref]][[nm]],
            file.path(OUT_DIR, sprintf("eigengenes_ref%s_%s.csv", ref, nm)))
}

# ==============================================================================
# K1.5.4 Primary: univariate Cox per eigengene per cohort
# ==============================================================================

cat(rep("=", 66), "\n", sep = ""); cat("K1.5.4 UNIVARIATE EIGENGENE-SURVIVAL\n")
cat(rep("=", 66), "\n", sep = "")

rows <- list()
for (ref in REF_COHORTS) {
  for (nm in ALL_COHORTS) {
    E <- MEs[[ref]][[nm]]
    s <- surv[[nm]]
    for (mecol in colnames(E)) {
      # Scale per cohort so HRs are per SD and comparable across cohorts
      x <- as.numeric(scale(E[[mecol]]))
      if (sd(x, na.rm = TRUE) < 1e-10) next
      fit <- tryCatch(
        coxph(Surv(s$time, s$event) ~ x),
        error = function(e) NULL)
      if (is.null(fit)) next
      sm <- summary(fit)
      rows[[paste(ref, nm, mecol)]] <- data.frame(
        reference = ref, cohort = nm,
        module = sub("^ME", "", mecol),
        beta = sm$coefficients[1, "coef"],
        HR = sm$coefficients[1, "exp(coef)"],
        HR_lo = sm$conf.int[1, "lower .95"],
        HR_hi = sm$conf.int[1, "upper .95"],
        p = sm$coefficients[1, "Pr(>|z|)"],
        concordance = sm$concordance[1],
        n = fit$n, n_events = fit$nevent
      )
    }
  }
}
uni <- bind_rows(rows) %>%
  group_by(reference, cohort) %>%
  mutate(q = p.adjust(p, method = "BH")) %>%
  ungroup()

write.csv(uni, file.path(OUT_DIR, "eigengene_univariate_cox.csv"), row.names = FALSE)

for (ref in REF_COHORTS) {
  cat("\n### Modules defined in", ref, "###\n")
  d <- uni[uni$reference == ref, ]
  for (m in unique(d$module)) {
    dm <- d[d$module == m, ]
    cat(sprintf("\n  %s (size %d)\n", m,
                qualifying$size[qualifying$reference == ref &
                                qualifying$module == m]))
    cat(sprintf("    %-10s %7s %-18s %9s %8s %7s\n",
                "cohort", "HR", "95% CI", "p", "q", "C-idx"))
    for (i in seq_len(nrow(dm))) {
      cat(sprintf("    %-10s %7.3f (%6.3f-%6.3f) %9.4f %8.3f %7.3f\n",
                  dm$cohort[i], dm$HR[i], dm$HR_lo[i], dm$HR_hi[i],
                  dm$p[i], dm$q[i], dm$concordance[i]))
    }
    same <- all(sign(dm$beta) == sign(dm$beta[1]))
    cat(sprintf("    direction consistent across all %d cohorts: %s\n",
                nrow(dm), ifelse(same, "YES", "no")))
  }
}

cat("\n--- Direction agreement summary ---\n")
dir_tab <- uni %>%
  group_by(reference, module) %>%
  summarise(n_cohorts = n(), n_positive = sum(beta > 0),
            consistent = n_positive %in% c(0, n_cohorts),
            n_nominal = sum(p < 0.05), n_q05 = sum(q < 0.05), .groups = "drop")
print(as.data.frame(dir_tab), row.names = FALSE)
cat("\n  Direction consistency here is meaningful ONLY because eigengenes were\n")
cat("  sign-aligned along module average expression (K1.5.3). Without that,\n")
cat("  PC1's arbitrary sign would make this comparison vacuous.\n")

# ==============================================================================
# K1.5.5 Secondary: fixed-coefficient transfer from ORIEN
# ==============================================================================

cat("\n", rep("=", 66), "\n", sep = "")
cat("K1.5.5 FIXED-COEFFICIENT TRANSFER (source =", SOURCE_COHORT, ")\n")
cat(rep("=", 66), "\n", sep = "")
cat("Multivariable Cox fit ONCE in", SOURCE_COHORT,
    "; coefficients applied unchanged\nelsewhere. No refitting in any test cohort.\n\n")

transfer_rows <- list()
for (ref in REF_COHORTS) {
  Esrc <- MEs[[ref]][[SOURCE_COHORT]]
  ssrc <- surv[[SOURCE_COHORT]]

  # Standardise using the SOURCE cohort's own mean/sd, then apply those same
  # constants to every test cohort -- refitting the scaling would leak test
  # information into the score.
  mu <- colMeans(Esrc); sdv <- apply(Esrc, 2, sd)
  Xsrc <- scale(Esrc, center = mu, scale = sdv)

  fit <- coxph(Surv(ssrc$time, ssrc$event) ~ ., data = as.data.frame(Xsrc))
  cat("--- modules from", ref, "| fitted in", SOURCE_COHORT, "---\n")
  print(summary(fit)$coefficients)

  cf <- coef(fit)
  for (nm in ALL_COHORTS) {
    E <- MEs[[ref]][[nm]][, names(cf), drop = FALSE]
    X <- scale(E, center = mu[names(cf)], scale = sdv[names(cf)])
    score <- as.numeric(as.matrix(X) %*% cf)
    s <- surv[[nm]]
    cidx <- summary(coxph(Surv(s$time, s$event) ~ score))$concordance[1]
    transfer_rows[[paste(ref, nm)]] <- data.frame(
      reference = ref, cohort = nm, role = ifelse(nm == SOURCE_COHORT,
                                                  "SOURCE (in-sample)", "transfer"),
      n = nrow(s), n_events = sum(s$event), c_index = cidx)
  }
  cat("\n")
}

tr <- bind_rows(transfer_rows)
write.csv(tr, file.path(OUT_DIR, "eigengene_score_transfer.csv"), row.names = FALSE)

cat("--- Transferred score C-index ---\n")
cat(sprintf("  %-6s %-10s %-20s %6s %8s %8s\n",
            "ref", "cohort", "role", "n", "events", "C-index"))
for (i in seq_len(nrow(tr))) {
  cat(sprintf("  %-6s %-10s %-20s %6d %8d %8.3f\n",
              tr$reference[i], tr$cohort[i], tr$role[i], tr$n[i],
              tr$n_events[i], tr$c_index[i]))
}
cat("\n  Benchmark: age-alone C-index floor is", AGE_FLOOR, "(Rung 1).\n")
cat("  The SOURCE row is in-sample and is NOT evidence of transportability.\n")

cat("\nFinished:", format(Sys.time()), "\n")
cat("--- sessionInfo ---\n"); print(sessionInfo())
sink(type = "message"); sink(); close(log_con)
