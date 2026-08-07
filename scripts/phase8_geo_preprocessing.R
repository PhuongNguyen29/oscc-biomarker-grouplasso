#!/usr/bin/env Rscript
# ==============================================================================
# Phase 8: GEO Data Preprocessing for External Validation
# ==============================================================================
#
# Steps 8.1 and 8.2 of the corrected pipeline.
#
# Step 8.1: Extract expression and clinical data from GSE41613 and GSE42743
# Step 8.2: Probe-to-gene mapping using Jetset (Li et al., 2011) with
#           highest-mean-expression fallback
#
# Input:
#   - GEO series matrices (via GEOquery)
#   - GSE42743 RMA matrix from phase8_rma_gse42743.R
#   - Retained gene list from the CORRECTED filter (post-ComBat, pre-z)
#   - Gene annotation: data/raw/protein_coding_genes_annotation.csv
#
# Output (data/external/):
#   GSE41613_expression.csv / _survival.csv / _clinical.csv
#   GSE42743_expression.csv / _survival.csv / _clinical.csv
#   probe_mapping_report.txt
#
# FIXES IN THIS REVISION
# ----------------------
# 1. CONSENSUS_PATH now points at the corrected post-ComBat pre-z retained set.
#    The previous value (data/processed/iqr_filtered/) is the frozen OLD post-z
#    list and is the reason these matrices must be rebuilt.
#
# 2. GSE42743 expression columns are aligned to pData BY NAME before the
#    oral-cavity mask is applied. Previously the mask was built from pData
#    (GEO order) and applied positionally to the RMA matrix (list.files order).
#    The failure mode was not scrambled labels -- the later intersect() fixes
#    names -- but silently retaining the WRONG subset of samples.
#
# 3. expr42743_oc was referenced at the extraction step but never created.
#
# 4. Values are clipped to +/-3 SD to match the development matrices. Without
#    this the model is evaluated outside its training input domain, and the
#    affected patients are the extreme-expression tail the signature is meant
#    to identify. The count is taken BEFORE clipping (the previous form
#    measured values sitting AT +/-3, i.e. after the operation).
#
# 5. The per-gene section of probe_mapping_report.txt referenced
#    row$jetset_overall, a column that does not exist. sprintf() with a
#    zero-length argument returns character(0), so every per-gene line was
#    silently dropped and the report appeared to end early.
#
# 6. GSE41613 scale is now recorded. It is a submitter-processed series matrix,
#    unlike GSE42743 (RMA from CEL). If it is not log2, z-scoring a
#    right-skewed linear-intensity matrix is a materially different operation
#    and the two cohorts are not interchangeable.
#
# 7. Logging via sink(); sessionInfo() recorded (annotation package versions
#    determine the probe mapping and were unrecoverable for the previous run).
#
# 8. library(Seurat) removed -- unused, heavy, and an unnecessary dependency.
#
# References:
#   Li Q et al. (2011) BMC Bioinformatics 12:474 -- Jetset probe selection
#   Davis S, Meltzer PS (2007) Bioinformatics 23:1846 -- GEOquery
#   Irizarry RA et al. (2003) Biostatistics 4:249 -- RMA
#
# Usage:
#   cd ~/hpchome/oscc-biomarker-grouplasso
#   Rscript scripts/phase8_geo_preprocessing.R
#
# Author: Phuong Nguyen
# ==============================================================================

library(GEOquery)
library(jetset)
library(hgu133plus2.db)

# ==============================================================================
# Configuration
# ==============================================================================

CONSENSUS_PATH  <- "data/processed/iqr_filtered_v4_prez/retained_genes.txt"
ANNOTATION_PATH <- "data/raw/protein_coding_genes_annotation.csv"
RMA_PATH        <- "data/external/GSE42743_rma_expression.csv"
OUTPUT_DIR      <- "data/external"
CLIP_SD         <- 3          # matches Phase 1 development preprocessing

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# --- Logging ------------------------------------------------------------------
# Note: split = TRUE works for stdout but NOT for messages -- R permits only one
# message sink and it cannot split. Warnings therefore go to the file only.
LOG_DIR <- "logs"
dir.create(LOG_DIR, showWarnings = FALSE, recursive = TRUE)
log_file <- file.path(
  LOG_DIR,
  sprintf("phase8_geo_preprocessing_%s.log", format(Sys.time(), "%Y%m%d_%H%M%S"))
)
log_con <- file(log_file, open = "wt")
sink(log_con, split = TRUE)
sink(log_con, type = "message")

cat("Log:", log_file, "\n")
cat("Started:", format(Sys.time()), "\n")
cat("R:", R.version.string, "\n")
cat("CONSENSUS_PATH: ", CONSENSUS_PATH, "\n")
cat("ANNOTATION_PATH:", ANNOTATION_PATH, "\n")
cat("RMA_PATH:       ", RMA_PATH, "\n")
cat("CLIP_SD:        ", CLIP_SD, "\n\n")

cat(rep("=", 70), "\n", sep = "")
cat("Phase 8: GEO Data Preprocessing\n")
cat(rep("=", 70), "\n", sep = "")

# ==============================================================================
# Load retained genes and annotation
# ==============================================================================

cat("\n--- Loading retained genes and annotation ---\n")

if (!file.exists(CONSENSUS_PATH)) {
  stop("Retained gene list not found: ", CONSENSUS_PATH,
       "\nThis must be the CORRECTED post-ComBat pre-z list, not iqr_filtered/.")
}
consensus_ensg <- readLines(CONSENSUS_PATH)
consensus_ensg <- consensus_ensg[nchar(consensus_ensg) > 0]
cat("Retained genes (ENSG):", length(consensus_ensg), "\n")

annot <- read.csv(ANNOTATION_PATH, stringsAsFactors = FALSE)

# Deduplicate the annotation at source. PAR genes are annotated on both chrX and
# chrY and would otherwise inflate the join (plan section C3).
n_annot_before <- nrow(annot)
annot <- annot[!duplicated(annot$gene_id), ]
if (nrow(annot) < n_annot_before) {
  cat("Annotation: removed", n_annot_before - nrow(annot),
      "duplicate gene_id rows\n")
}
ensg_to_symbol <- setNames(annot$gene_symbol, annot$gene_id)

consensus_symbols <- ensg_to_symbol[consensus_ensg]
mapped_mask <- !is.na(consensus_symbols) & nchar(consensus_symbols) > 0
cat("Mapped to symbols:", sum(mapped_mask), "/", length(consensus_ensg), "\n")

dup_mask <- duplicated(consensus_symbols[mapped_mask])
if (any(dup_mask)) {
  cat("  Duplicate symbols (keeping first ENSG each):", sum(dup_mask), "\n")
  cat("  ", paste(unique(consensus_symbols[mapped_mask][dup_mask]),
                  collapse = ", "), "\n")
}
if (any(!mapped_mask)) {
  cat("  ENSG without symbol:", sum(!mapped_mask), "\n")
}

mapped_ensg    <- consensus_ensg[mapped_mask][!dup_mask]
mapped_symbols <- consensus_symbols[mapped_mask][!dup_mask]
target_symbols <- mapped_symbols
cat("Target gene symbols:", length(target_symbols), "\n")

# Attrition is recorded at each step separately -- a single coverage figure
# collapses four distinct losses (no symbol, duplicate symbol, no GPL570 probe,
# zero variance) that have different implications.
attrition <- list(
  retained_ensg      = length(consensus_ensg),
  with_symbol        = sum(mapped_mask),
  after_dedup_symbol = length(target_symbols)
)

# ==============================================================================
# Probe-to-gene mapping: Jetset
# ==============================================================================

cat("\n--- Probe-to-gene mapping (Jetset + fallback) ---\n")

jetset_best <- data.frame(
  symbol         = target_symbols,
  ensg           = mapped_ensg,
  jetset_probe   = NA_character_,
  fallback_probe = NA_character_,
  final_probe    = NA_character_,
  mapping_method = NA_character_,
  stringsAsFactors = FALSE
)

for (i in seq_along(target_symbols)) {
  probe <- tryCatch(
    jmap("hgu133plus2", symbol = target_symbols[i]),
    error = function(e) NA_character_
  )
  if (length(probe) == 1 && !is.na(probe) && nchar(probe) > 0) {
    jetset_best$jetset_probe[i] <- probe
  }
}
cat("Jetset mapped:", sum(!is.na(jetset_best$jetset_probe)), "/",
    length(target_symbols), "genes\n")

# ==============================================================================
# Load GEO expression data
# ==============================================================================

cat("\n--- Loading GEO expression data ---\n")

gse41613  <- getGEO("GSE41613", GSEMatrix = TRUE)
expr41613 <- exprs(gse41613[[1]])
meta41613 <- pData(gse41613[[1]])
cat("GSE41613 expression:", nrow(expr41613), "probes x",
    ncol(expr41613), "samples\n")

# GSE41613 is a submitter-processed series matrix; GSE42743 is RMA from CEL.
# If the scales differ, z-scoring is not the same operation on the two cohorts,
# and the fallback probe rule below (which uses GSE41613 as reference for BOTH)
# would be reading "highest mean" off the wrong scale for GSE42743.
r41613 <- range(expr41613, na.rm = TRUE)
cat("GSE41613 value range:", round(r41613, 3), "\n")
if (r41613[2] > 30) {
  cat("  *** WARNING: max >30 suggests LINEAR intensities, not log2.\n")
  cat("  *** GSE42743 is log2 (RMA). The cohorts are NOT comparable as-is.\n")
  cat("  *** Record this before interpreting any cross-cohort result.\n")
} else {
  cat("  Range is consistent with a log2 scale (matches GSE42743 RMA).\n")
}
cat("GSE41613 data_processing field:\n")
print(unique(as.character(meta41613$data_processing))[1])

gse42743  <- getGEO("GSE42743", GSEMatrix = TRUE)
meta42743 <- pData(gse42743[[1]])

if (!file.exists(RMA_PATH)) {
  stop("RMA matrix not found: ", RMA_PATH,
       "\nRun scripts/phase8_rma_gse42743.R first.")
}
expr42743 <- as.matrix(read.csv(RMA_PATH, row.names = 1, check.names = FALSE))
cat("GSE42743 RMA expression:", nrow(expr42743), "probes x",
    ncol(expr42743), "samples\n")
cat("GSE42743 value range:", round(range(expr42743, na.rm = TRUE), 3),
    "(log2, RMA)\n")

probe_to_symbol_vec <- unlist(as.list(hgu133plus2SYMBOL))

# ==============================================================================
# Fallback probe: highest mean expression
# ==============================================================================

needs_fallback <- which(is.na(jetset_best$jetset_probe))
cat("\nGenes needing fallback:", length(needs_fallback), "\n")

if (length(needs_fallback) > 0) {
  keep <- !is.na(probe_to_symbol_vec)
  symbol_to_probes <- split(names(probe_to_symbol_vec)[keep],
                            probe_to_symbol_vec[keep])

  for (i in needs_fallback) {
    candidate_probes <- symbol_to_probes[[jetset_best$symbol[i]]]
    if (!is.null(candidate_probes) && length(candidate_probes) > 0) {
      available <- candidate_probes[candidate_probes %in% rownames(expr41613)]
      if (length(available) > 0) {
        means <- rowMeans(expr41613[available, , drop = FALSE], na.rm = TRUE)
        jetset_best$fallback_probe[i] <- names(which.max(means))
      }
    }
  }
  cat("Fallback mapped:", sum(!is.na(jetset_best$fallback_probe)), "/",
      length(needs_fallback), "genes\n")
  cat("  NOTE: fallback uses GSE41613 as reference for BOTH cohorts.\n")
}

jetset_best$final_probe <- ifelse(
  !is.na(jetset_best$jetset_probe),
  jetset_best$jetset_probe,
  jetset_best$fallback_probe
)
jetset_best$mapping_method <- ifelse(
  !is.na(jetset_best$jetset_probe), "jetset",
  ifelse(!is.na(jetset_best$fallback_probe), "fallback_highest_mean", "unmapped")
)

cat("\n--- Probe mapping summary ---\n")
cat("  Jetset:  ", sum(jetset_best$mapping_method == "jetset"), "\n")
cat("  Fallback:", sum(jetset_best$mapping_method == "fallback_highest_mean"), "\n")
cat("  Unmapped:", sum(jetset_best$mapping_method == "unmapped"), "\n")

unmapped_genes <- jetset_best$symbol[jetset_best$mapping_method == "unmapped"]
if (length(unmapped_genes) > 0) {
  cat("  Unmapped symbols:", paste(unmapped_genes, collapse = ", "), "\n")
}

mapped_genes <- jetset_best[jetset_best$mapping_method != "unmapped", ]
cat("Final mapped genes:", nrow(mapped_genes), "/", length(target_symbols), "\n")
attrition$with_probe <- nrow(mapped_genes)

# ==============================================================================
# Expression extraction: subset, z-score, clip
# ==============================================================================

extract_expression <- function(expr_matrix, mapped_genes, dataset_name) {
  cat("\n--- Extracting expression:", dataset_name, "---\n")

  probes_to_use <- mapped_genes$final_probe
  available <- probes_to_use[probes_to_use %in% rownames(expr_matrix)]
  if (length(available) < length(probes_to_use)) {
    cat("  WARNING:", length(probes_to_use) - length(available),
        "probes not present in this expression matrix\n")
  }

  idx <- match(available, mapped_genes$final_probe)
  expr_subset <- expr_matrix[available, , drop = FALSE]
  rownames(expr_subset) <- mapped_genes$ensg[idx]

  cat("  Matrix:", nrow(expr_subset), "genes x", ncol(expr_subset), "samples\n")
  cat("  Range (pre z-score):", round(range(expr_subset, na.rm = TRUE), 3), "\n")

  gene_means <- rowMeans(expr_subset, na.rm = TRUE)
  gene_sds   <- apply(expr_subset, 1, sd, na.rm = TRUE)

  zero_var <- gene_sds == 0 | is.na(gene_sds)
  if (any(zero_var)) {
    cat("  WARNING:", sum(zero_var), "zero-variance genes removed\n")
    expr_subset <- expr_subset[!zero_var, , drop = FALSE]
    gene_means  <- gene_means[!zero_var]
    gene_sds    <- gene_sds[!zero_var]
  }

  expr_zscored <- (expr_subset - gene_means) / gene_sds

  # Count BEFORE clipping. The previous form measured values sitting AT +/-3,
  # which is the result of the operation rather than its input.
  n_clip <- sum(abs(expr_zscored) > CLIP_SD, na.rm = TRUE)
  expr_zscored <- pmax(pmin(expr_zscored, CLIP_SD), -CLIP_SD)
  cat("  Clipped:", n_clip, "values (",
      round(100 * n_clip / length(expr_zscored), 2),
      "% ) -- compare TCGA 0.40%, ORIEN 0.32%\n")

  cat("  Final:", nrow(expr_zscored), "genes x", ncol(expr_zscored), "samples\n")
  cat("  Z-score range:", round(range(expr_zscored, na.rm = TRUE), 2), "\n")

  expr_zscored
}

# ==============================================================================
# GSE41613
# ==============================================================================

cat("\n", rep("=", 70), "\n", sep = "")
cat("Processing GSE41613\n")
cat(rep("=", 70), "\n", sep = "")

clinical41613 <- data.frame(
  sampleID       = rownames(meta41613),
  geo_accession  = meta41613$geo_accession,
  vital          = meta41613$`vital:ch1`,
  fu_time_months = as.numeric(meta41613$`fu time:ch1`),
  sex            = meta41613$`Sex:ch1`,
  age_bin        = meta41613$`age:ch1`,
  tumor_stage    = meta41613$`tumor stage:ch1`,
  treatment      = meta41613$`treatment:ch1`,
  stringsAsFactors = FALSE
)

# Alive = 0; all Dead-* categories = 1
clinical41613$event <- ifelse(clinical41613$vital == "Alive", 0, 1)
clinical41613$time  <- clinical41613$fu_time_months   # MONTHS

cat("GSE41613 vital status:\n"); print(table(clinical41613$vital))
cat("Events:", sum(clinical41613$event, na.rm = TRUE), "/",
    nrow(clinical41613), "\n")

valid41613 <- !is.na(clinical41613$time) & clinical41613$time > 0 &
              !is.na(clinical41613$event)
cat("Valid survival:", sum(valid41613), "/", nrow(clinical41613), "\n")
clinical41613 <- clinical41613[valid41613, ]

surv41613 <- data.frame(
  sampleID = clinical41613$sampleID,
  time     = clinical41613$time,
  event    = clinical41613$event,
  stringsAsFactors = FALSE
)
clin41613 <- data.frame(
  sampleID    = clinical41613$sampleID,
  gender      = ifelse(clinical41613$sex == "M", 1, 0),
  tumor_stage = clinical41613$tumor_stage,
  age_bin     = clinical41613$age_bin,
  stringsAsFactors = FALSE
)

expr41613_z <- extract_expression(expr41613, mapped_genes, "GSE41613")

common41613 <- intersect(colnames(expr41613_z), clinical41613$sampleID)
expr41613_z <- expr41613_z[, common41613, drop = FALSE]
surv41613   <- surv41613[surv41613$sampleID %in% common41613, ]
clin41613   <- clin41613[clin41613$sampleID %in% common41613, ]

cat("GSE41613 final: ", ncol(expr41613_z), " samples, ",
    nrow(expr41613_z), " genes, ", sum(surv41613$event), " events\n", sep = "")

# ==============================================================================
# GSE42743
# ==============================================================================

cat("\n", rep("=", 70), "\n", sep = "")
cat("Processing GSE42743\n")
cat(rep("=", 70), "\n", sep = "")

# The original RMA script saved names as GSM#######_MDA-HNS-###; the corrected
# one strips to bare GSM IDs. Handle both so this script works with either.
colnames(expr42743) <- sub("_.*", "", colnames(expr42743))
stopifnot(all(grepl("^GSM[0-9]+$", colnames(expr42743))))
stopifnot(!any(duplicated(colnames(expr42743))))

# The RMA matrix column order comes from list.files(); pData is in GEO order.
# Align BY NAME before any positional operation. Applying a pData-derived mask
# positionally to a differently-ordered matrix silently selects the wrong
# samples -- and the later intersect() would repair the names, hiding it.
if (!all(rownames(meta42743) %in% colnames(expr42743))) {
  missing_gsm <- setdiff(rownames(meta42743), colnames(expr42743))
  stop(length(missing_gsm), " GSM IDs in pData are absent from the RMA matrix. ",
       "First few: ", paste(head(missing_gsm, 5), collapse = ", "),
       "\nCheck the colname cleaning step in phase8_rma_gse42743.R.")
}
expr42743 <- expr42743[, rownames(meta42743), drop = FALSE]
stopifnot(identical(colnames(expr42743), rownames(meta42743)))
cat("Aligned RMA columns to pData order:", ncol(expr42743), "samples\n")

oc_mask <- meta42743$`tu-vs-nl:ch1` == "cancer" &
           meta42743$`oc vs. op:ch1` == "Oral cavity"
oc_mask[is.na(oc_mask)] <- FALSE
cat("Filtering: cancer + oral cavity =", sum(oc_mask), "samples\n")

meta42743_oc <- meta42743[oc_mask, ]
expr42743_oc <- expr42743[, oc_mask, drop = FALSE]
stopifnot(identical(colnames(expr42743_oc), rownames(meta42743_oc)))

clinical42743 <- data.frame(
  sampleID      = rownames(meta42743_oc),
  geo_accession = meta42743_oc$geo_accession,
  alldeath      = as.integer(meta42743_oc$`alldeath:ch1`),
  futime_days   = as.numeric(meta42743_oc$`futime:ch1`),
  vital         = meta42743_oc$`vital:ch1`,
  gender        = meta42743_oc$`gender:ch1`,
  age           = as.numeric(meta42743_oc$`age@dx:ch1`),
  smoking       = meta42743_oc$`smoking status:ch1`,
  t_stage       = meta42743_oc$`t stage:ch1`,
  cn_stage      = meta42743_oc$`cn stage:ch1`,
  pn_stage      = meta42743_oc$`pn stage:ch1`,
  stringsAsFactors = FALSE
)

# TIME UNIT: DAYS. GSE41613 is in MONTHS. C-index and HRs are invariant to
# monotone rescaling, so 8.3/8.4/8.5 are unaffected -- but anything evaluated
# at a FIXED HORIZON is not. Every downstream script must handle this
# explicitly (see km_auc_stratification.py TIME_HORIZONS).
clinical42743$event <- clinical42743$alldeath
clinical42743$time  <- clinical42743$futime_days

cat("GSE42743 vital status:\n"); print(table(clinical42743$vital))
cat("alldeath:\n");              print(table(clinical42743$alldeath))
cat("Events:", sum(clinical42743$event, na.rm = TRUE), "/",
    nrow(clinical42743), "\n")

valid42743 <- !is.na(clinical42743$time) & clinical42743$time > 0 &
              !is.na(clinical42743$event)
cat("Valid survival:", sum(valid42743), "/", nrow(clinical42743), "\n")
clinical42743 <- clinical42743[valid42743, ]

surv42743 <- data.frame(
  sampleID = clinical42743$sampleID,
  time     = clinical42743$time,
  event    = clinical42743$event,
  stringsAsFactors = FALSE
)
clin42743 <- data.frame(
  sampleID = clinical42743$sampleID,
  gender   = ifelse(clinical42743$gender == "Male", 1, 0),
  age      = clinical42743$age,
  smoking  = ifelse(clinical42743$smoking == "NeverSmoker", "Never", "Ever"),
  T_stage  = ifelse(clinical42743$t_stage %in% c("1", "2"), "T1-T2", "T3-T4"),
  N_stage  = ifelse(clinical42743$pn_stage == "0", "N0", "N+"),
  stringsAsFactors = FALSE
)

expr42743_z <- extract_expression(expr42743_oc, mapped_genes, "GSE42743")

common42743 <- intersect(colnames(expr42743_z), clinical42743$sampleID)
expr42743_z <- expr42743_z[, common42743, drop = FALSE]
surv42743   <- surv42743[surv42743$sampleID %in% common42743, ]
clin42743   <- clin42743[clin42743$sampleID %in% common42743, ]

cat("GSE42743 final: ", ncol(expr42743_z), " samples, ",
    nrow(expr42743_z), " genes, ", sum(surv42743$event), " events\n", sep = "")

# ==============================================================================
# Sanity checks
# ==============================================================================

cat("\n--- Sanity checks ---\n")
stopifnot(ncol(expr41613_z) == nrow(surv41613))
stopifnot(ncol(expr42743_z) == nrow(surv42743))
stopifnot(identical(colnames(expr41613_z), surv41613$sampleID))
stopifnot(identical(colnames(expr42743_z), surv42743$sampleID))
cat("Expression/survival alignment: OK\n")

shared_genes <- intersect(rownames(expr41613_z), rownames(expr42743_z))
cat("Genes present in BOTH cohorts:", length(shared_genes), "\n")
attrition$in_gse41613  <- nrow(expr41613_z)
attrition$in_gse42743  <- nrow(expr42743_z)
attrition$in_both      <- length(shared_genes)

cat("\nCoverage attrition (each step recorded separately):\n")
for (k in names(attrition)) cat(sprintf("  %-20s %d\n", k, attrition[[k]]))

# ==============================================================================
# Save outputs
# ==============================================================================

cat("\n--- Saving outputs ---\n")

write.csv(expr41613_z, file.path(OUTPUT_DIR, "GSE41613_expression.csv"))
write.csv(surv41613,   file.path(OUTPUT_DIR, "GSE41613_survival.csv"), row.names = FALSE)
write.csv(clin41613,   file.path(OUTPUT_DIR, "GSE41613_clinical.csv"), row.names = FALSE)
write.csv(expr42743_z, file.path(OUTPUT_DIR, "GSE42743_expression.csv"))
write.csv(surv42743,   file.path(OUTPUT_DIR, "GSE42743_survival.csv"), row.names = FALSE)
write.csv(clin42743,   file.path(OUTPUT_DIR, "GSE42743_clinical.csv"), row.names = FALSE)

report_lines <- c(
  paste(rep("=", 70), collapse = ""),
  "Phase 8: Probe Mapping Report",
  paste(rep("=", 70), collapse = ""),
  "",
  paste("Generated:", format(Sys.time())),
  paste("Gene list:", CONSENSUS_PATH),
  paste("Clipping:  +/-", CLIP_SD, "SD (matches development preprocessing)"),
  "",
  "Coverage attrition:",
  paste("  Retained ENSG:            ", attrition$retained_ensg),
  paste("  With gene symbol:         ", attrition$with_symbol),
  paste("  After duplicate-symbol:   ", attrition$after_dedup_symbol),
  paste("  With GPL570 probe:        ", attrition$with_probe),
  paste("  Present in GSE41613:      ", attrition$in_gse41613),
  paste("  Present in GSE42743:      ", attrition$in_gse42743),
  paste("  Present in BOTH:          ", attrition$in_both),
  "",
  "Probe mapping method:",
  paste("  Jetset:  ", sum(jetset_best$mapping_method == "jetset")),
  paste("  Fallback:", sum(jetset_best$mapping_method == "fallback_highest_mean")),
  paste("  Unmapped:", sum(jetset_best$mapping_method == "unmapped")),
  "  NOTE: fallback selects the highest-mean probe using GSE41613 as",
  "        reference for BOTH cohorts.",
  "",
  "Unmapped gene symbols:",
  if (length(unmapped_genes) > 0) paste("  ", unmapped_genes) else "  None",
  "",
  "GSE41613:",
  paste("  Samples:", ncol(expr41613_z)),
  paste("  Genes:  ", nrow(expr41613_z)),
  paste("  Events: ", sum(surv41613$event), "/", nrow(surv41613)),
  "  Survival time unit: MONTHS",
  paste("  Source: GEO series matrix (submitter-processed); value range",
        paste(round(r41613, 2), collapse = " to ")),
  "",
  "GSE42743:",
  paste("  Samples:", ncol(expr42743_z), "(oral cavity cancer only)"),
  paste("  Genes:  ", nrow(expr42743_z)),
  paste("  Events: ", sum(surv42743$event), "/", nrow(surv42743)),
  "  Survival time unit: DAYS",
  "  Source: RMA from CEL files (phase8_rma_gse42743.R)",
  paste("  Filtered from", ncol(expr42743), "total samples"),
  "",
  paste(rep("=", 70), collapse = ""),
  "Per-gene mapping details:",
  paste(rep("-", 70), collapse = "")
)

# The previous version referenced row$jetset_overall, which does not exist in
# jetset_best. sprintf() with a zero-length argument returns character(0), so
# every per-gene line was silently dropped.
for (i in seq_len(nrow(jetset_best))) {
  report_lines <- c(report_lines, sprintf(
    "  %s (%s): %s -> %s",
    jetset_best$ensg[i],
    jetset_best$symbol[i],
    jetset_best$mapping_method[i],
    ifelse(!is.na(jetset_best$final_probe[i]),
           jetset_best$final_probe[i], "UNMAPPED")
  ))
}

writeLines(report_lines, file.path(OUTPUT_DIR, "probe_mapping_report.txt"))
write.csv(jetset_best, file.path(OUTPUT_DIR, "probe_mapping_table.csv"),
          row.names = FALSE)

cat("\nFiles saved to:", OUTPUT_DIR, "\n")
for (f in c("GSE41613_expression.csv", "GSE41613_survival.csv",
            "GSE41613_clinical.csv", "GSE42743_expression.csv",
            "GSE42743_survival.csv", "GSE42743_clinical.csv",
            "probe_mapping_report.txt", "probe_mapping_table.csv")) {
  cat("  ", f, "\n", sep = "")
}

cat("\n", rep("=", 70), "\n", sep = "")
cat("Phase 8 preprocessing complete.\n")
cat(rep("=", 70), "\n", sep = "")

cat("\nFinished:", format(Sys.time()), "\n")
cat("--- sessionInfo ---\n")
print(sessionInfo())

sink(type = "message")
sink()
close(log_con)
