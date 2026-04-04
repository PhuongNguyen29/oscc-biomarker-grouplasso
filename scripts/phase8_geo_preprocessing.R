#!/usr/bin/env Rscript
# ==============================================================================
# Phase 8: GEO Data Preprocessing for External Validation
# ==============================================================================
#
# Steps 8.1 and 8.2 of the v4 pipeline.
#
# Step 8.1: Extract expression and clinical data from GSE41613 and GSE42743
# Step 8.2: Probe-to-gene mapping using Jetset (Li et al., 2011, BMC Bioinformatics)
#           with highest-mean-expression fallback
#
# Input:
#   - GEO series matrix files (downloaded via GEOquery)
#   - Consensus gene list: results/phase4_stability_mad3000/consensus/cox_consensus_pi080.txt
#   - Gene annotation: data/raw/protein_coding_genes_annotation.csv
#
# Output:
#   - data/external/GSE41613_expression.csv   (genes × samples, z-scored)
#   - data/external/GSE41613_survival.csv     (sampleID, time, event)
#   - data/external/GSE41613_clinical.csv     (sampleID, clinical vars)
#   - data/external/GSE42743_expression.csv
#   - data/external/GSE42743_survival.csv
#   - data/external/GSE42743_clinical.csv
#   - data/external/probe_mapping_report.txt
#
# References:
#   - Li et al. (2011) BMC Bioinformatics — Jetset probe selection
#   - Davis & Meltzer (2007) Bioinformatics — GEOquery
#
# Usage:
#   cd ~/hpchome/oscc-biomarker-elasticdeepsurv
#   Rscript scripts/phase8_geo_preprocessing.R
#
# Author: Phuong Nguyen
# Date: February 2026
# ==============================================================================

library(GEOquery)
library(jetset)
library(hgu133plus2.db)

# ==============================================================================
# Configuration
# ==============================================================================

CONSENSUS_PATH <- "data/processed/iqr_filtered/retained_genes.txt"
ANNOTATION_PATH <- "data/raw/protein_coding_genes_annotation.csv"
OUTPUT_DIR <- "data/external"

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat("=" , rep("=", 69), "\n", sep = "")
cat("Phase 8: GEO Data Preprocessing\n")
cat("=" , rep("=", 69), "\n", sep = "")

# ==============================================================================
# Load consensus genes and annotation
# ==============================================================================

cat("\n--- Loading consensus genes and annotation ---\n")

# Consensus ENSG IDs
consensus_ensg <- readLines(CONSENSUS_PATH)
consensus_ensg <- consensus_ensg[nchar(consensus_ensg) > 0]
cat("Consensus genes (ENSG):", length(consensus_ensg), "\n")

# Gene annotation: ENSG → symbol
annot <- read.csv(ANNOTATION_PATH, stringsAsFactors = FALSE)
ensg_to_symbol <- setNames(annot$gene_symbol, annot$gene_id)

# Map consensus ENSG to symbols
consensus_symbols <- ensg_to_symbol[consensus_ensg]
mapped_mask <- !is.na(consensus_symbols)
cat("Mapped to symbols:", sum(mapped_mask), "/", length(consensus_ensg), "\n")

# Remove duplicate symbols (keep first ENSG per symbol)
dup_mask <- duplicated(consensus_symbols[mapped_mask])
if (any(dup_mask)) {
  cat("  Duplicate symbols (removing):", sum(dup_mask), "\n")
  # Show which ones
  dup_syms <- consensus_symbols[mapped_mask][dup_mask]
  cat("  ", paste(dup_syms, collapse = ", "), "\n")
}

if (any(!mapped_mask)) {
    cat("  Unmapped ENSG IDs:\n")
    cat("  ", consensus_ensg[!mapped_mask], "\n")
}

# Create lookup tables
symbol_to_ensg <- setNames(consensus_ensg[mapped_mask], consensus_symbols[mapped_mask])
target_symbols <- consensus_symbols[mapped_mask]
cat("Target gene symbols:", length(target_symbols), "\n")

# ==============================================================================
# Probe-to-gene mapping using Jetset
# ==============================================================================

cat("\n--- Probe-to-gene mapping (Jetset + fallback) ---\n")

# Use jmap to get best probe per gene directly
jetset_best <- data.frame(
  symbol = target_symbols,
  ensg = consensus_ensg[mapped_mask],
  jetset_probe = NA_character_,
  fallback_probe = NA_character_,
  final_probe = NA_character_,
  mapping_method = NA_character_,
  stringsAsFactors = FALSE
)

for (i in seq_along(target_symbols)) {
  sym <- target_symbols[i]
  probe <- tryCatch(
    jmap("hgu133plus2", symbol = sym),
    error = function(e) NA_character_
  )
  if (!is.na(probe) && nchar(probe) > 0 && probe != "") {
    jetset_best$jetset_probe[i] <- probe
  }
}

cat("Jetset mapped:", sum(!is.na(jetset_best$jetset_probe)), "/",
    length(target_symbols), "genes\n")

# ==============================================================================
# Fallback: highest mean expression for unmapped genes
# ==============================================================================

library(GEOquery)
library(Seurat)
library(dplyr)


# We need expression data to compute fallback — load GSE41613 as reference
cat("\n--- Loading GEO expression data ---\n")

gse41613 <- getGEO("GSE41613", GSEMatrix = TRUE)
expr41613 <- exprs(gse41613[[1]])
meta41613 <- pData(gse41613[[1]])
cat("GSE41613 expression:", nrow(expr41613), "probes ×", ncol(expr41613), "samples\n")

gse42743 <- getGEO("GSE42743", GSEMatrix = TRUE)
# expr42743 <- exprs(gse42743[[1]])
meta42743 <- pData(gse42743[[1]])
# cat("GSE42743 expression:", nrow(expr42743), "probes ×", ncol(expr42743), "samples\n")

# Load full RMA expression (all probes)
expr42743_full <- read.csv("data/external/GSE42743_rma_expression.csv", row.names = 1)
dim(expr42743_full)
head(rownames(expr42743_full), 10)
expr42743 <- expr42743_full
# Build probe-to-symbol mapping from hgu133plus2.db
probe_to_symbol <- as.list(hgu133plus2SYMBOL)
# Flatten: some probes map to NA
probe_to_symbol_vec <- unlist(probe_to_symbol)

# For genes not mapped by Jetset, find probe with highest mean expression
# Use GSE41613 as reference (larger dataset with all tumor samples)
needs_fallback <- which(is.na(jetset_best$jetset_probe))
cat("Genes needing fallback:", length(needs_fallback), "\n")

if (length(needs_fallback) > 0) {
    # Build reverse mapping: symbol → all probes
    symbol_to_probes <- list()
    for (probe_id in names(probe_to_symbol_vec)) {
        sym <- probe_to_symbol_vec[probe_id]
        if (!is.na(sym)) {
            if (is.null(symbol_to_probes[[sym]])) {
                symbol_to_probes[[sym]] <- c()
            }
            symbol_to_probes[[sym]] <- c(symbol_to_probes[[sym]], probe_id)
        }
    }
    
    for (i in needs_fallback) {
        sym <- jetset_best$symbol[i]
        candidate_probes <- symbol_to_probes[[sym]]
        
        if (!is.null(candidate_probes) && length(candidate_probes) > 0) {
            # Filter to probes present in expression data
            available <- candidate_probes[candidate_probes %in% rownames(expr41613)]
            
            if (length(available) > 0) {
                # Select probe with highest mean expression
                means <- rowMeans(expr41613[available, , drop = FALSE], na.rm = TRUE)
                best <- names(which.max(means))
                jetset_best$fallback_probe[i] <- best
            }
        }
    }
    
    cat("Fallback mapped:", sum(!is.na(jetset_best$fallback_probe)), "/",
        length(needs_fallback), "genes\n")
}

# Assign final probe: Jetset preferred, fallback if needed
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
cat("  Jetset:", sum(jetset_best$mapping_method == "jetset"), "\n")
cat("  Fallback (highest mean):", sum(jetset_best$mapping_method == "fallback_highest_mean"), "\n")
cat("  Unmapped:", sum(jetset_best$mapping_method == "unmapped"), "\n")

unmapped_genes <- jetset_best$symbol[jetset_best$mapping_method == "unmapped"]
if (length(unmapped_genes) > 0) {
    cat("  Unmapped gene symbols:", paste(unmapped_genes, collapse = ", "), "\n")
}

# Filter to successfully mapped genes only
mapped_genes <- jetset_best[jetset_best$mapping_method != "unmapped", ]
cat("Final mapped genes:", nrow(mapped_genes), "/", length(target_symbols), "\n")

# ==============================================================================
# Extract expression matrices (mapped consensus genes only, z-scored)
# ==============================================================================

extract_expression <- function(expr_matrix, mapped_genes, dataset_name) {
    cat("\n--- Extracting expression:", dataset_name, "---\n")
    
    # Subset to final probes
    probes_to_use <- mapped_genes$final_probe
    available <- probes_to_use[probes_to_use %in% rownames(expr_matrix)]
    
    if (length(available) < length(probes_to_use)) {
        missing_probes <- probes_to_use[!probes_to_use %in% rownames(expr_matrix)]
        cat("  WARNING:", length(missing_probes), "probes not in expression matrix\n")
    }
    
    # Subset expression
    idx <- match(available, mapped_genes$final_probe)
    expr_subset <- expr_matrix[available, , drop = FALSE]
    
    # Rename rows from probe IDs to ENSG IDs
    rownames(expr_subset) <- mapped_genes$ensg[idx]
    
    cat("  Expression matrix:", nrow(expr_subset), "genes ×", ncol(expr_subset), "samples\n")
    cat("  Expression range (pre z-score):", range(expr_subset, na.rm = TRUE), "\n")
    
    # Z-score each gene across samples (zero mean, unit variance)
    # Same as development cohort preprocessing
    gene_means <- rowMeans(expr_subset, na.rm = TRUE)
    gene_sds <- apply(expr_subset, 1, sd, na.rm = TRUE)
    
    # Handle zero-variance genes
    zero_var <- gene_sds == 0 | is.na(gene_sds)
    if (any(zero_var)) {
        cat("  WARNING:", sum(zero_var), "genes with zero variance, removing\n")
        expr_subset <- expr_subset[!zero_var, , drop = FALSE]
        gene_means <- gene_means[!zero_var]
        gene_sds <- gene_sds[!zero_var]
    }
    
    expr_zscored <- (expr_subset - gene_means) / gene_sds
    
    cat("  After z-scoring:", nrow(expr_zscored), "genes ×", ncol(expr_zscored), "samples\n")
    cat("  Z-score range:", round(range(expr_zscored, na.rm = TRUE), 2), "\n")
    
    return(expr_zscored)
}

# ==============================================================================
# GSE41613: Process clinical data and expression
# ==============================================================================

cat("\n", rep("=", 70), "\n", sep = "")
cat("Processing GSE41613\n")
cat(rep("=", 70), "\n", sep = "")

# Clinical data extraction
# Survival: Recode vital: Alive=0, Dead-oral ca=1, Dead-non OC=1
# fu time: months
clinical41613 <- data.frame(
    sampleID = rownames(meta41613),
    geo_accession = meta41613$geo_accession,
    vital = meta41613$`vital:ch1`,
    fu_time_months = as.numeric(meta41613$`fu time:ch1`),
    sex = meta41613$`Sex:ch1`,
    age_bin = meta41613$`age:ch1`,
    tumor_stage = meta41613$`tumor stage:ch1`,
    treatment = meta41613$`treatment:ch1`,
    stringsAsFactors = FALSE
)

# Construct survival endpoint per v4 document:
# Alive = 0, Dead-oral ca = 1, Dead-non OC = 1, Dead-unk cause = 1
clinical41613$event <- ifelse(clinical41613$vital == "Alive", 0, 1)
clinical41613$time <- clinical41613$fu_time_months

# Handle edge cases
cat("GSE41613 vital status:\n")
print(table(clinical41613$vital))
cat("Events:", sum(clinical41613$event, na.rm = TRUE), "/", nrow(clinical41613), "\n")

# Check for missing/invalid survival data
valid41613 <- !is.na(clinical41613$time) & clinical41613$time > 0 & !is.na(clinical41613$event)
cat("Valid survival data:", sum(valid41613), "/", nrow(clinical41613), "\n")
clinical41613 <- clinical41613[valid41613, ]

# Survival file
surv41613 <- data.frame(
    sampleID = clinical41613$sampleID,
    time = clinical41613$time,
    event = clinical41613$event,
    stringsAsFactors = FALSE
)

# Clinical file (for stratified analysis)
clin41613 <- data.frame(
    sampleID = clinical41613$sampleID,
    gender = ifelse(clinical41613$sex == "M", 1, 0),  # 0=F, 1=M
    tumor_stage = clinical41613$tumor_stage,  # I/II vs III/IV
    age_bin = clinical41613$age_bin,
    stringsAsFactors = FALSE
)

# Expression: all 97 samples (all HPV-negative OSCC)
expr41613_z <- extract_expression(expr41613, mapped_genes, "GSE41613")

# Align samples between expression and clinical
common41613 <- intersect(colnames(expr41613_z), clinical41613$sampleID)
expr41613_z <- expr41613_z[, common41613, drop = FALSE]
surv41613 <- surv41613[surv41613$sampleID %in% common41613, ]
clin41613 <- clin41613[clin41613$sampleID %in% common41613, ]

cat("GSE41613 final: ", ncol(expr41613_z), " samples, ", nrow(expr41613_z), " genes, ",
    sum(surv41613$event), " events\n", sep = "")

# ==============================================================================
# GSE42743: Process clinical data and expression
# ==============================================================================

cat("\n", rep("=", 70), "\n", sep = "")
cat("Processing GSE42743\n")
cat(rep("=", 70), "\n", sep = "")

# Filter to oral cavity cancer only (per v4 document)
oc_mask <- meta42743$`tu-vs-nl:ch1` == "cancer" & meta42743$`oc vs. op:ch1` == "Oral cavity"
cat("Filtering: cancer + oral cavity =", sum(oc_mask), "samples\n")

meta42743_oc <- meta42743[oc_mask, ]
expr42743_oc <- expr42743[, oc_mask, drop = FALSE]

# Clinical data extraction
# Survival: alldeath (0/1), futime (days)
clinical42743 <- data.frame(
    sampleID = rownames(meta42743_oc),
    geo_accession = meta42743_oc$geo_accession,
    alldeath = as.integer(meta42743_oc$`alldeath:ch1`),
    futime_days = as.numeric(meta42743_oc$`futime:ch1`),
    vital = meta42743_oc$`vital:ch1`,
    gender = meta42743_oc$`gender:ch1`,
    age = as.numeric(meta42743_oc$`age@dx:ch1`),
    smoking = meta42743_oc$`smoking status:ch1`,
    t_stage = meta42743_oc$`t stage:ch1`,
    cn_stage = meta42743_oc$`cn stage:ch1`,
    pn_stage = meta42743_oc$`pn stage:ch1`,
    stringsAsFactors = FALSE
)

# Survival endpoint: alldeath as event, futime as time (days)
clinical42743$event <- clinical42743$alldeath
clinical42743$time <- clinical42743$futime_days

cat("GSE42743 vital status:\n")
print(table(clinical42743$vital))
cat("alldeath:\n")
print(table(clinical42743$alldeath))
cat("Events:", sum(clinical42743$event, na.rm = TRUE), "/", nrow(clinical42743), "\n")

# Check for missing/invalid survival data
valid42743 <- !is.na(clinical42743$time) & clinical42743$time > 0 & !is.na(clinical42743$event)
cat("Valid survival data:", sum(valid42743), "/", nrow(clinical42743), "\n")
clinical42743 <- clinical42743[valid42743, ]

# Survival file
surv42743 <- data.frame(
  sampleID = clinical42743$sampleID,
  time = clinical42743$time,
  event = clinical42743$event,
  stringsAsFactors = FALSE
)

cat("Survival: ", nrow(surv42743), " samples, ", sum(surv42743$event), " events\n", sep = "")

# Clinical file (for stratified analysis)
# Encode per v4: gender binary, smoking (collapse Current/Former → Ever),
# T-stage (1-2 vs 3-4), N-stage from pn_stage (0 vs 1+)
clin42743 <- data.frame(
    sampleID = clinical42743$sampleID,
    gender = ifelse(clinical42743$gender == "Male", 1, 0),
    age = clinical42743$age,
    smoking = ifelse(clinical42743$smoking == "NeverSmoker", "Never", "Ever"),
    T_stage = ifelse(clinical42743$t_stage %in% c("1", "2"), "T1-T2", "T3-T4"),
    N_stage = ifelse(clinical42743$pn_stage == "0", "N0", "N+"),
    stringsAsFactors = FALSE
)

# Expression
expr42743_filtered <- expr42743_oc
expr42743_z <- extract_expression(expr42743_filtered, mapped_genes, "GSE42743")

colnames(expr42743_z) <- sub("_.*", "", colnames(expr42743_z))

# Now re-align
common42743 <- intersect(colnames(expr42743_z), clinical42743$sampleID)
expr42743_z <- expr42743_z[, common42743, drop = FALSE]
surv42743 <- surv42743[surv42743$sampleID %in% common42743, ]
clin42743 <- clin42743[clin42743$sampleID %in% common42743, ]

cat("GSE42743 final: ", ncol(expr42743_z), " samples, ", nrow(expr42743_z), " genes, ",
    sum(surv42743$event), " events\n", sep = "")

# ==============================================================================
# Save outputs
# ==============================================================================

cat("\n--- Saving outputs ---\n")

# GSE41613
write.csv(expr41613_z, file.path(OUTPUT_DIR, "GSE41613_expression.csv"))
write.csv(surv41613, file.path(OUTPUT_DIR, "GSE41613_survival.csv"), row.names = FALSE)
write.csv(clin41613, file.path(OUTPUT_DIR, "GSE41613_clinical.csv"), row.names = FALSE)

# GSE42743
write.csv(expr42743_z, file.path(OUTPUT_DIR, "GSE42743_expression.csv"))
write.csv(surv42743, file.path(OUTPUT_DIR, "GSE42743_survival.csv"), row.names = FALSE)
write.csv(clin42743, file.path(OUTPUT_DIR, "GSE42743_clinical.csv"), row.names = FALSE)

# Probe mapping report
report_lines <- c(
    paste(rep("=", 70), collapse = ""),
    "Phase 8: Probe Mapping Report",
    paste(rep("=", 70), collapse = ""),
    "",
    paste("Consensus genes (ENSG IDs):", length(consensus_ensg)),
    paste("Mapped to gene symbols:", sum(mapped_mask)),
    "",
    "Probe mapping method:",
    paste("  Jetset:", sum(mapped_genes$mapping_method == "jetset")),
    paste("  Fallback (highest mean expression):", 
          sum(mapped_genes$mapping_method == "fallback_highest_mean")),
    paste("  Unmapped:", sum(jetset_best$mapping_method == "unmapped")),
    "",
    "Unmapped gene symbols:",
    if (length(unmapped_genes) > 0) paste("  ", unmapped_genes) else "  None",
    "",
    "GSE41613:",
    paste("  Samples:", ncol(expr41613_z)),
    paste("  Genes mapped:", nrow(expr41613_z)),
    paste("  Events:", sum(surv41613$event), "/", nrow(surv41613)),
    paste("  Survival time unit: months"),
    paste("  All HPV-negative OSCC"),
    "",
    "GSE42743:",
    paste("  Samples:", ncol(expr42743_z), "(oral cavity cancer only)"),
    paste("  Genes mapped:", nrow(expr42743_z)),
    paste("  Events:", sum(surv42743$event), "/", nrow(surv42743)),
    paste("  Survival time unit: days"),
    paste("  Filtered from", sum(oc_mask), "oral cavity cancer samples"),
    "",
    paste(rep("=", 70), collapse = ""),
    "Per-gene mapping details:",
    paste(rep("-", 70), collapse = "")
)

# Add per-gene details
for (i in seq_len(nrow(jetset_best))) {
    row <- jetset_best[i, ]
    report_lines <- c(report_lines,
        sprintf("  %s (%s): %s → %s [%s]",
                row$ensg, row$symbol, row$mapping_method,
                ifelse(!is.na(row$final_probe), row$final_probe, "UNMAPPED"),
                ifelse(!is.na(row$jetset_overall), 
                       sprintf("Jetset score=%.3f", row$jetset_overall), "N/A"))
    )
}

writeLines(report_lines, file.path(OUTPUT_DIR, "probe_mapping_report.txt"))

cat("\nFiles saved to:", OUTPUT_DIR, "\n")
cat("  GSE41613_expression.csv\n")
cat("  GSE41613_survival.csv\n")
cat("  GSE41613_clinical.csv\n")
cat("  GSE42743_expression.csv\n")
cat("  GSE42743_survival.csv\n")
cat("  GSE42743_clinical.csv\n")
cat("  probe_mapping_report.txt\n")

cat("\n", rep("=", 70), "\n", sep = "")
cat("Phase 8 preprocessing complete.\n")
cat(rep("=", 70), "\n", sep = "")
