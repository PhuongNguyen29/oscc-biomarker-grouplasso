################################################################################
# Phase 1: Data Preparation for OSCC Biomarker Discovery (v4 Pipeline)
# 
# This script implements Phase 1 of the corrected analysis pipeline:
#   Step 1.1: Gene filtering (biotype + TPM expression filtering)
#   Step 1.2: Preprocessing (log2 transform, batch correction, z-score)
#   Step 1.3: Clinical data alignment
#
# Key methodological corrections in v4:
#   - SVA for TCGA: NO survival variables in model (unsupervised)
#   - ComBat for ORIEN: NO survival variables in model (unsupervised)
#   - No IQR filtering, no univariate Cox pre-screening
#
# Input files (pre-batch TPM):
#   - expr_tmp_tcga.csv
#   - expr_tmp_orien_1112.csv
#   - clinical_tcga.csv
#   - clinical_orien_updated.csv
#   - surv_tcga.csv
#   - surv_orien_update.csv
#   - batch_info_orien_all.csv
#
# References:
#   - TPM filtering: GTEx Consortium methodology; Chen et al. (2016) F1000Research
#   - SVA: Leek & Storey (2007) PLoS Genetics; Leek et al. (2012) Bioinformatics
#   - ComBat: Johnson et al. (2007) Biostatistics
#
# Author: [Your Name]
# Date: [Date]
# Version: 4.0
################################################################################

# =============================================================================
# 0. SETUP AND CONFIGURATION
# =============================================================================

# Clear environment
rm(list = ls())

# Load required libraries
library(data.table)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(sva)
library(limma)
library(VennDiagram)
library(matrixStats)

# Set paths - UPDATE THESE FOR YOUR ENVIRONMENT
DATA_DIR <- "data/raw"
OUTPUT_DIR <- "data/processed"
PLOT_DIR <- "results/phase1_plots"

# Create output directories if they don't exist
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(PLOT_DIR, recursive = TRUE, showWarnings = FALSE)

# Set random seed for reproducibility
set.seed(42)

# =============================================================================
# 1. LOAD INPUT DATA
# =============================================================================

cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("PHASE 1: DATA PREPARATION\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n\n")

cat("Step 0: Loading input data...\n")

# Load pre-batch TPM expression data
expr_tmp_tcga <- read.csv(file.path(DATA_DIR, "expr_tmp_tcga.csv"), 
                          row.names = 1, check.names = FALSE)
expr_tmp_orien <- read.csv(file.path(DATA_DIR, "expr_tmp_orien_1112.csv"), 
                           row.names = 1, check.names = FALSE)

protein_coding_genes <- read.table(file.path(DATA_DIR, "protein_coding_genes_ensg.txt"))

# Load clinical and survival data
clinical_tcga <- read.csv(file.path(DATA_DIR, "clinical_tcga.csv"), row.names = 1)
clinical_orien <- read.csv(file.path(DATA_DIR, "clinical_orien_updated.csv"), row.names = 1)
surv_tcga <- read.csv(file.path(DATA_DIR, "surv_tcga.csv"), row.names = 1)
surv_orien <- read.csv(file.path(DATA_DIR, "surv_orien_update.csv"), row.names = 1)

# Load ORIEN batch information
batch_info_orien <- read.csv(file.path(DATA_DIR, "batch_info_orien_all.csv"), row.names = 1)
batch_info_tcga <- read.csv(file.path(DATA_DIR, "batch_info_tcga.csv"), row.names = 1)

cat(sprintf("  TCGA expression: %d genes x %d samples\n", 
            nrow(expr_tmp_tcga), ncol(expr_tmp_tcga)))
cat(sprintf("  ORIEN expression: %d genes x %d samples\n", 
            nrow(expr_tmp_orien), ncol(expr_tmp_orien)))
cat(sprintf("  TCGA survival: %d samples\n", nrow(surv_tcga)))
cat(sprintf("  ORIEN survival: %d samples\n", nrow(surv_orien)))
cat(sprintf(" Protein-coding gene: %d genes\n", length(protein_coding_genes$V1)))

tcga_genes <- rownames(expr_tmp_tcga)
orien_genes <- rownames(expr_tmp_orien)

cat("=== GENE SET SIZES ===\n")
cat(sprintf("TCGA: %d genes\n", length(tcga_genes)))
cat(sprintf("ORIEN: %d genes\n", length(orien_genes)))

# Intersect all three
common_genes_all <- Reduce(intersect, list(tcga_genes, orien_genes, protein_coding_genes$V1))
cat(sprintf("\nCommon genes (TCGA ∩ ORIEN ∩ Institutional): %d\n", length(common_genes_all)))

# =============================================================================
# 2. STEP 1.1: GENE FILTERING
# =============================================================================


cat("\n", "=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("STEP 1.1: GENE FILTERING\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")

#' Filter genes by TPM expression threshold
#' 
#' Retains genes with TPM > min_tpm in at least min_sample_fraction of samples.
#' This threshold is more conservative than GTEx (TPM > 0.1) for expression level,
#' while using a permissive sample fraction appropriate for heterogeneous cancer cohorts.
#' 
#' @references 
#'   - GTEx Consortium methodology (TPM > 0.1 in 20% samples)
#'   - Chen et al. (2016) F1000Research - edgeR filtering principles
#'
#' @param expr_tpm Expression matrix (genes x samples) in TPM units
#' @param min_tpm Minimum TPM threshold (default: 0.5)
#' @param min_sample_fraction Minimum fraction of samples (default: 0.1 = 10%)
#' @param dataset_name Name for logging
#' @return List with filtered expression and metadata
filter_genes_by_expression <- function(expr_tpm,
                                       common_genes_all,
                                       min_tpm = 0.5, 
                                       min_sample_fraction = 0.1,
                                       dataset_name = "Dataset") {
  
  cat(sprintf("\nProcessing %s\n", dataset_name))
  cat("-" %>% rep(50) %>% paste(collapse = ""), "\n")
  
  # Ensure matrix format

  expr_mat <- as.matrix(expr_tpm)
  
  # Initial dimensions
  cat(sprintf("Initial dimensions: %d genes x %d samples\n", 
              nrow(expr_mat), ncol(expr_mat)))
  
  # Remove genes with NA values
  if (any(is.na(expr_mat))) {
    genes_with_na <- rowSums(is.na(expr_mat)) > 0
    n_genes_with_na <- sum(genes_with_na)
    cat(sprintf("Removing %d genes with NA values (%.2f%%)\n", 
                n_genes_with_na, 
                n_genes_with_na / nrow(expr_mat) * 100))
    expr_mat <- expr_mat[!genes_with_na, ]
  }
  
  expr_filtered <- expr_mat[common_genes_all, ]
  cat(sprintf("Filtered dimensions: %d genes x %d samples\n", 
              nrow(expr_filtered), ncol(expr_filtered)))

  
  # Calculate minimum number of samples required
  min_samples <- ceiling(min_sample_fraction * ncol(expr_filtered))
  cat(sprintf("Requiring TPM > %.1f in >= %d samples (%.0f%% of %d)\n",
              min_tpm, min_samples, min_sample_fraction * 100, ncol(expr_filtered)))
  
  # Count samples exceeding threshold for each gene
  genes_passing <- rowSums(expr_filtered > min_tpm) >= min_samples
  
  # Filter genes
  expr_filtered <- expr_filtered[genes_passing, ]
  
  # Summary statistics
  cat(sprintf("\nFiltering Results:\n"))
  cat(sprintf("  Genes passing TPM filter: %d (%.1f%%)\n", 
              sum(genes_passing), 
              sum(genes_passing) / nrow(expr_filtered) * 100))
  cat(sprintf("  Genes removed: %d\n", 
              nrow(expr_filtered) - sum(genes_passing)))
  
  return(list(
    expr_filtered = expr_filtered,
    retained_genes = rownames(expr_filtered),
    n_genes_initial = nrow(expr_mat),
    n_genes_retained = sum(genes_passing),
    params = list(min_tpm = min_tpm, min_sample_fraction = min_sample_fraction)
  ))
}

# Apply filtering to each cohort independently
tcga_filter_result <- filter_genes_by_expression(
  expr_tmp_tcga,
  common_genes_all,
  min_tpm = 0.5,
  min_sample_fraction = 0.1,
  dataset_name = "TCGA"
)

orien_filter_result <- filter_genes_by_expression(
  expr_tmp_orien,
  common_genes_all,
  min_tpm = 0.5,
  min_sample_fraction = 0.1,
  dataset_name = "ORIEN"
)

# Find intersection of retained genes (shared gene set)
common_genes <- intersect(tcga_filter_result$retained_genes, 
                          orien_filter_result$retained_genes)

cat("\n", "-" %>% rep(50) %>% paste(collapse = ""), "\n")
cat("GENE INTERSECTION ANALYSIS\n")
cat("-" %>% rep(50) %>% paste(collapse = ""), "\n")
cat(sprintf("TCGA retained: %d genes\n", tcga_filter_result$n_genes_retained))
cat(sprintf("ORIEN retained: %d genes\n", orien_filter_result$n_genes_retained))
cat(sprintf("Common genes (intersection): %d\n", length(common_genes)))
cat(sprintf("TCGA-specific: %d\n", 
            length(setdiff(tcga_filter_result$retained_genes, common_genes))))
cat(sprintf("ORIEN-specific: %d\n", 
            length(setdiff(orien_filter_result$retained_genes, common_genes))))

# Create Venn diagram
venn.plot <- venn.diagram(
  x = list(
    TCGA = tcga_filter_result$retained_genes,
    ORIEN = orien_filter_result$retained_genes
  ),
  filename = file.path(PLOT_DIR, "gene_filtering_venn.png"),
  output = TRUE,
  imagetype = "png",
  height = 2400,
  width = 2400,
  resolution = 300,
  fill = c("#E41A1C", "#377EB8"),
  alpha = 0.5,
  cex = 1.5,
  cat.cex = 1.5,
  cat.fontface = "bold",
  main = "Gene Retention After TPM Filtering\n(TPM > 0.5 in ≥10% samples)",
  main.cex = 1.5
)
cat(sprintf("\nVenn diagram saved to: %s\n", file.path(PLOT_DIR, "gene_filtering_venn.png")))

# Subset to common genes
tcga_filtered <- tcga_filter_result$expr_filtered[common_genes, ]
orien_filtered <- orien_filter_result$expr_filtered[common_genes, ]

cat(sprintf("\nFinal dimensions after filtering to common genes:\n"))
cat(sprintf("  TCGA: %d genes x %d samples\n", nrow(tcga_filtered), ncol(tcga_filtered)))
cat(sprintf("  ORIEN: %d genes x %d samples\n", nrow(orien_filtered), ncol(orien_filtered)))

# =============================================================================
# 3. STEP 1.2: PREPROCESSING
# =============================================================================

cat("\n", "=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("STEP 1.2: PREPROCESSING\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")

# -----------------------------------------------------------------------------
# 3.1 Log2 Transformation
# -----------------------------------------------------------------------------

cat("\n3.1 Log2(TPM + 1) Transformation\n")
cat("-" %>% rep(50) %>% paste(collapse = ""), "\n")

tcga_log <- log2(tcga_filtered + 1)
orien_log <- log2(orien_filtered + 1)

cat("Log2 transformation applied to both cohorts.\n")

# -----------------------------------------------------------------------------
# 3.2 Batch Correction - ORIEN (ComBat with known batches)
# -----------------------------------------------------------------------------

cat("\n3.2 Batch Correction - ORIEN (ComBat)\n")
cat("-" %>% rep(50) %>% paste(collapse = ""), "\n")

#' Apply ComBat batch correction WITHOUT survival variables
#' 
#' CRITICAL: v4 pipeline requires batch correction without outcome variables
#' to avoid removing survival-associated biological signal that correlates with batch.
#' 
#' @references Johnson et al. (2007) Biostatistics
apply_combat_correction <- function(expr_log, batch_info, 
                                    batch_col = "NexusBatch",
                                    sample_col = "ORIENAvatarKey",
                                    min_batch_size = 3) {
  
  # Align batch info to expression samples
  expr_samples <- colnames(expr_log)
  
  # Match by column if it exists, otherwise by rownames
  if (sample_col %in% colnames(batch_info)) {
    batch_info_aligned <- batch_info[match(expr_samples, batch_info[[sample_col]]), ]
    na_samples <- sum(is.na(batch_info_aligned[[sample_col]]))
  } else {
    batch_info_aligned <- batch_info[match(expr_samples, rownames(batch_info)), ]
    na_samples <- sum(is.na(batch_info_aligned[[batch_col]]))
  }
  
  # Check for missing batch info
  if (na_samples > 0) {
    cat(sprintf("Warning: %d samples have no batch information - removing\n", na_samples))
    keep_samples <- !is.na(batch_info_aligned[[batch_col]])
    expr_log <- expr_log[, keep_samples]
    batch_info_aligned <- batch_info_aligned[keep_samples, ]
  }
  
  # Extract batch variable
  batch <- factor(batch_info_aligned[[batch_col]])
  
  cat(sprintf("Samples with batch info: %d\n", ncol(expr_log)))
  cat(sprintf("Number of batches: %d\n", length(unique(batch))))
  
  # Handle small batches
  batch_sizes <- table(batch)
  small_batches <- names(batch_sizes)[batch_sizes < min_batch_size]
  
  if (length(small_batches) > 0) {
    cat(sprintf("Merging %d small batches (< %d samples) into 'SmallBatches'\n",
                length(small_batches), min_batch_size))
    batch_merged <- as.character(batch)
    batch_merged[batch_merged %in% small_batches] <- "SmallBatches"
    batch <- factor(batch_merged)
    cat(sprintf("Final number of batches: %d\n", length(unique(batch))))
  }
  
  # Apply ComBat with intercept-only model (NO survival variables)
  cat("\nApplying ComBat (intercept-only model - NO survival variables)...\n")
  mod <- model.matrix(~ 1, data = data.frame(sample = colnames(expr_log)))
  
  expr_corrected <- ComBat(
    dat = as.matrix(expr_log),
    batch = batch,
    mod = mod,
    par.prior = TRUE,
    prior.plots = FALSE
  )
  
  cat("ComBat correction complete.\n")
  
  return(list(
    expr_corrected = as.data.frame(expr_corrected),
    batch = batch,
    n_batches = length(unique(batch))
  ))
}

orien_combat_result <- apply_combat_correction(
  orien_log, batch_info_orien,
  batch_col = "NexusBatch", sample_col = "ORIENAvatarKey"
)
orien_corrected <- orien_combat_result$expr_corrected

tcga_combat_result <- apply_combat_correction(
  tcga_log, batch_info_tcga,
  batch_col = "plate_id", sample_col = "sampleID"
)
tcga_corrected <- tcga_combat_result$expr_corrected

# -----------------------------------------------------------------------------
# 3.4 Z-score Standardization
# -----------------------------------------------------------------------------

cat("\n3.4 Z-score Standardization\n")
cat("-" %>% rep(50) %>% paste(collapse = ""), "\n")

#' Z-score standardization with outlier clipping
#' 
#' Standardize each gene to mean=0, SD=1 within each cohort.
#' Clip outliers to ±3 SD to reduce influence of extreme values.
#' 
#' @param expr_df Expression data frame (genes x samples)
#' @param clip_sd SD threshold for clipping (default: 3)
#' @return Standardized expression data frame
standardize_expression <- function(expr_df, clip_sd = 3) {
  
  expr_mat <- as.matrix(expr_df)
  
  # Z-score standardization (per gene)
  gene_means <- rowMeans(expr_mat)
  gene_sds <- rowSds(expr_mat)
  
  # Avoid division by zero for zero-variance genes
  gene_sds[gene_sds == 0] <- 1
  
  expr_zscore <- (expr_mat - gene_means) / gene_sds
  
  # Clip outliers to ±clip_sd
  n_clipped <- sum(abs(expr_zscore) > clip_sd, na.rm = TRUE)
  pct_clipped <- n_clipped / length(expr_zscore) * 100
  
  expr_zscore[expr_zscore > clip_sd] <- clip_sd
  expr_zscore[expr_zscore < -clip_sd] <- -clip_sd
  
  cat(sprintf("Clipped %.2f%% of values to ±%.0f SD\n", pct_clipped, clip_sd))
  
  return(as.data.frame(expr_zscore))
}

tcga_final <- standardize_expression(tcga_corrected, clip_sd = 3)
orien_final <- standardize_expression(orien_corrected, clip_sd = 3)

cat(sprintf("\nFinal preprocessed dimensions:\n"))
cat(sprintf("  TCGA: %d genes x %d samples\n", nrow(tcga_final), ncol(tcga_final)))
cat(sprintf("  ORIEN: %d genes x %d samples\n", nrow(orien_final), ncol(orien_final)))

# =============================================================================
# 4. STEP 1.3: SAMPLE ALIGNMENT WITH SURVIVAL DATA
# =============================================================================

cat("\n", "=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("STEP 1.3: SAMPLE ALIGNMENT\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")

# Align TCGA samples
tcga_common_samples <- intersect(colnames(tcga_final), surv_tcga$sampleID)
tcga_final_aligned <- tcga_final[, tcga_common_samples]
surv_tcga_aligned <- surv_tcga[match(tcga_common_samples, surv_tcga$sampleID), ]
clinical_tcga_aligned <- clinical_tcga[match(tcga_common_samples, clinical_tcga$sampleID), ]

cat(sprintf("TCGA samples aligned: %d\n", length(tcga_common_samples)))

# Align ORIEN samples
orien_common_samples <- intersect(colnames(orien_final), surv_orien$sampleID)
orien_final_aligned <- orien_final[, orien_common_samples]
surv_orien_aligned <- surv_orien[match(orien_common_samples, surv_orien$sampleID), ]
clinical_orien_aligned <- clinical_orien[match(orien_common_samples, clinical_orien$sampleID), ]

cat(sprintf("ORIEN samples aligned: %d\n", length(orien_common_samples)))

# Summary statistics
cat("\nSurvival Summary:\n")
cat(sprintf("  TCGA: %d events / %d samples (%.1f%% event rate)\n",
            sum(surv_tcga_aligned$event), nrow(surv_tcga_aligned),
            sum(surv_tcga_aligned$event) / nrow(surv_tcga_aligned) * 100))
cat(sprintf("  ORIEN: %d events / %d samples (%.1f%% event rate)\n",
            sum(surv_orien_aligned$event), nrow(surv_orien_aligned),
            sum(surv_orien_aligned$event) / nrow(surv_orien_aligned) * 100))

# =============================================================================
# 5. QUALITY CONTROL PLOTS
# =============================================================================

cat("\n", "=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("QUALITY CONTROL PLOTS\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")

#' Create PCA plot for batch effect visualization
create_pca_batch_plot <- function(expr_before, expr_after,
                                  batch_vector,
                                  dataset_name,
                                  n_top_batches = 6,
                                  output_file = NULL) {
  
  # Simplify batch labels: keep top N, rest as "Other"
  batch_sizes <- sort(table(batch_vector), decreasing = TRUE)
  top_batches <- names(batch_sizes)[1:min(n_top_batches, length(batch_sizes))]
  batch_simplified <- ifelse(batch_vector %in% top_batches, 
                             as.character(batch_vector), "Other")
  batch_simplified <- factor(batch_simplified, 
                             levels = c(top_batches, "Other"))
  
  # PCA before correction
  pca_before <- prcomp(t(as.matrix(expr_before)), scale. = TRUE, center = TRUE)
  var_before <- summary(pca_before)$importance[2, 1:2] * 100
  
  # PCA after correction
  pca_after <- prcomp(t(as.matrix(expr_after)), scale. = TRUE, center = TRUE)
  var_after <- summary(pca_after)$importance[2, 1:2] * 100
  
  # Create plot data
  plot_data <- rbind(
    data.frame(PC1 = pca_before$x[, 1], PC2 = pca_before$x[, 2],
               Batch = batch_simplified, Stage = "Before Correction"),
    data.frame(PC1 = pca_after$x[, 1], PC2 = pca_after$x[, 2],
               Batch = batch_simplified, Stage = "After Correction")
  )
  plot_data$Stage <- factor(plot_data$Stage,
                            levels = c("Before Correction", "After Correction"))
  
  # Color palette: distinct colors for top batches, grey for Other
  n_colors <- length(top_batches)
  batch_colors <- c(scales::hue_pal()(n_colors), "grey70")
  names(batch_colors) <- c(top_batches, "Other")
  
  p <- ggplot(plot_data, aes(x = PC1, y = PC2, color = Batch)) +
    geom_point(alpha = 0.6, size = 1.5) +
    facet_wrap(~ Stage, scales = "free") +
    scale_color_manual(values = batch_colors) +
    theme_minimal() +
    theme(
      strip.text = element_text(size = 12, face = "bold"),
      axis.title = element_text(size = 11),
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      legend.position = "bottom"
    ) +
    labs(
      title = paste(dataset_name, "- Batch Correction Effect"),
      x = sprintf("PC1 (%.1f%% / %.1f%% var)", var_before[1], var_after[1]),
      y = sprintf("PC2 (%.1f%% / %.1f%% var)", var_before[2], var_after[2]),
      color = "Batch"
    )
  
  if (!is.null(output_file)) {
    ggsave(output_file, p, width = 12, height = 6, dpi = 300)
    cat(sprintf("PCA plot saved to: %s\n", output_file))
  }
  
  return(p)
}
# TCGA batch vector
tcga_batch_aligned <- batch_info_tcga[match(colnames(tcga_log), 
                                                  rownames(batch_info_tcga)), ]
tcga_batch_vector <- tcga_batch_aligned$plate_id

# ORIEN batch vector
orien_batch_aligned <- batch_info_orien[match(colnames(orien_log), 
                                              batch_info_orien$ORIENAvatarKey), ]
orien_batch_vector <- orien_batch_aligned$NexusBatch

# Generate PCA plots
p_tcga <- create_pca_batch_plot(
  tcga_log, tcga_corrected,
  tcga_batch_vector, "TCGA (ComBat - Plate ID)",
  n_top_batches = 6,
  output_file = file.path(PLOT_DIR, "pca_tcga_batch_correction.png")
)

p_orien <- create_pca_batch_plot(
  orien_log, orien_corrected,
  orien_batch_vector, "ORIEN (ComBat - NexusBatch)",
  n_top_batches = 6,
  output_file = file.path(PLOT_DIR, "pca_orien_batch_correction.png")
)

# -----------------------------------------------------------------------------
# 5.1 ANOVA: Quantitative Batch Effect Assessment
# -----------------------------------------------------------------------------
# Tests whether batch variable explains significant variance in top PCs
# before and after ComBat correction.
# Reference: Leek et al. (2010) Nature Reviews Genetics - batch effect detection

cat("\n5.1 ANOVA: Quantitative Batch Effect Assessment\n")
cat("-" %>% rep(50) %>% paste(collapse = ""), "\n")

assess_batch_anova <- function(expr_before, expr_after, batch_vector, dataset_name) {
  
  cat(sprintf("\n%s Batch Effect ANOVA\n", dataset_name))
  cat("-" %>% rep(40) %>% paste(collapse = ""), "\n")
  
  # PCA before correction
  pca_before <- prcomp(t(as.matrix(expr_before)), scale. = TRUE, center = TRUE)
  # PCA after correction
  pca_after <- prcomp(t(as.matrix(expr_after)), scale. = TRUE, center = TRUE)
  
  results <- data.frame(
    Dataset = character(),
    Stage = character(),
    PC = character(),
    Batch_SS = numeric(),
    Residual_SS = numeric(),
    R_squared = numeric(),
    F_value = numeric(),
    p_value = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (pc_idx in 1:2) {
    pc_label <- paste0("PC", pc_idx)
    
    # Before correction
    aov_before <- summary(aov(pca_before$x[, pc_idx] ~ batch_vector))
    ss_batch_b <- aov_before[[1]]["batch_vector", "Sum Sq"]
    ss_resid_b <- aov_before[[1]]["Residuals", "Sum Sq"]
    r2_before <- ss_batch_b / (ss_batch_b + ss_resid_b)
    f_before <- aov_before[[1]]["batch_vector", "F value"]
    p_before <- aov_before[[1]]["batch_vector", "Pr(>F)"]
    
    # After correction
    aov_after <- summary(aov(pca_after$x[, pc_idx] ~ batch_vector))
    ss_batch_a <- aov_after[[1]]["batch_vector", "Sum Sq"]
    ss_resid_a <- aov_after[[1]]["Residuals", "Sum Sq"]
    r2_after <- ss_batch_a / (ss_batch_a + ss_resid_a)
    f_after <- aov_after[[1]]["batch_vector", "F value"]
    p_after <- aov_after[[1]]["batch_vector", "Pr(>F)"]
    
    results <- rbind(results, data.frame(
      Dataset = dataset_name,
      Stage = "Before",
      PC = pc_label,
      Batch_SS = round(ss_batch_b, 1),
      Residual_SS = round(ss_resid_b, 1),
      R_squared = round(r2_before, 4),
      F_value = round(f_before, 3),
      p_value = p_before
    ))
    results <- rbind(results, data.frame(
      Dataset = dataset_name,
      Stage = "After",
      PC = pc_label,
      Batch_SS = round(ss_batch_a, 1),
      Residual_SS = round(ss_resid_a, 1),
      R_squared = round(r2_after, 4),
      F_value = round(f_after, 3),
      p_value = p_after
    ))
    
    cat(sprintf("  %s Before: R²=%.3f, F=%.2f, p=%.4g\n", 
                pc_label, r2_before, f_before, p_before))
    cat(sprintf("  %s After:  R²=%.3f, F=%.2f, p=%.4g\n", 
                pc_label, r2_after, f_after, p_after))
  }
  
  return(results)
}

anova_tcga <- assess_batch_anova(tcga_log, tcga_corrected, tcga_batch_vector, "TCGA")
anova_orien <- assess_batch_anova(orien_log, orien_corrected, orien_batch_vector, "ORIEN")

# Combine and save
anova_results <- rbind(anova_tcga, anova_orien)
write.csv(anova_results, file.path(PLOT_DIR, "batch_anova_results.csv"), row.names = FALSE)
cat(sprintf("\nANOVA results saved to: %s\n", file.path(PLOT_DIR, "batch_anova_results.csv")))

# =============================================================================
# 6. SAVE OUTPUTS
# =============================================================================

cat("\n", "=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("SAVING OUTPUTS\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")

# Save preprocessed expression data
write.csv(tcga_final_aligned, 
          file.path(OUTPUT_DIR, "tcga_preprocessed_v4.csv"))
write.csv(orien_final_aligned, 
          file.path(OUTPUT_DIR, "orien_preprocessed_v4.csv"))

# Save aligned survival data
write.csv(surv_tcga_aligned, 
          file.path(OUTPUT_DIR, "surv_tcga_aligned_v4.csv"))
write.csv(surv_orien_aligned, 
          file.path(OUTPUT_DIR, "surv_orien_aligned_v4.csv"))

# Save aligned clinical data
write.csv(clinical_tcga_aligned, 
          file.path(OUTPUT_DIR, "clinical_tcga_aligned_v4.csv"))
write.csv(clinical_orien_aligned, 
          file.path(OUTPUT_DIR, "clinical_orien_aligned_v4.csv"))

# Save common genes list
write.table(common_genes, 
            file.path(OUTPUT_DIR, "common_genes_v4.txt"),
            row.names = FALSE, col.names = FALSE, quote = FALSE)

# Save preprocessing metadata
metadata <- list(
  pipeline_version = "4.0",
  date_processed = Sys.Date(),
  gene_filtering = list(
    min_tpm = 0.5,
    min_sample_fraction = 0.1,
    tcga_genes_initial = tcga_filter_result$n_genes_initial,
    tcga_genes_retained = tcga_filter_result$n_genes_retained,
    orien_genes_initial = orien_filter_result$n_genes_initial,
    orien_genes_retained = orien_filter_result$n_genes_retained,
    common_genes = length(common_genes)
  ),
  batch_correction = list(
    tcga_method = "ComBat (plate ID, intercept-only model)",
    tcga_n_batches = tcga_combat_result$n_batches,
    orien_method = "ComBat (intercept-only model)",
    orien_n_batches = orien_combat_result$n_batches
  ),
  standardization = list(
    method = "z-score per gene",
    clip_sd = 3
  ),
  final_dimensions = list(
    tcga_genes = nrow(tcga_final_aligned),
    tcga_samples = ncol(tcga_final_aligned),
    orien_genes = nrow(orien_final_aligned),
    orien_samples = ncol(orien_final_aligned)
  ),
  survival_summary = list(
    tcga_events = sum(surv_tcga_aligned$event),
    tcga_total = nrow(surv_tcga_aligned),
    orien_events = sum(surv_orien_aligned$event),
    orien_total = nrow(surv_orien_aligned)
  )
)

saveRDS(metadata, file.path(OUTPUT_DIR, "preprocessing_metadata_v4.rds"))

cat("\nOutput files saved:\n")
cat(sprintf("  %s/tcga_preprocessed_v4.csv\n", OUTPUT_DIR))
cat(sprintf("  %s/orien_preprocessed_v4.csv\n", OUTPUT_DIR))
cat(sprintf("  %s/surv_tcga_aligned_v4.csv\n", OUTPUT_DIR))
cat(sprintf("  %s/surv_orien_aligned_v4.csv\n", OUTPUT_DIR))
cat(sprintf("  %s/clinical_tcga_aligned_v4.csv\n", OUTPUT_DIR))
cat(sprintf("  %s/clinical_orien_aligned_v4.csv\n", OUTPUT_DIR))
cat(sprintf("  %s/common_genes_v4.txt\n", OUTPUT_DIR))
cat(sprintf("  %s/preprocessing_metadata_v4.rds\n", OUTPUT_DIR))

# =============================================================================
# 7. FINAL SUMMARY
# =============================================================================

cat("\n", "=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("PHASE 1 COMPLETE - SUMMARY\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")

cat("\nPreprocessing Pipeline v4:\n")
cat("  1. Gene Filtering: TPM > 0.5 in ≥10% samples (per cohort)\n")
cat("  2. Log Transformation: log2(TPM + 1)\n")
cat("  3. Batch Correction:\n")
cat("     - TCGA: SVA (unsupervised, NO survival variables)\n")
cat("     - ORIEN: ComBat (NO survival variables)\n")
cat("  4. Standardization: Z-score per gene, clipped to ±3 SD\n")

cat(sprintf("\nFinal Data:\n"))
cat(sprintf("  Common genes: %d\n", length(common_genes)))
cat(sprintf("  TCGA: %d samples (%d events, %.1f%%)\n", 
            ncol(tcga_final_aligned), 
            sum(surv_tcga_aligned$event),
            sum(surv_tcga_aligned$event) / nrow(surv_tcga_aligned) * 100))
cat(sprintf("  ORIEN: %d samples (%d events, %.1f%%)\n", 
            ncol(orien_final_aligned),
            sum(surv_orien_aligned$event),
            sum(surv_orien_aligned$event) / nrow(surv_orien_aligned) * 100))

cat("\n", "=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("Ready for Phase 2: Hyperparameter Tuning\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")
