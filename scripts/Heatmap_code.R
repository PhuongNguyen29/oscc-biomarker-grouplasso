# =============================================================
# ComplexHeatmap Visualization for 25-Gene PSN
# =============================================================


library(ComplexHeatmap)
library(circlize)
library(RColorBrewer)
library(grid)

# =============================================================
# Configuration
# =============================================================
BASE_DIR <- "results/mutation_analysis_25gene/r_heatmap_data"
OUTPUT_DIR <- "results/mutation_analysis_25gene/heatmaps"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Mutation type definitions
lof_types <- c("Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", 
               "Splice_Site", "NONSENSE", "FRAME_SHIFT_DEL", 
               "FRAME_SHIFT_INS", "SPLICE_SITE")
missense_types <- c("Missense_Mutation", "In_Frame_Del", "In_Frame_Ins",
                    "MISSENSE", "IN_FRAME_DEL", "IN_FRAME_INS")

# =============================================================
# Main heatmap function
# =============================================================
create_25gene_heatmap <- function(
    cohort_name,        # "TCGA" or "ORIEN"
    network_type,       # "expression" or "snf"
    mutation_data,      # Full mutation data frame for this cohort
    gene_col,           # Column name for gene symbol in mutation data
    sample_col,         # Column name for sample ID in mutation data
    variant_col,        # Column name for variant classification
    width = 3200,
    height = 2200,
    res = 150
) {
  
  prefix <- paste0(tolower(cohort_name), "_", tolower(network_type))
  cat(sprintf("\n=== Creating heatmap: %s %s ===\n", cohort_name, network_type))
  
  # ---------------------------------------------------------
  # 1. Load data
  # ---------------------------------------------------------
  meta <- read.csv(file.path(BASE_DIR, paste0(prefix, "_patient_metadata.csv")),
                   stringsAsFactors = FALSE)
  expr <- read.csv(file.path(BASE_DIR, paste0(prefix, "_expression_ordered.csv")),
                   row.names = 1)
  mut_binary <- read.csv(file.path(BASE_DIR, paste0(prefix, "_mutation_binary.csv")),
                         row.names = 1)
  
  cat(sprintf("  Patients: %d, Genes: %d\n", nrow(meta), ncol(expr)))
  
  # ---------------------------------------------------------
  # 2. Order patients by community, then by risk group
  # ---------------------------------------------------------
  meta <- meta[order(meta$community, meta$risk_group, -meta$risk_score), ]
  ordered_patients <- meta$sampleID
  
  # Align expression and mutation matrices
  expr <- expr[ordered_patients, , drop = FALSE]
  mut_binary <- mut_binary[ordered_patients, , drop = FALSE]
  
  # ---------------------------------------------------------
  # 3. Build LoF and Missense mutation matrices from raw data
  # ---------------------------------------------------------
  gene_symbols <- colnames(expr)
  n_patients <- length(ordered_patients)
  
  # Initialize matrices
  lof_matrix <- matrix(0, nrow = n_patients, ncol = length(gene_symbols),
                       dimnames = list(ordered_patients, gene_symbols))
  missense_matrix <- matrix(0, nrow = n_patients, ncol = length(gene_symbols),
                            dimnames = list(ordered_patients, gene_symbols))
  
  # Filter mutation data to our patients and genes
  mut_filtered <- mutation_data[
    mutation_data[[sample_col]] %in% ordered_patients &
      mutation_data[[gene_col]] %in% gene_symbols, 
  ]
  
  if (nrow(mut_filtered) > 0) {
    for (i in 1:nrow(mut_filtered)) {
      s <- mut_filtered[[sample_col]][i]
      g <- mut_filtered[[gene_col]][i]
      vc <- mut_filtered[[variant_col]][i]
      
      if (s %in% ordered_patients && g %in% gene_symbols) {
        if (vc %in% lof_types) {
          lof_matrix[s, g] <- 1
        } else if (vc %in% missense_types) {
          missense_matrix[s, g] <- 1
        }
      }
    }
  }
  
  cat(sprintf("  LoF mutations: %d, Missense mutations: %d\n",
              sum(lof_matrix), sum(missense_matrix)))
  
  # ---------------------------------------------------------
  # 4. Scale expression for heatmap
  # ---------------------------------------------------------
  expr_matrix <- as.matrix(expr)
  # Clip to [-2, 2] for visualization
  expr_matrix[expr_matrix > 2] <- 2
  expr_matrix[expr_matrix < -2] <- -2
  
  # ---------------------------------------------------------
  # 5. Community split factor
  # ---------------------------------------------------------
  community_factor <- factor(meta$community, 
                             levels = sort(unique(meta$community)))
  
  # ---------------------------------------------------------
  # 6. Row annotations (patient metadata)
  # ---------------------------------------------------------
  
  # Community colors
  n_comm <- length(unique(meta$community))
  comm_colors <- setNames(
    colorRampPalette(brewer.pal(min(n_comm, 12), "Set3"))(n_comm),
    sort(unique(meta$community))
  )
  
  # Risk group colors
  risk_colors <- c("High Risk" = "#E74C3C", "Low Risk" = "#2E86DE")
  
  # Risk score color function
  risk_score_range <- range(meta$risk_score, na.rm = TRUE)
  risk_score_col <- colorRamp2(
    c(risk_score_range[1], 0, risk_score_range[2]),
    c("#2166AC", "white", "#B2182B")
  )
  
  # Clinical variable colors
  gender_colors <- c("0" = "#FF9999", "1" = "#6699CC")
  smoking_colors <- c("Ever" = "#E69F00", "Never" = "#009E73", "Unknown" = "#999999")
  alcohol_colors <- c("Ever" = "#E69F00", "Never" = "#009E73", "Unknown" = "#999999")
  t_stage_colors <- c("T1-T2" = "#56B4E9", "T3-T4" = "#D55E00", "Unknown" = "#999999")
  n_stage_colors <- c("N0" = "#56B4E9", "N1" = "#F0E442", 
                      "N2-3" = "#D55E00", "Unknown" = "#999999")
  event_colors <- c("0" = "#CCCCCC", "1" = "#333333")
  
  # Handle potential NA or missing values
  meta$gender <- as.character(meta$gender)
  meta$smoking[is.na(meta$smoking)] <- "Unknown"
  meta$alcohol[is.na(meta$alcohol)] <- "Unknown"
  meta$T_stage[is.na(meta$T_stage)] <- "Unknown"
  meta$N_stage[is.na(meta$N_stage)] <- "Unknown"
  meta$event <- as.character(as.integer(meta$event))
  
  # Mutation count
  mutation_count <- rowSums(mut_binary)
  mut_count_col <- colorRamp2(c(0, max(mutation_count, 1)), c("white", "#8B0000"))
  
  row_ha <- rowAnnotation(
    Community = factor(meta$community, levels = sort(unique(meta$community))),
    RiskGroup = meta$risk_group,
    RiskScore = meta$risk_score,
    Age = meta$age,
    Gender = meta$gender,
    Smoking = meta$smoking,
    Alcohol = meta$alcohol,
    T_stage = meta$T_stage,
    N_stage = meta$N_stage,
    Time = meta$time,
    Event = meta$event,
    Mutations = mutation_count,
    col = list(
      Community = comm_colors,
      RiskGroup = risk_colors,
      RiskScore = risk_score_col,
      Age = colorRamp2(c(20, 50, 80), c("#FFFFCC", "#FD8D3C", "#800026")),
      Gender = gender_colors,
      Smoking = smoking_colors,
      Alcohol = alcohol_colors,
      T_stage = t_stage_colors,
      N_stage = n_stage_colors,
      Time = colorRamp2(c(0, max(meta$time, na.rm = TRUE)), c("white", "#006400")),
      Event = event_colors,
      Mutations = mut_count_col
    ),
    show_annotation_name = TRUE,
    annotation_name_side = "top",
    annotation_name_gp = gpar(fontsize = 8),
    gap = unit(1, "mm"),
    width = unit(6, "cm")
  )
  
  # ---------------------------------------------------------
  # 7. Build heatmap list
  # ---------------------------------------------------------
  
  # Convert to character matrices for mutation heatmaps
  lof_char <- matrix(as.character(lof_matrix), nrow = nrow(lof_matrix),
                     dimnames = dimnames(lof_matrix))
  mis_char <- matrix(as.character(missense_matrix), nrow = nrow(missense_matrix),
                     dimnames = dimnames(missense_matrix))
  
  # LoF heatmap
  ht_lof <- Heatmap(
    lof_char,
    name = "LoF",
    col = c("0" = "white", "1" = "#D55E00"),
    cluster_rows = FALSE,
    cluster_columns = TRUE,
    show_row_names = FALSE,
    show_column_names = TRUE,
    column_names_gp = gpar(fontsize = 7),
    column_names_rot = 45,
    column_names_side = "top",
    row_split = community_factor,
    column_title = "LoF Mutations",
    column_title_side = "top",
    column_title_gp = gpar(fontsize = 10),
    row_title_gp = gpar(fontsize = 8),
    border = TRUE,
    height = unit(12, "cm"),
    heatmap_legend_param = list(
      title = "LoF",
      labels = c("WT", "Mutated"),
      at = c("0", "1")
    )
  )
  
  # Missense heatmap
  ht_missense <- Heatmap(
    mis_char,
    name = "Missense",
    col = c("0" = "white", "1" = "#0072B2"),
    cluster_rows = FALSE,
    cluster_columns = TRUE,
    show_row_names = FALSE,
    show_column_names = TRUE,
    column_names_gp = gpar(fontsize = 7),
    column_names_rot = 45,
    column_names_side = "top",
    row_split = community_factor,
    column_title = "Missense Mutations",
    column_title_side = "top",
    column_title_gp = gpar(fontsize = 10),
    row_title_gp = gpar(fontsize = 8),
    border = TRUE,
    height = unit(12, "cm"),
    heatmap_legend_param = list(
      title = "Missense",
      labels = c("WT", "Mutated"),
      at = c("0", "1")
    )
  )
  
  # Expression heatmap
  ht_expr <- Heatmap(
    expr_matrix,
    name = "Expression",
    col = colorRamp2(c(-2, 0, 2), c("#2166AC", "white", "#B2182B")),
    cluster_rows = FALSE,
    cluster_columns = TRUE,
    show_row_names = FALSE,
    show_column_names = TRUE,
    column_names_gp = gpar(fontsize = 7),
    column_names_rot = 45,
    column_names_side = "top",
    row_split = community_factor,
    column_title = "Gene Expression",
    column_title_side = "top",
    column_title_gp = gpar(fontsize = 10),
    row_title_gp = gpar(fontsize = 8),
    border = TRUE,
    use_raster = FALSE,
    height = unit(12, "cm"),
    heatmap_legend_param = list(
      title = "Expression\n(Z-score)",
      at = c(-2, -1, 0, 1, 2)
    )
  )
  
  # Combine: annotations + LoF + Missense + Expression
  heatmap_list <- row_ha + ht_lof + ht_missense + ht_expr
  
  # ---------------------------------------------------------
  # 8. Draw and save
  # ---------------------------------------------------------
  plot_title <- sprintf(
    "%s %s Network Communities (n=%d, Communities=%d)",
    cohort_name,
    tools::toTitleCase(network_type),
    n_patients,
    n_comm
  )
  
  # Save to file
  filename <- file.path(OUTPUT_DIR, 
                        paste0(prefix, "_network_cluster_heatmap.png"))
  
  png(filename, width = width, height = height, res = res)
  
  draw(heatmap_list,
       column_title = plot_title,
       column_title_gp = gpar(fontsize = 14, fontface = "bold"),
       merge_legend = TRUE,
       heatmap_legend_side = "bottom",
       annotation_legend_side = "right",
       padding = unit(c(2, 2, 2, 5), "mm"))
  
  dev.off()
  
  cat(sprintf("  Saved: %s\n", filename))
  
  # Also display
  draw(heatmap_list,
       column_title = plot_title,
       column_title_gp = gpar(fontsize = 14, fontface = "bold"),
       merge_legend = TRUE,
       heatmap_legend_side = "bottom",
       annotation_legend_side = "right",
       padding = unit(c(2, 2, 2, 5), "mm"))
  
  return(invisible(NULL))
}

# =============================================================
# Load mutation data
# =============================================================
# TCGA
mut_tcga <- read.csv("data/processed/TCGA_somatic_mutation.csv", row.names = 1)
mut_tcga$Processed_SampleID <- gsub("-", ".", substr(mut_tcga$Tumor_Sample_Barcode, 1, 12))

# ORIEN
mut_orien <- read.csv("data/processed/all_mutations_combined.tsv", sep = "\t",
                      stringsAsFactors = FALSE)

# =============================================================
# Generate heatmaps
# =============================================================

# TCGA Expression
create_25gene_heatmap(
  cohort_name = "TCGA",
  network_type = "expression",
  mutation_data = mut_tcga,
  gene_col = "Hugo_Symbol",
  sample_col = "Processed_SampleID",
  variant_col = "Variant_Classification"
)

# ORIEN Expression
create_25gene_heatmap(
  cohort_name = "ORIEN",
  network_type = "expression",
  mutation_data = mut_orien,
  gene_col = "gene",
  sample_col = "sampleID",
  variant_col = "variant_class"
)

# TCGA SNF
create_25gene_heatmap(
  cohort_name = "TCGA",
  network_type = "snf",
  mutation_data = mut_tcga,
  gene_col = "Hugo_Symbol",
  sample_col = "Processed_SampleID",
  variant_col = "Variant_Classification"
)

# ORIEN SNF
create_25gene_heatmap(
  cohort_name = "ORIEN",
  network_type = "snf",
  mutation_data = mut_orien,
  gene_col = "gene",
  sample_col = "sampleID",
  variant_col = "variant_class"
)

cat("\n=== All heatmaps generated ===\n")