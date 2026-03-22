# ==============================================================================
# GSE42743 RMA Normalization from CEL Files
# ==============================================================================
# Purpose: Produce probes × samples expression matrix (log2, RMA normalized)
#          so existing phase8_geo_preprocessing.R can continue with:
#          expr42743 <- read.csv("GSE42743/GSE42743_rma_expression.csv", row.names = 1)
#
# Reference: Irizarry et al. (2003) Biostatistics, 4(2):249-264
# Platform:  GPL570 — Affymetrix HG-U133 Plus 2.0
# ==============================================================================

library(affy)

# --- Step 1: Extract CEL files ------------------------------------------------
cat("Step 1: Extracting CEL files from GSE42743_RAW.tar...\n")
cel_dir <- "GSE42743/CEL"
dir.create(cel_dir, showWarnings = FALSE, recursive = TRUE)
untar("GSE42743/GSE42743_RAW.tar", exdir = cel_dir)

# Check what was extracted — CEL files may be gzipped
cel_files <- list.files(cel_dir, pattern = "\\.CEL\\.gz$|\\.cel\\.gz$|\\.CEL$|\\.cel$",
                        full.names = TRUE)
cat("Found", length(cel_files), "CEL file(s)\n")

if (length(cel_files) == 0) {
  stop("No CEL files found in ", cel_dir, ". Check tar contents.")
}

# --- Step 2: RMA normalization ------------------------------------------------
cat("Step 2: Reading CEL files and performing RMA normalization...\n")
cat("  This may take several minutes...\n")
raw_data <- ReadAffy(filenames = cel_files)
cat("  Chip type:", cdfName(raw_data), "\n")
cat("  Samples:", ncol(raw_data), "\n")

eset <- rma(raw_data)  # Background correction + quantile normalization + summarization
expr42743 <- exprs(eset)  # probes × samples, log2 scale

cat("Step 2 complete: RMA expression matrix:", nrow(expr42743), "probes ×",
    ncol(expr42743), "samples\n")
cat("  Expression range:", round(min(expr42743), 3), "to",
    round(max(expr42743), 3), "(log2)\n")

# --- Step 3: Clean sample names -----------------------------------------------
# CEL file names often have .CEL.gz suffix — strip to match GEO sample IDs
colnames(expr42743) <- gsub("\\.CEL\\.gz$|\\.cel\\.gz$|\\.CEL$|\\.cel$", "",
                            colnames(expr42743))
colnames(expr42743) <- sub("_.*", "", colnames(expr42743))
cat("Step 3: Sample name examples (first 5):\n")
print(head(colnames(expr42743), 5))

# --- Step 4: Save -------------------------------------------------------------
out_file <- "GSE42743/GSE42743_rma_expression.csv"
write.csv(expr42743, out_file)
cat("Step 4: Saved to", out_file, "\n")
cat("  Dimensions:", nrow(expr42743), "probes ×", ncol(expr42743), "samples\n")
cat("Done.\n")
