#!/usr/bin/env Rscript
# ==============================================================================
# GSE42743 RMA Normalization from CEL Files
# ==============================================================================
# Purpose: Produce a probes x samples expression matrix (log2, RMA normalized)
#          for phase8_geo_preprocessing.R.
#
# Reference: Irizarry et al. (2003) Biostatistics 4(2):249-264 -- RMA
# Platform:  GPL570 -- Affymetrix HG-U133 Plus 2.0
#
# FIXES IN THIS REVISION
# ----------------------
# 1. cat("CONSENSUS_PATH: ", CONSENSUS_PATH) was copied from the preprocessing
#    script. That object does not exist here, so the script errored on line 37
#    before doing any work.
#
# 2. The log filename said "phase8_geo_preprocessing_*", which would interleave
#    with the other script's logs in the same directory and make the audit
#    trail ambiguous.
#
# 3. OUT_FILE wrote to GSE42743/, but phase8_geo_preprocessing.R reads from
#    data/external/. The file had to be moved by hand between runs -- an
#    unrecorded manual step in the middle of the pipeline.
#
# 4. Sample names: list.files(full.names = TRUE) means ReadAffy may carry the
#    directory prefix into sampleNames. basename() is applied explicitly before
#    stripping suffixes, so the GSM assertion tests the right string. Without
#    it, colnames could become "GSE42743/CEL/GSM1234567" and every downstream
#    name match against pData would silently fail.
#
# 5. ncol() on an AffyBatch returns the CHIP GRID width, not the sample count.
#    "Samples: 1164" was reporting array geometry. Uses length(sampleNames()).
#
# 6. on.exit() at top level is a no-op under Rscript -- it registers a handler
#    on a function frame, and there is none. Sinks are closed explicitly at the
#    end instead.
#
# Usage:
#   cd ~/hpchome/oscc-biomarker-grouplasso
#   Rscript scripts/phase8_rma_gse42743.R
# ==============================================================================

library(affy)

# ==============================================================================
# Configuration
# ==============================================================================

TAR_FILE <- "GSE42743/GSE42743_RAW.tar"
CEL_DIR  <- "GSE42743/CEL"
OUT_DIR  <- "data/external"
OUT_FILE <- file.path(OUT_DIR, "GSE42743_rma_expression.csv")

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# --- Logging ------------------------------------------------------------------
# split = TRUE works for stdout but NOT for messages: R permits only one message
# sink and it cannot split. Warnings therefore go to the file only.
LOG_DIR <- "logs"
dir.create(LOG_DIR, showWarnings = FALSE, recursive = TRUE)
log_file <- file.path(
  LOG_DIR,
  sprintf("phase8_rma_gse42743_%s.log", format(Sys.time(), "%Y%m%d_%H%M%S"))
)
log_con <- file(log_file, open = "wt")
sink(log_con, split = TRUE)
sink(log_con, type = "message")

cat("Log:", log_file, "\n")
cat("Started:", format(Sys.time()), "\n")
cat("R:", R.version.string, "\n")
cat("TAR_FILE:", TAR_FILE, "\n")
cat("OUT_FILE:", OUT_FILE, "\n\n")

# ==============================================================================
# Step 1: Extract CEL files
# ==============================================================================

cat("Step 1: Extracting CEL files\n")

if (!file.exists(TAR_FILE)) {
  stop("Archive not found: ", TAR_FILE)
}
dir.create(CEL_DIR, showWarnings = FALSE, recursive = TRUE)

cel_pattern <- "\\.CEL\\.gz$|\\.cel\\.gz$|\\.CEL$|\\.cel$"
existing <- list.files(CEL_DIR, pattern = cel_pattern)
if (length(existing) > 0) {
  cat("  ", length(existing), "CEL files already present; skipping untar\n")
} else {
  untar(TAR_FILE, exdir = CEL_DIR)
}

cel_files <- list.files(CEL_DIR, pattern = cel_pattern, full.names = TRUE)
cat("  Found", length(cel_files), "CEL file(s)\n")
if (length(cel_files) == 0) {
  stop("No CEL files found in ", CEL_DIR, ". Check tar contents.")
}

# ==============================================================================
# Step 2: RMA normalization
# ==============================================================================

cat("\nStep 2: Reading CEL files and running RMA (several minutes)...\n")
raw_data <- ReadAffy(filenames = cel_files)

cat("  Chip type:", cdfName(raw_data), "\n")
# NOT ncol(raw_data): on an AffyBatch that returns the chip grid width, not the
# number of samples.
cat("  Samples:", length(sampleNames(raw_data)), "\n")

eset <- rma(raw_data)          # background correction + quantile norm + summarization
expr42743 <- exprs(eset)       # probes x samples, log2

cat("  RMA matrix:", nrow(expr42743), "probes x", ncol(expr42743), "samples\n")
cat("  Expression range:", round(min(expr42743), 3), "to",
    round(max(expr42743), 3), "(log2)\n")

# ==============================================================================
# Step 3: Clean sample names
# ==============================================================================

cat("\nStep 3: Cleaning sample names\n")
cat("  Raw examples:", paste(head(colnames(expr42743), 2), collapse = " | "), "\n")

# basename() first: filenames were passed with full paths, so the directory
# prefix can survive into sampleNames and would break every downstream name
# match against pData.
colnames(expr42743) <- basename(colnames(expr42743))
colnames(expr42743) <- sub(cel_pattern, "", colnames(expr42743))
colnames(expr42743) <- sub("_.*", "", colnames(expr42743))

bad <- !grepl("^GSM[0-9]+$", colnames(expr42743))
if (any(bad)) {
  stop(sum(bad), " sample name(s) are not GSM IDs after cleaning. ",
       "First few: ", paste(head(colnames(expr42743)[bad], 5), collapse = ", "),
       "\nphase8_geo_preprocessing.R matches these against pData rownames.")
}
if (any(duplicated(colnames(expr42743)))) {
  stop("Duplicate GSM IDs after cleaning: ",
       paste(unique(colnames(expr42743)[duplicated(colnames(expr42743))]),
             collapse = ", "))
}
cat("  Cleaned examples:", paste(head(colnames(expr42743), 5), collapse = ", "), "\n")
cat("  All", ncol(expr42743), "names are unique GSM IDs: OK\n")

# ==============================================================================
# Step 4: Save
# ==============================================================================

cat("\nStep 4: Saving\n")
write.csv(expr42743, OUT_FILE)
cat("  Saved:", OUT_FILE, "\n")
cat("  Dimensions:", nrow(expr42743), "probes x", ncol(expr42743), "samples\n")

cat("\nFinished:", format(Sys.time()), "\n")
cat("--- sessionInfo ---\n")
print(sessionInfo())

sink(type = "message")
sink()
close(log_con)
