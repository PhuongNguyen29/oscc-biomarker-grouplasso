# ==============================================================================
# Figure 4.1 -- Module preservation against permutation null ceilings
#
# Reads results/phaseK1_module_preservation/preservation_statistics.csv.
# Column names vary between WGCNA export conventions, so they are auto-detected
# and the script stops with the available names if it cannot find them.
#
# DESIGN: one facet per reference-test comparison; each module a point at its
# Zsummary; the dashed line is that comparison's permutation null ceiling.
# Points above the line are preserved. This makes the "5 of 9 / 6 of 7 modules
# preserved in every test cohort" claim visible rather than asserted.
#
# medianRank NOTE: Langfelder et al. (2011) recommend reporting medianRank
# alongside Zsummary, because Zsummary increases with module size while
# medianRank does not. If the file carries a medianRank column, this script
# writes a SECOND panel for it; if not, the omission should be acknowledged in
# the methods text rather than passed over.
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr)
})

K1_FILE <- "results/phaseK1_module_preservation/preservation_statistics.csv"
OUT_DIR <- "results/figures"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

ps <- read.csv(K1_FILE)
pick <- function(cands, what) {
  hit <- intersect(cands, names(ps))
  if (!length(hit))
    stop("No ", what, " column; available: ", paste(names(ps), collapse = ", "))
  hit[1]
}
c_ref  <- pick(c("reference", "ref", "reference_cohort"), "reference")
c_test <- pick(c("test", "test_cohort"), "test")
c_mod  <- pick(c("module", "moduleColor", "color"), "module")
c_z    <- pick(c("Zsummary", "Zsummary.pres", "zsummary"), "Zsummary")
c_ceil <- pick(c("null_ceiling", "ceiling", "max_null_Z", "null_max"), "null ceiling")
c_size <- intersect(c("moduleSize", "size", "n_genes"), names(ps))

d <- data.frame(reference = ps[[c_ref]], test = ps[[c_test]],
                module = ps[[c_mod]], Z = ps[[c_z]], ceiling = ps[[c_ceil]])
if (length(c_size)) d$size <- ps[[c_size[1]]]
d <- d[d$module != "grey" & d$module != "gold", ]

# Provenance tripwire: the recorded ORIEN -> TCGA ceiling is 10.3
chk <- unique(d$ceiling[d$reference == "ORIEN" & d$test == "TCGA"])
stopifnot(length(chk) == 1, abs(chk - 10.3) < 0.3)

d$comparison <- factor(paste(d$reference, "\u2192", d$test),
                       levels = unique(paste(d$reference, "\u2192", d$test)))
d$preserved <- ifelse(d$Z > d$ceiling, "Above null ceiling", "Not preserved")

ceil <- d %>% group_by(comparison) %>% summarise(ceiling = first(ceiling))

gg <- ggplot(d, aes(x = Z, y = reorder(module, Z), colour = preserved)) +
  geom_vline(data = ceil, aes(xintercept = ceiling),
             linetype = "dashed", colour = "grey35", linewidth = 0.45) +
  geom_point(size = if ("size" %in% names(d)) 2.6 else 2.6) +
  facet_wrap(~comparison, scales = "free_y", ncol = 3) +
  scale_colour_manual(values = c("Above null ceiling" = "#1B7837",
                                 "Not preserved" = "grey55"), name = NULL) +
  labs(x = "Preservation Zsummary (dashed line: permutation null ceiling)",
       y = NULL,
       title = "Five TCGA and six ORIEN modules preserve in every test cohort",
       subtitle = paste0("200 permutations per comparison; ",
                         "grey module excluded")) +
  theme_bw(base_size = 11) +
  theme(strip.background = element_blank(),
        strip.text = element_text(face = "bold", size = 9.5),
        panel.grid.minor = element_blank(),
        legend.position = "bottom",
        plot.title = element_text(face = "bold", size = 12),
        plot.subtitle = element_text(size = 9, colour = "grey30"))

ggsave(file.path(OUT_DIR, "fig4_1_preservation.pdf"), gg, width = 11, height = 6.5)
ggsave(file.path(OUT_DIR, "fig4_1_preservation.png"), gg, width = 11, height = 6.5, dpi = 300)
cat("Written:", file.path(OUT_DIR, "fig4_1_preservation.{pdf,png}"), "\n")

# --- medianRank companion, if available --------------------------------------
c_mr <- intersect(c("medianRank", "medianRank.pres", "median_rank"), names(ps))
if (length(c_mr)) {
  d$mr <- ps[[c_mr[1]]][match(paste(d$reference, d$test, d$module),
                              paste(ps[[c_ref]], ps[[c_test]], ps[[c_mod]]))]
  gg2 <- ggplot(d, aes(x = mr, y = reorder(module, -mr))) +
    geom_point(size = 2.4, colour = "#3C6E71") +
    facet_wrap(~comparison, scales = "free_y", ncol = 3) +
    labs(x = "Preservation medianRank (lower = better preserved)", y = NULL,
         title = "Preservation medianRank, independent of module size",
         subtitle = "Companion to Zsummary (Langfelder et al. 2011)") +
    theme_bw(base_size = 11) +
    theme(strip.background = element_blank(),
          strip.text = element_text(face = "bold", size = 9.5),
          panel.grid.minor = element_blank(),
          plot.title = element_text(face = "bold", size = 12),
          plot.subtitle = element_text(size = 9, colour = "grey30"))
  ggsave(file.path(OUT_DIR, "fig4_1b_medianrank.pdf"), gg2, width = 11, height = 6.5)
  cat("Written: fig4_1b_medianrank.pdf\n")
} else {
  cat("\nNOTE: no medianRank column found in", K1_FILE, "\n")
  cat("Langfelder et al. recommend reporting it alongside Zsummary because\n")
  cat("Zsummary grows with module size. Either export it from the K1 run or\n")
  cat("acknowledge the omission in Methods 4.2.2.\n")
}
