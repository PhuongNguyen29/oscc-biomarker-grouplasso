# ==============================================================================
# Figure 4.5 -- Per-pair enrichment dot plot
#
# Reads results/module_decomposition/pair_enrichment.csv (produced by
# pair_characterization_hub250.R). No transcription.
#
# DESIGN: showing all 468 significant terms is unreadable, so the figure shows
# the TOP_N most significant terms per pair per collection. That is a display
# choice, not a filter on the analysis -- the full table is the supplement, and
# the caption must say so. Terms are ordered within pair by q so the reader can
# see the programme identity at a glance.
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr)
})

ENR_FILE <- "results/module_decomposition/pair_enrichment.csv"
OUT_DIR  <- "results/figures"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

TOP_N  <- 4       # terms per pair per collection
Q_CUT  <- 0.10

enr <- read.csv(ENR_FILE)
stopifnot(all(c("collection", "pair", "term", "overlap", "fold", "q") %in% names(enr)))

# Provenance tripwire: the interferon result must be present as recorded
ifn <- enr[enr$pair == "pink_black" &
           grepl("INTERFERON_GAMMA_RESPONSE", enr$term), ]
stopifnot(nrow(ifn) >= 1, min(ifn$q) < 1e-40)

# Human-readable programme labels (from the enrichment itself)
PROGRAM <- c(brown_yellow  = "Adaptive immune / T cell",
             turquoise_blue = "ECM / mesenchymal",
             blue_green    = "Epithelial differentiation",
             pink_black    = "Interferon",
             green_red     = "Myogenesis")

top <- enr %>%
  filter(q < Q_CUT) %>%
  group_by(pair, collection) %>%
  slice_min(q, n = TOP_N, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(
    pair_lab = ifelse(pair %in% names(PROGRAM),
                      sprintf("%s\n(%s)", PROGRAM[pair], pair), pair),
    # Trim long GO names; keep them recognizable
    term_lab = gsub("_", " ", sub("^(GOBP|HALLMARK)_", "", term)),
    # term_lab = ifelse(nchar(term_lab) > 58,
    #                   paste0(substr(term_lab, 1, 56), "..."), term_lab),
    logq = -log10(pmax(q, 1e-60)))

# Order terms by significance within pair for readability
top <- top %>% arrange(pair_lab, logq) %>%
  mutate(row = factor(paste(pair_lab, term_lab), levels = unique(paste(pair_lab, term_lab))))

top$collection_lab <- ifelse(grepl("^h\\.all", top$collection),
                             "Hallmark", "GO BP")

gg <- ggplot(top, aes(x = fold, y = row, size = overlap, colour = logq,
                      shape = collection_lab)) +
  geom_point() +
  scale_shape_manual(values = c("Hallmark" = 16, "GO BP" = 17), name = NULL) +
  facet_grid(pair_lab ~ ., scales = "free_y", space = "free_y", switch = "y") +
  scale_y_discrete(labels = function(x) sub("^.*\\)\\s*", "", x)) +
  scale_colour_viridis_c(option = "plasma", direction = -1,
                         name = expression(-log[10](q))) +
  scale_size_continuous(name = "Genes in\noverlap", range = c(2, 6)) +
  labs(x = "Fold enrichment over the 4,702-gene analysis universe", y = NULL,
       title = "Each matched module pair is a distinct biological programme",
       subtitle = paste0("Top ", TOP_N, " terms per pair per collection ",
                         "(Hallmark, GO BP); full table in the supplement")) +
  theme_bw(base_size = 12) +
  theme(strip.placement = "outside",
        strip.background = element_blank(),
        strip.text.y.left = element_text(angle = 0, face = "bold", size = 10),
        panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold", size = 12),
        plot.subtitle = element_text(size = 9, colour = "grey30"),
        legend.position = "right")

# NOTE ON SIZING: rendered at 13 x 8.5 in. If the dissertation places this on a
# portrait page (~6.5 in text width) it will be scaled to ~50%, halving the
# effective font size -- either place it landscape / full-page, or re-render at
# width = 7.5 with base_size = 9. Decide once and keep it consistent.
ggsave(file.path(OUT_DIR, "fig4_5_enrichment_dots.pdf"), gg, width = 14, height = 8.5)
ggsave(file.path(OUT_DIR, "fig4_5_enrichment_dots.png"), gg, width = 14, height = 8.5, dpi = 300)
cat("Written:", file.path(OUT_DIR, "fig4_5_enrichment_dots.{pdf,png}"), "\n")
cat(sprintf("Terms shown: %d of %d with q < %.2f\n",
            nrow(top), sum(enr$q < Q_CUT), Q_CUT))
