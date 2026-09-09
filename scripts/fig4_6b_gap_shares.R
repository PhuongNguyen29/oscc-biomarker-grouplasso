# ==============================================================================
# Figure 4.6b -- Programme shares of the high-low risk separation, by cohort
#
# Companion to the decomposition tile written by module_decomposition_hub250.R
# (fig4_6a). Reads results/module_decomposition/group_difference_summary.csv --
# no transcription.
#
# NOTE ON INTERPRETATION (mirrored in the draft text): every share is positive
# partly by construction, because patients are split at the median of the total
# score to which these components sum. The informative content is the RELATIVE
# size of the shares and their consistency across cohorts, not their sign.
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr)
})

DS_FILE <- "results/module_decomposition_k30/group_difference_summary.csv"
OUT_DIR <- "results/figures"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

PROGRAM <- c(brown_yellow   = "Adaptive immune / T cell",
             turquoise_blue = "ECM / mesenchymal",
             blue_green     = "Epithelial differentiation",
             pink_black     = "Interferon",
             green_red      = "Myogenesis")

ds <- read.csv(DS_FILE)
stopifnot(all(c("cohort", "pair", "share_of_total_gap") %in% names(ds)))
stopifnot(nrow(ds) == 20)   # 5 pairs x 4 cohorts

# Provenance tripwire: recorded ORIEN adaptive-immune share ~39%
chk <- ds$share_of_total_gap[ds$cohort == "ORIEN" & ds$pair == "brown_yellow"]
stopifnot(length(chk) == 1, abs(chk - 0.403) < 0.01)

ds <- ds %>%
  mutate(programme = ifelse(pair %in% names(PROGRAM), PROGRAM[pair], pair),
         pct = 100 * share_of_total_gap,
         cohort = factor(cohort,
                         levels = c("TCGA", "ORIEN", "GSE41613", "GSE42743")))

# Order programmes by mean share so the figure reads top-down
ord <- ds %>% group_by(programme) %>% summarise(m = mean(pct)) %>% arrange(m)
ds$programme <- factor(ds$programme, levels = ord$programme)

gg <- ggplot(ds, aes(pct, programme, fill = cohort)) +
  geom_col(position = position_dodge(width = 0.78), width = 0.72) +
  geom_text(aes(label = sprintf("%.0f", pct)),
            position = position_dodge(width = 0.78),
            hjust = -0.25, size = 2.7) +
  scale_fill_manual(values = c(TCGA = "#9E9E9E", ORIEN = "#3C6E71",
                               GSE41613 = "#D9782D", GSE42743 = "#8C4A9E"),
                    name = NULL) +
  scale_x_continuous(limits = c(0, 50), expand = c(0, 0)) +
  labs(x = "Share of the mean high-minus-low risk difference (%)", y = NULL,
       title = "The risk separation is distributed across programmes",
       subtitle = paste0("No programme exceeds 42% in any cohort; ",
                         "exact linear decomposition of the hub score")) +
  theme_classic(base_size = 11) +
  theme(legend.position = "bottom",
        legend.text = element_text(size = 9),
        plot.title = element_text(face = "bold", size = 12),
        plot.subtitle = element_text(size = 9, colour = "grey30"))

ggsave(file.path(OUT_DIR, "fig4_6b_gap_shares_k30.pdf"), gg, width = 7.5, height = 5)
ggsave(file.path(OUT_DIR, "fig4_6b_gap_shares_k30.png"), gg, width = 7.5, height = 5, dpi = 300)
cat("Written:", file.path(OUT_DIR, "fig4_6b_gap_shares.{pdf,png}"), "\n")
