# ==============================================================================
# Figure 3.2 -- Selection agreement scales with reused data
#
# Panel A: chance-corrected agreement (Kuncheva, top-k) vs subsampling fraction,
#          both cohorts, with the f^2 expectation curve as reference.
# Panel B: agreement vs penalty multiplier at fraction 0.5 -- the sparsity
#          direction, which is opposite in the two cohorts (an SNR signature).
#
# Reads results/phase4b_fraction_stability_v2/stability_by_fraction.csv
# (rev b: corrected weight direction, beta-scale ranking, per-cohort seeds).
# No transcription; a tripwire asserts the recorded fraction-0.5 values.
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(patchwork)
})

IN_FILE <- "results/phase4b_fraction_stability_v2/stability_by_fraction.csv"
OUT_DIR <- "results/figures"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

st <- read.csv(IN_FILE)
stopifnot(all(c("cohort", "fraction", "multiplier", "estimable",
                "kuncheva_topk") %in% names(st)))
st <- st[st$estimable == TRUE, ]

COH <- c(tcga = "TCGA", orien = "ORIEN")
PAL <- c(TCGA = "#B2182B", ORIEN = "#2166AC")
st$Cohort <- factor(ifelse(st$cohort %in% names(COH), COH[st$cohort], st$cohort),
                    levels = c("TCGA", "ORIEN"))

# --- Panel A: mean over multipliers, by fraction ------------------------------
byf <- st %>% group_by(Cohort, fraction) %>%
  summarise(k = mean(kuncheva_topk), .groups = "drop")

# Tripwire: recorded means at fraction 0.5
chk <- byf$k[byf$fraction == 0.5]
stopifnot(length(chk) == 2,
          any(abs(chk - 0.045) < 0.005), any(abs(chk - 0.097) < 0.005))

# f^2 reference, scaled to pass through the ORIEN point at f = 0.5 so that the
# curve shows the SHAPE of the expectation, not an absolute prediction. Label it
# as such in the caption -- it is a reference, not a fitted model.
scale_f2 <- byf$k[byf$Cohort == "ORIEN" & byf$fraction == 0.5] / 0.25
ref <- data.frame(fraction = seq(0.5, 0.9, 0.01))
ref$k <- scale_f2 * ref$fraction^2

pA <- ggplot(byf, aes(fraction, k, colour = Cohort)) +
  geom_line(data = ref, aes(fraction, k), inherit.aes = FALSE,
            linetype = "dashed", colour = "grey45", linewidth = 0.5) +
  annotate("text", x = 0.875, y = scale_f2 * 0.875^2 + 0.018,
           label = "expected shared data (f\u00b2)", colour = "grey35",
           size = 3.1, hjust = 1) +
  geom_line(linewidth = 0.8) + geom_point(size = 2.4) +
  scale_colour_manual(values = PAL, name = NULL) +
  scale_x_continuous(breaks = seq(0.5, 0.9, 0.1)) +
  scale_y_continuous(limits = c(0, 0.32)) +
  labs(x = "Subsampling fraction", y = "Cross-subsample agreement\n(Kuncheva, chance-corrected)",
       title = "A. Apparent stability tracks how much data subsamples share") +
  theme_classic(base_size = 11) +
  theme(legend.position = c(0.14, 0.86),
        plot.title = element_text(face = "bold", size = 11))

# --- Panel B: sparsity direction at fraction 0.5 ------------------------------
half <- st %>% filter(abs(fraction - 0.5) < 1e-9) %>%
  group_by(Cohort, multiplier) %>%
  summarise(k = mean(kuncheva_topk), .groups = "drop")

pB <- ggplot(half, aes(multiplier, k, colour = Cohort)) +
  geom_line(linewidth = 0.8) + geom_point(size = 2.2) +
  scale_colour_manual(values = PAL, guide = "none") +
  scale_y_continuous(limits = c(0, 0.15)) +
  labs(x = "Penalty multiplier (sparser \u2192)",
       y = "Agreement at fraction 0.5",
       title = "B. Sparser selection helps ORIEN, hurts TCGA") +
  theme_classic(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 11))

gg <- pA + pB + plot_layout(widths = c(1.35, 1))
ggsave(file.path(OUT_DIR, "fig3_2_fraction_stability.pdf"), gg, width = 10, height = 4.3)
ggsave(file.path(OUT_DIR, "fig3_2_fraction_stability.png"), gg, width = 10, height = 4.3, dpi = 300)
cat("Written:", file.path(OUT_DIR, "fig3_2_fraction_stability.{pdf,png}"), "\n")

# --- Console: the ratio table quoted in the text ------------------------------
cat("\nKuncheva / f^2 by fraction (the quantity quoted in 3.3.4):\n")
byf %>% mutate(ratio = k / fraction^2) %>%
  tidyr::pivot_wider(names_from = Cohort, values_from = c(k, ratio)) %>%
  as.data.frame() %>% print(row.names = FALSE, digits = 3)
