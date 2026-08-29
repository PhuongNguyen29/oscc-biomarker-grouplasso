# ==============================================================================
# Figure 4.2 -- Hub vs corrected-lasso cross-cohort selection agreement
#
# Grouped bars: Kuncheva chance-corrected index at matched pooled set sizes
# (50 / 100 / 250), hub (kME top-k, matched module pairs) vs corrected-lasso
# top-N (CPSS selection frequencies, canonical selected cell).
#
# DATA PROVENANCE: the twelve values below are transcribed from the verified
# 27 Aug 2026 rerun of phaseK1_6_hub_overlap.R (post-audit, canonical
# artifacts). CROSS-CHECK before finalizing the figure: hub union rows in the
# K1.6.4 console log (0.232 / 0.264 / 0.510) and the comparator block
# (-0.004 / -0.001 / -0.001 at sizes 50 / 100 / 250; per-size values 10..250
# ranged -0.004 to 0.009). If hub_overlap.csv in results/phaseK1_6_hub_overlap
# contains these rows in machine-readable form, prefer reading them and delete
# the inline block -- the stopifnot below is the tripwire either way.
# ==============================================================================

suppressPackageStartupMessages(library(ggplot2))
OUT_DIR <- "results/figures"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

df <- data.frame(
  size     = factor(rep(c("50", "100", "250"), 2), levels = c("50", "100", "250")),
  selector = rep(c("Module membership (kME hubs)",
                   "Penalized regression (lasso top-N)"), each = 3),
  kuncheva = c(0.232, 0.264, 0.510,      # hub union, k = 10/20/50 per pair
               -0.004, -0.001, -0.001))  # corrected-lasso top-N, matched sizes

# Tripwire: these must match the K1.6.4 rerun log exactly
stopifnot(identical(df$kuncheva[1:3], c(0.232, 0.264, 0.510)))

gg <- ggplot(df, aes(size, kuncheva, fill = selector)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.62) +
  geom_hline(yintercept = 0, linewidth = 0.4) +
  geom_text(aes(label = sprintf("%.3f", kuncheva),
                vjust = ifelse(kuncheva >= 0, -0.45, 1.4)),
            position = position_dodge(width = 0.7), size = 3.4) +
  scale_fill_manual(values = c("Module membership (kME hubs)" = "#1B7837",
                               "Penalized regression (lasso top-N)" = "#762A83"),
                    name = NULL) +
  scale_y_continuous(limits = c(-0.05, 0.60), expand = c(0, 0)) +
  labs(x = "Matched selection size (genes per cohort, pooled over module pairs)",
       y = "Cross-cohort agreement\n(Kuncheva chance-corrected index)",
       title = "Selection principle, not data, determines cross-cohort reproducibility",
       subtitle = paste("Same cohorts, same 4,702-gene feature space, same index;",
                        "0 = chance agreement at matched size")) +
  theme_classic(base_size = 12) +
  theme(legend.position = c(0.28, 0.92),
        legend.background = element_blank(),
        plot.title = element_text(face = "bold", size = 12))

ggsave(file.path(OUT_DIR, "fig4_2_kuncheva_bars.pdf"), gg, width = 6.5, height = 4.6)
ggsave(file.path(OUT_DIR, "fig4_2_kuncheva_bars.png"), gg, width = 6.5, height = 4.6, dpi = 300)
cat("Written:", file.path(OUT_DIR, "fig4_2_kuncheva_bars.{pdf,png}"), "\n")
