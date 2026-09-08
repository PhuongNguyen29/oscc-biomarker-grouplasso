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
# K1.6.4 console log (0.232 / 0.264 / 0.510) and the comparator block at the
# MATCHED UNION sizes (0.009 / -0.001 / -0.001 at 50 / 100 / 250). The
# comparator block prints in PAIRS -- per-module size then union size -- so the
# per-module rows (-0.002 at 10, -0.004 at 20, 0.009 at 50) must NOT be matched
# to the hub union bars. Across all comparator sizes the range is -0.004 to
# 0.009, all chance-level. If hub_overlap.csv in results/phaseK1_6_hub_overlap
# contains these rows in machine-readable form, prefer reading them and delete
# the inline block -- the stopifnot below is the tripwire either way.
# ==============================================================================

suppressPackageStartupMessages({library(ggplot2); library(dplyr)})
K164_DIR <- "results/phaseK1_6_hub_overlap_k30"
OUT_DIR <- "results/figures"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

ov <- read.csv(file.path(K164_DIR, "hub_overlap.csv"))

# Number of matched pairs, read from the data rather than assumed
n_pairs <- ov %>% filter(level == "per_module") %>%
  distinct(tcga_module, orien_module) %>% nrow()
stopifnot(n_pairs == 5)

# Hub: union rows. Union size = n_pairs * k (pairs are disjoint gene sets).
hub <- ov %>% filter(level == "union") %>%
  transmute(size = n_pairs * k, kuncheva,
            selector = "Module membership (kME hubs)")

# Lasso: size is in the level string ("lasso_n250"), NOT the k column.
las <- ov %>% filter(grepl("^lasso_n", level)) %>%
  transmute(size = as.integer(sub("^lasso_n", "", level)), kuncheva,
            selector = "Penalized regression (lasso top-N)") %>%
  filter(size %in% hub$size)          # keep only union-matched sizes

df <- bind_rows(hub, las) %>% arrange(size)
df$size <- factor(df$size, levels = sort(unique(df$size)))
stopifnot(nrow(df) == 6)
print(df)                              # verify before plotting    

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
       title = "Selection principle, not data, determines\ncross-cohort reproducibility",
       subtitle = paste0("Same cohorts, same 4,702-gene feature space, same index\n",
                         "0 = chance agreement at matched set size")) +
  theme_classic(base_size = 12) +
  theme(legend.position = c(0.32, 0.88),
        legend.background = element_blank(),
        legend.text = element_text(size = 9),
        legend.key.size = unit(0.45, "cm"),
        plot.title = element_text(face = "bold", size = 12),
        plot.subtitle = element_text(size = 9, colour = "grey30"))

ggsave(file.path(OUT_DIR, "fig4_2_kuncheva_bars_k30.pdf"), gg, width = 6.5, height = 4.6)
ggsave(file.path(OUT_DIR, "fig4_2_kuncheva_bars_k30.png"), gg, width = 6.5, height = 4.6, dpi = 300)
cat("Written:", file.path(OUT_DIR, "fig4_2_kuncheva_bars.{pdf,png}"), "\n")
