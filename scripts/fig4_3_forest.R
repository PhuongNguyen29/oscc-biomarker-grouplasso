# ==============================================================================
# Figure 4.3 -- Forest plot: three scores x four cohorts, bootstrap 95% CIs
#                against the age-alone clinical floor
#
# Reads results/phaseK3_bootstrap_variance/bootstrap_cis.csv directly (no
# transcription). Column names are auto-detected and validated; the script
# stops if the expected 12 rows or the hub external CIs are not found --
# the hub GSE41613 interval [0.608, 0.749] serves as the provenance tripwire.
# ==============================================================================

suppressPackageStartupMessages(library(ggplot2))
K3_FILE <- "results/phaseK3_bootstrap_variance/bootstrap_cis.csv"
OUT_DIR <- "results/figures"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

FLOOR_LO <- 0.551; FLOOR_HI <- 0.565

bt <- read.csv(K3_FILE)
pick <- function(cands) {
  hit <- intersect(cands, names(bt))
  if (!length(hit)) stop("Column not found; have: ", paste(names(bt), collapse = ", "))
  hit[1]
}
c_score  <- pick(c("score", "model", "signature"))
c_cohort <- pick(c("cohort", "dataset"))
c_ci     <- pick(c("cindex", "C.index", "c_index", "C"))
c_lo     <- pick(c("ci_lo", "ci_lower", "lower", "lo"))
c_hi     <- pick(c("ci_hi", "ci_upper", "upper", "hi"))

d <- data.frame(score = bt[[c_score]], cohort = bt[[c_cohort]],
                C = bt[[c_ci]], lo = bt[[c_lo]], hi = bt[[c_hi]])
stopifnot(nrow(d) == 12)

# Provenance tripwire: hub external CI must match the recorded result
hub41613 <- d[grepl("hub", d$score, ignore.case = TRUE) &
              grepl("41613", d$cohort), ]
stopifnot(nrow(hub41613) == 1,
          abs(hub41613$lo - 0.608) < 0.002, abs(hub41613$hi - 0.749) < 0.002)

score_labs <- c("eigengene_K1.5" = "Eigengene score",
                "hub_250"        = "Hub score (309 genes)",
                "lasso_250"      = "Lasso score (matched size)")
d$score_lab <- ifelse(d$score %in% names(score_labs),
                      score_labs[d$score], d$score)
d$score_lab <- factor(d$score_lab, levels = rev(c(
  "Hub score (309 genes)", "Eigengene score", "Lasso score (matched size)")))
d$cohort <- factor(d$cohort, levels = c("TCGA", "ORIEN", "GSE41613", "GSE42743"),
                   labels = c("TCGA (transfer)", "ORIEN (source, in-sample)",
                              "GSE41613 (external)", "GSE42743 (external)"))
d$clears <- ifelse(d$lo > FLOOR_HI, "CI above floor", "CI overlaps floor")

gg <- ggplot(d, aes(C, score_lab, colour = clears)) +
  annotate("rect", xmin = FLOOR_LO, xmax = FLOOR_HI, ymin = -Inf, ymax = Inf,
           fill = "grey75", alpha = 0.5) +
  geom_vline(xintercept = 0.5, linetype = "dotted", linewidth = 0.4) +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.18, linewidth = 0.7) +
  geom_point(size = 2.4) +
  facet_wrap(~cohort, ncol = 1) +
  scale_colour_manual(values = c("CI above floor" = "#1B7837",
                                 "CI overlaps floor" = "grey35"), name = NULL) +
  scale_x_continuous(limits = c(0.45, 0.85), breaks = seq(0.45, 0.85, 0.10)) +
  labs(x = "Concordance index (95% bootstrap CI, 2,000 resamples, coefficients fixed)",
       y = NULL,
       title = "Only the hub score clears the clinical floor in both external cohorts",
       subtitle = "Grey band: age-alone floor (0.551\u20130.565); dotted line: chance") +
  theme_classic(base_size = 11) +
  theme(legend.position = "bottom",
        strip.background = element_blank(),
        strip.text = element_text(face = "bold", hjust = 0),
        plot.title = element_text(face = "bold", size = 12))

ggsave(file.path(OUT_DIR, "fig4_3_forest.pdf"), gg, width = 6.5, height = 7.5)
ggsave(file.path(OUT_DIR, "fig4_3_forest.png"), gg, width = 6.5, height = 7.5, dpi = 300)
cat("Written:", file.path(OUT_DIR, "fig4_3_forest.{pdf,png}"), "\n")
