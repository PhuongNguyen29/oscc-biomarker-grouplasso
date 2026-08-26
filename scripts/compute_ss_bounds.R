suppressMessages(library(stabs))
p <- 4835; B <- 50
cells <- read.csv("cells.csv")

# --- Validation 1: stabs MB bound must equal q^2/((2*pi-1)*p) (our transcription)
mb_check <- function(q, pi) {
  a <- stabsel_parameters(p=p, q=q, cutoff=pi, B=B, sampling.type="MB")$PFER
  b <- q^2/((2*pi-1)*p)
  abs(a-b)
}
cat("MB validation max abs diff:",
    max(mb_check(96, 0.60), mb_check(20, 0.95), mb_check(142, 0.75)), "\n")

# --- Validation 2: ordering none >= unimodal >= r.concave at same (q, pi)
v <- sapply(c("none","unimodal","r-concave"), function(a)
  stabsel_parameters(p=p, q=96, cutoff=0.60, B=B, sampling.type="SS", assumption=a)$PFER)
cat("SS ordering check (none, unimodal, r-concave):", round(v,3),
    " monotone:", all(diff(v) <= 1e-9), "\n")

# --- Compute SS r-concave PFER per cell, q = ceiling(mean q) [conservative]
ss <- function(q, pi) stabsel_parameters(p=p, q=ceiling(q), cutoff=pi, B=B,
        sampling.type="SS", assumption="r-concave")$PFER
cells$ev_tcga_ss  <- mapply(ss, cells$q_tcga,  cells$pi)
cells$ev_orien_ss <- mapply(ss, cells$q_orien, cells$pi)
write.csv(cells, "cells_ss.csv", row.names=FALSE)

adm <- subset(cells, ev_tcga_ss <= 10 & ev_orien_ss <= 10)
adm <- adm[order(adm$pi, adm$mult), ]
cat("\nAdmissible under SS r-concave (E(V)<=10 both), pi-first order:\n")
print(adm[, c("mult","pi","ev_tcga_ss","ev_orien_ss","n_t","n_o")], row.names=FALSE)
cat("\nSELECTED (SS):", adm$mult[1], adm$pi[1], "\n")
