# ==============================================================================
# ECMS simulation, Stage 0: synthetic factor model data generator   (rev 2)
# Implements ecms_simulation_design.md section 3a for ONE replicate:
# a discovery (reference) cohort plus C test cohorts, planted module labels,
# and the ground truth table. No WGCNA call in this file.
#
# rev 2 (6 Oct 2026), aligned with the updated design (C0, 7b, ECMS-joint) and
# with ecms_null_run.R / ecms_null_aggregate.R:
#   1. truth now records each module's DISCOVERY properties (mean within-module
#      r, eigengene variance explained). Needed if the relabel check (7b) finds
#      the size-matched null anti-conservative and the null must also be matched
#      on discovery density / connectivity.
#   2. ecms_relabel_test(): simulated analogue of the 7b relabel check. Shuffles
#      gene identities inside one test cohort (keeps its correlation structure,
#      destroys correspondence with discovery). Also the source of whole-run
#      joint null vectors for ECMS-joint (S6).
#   3. ecms_null_labels(): random partition for the matched-size pipeline null,
#      the SAME algorithm as random_partition() in ecms_null_run.R, with the size
#      set taken from the planted sizes (ecms_size_set()). Every planted size is
#      in the null size set, so no module gets an NA location/scale in the
#      aggregate step.
#   4. ecms_null_split(): calls 1..n_holdout are held out (true negatives for
#      gate (a)), as in ecms_null_aggregate.R (N_HOLDOUT = 5).
#
# rev 3 (7 Oct 2026): lambda and gamma calibrated from the real K1 modules
# (base lambda 0.98, gamma 0.154); scenario levels in ecms_levels;
# ecms_check_base_target(). Module sizes remain OPEN (refuses to run).
#
# Model, gene g in module k, sample i, cohort c:
#   x_gi = m_kc * lambda_c * w_g * f_ki + gamma_c * h_i + e_gi,  e ~ N(0, sigma_c^2)
#   background genes:        gamma_c * h_i + e_gi
#   m_kc = 1 preserved, 0.5 weakened, 0 absent (discovery cohort: always 1)
#   w_g ~ U(w_range), drawn once per replicate and shared by all cohorts
#   module factors f_k ~ N(0, Sigma_f); Sigma_f exchangeable rho_M on dep_modules
#
# Output plugs straight into modulePreservation():
#   sim$multiExpr  list of list(data = samples x genes data.frame), reference first
#   sim$multiColor list(disc = numeric labels), 0 = background (WGCNA grey)
#   sim$truth      one row per module: size, pattern, n_preserved, true_r1..true_rC,
#                  disc_mean_r, disc_ve
#
# Usage:
#   source("scripts/ecms_sim_generator.R")
#   chk <- ecms_check_many(ecms_demo_params(), seeds = 1:20)   # self test
#   chk$summary; chk$random_set_mean_r
# ==============================================================================

# ---- Parameters ---------------------------------------------------------------

# Base scenario skeleton. NA entries are OPEN DECISIONS (Stage 0) and must be
# fixed in writing before any base run; the generator refuses to run with NA.
ecms_base_params <- function() list(
  p          = 2000,
  C          = 3,
  sizes      = NA,             # OPEN: 15 sizes in 30 to 500 do not fit in 2,000 genes
  n          = rep(300, 4),    # disc, t1, t2, t3
  lambda     = rep(0.98, 4),   # FIXED 7 Oct: pooled median of 16 real modules
  gamma      = rep(0.154, 4),  # FIXED 7 Oct: TCGA background r 0.023, gamma = sqrt(r/(1-r))
  sigma      = rep(1, 4),
  w_range    = c(0.5, 1),
  n_by_npres = c(3, 3, 3, 6),  # modules preserved in 0, 1, 2, 3 test cohorts (20/20/20/40%)
  p_weaken   = 0,              # OPEN: is the weakened level used in base?
  rho_M      = 0,
  dep_modules = integer(0),
  zscore     = TRUE,           # per gene within cohort, as in the real pipeline
  clip       = NA)             # real pipeline clips at 3; NA = no clipping

# Calibrated scenario levels (fixed 7 Oct 2026, from the real K1 modules;
# sigma = 1, w ~ U(0.5, 1)). Population mean within-module r in brackets,
# at gamma = 0.154 unless stated (computed analytically, 4,000 w draws).
ecms_levels <- list(
  S3_lambda = c(weak = 0.63, moderate = 0.98, strong = 1.51),   # r 0.19 / 0.35 / 0.55
  S5_gamma  = c(none = 0, tcga = 0.154, orien = 0.457),         # at lambda 0.98: r 0.34 / 0.35 / 0.42
  # Real per-cohort gamma (TCGA, ORIEN, GSE41613, GSE42743): a data-anchored
  # option for S7 heterogeneity (cohort order in the sim is disc, t1, t2, t3).
  real_gamma = c(TCGA = 0.154, ORIEN = 0.457, GSE41613 = 0.110, GSE42743 = 0.225))

# Pre-registered generator check for the base parameters: population mean
# within-module r at lambda 0.98, gamma 0.154 is 0.351 (target from real data
# about 0.35). ecms_expected_r() on fresh w draws should return 0.35 +/- 0.01.
ecms_check_base_target <- function(n_w = 2000, seed = 1) {
  set.seed(seed); w <- runif(n_w, 0.5, 1)
  r <- ecms_expected_r(w, 0.98, 0.154, 1)
  cat(sprintf("[target] base mean within-module r = %.3f (expected 0.351)\n", r))
  abs(r - 0.351) <= 0.01
}

# SELF TEST ONLY. Arbitrary values, not pre-registered, not for any reported run.
ecms_demo_params <- function() {
  par <- ecms_base_params()
  par$sizes      <- c(30, 40, 50, 60, 70, 80, 90, 100, 120, 140)
  par$n_by_npres <- c(2, 2, 2, 4)
  par$lambda     <- rep(0.8, 4)
  par$gamma      <- rep(0.2, 4)
  par
}

ecms_validate_params <- function(par) {
  need <- c("p", "C", "sizes", "n", "lambda", "gamma", "sigma", "w_range",
            "n_by_npres", "p_weaken", "rho_M", "dep_modules", "zscore", "clip")
  miss <- setdiff(need, names(par))
  if (length(miss)) stop("missing parameters: ", paste(miss, collapse = ", "))
  if (anyNA(c(par$sizes, par$lambda, par$gamma, par$sigma, par$n)))
    stop("sizes, lambda, gamma, sigma or n still NA: fix the Stage 0 open decisions first")
  C <- par$C; K <- length(par$sizes)
  stopifnot(length(par$n) == C + 1, length(par$lambda) == C + 1,
            length(par$gamma) == C + 1, length(par$sigma) == C + 1,
            length(par$n_by_npres) == C + 1, sum(par$n_by_npres) == K,
            sum(par$sizes) < par$p, all(par$sizes >= 2),
            par$p_weaken >= 0, par$p_weaken <= 1,
            par$rho_M >= 0, par$rho_M < 1,
            all(par$dep_modules %in% seq_len(K)))
  invisible(TRUE)
}

# ---- Preservation patterns ----------------------------------------------------

# K x C matrix of multipliers. Counts per "number of test cohorts preserved" are
# fixed (n_by_npres), so pi1 is exact in every replicate; which modules get which
# count, and which cohorts, is random, so pattern is not tied to module size.
ecms_assign_patterns <- function(K, C, n_by_npres, p_weaken) {
  npres <- sample(rep(0:C, n_by_npres))
  M <- matrix(unlist(lapply(npres, function(j) {
    m <- numeric(C); if (j > 0) m[sample.int(C, j)] <- 1; m
  })), nrow = K, byrow = TRUE)
  if (p_weaken > 0) {
    pres <- which(M == 1)
    M[pres[runif(length(pres)) < p_weaken]] <- 0.5
  }
  colnames(M) <- paste0("t", seq_len(C))
  M
}

ecms_truth_table <- function(sizes, M) {
  C <- ncol(M); npres <- rowSums(M > 0)
  truth <- data.frame(module = seq_along(sizes), size = sizes, n_preserved = npres,
                      pattern = apply(M, 1, paste, collapse = "/"))
  for (r in seq_len(C)) truth[[paste0("true_r", r)]] <- npres >= r
  truth
}

# ---- Generator ------------------------------------------------------------------

ecms_generate <- function(par, seed, verbose = TRUE) {
  ecms_validate_params(par)
  set.seed(seed)
  C <- par$C; K <- length(par$sizes); p <- par$p
  coh <- c("disc", paste0("t", seq_len(C)))

  labels <- c(rep(seq_len(K), par$sizes), rep(0L, p - sum(par$sizes)))
  in_mod <- which(labels > 0)
  w <- numeric(p); w[in_mod] <- runif(length(in_mod), par$w_range[1], par$w_range[2])

  M <- ecms_assign_patterns(K, C, par$n_by_npres, par$p_weaken)

  Sf <- diag(K)
  if (par$rho_M > 0 && length(par$dep_modules) > 1) {
    d <- par$dep_modules; Sf[d, d] <- par$rho_M; diag(Sf) <- 1
  }
  Rf <- chol(Sf)

  gene_ids  <- sprintf("g%04d", seq_len(p))
  multiExpr <- setNames(vector("list", C + 1), coh)
  for (j in seq_along(coh)) {
    n    <- par$n[j]
    mult <- if (j == 1) rep(1, K) else M[, j - 1]
    Fk <- matrix(rnorm(n * K), n, K) %*% Rf
    h  <- rnorm(n)
    X  <- matrix(rnorm(n * p, sd = par$sigma[j]), n, p) + par$gamma[j] * h  # h recycles down columns
    L  <- matrix(0, K, p)
    L[cbind(labels[in_mod], in_mod)] <- mult[labels[in_mod]] * par$lambda[j] * w[in_mod]
    X  <- X + Fk %*% L
    if (par$zscore) {
      X <- scale(X); attr(X, "scaled:center") <- NULL; attr(X, "scaled:scale") <- NULL
    }
    if (!is.na(par$clip)) X <- pmin(pmax(X, -par$clip), par$clip)
    dimnames(X) <- list(sprintf("%s_s%04d", coh[j], seq_len(n)), gene_ids)
    multiExpr[[j]] <- list(data = as.data.frame(X))
    if (verbose) cat(sprintf("[gen] seed %d | %-4s n=%d | present %d, absent %d modules\n",
                             seed, coh[j], n, sum(mult > 0), sum(mult == 0)))
  }

  truth <- ecms_truth_table(par$sizes, M)

  # Discovery properties (rev 2): observed in the generated discovery data.
  Xd <- as.matrix(multiExpr$disc$data)
  truth$disc_mean_r <- vapply(seq_len(K), function(k)
    mean_offdiag_r(Xd[, labels == k, drop = FALSE]), numeric(1))
  truth$disc_ve <- vapply(seq_len(K), function(k)
    eigengene_ve(Xd[, labels == k, drop = FALSE]), numeric(1))

  list(multiExpr = multiExpr, multiColor = list(disc = labels), truth = truth,
       w = w, M = M, seed = seed, params = par, relabelled = character(0))
}

# ---- Relabel check analogue (design 7b) -----------------------------------------

# Shuffle gene identities inside one or more test cohorts. Each cohort keeps its
# own correlation structure; correspondence with discovery is destroyed, so every
# discovery module becomes ABSENT in that cohort by construction (a shuffled
# module's genes can by chance include some genes of a planted module, so this is
# absence of correspondence, the object 7b tests, not zero correlation).
# Truth is updated: the shuffled cohorts' multipliers are set to 0.
ecms_relabel_test <- function(sim, cohorts, seed) {
  stopifnot(all(cohorts %in% colnames(sim$M)))
  set.seed(seed)
  for (cc in cohorts) {
    d  <- sim$multiExpr[[cc]]$data
    nm <- colnames(d)
    d  <- d[, sample(ncol(d))]
    colnames(d) <- nm                     # same names, permuted content
    sim$multiExpr[[cc]]$data <- d
    sim$M[, cc] <- 0
  }
  keep <- sim$truth[, c("disc_mean_r", "disc_ve")]
  sim$truth <- cbind(ecms_truth_table(sim$params$sizes, sim$M), keep)
  sim$relabelled <- union(sim$relabelled, cohorts)
  sim$relabel_seed <- seed
  sim
}

# ---- Matched-size null labels (mirrors ecms_null_run.R) -------------------------

# Null size set for one replicate: the planted sizes, plus an optional cap
# (ecms_null_run.R adds 1000, WGCNA's cap; with p = 2000 that is half the
# universe, an OPEN decision).
ecms_size_set <- function(sim, cap = NA) {
  s <- unique(sim$params$sizes)
  if (!is.na(cap)) s <- c(s, cap)
  sort(unique(s))
}

# Same algorithm as random_partition() in ecms_null_run.R: shuffle genes, then
# cut consecutive chunks whose sizes are drawn from `sizes`; leftovers get 0.
ecms_random_partition <- function(genes, sizes, seed) {
  set.seed(seed)
  perm <- sample(genes)
  lab  <- integer(length(perm))
  i <- 1L; k <- 0L
  repeat {
    remaining <- length(perm) - i + 1L
    fits <- sizes[sizes <= remaining]
    if (!length(fits)) break
    s <- fits[sample.int(length(fits), 1L)]
    k <- k + 1L
    lab[i:(i + s - 1L)] <- k
    i <- i + s
    if (i > length(perm)) break
  }
  out <- integer(length(genes))
  out[match(perm, genes)] <- lab
  out
}

# multiColor for one null call on this replicate's data (reference = disc).
ecms_null_labels <- function(sim, call, cap = NA, seed_base = 42L * 10000L) {
  genes <- colnames(sim$multiExpr$disc$data)
  list(disc = ecms_random_partition(genes, ecms_size_set(sim, cap), seed_base + call))
}

# Hold-out split, as in ecms_null_aggregate.R: first n_holdout calls are held out.
ecms_null_split <- function(calls, n_holdout = 5) {
  calls <- sort(unique(calls))
  stopifnot(length(calls) > n_holdout + 5)
  list(holdout = head(calls, n_holdout), train = tail(calls, -n_holdout))
}

# ---- Implied correlations and calibration --------------------------------------

# Mean off-diagonal population correlation among genes with weights w,
# effective loading a = m * lambda.
ecms_expected_r <- function(w, a, gamma, sigma) {
  num <- a^2 * outer(w, w) + gamma^2
  d   <- sqrt(a^2 * w^2 + gamma^2 + sigma^2)
  R   <- num / outer(d, d)
  mean(R[upper.tri(R)])
}

# lambda giving a target mean within-module correlation (population value).
ecms_calibrate_lambda <- function(target_r, gamma, sigma = 1, w_range = c(0.5, 1),
                                  n_w = 300, seed = 1) {
  set.seed(seed); w <- runif(n_w, w_range[1], w_range[2])
  f <- function(l) ecms_expected_r(w, l, gamma, sigma) - target_r
  if (f(0) >= 0) stop("target_r is at or below the background correlation for this gamma")
  uniroot(f, c(0, 50))$root
}

# ---- Helpers ------------------------------------------------------------------------

mean_offdiag_r <- function(X) { R <- cor(X); mean(R[upper.tri(R)]) }

# Fraction of variance explained by the first PC of the standardized genes
# (the quantity WGCNA reports as eigengene variance explained).
eigengene_ve <- function(X) {
  ev <- eigen(cor(X), symmetric = TRUE, only.values = TRUE)$values
  ev[1] / sum(ev)
}

# ---- Self check -------------------------------------------------------------------

# Observed vs expected mean correlation per planted module, background, and a
# random gene set (the random set has no closed form; it is the gamma
# calibration target, to compare with the real 0.17 ORIEN / 0.01 to 0.05 others).
ecms_check_sim <- function(sim, n_random = 200) {
  par <- sim$params; lab <- sim$multiColor$disc; K <- length(par$sizes)
  out <- list()
  for (j in seq_along(sim$multiExpr)) {
    coh  <- names(sim$multiExpr)[j]
    X    <- as.matrix(sim$multiExpr[[j]]$data)
    mult <- if (j == 1) rep(1, K) else sim$M[, j - 1]
    for (k in seq_len(K)) {
      idx <- which(lab == k)
      out[[length(out) + 1]] <- data.frame(
        cohort = coh, set = paste0("module_", k),
        type = if (mult[k] > 0) "module_present" else "module_absent",
        observed = mean_offdiag_r(X[, idx]),
        expected = ecms_expected_r(sim$w[idx], mult[k] * par$lambda[j],
                                   par$gamma[j], par$sigma[j]))
    }
    bg <- which(lab == 0)
    if (length(bg) >= 2) {
      idx <- if (length(bg) > n_random) sample(bg, n_random) else bg
      out[[length(out) + 1]] <- data.frame(
        cohort = coh, set = "background", type = "background",
        observed = mean_offdiag_r(X[, idx]),
        expected = par$gamma[j]^2 / (par$gamma[j]^2 + par$sigma[j]^2))
    }
    idx <- sample(ncol(X), n_random)
    out[[length(out) + 1]] <- data.frame(
      cohort = coh, set = "random_set", type = "random_set",
      observed = mean_offdiag_r(X[, idx]), expected = NA_real_)
  }
  do.call(rbind, out)
}

# Pre-registered pass rule: |mean(observed - expected)| <= 0.01 for every
# cohort x type (module_present, module_absent, background) over 20 seeds.
# Only meaningful on UNshuffled replicates (expected r assumes correspondence).
ecms_check_many <- function(par, seeds = 1:20) {
  all <- do.call(rbind, lapply(seeds, function(s) {
    cat(sprintf("[check] seed %d  %s\n", s, format(Sys.time(), "%H:%M:%S")))
    ch <- ecms_check_sim(ecms_generate(par, s, verbose = FALSE)); ch$seed <- s; ch
  }))
  all$diff <- all$observed - all$expected
  tested <- all[all$type != "random_set", ]
  summ <- aggregate(cbind(observed, expected, diff) ~ cohort + type, data = tested, FUN = mean)
  summ$pass <- abs(summ$diff) <= 0.01
  rnd <- aggregate(observed ~ cohort, data = all[all$type == "random_set", ], FUN = mean)
  cat(sprintf("[check] %d of %d cohort x type cells pass\n", sum(summ$pass), nrow(summ)))
  list(per_set = all, summary = summ, random_set_mean_r = rnd)
}

# Relabel self check (rev 2): after shuffling a test cohort, its mean pairwise r
# distribution must be unchanged (same structure) and every module must be
# recorded as absent there. Returns TRUE/FALSE per check.
ecms_check_relabel <- function(sim, cohort, seed) {
  before <- as.matrix(sim$multiExpr[[cohort]]$data)
  s2     <- ecms_relabel_test(sim, cohort, seed)
  after  <- as.matrix(s2$multiExpr[[cohort]]$data)
  c(same_values   = isTRUE(all.equal(sort(as.vector(before)), sort(as.vector(after)))),
    same_names    = identical(colnames(before), colnames(after)),
    content_moved = !isTRUE(all.equal(before, after)),
    truth_absent  = all(s2$M[, cohort] == 0))
}
