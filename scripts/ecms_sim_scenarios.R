# ==============================================================================
# ECMS simulation v2: scenario definitions (design Part I, V2.3). Sourced by
# ecms_sim_call.R and ecms_sim_score.R. Requires ecms_sim_generator.R (rev 5).
#
# All scenarios start from the frozen base (ecms_base_params()) and change ONE
# thing. Size vectors for C are evenly spaced over the pre-registered ranges.
# ==============================================================================

ECMS_NPERM  <- 200L        # Part II §7d rule; data and null calls always match
ECMS_ALPHAS <- c(0.01, 0.05, 0.10)

ecms_scenarios <- function() {
  base <- ecms_base_params()
  list(
    A        = modifyList(base, list(n_by_npres = c(9, 0, 0, 0))),               # global null, pure absence
    B        = base,                                                              # base
    C_small  = modifyList(base, list(sizes = c(30, 36, 42, 49, 55, 61, 67, 74, 80))),
    C_large  = modifyList(base, list(sizes = c(300, 338, 375, 413, 450, 488, 525, 563, 600))),
    D_weak   = modifyList(base, list(lambda = rep(0.63, 4))),
    D_strong = modifyList(base, list(lambda = rep(1.51, 4))),
    E        = modifyList(base, list(n_by_npres = c(9, 0, 0, 0), reorg_phi = 0.45)) # hard null
  )
}

# Replicates (V2.3) and reused-null calls per scenario. C_large gets more null
# calls because fewer large random modules fit in each call (about 8 vs 22).
ecms_n_data <- c(A = 1000, B = 1000, C_small = 200, C_large = 200,
                 D_weak = 200, D_strong = 200, E = 1000)
ecms_n_null <- c(A = 100, B = 100, C_small = 100, C_large = 200,
                 D_weak = 100, D_strong = 100, E = 100)

# Seeds: disjoint blocks per scenario and call type, so no two calls share data.
ecms_seed <- function(scenario, type, index) {
  s <- match(scenario, names(ecms_scenarios()))
  stopifnot(!is.na(s), type %in% c("data", "null"), index >= 1, index < 400000)
  as.integer(1e6 * s + (if (type == "null") 500000L else 0L) + index)
}
