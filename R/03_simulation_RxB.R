###############################################################################
# 03_simulation_RxB.R - R x B Simulation (resampling-based)
#
# Key changes from original:
#   - Non-split (no-refit) model is fit ONCE per MC replicate (outside B loop)
#     because without resampling / splitting the result is deterministic.
#   - Inside the B loop: only sample-split evaluation is repeated.
#   - Uses unified fit_and_evaluate_model() for all models.
#   - Total inference samples = R * B for split results.
#   - Includes timing for a single iteration to help tune R and B.
###############################################################################

#' Run R x B simulation for nested AUC difference
#'
#' For each of R Monte Carlo replicates:
#'   1. Generate fresh data (n1 + n2 observations).
#'   2. Fit all models once on the full data -> non-split AUC (deterministic).
#'   3. For each of B resample-split iterations:
#'      a. Resample with replacement.
#'      b. Split into train/test.
#'      c. Fit models on train, evaluate on test -> split AUC + DeLong.
#'
#' @param n1  Number of class-1 samples
#' @param n2  Number of class-0 samples
#' @param R   Number of Monte Carlo replicates
#' @param B   Number of resample-split iterations per replicate
#' @param p   Number of noise features (default 100)
#' @param mu  Mean shift (default 1)
#' @param sig SD of noise features (default 10)
#' @param test_frac  Fraction for test split (default 0.25)
#' @param model_types Models to run (default: lda, logistic, svm, xgboost)
#' @param nrounds  XGBoost rounds (default 100)
#' @param reduced_formula Formula string for reduced model (default "Y ~ X1")
#' @return Named list with matrices of results (R x B for split, R x 1 for
#'         non-split), plus timing info.
run_simulation_RxB <- function(n1 = 1000, n2 = 1000,
                               R = 50, B = 50,
                               p = 100, mu = 1, sig = 10,
                               test_frac = 0.25,
                               model_types = c("lda", "logistic",
                                               "svm", "xgboost"),
                               nrounds = 100,
                               reduced_formula = "Y ~ X1") {

  n_total   <- n1 + n2
  n_test    <- round(n_total * test_frac)

  # --- Storage: non-split (R x 1), split (R x B) ---
  init_mat <- function(nr, nc) matrix(0, nrow = nr, ncol = nc)

  nosplit <- list()
  split_res <- list()
  for (m in model_types) {
    nosplit[[m]] <- list(
      auc_full     = rep(0, R),
      auc_redu     = rep(0, R),
      delong_lower = rep(0, R),
      delong_upper = rep(0, R)
    )
    split_res[[m]] <- list(
      auc_full     = init_mat(R, B),
      auc_redu     = init_mat(R, B),
      delong_lower = init_mat(R, B),
      delong_upper = init_mat(R, B)
    )
  }

  # --- Time a single iteration ---
  cat("Timing a single (i=1, j=1) iteration...\n")
  t_single <- time_single_iter(function() {
    df_tmp <- generate_sim_data(n1, n2, p, mu, sig)
    for (m in model_types) {
      fit_and_evaluate_model(m, df_tmp, df_tmp, df_tmp$Y,
                             reduced_formula = reduced_formula,
                             nrounds = nrounds, xgb_redu_cols = "X1")
    }
    test_ix   <- sample(1:n_total, n_test)
    df_test_  <- df_tmp[test_ix, ]
    df_train_ <- df_tmp[-test_ix, ]
    for (m in model_types) {
      fit_and_evaluate_model(m, df_train_, df_test_, df_test_$Y,
                             reduced_formula = reduced_formula,
                             nrounds = nrounds, xgb_redu_cols = "X1")
    }
  })
  est_total <- t_single * R * B
  cat(sprintf("  Single iteration: %.2f sec\n", t_single))
  cat(sprintf("  Estimated total (R=%d, B=%d): %.1f sec (%.1f min)\n",
              R, B, est_total, est_total / 60))

  # --- Main simulation loop ---
  t_start <- proc.time()

  for (i in 1:R) {
    if (i %% max(1, R %/% 10) == 0) {
      cat(sprintf("  MC replicate %d / %d\n", i, R))
    }

    # 1. Generate original data for this MC replicate
    df_ori <- generate_sim_data(n1, n2, p, mu, sig)

    # 2. Non-split: fit once on full original data (deterministic -> no B loop)
    for (m in model_types) {
      res_ns <- fit_and_evaluate_model(
        model_type      = m,
        df_train        = df_ori,
        df_test         = df_ori,
        y_test          = df_ori$Y,
        reduced_formula = reduced_formula,
        nrounds         = nrounds,
        direction       = "<",
        xgb_redu_cols   = "X1"
      )
      nosplit[[m]]$auc_full[i]     <- res_ns$auc_full
      nosplit[[m]]$auc_redu[i]     <- res_ns$auc_redu
      nosplit[[m]]$delong_lower[i] <- res_ns$delong_lower
      nosplit[[m]]$delong_upper[i] <- res_ns$delong_upper
    }

    # 3. Split: resample + train/test split B times
    for (j in 1:B) {
      # Resample with replacement
      id <- sample(1:n_total, replace = TRUE)
      df_boot <- df_ori[id, ]

      # Train/test split on the bootstrap sample
      test_ix  <- sample(1:n_total, n_test)
      df_test  <- df_boot[test_ix, ]
      df_train <- df_boot[-test_ix, ]
      y_test   <- df_test$Y

      for (m in model_types) {
        res_sp <- fit_and_evaluate_model(
          model_type      = m,
          df_train        = df_train,
          df_test         = df_test,
          y_test          = y_test,
          reduced_formula = reduced_formula,
          nrounds         = nrounds,
          direction       = "<",
          xgb_redu_cols   = "X1"
        )
        split_res[[m]]$auc_full[i, j]     <- res_sp$auc_full
        split_res[[m]]$auc_redu[i, j]     <- res_sp$auc_redu
        split_res[[m]]$delong_lower[i, j] <- res_sp$delong_lower
        split_res[[m]]$delong_upper[i, j] <- res_sp$delong_upper
      }
    }
  }

  elapsed <- (proc.time() - t_start)["elapsed"]
  cat(sprintf("Simulation completed in %.1f sec (%.1f min).\n",
              elapsed, elapsed / 60))

  return(list(
    nosplit       = nosplit,
    split         = split_res,
    params        = list(n1 = n1, n2 = n2, R = R, B = B, p = p,
                         mu = mu, sig = sig, test_frac = test_frac,
                         model_types = model_types),
    time_single   = t_single,
    time_total    = elapsed
  ))
}
