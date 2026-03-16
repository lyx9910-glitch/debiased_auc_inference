###############################################################################
# 02_ground_truth.R - Compute ground truth Delta AUC via large Monte Carlo
#
# Key changes from original:
#   - Separate test sample sizes: n1_test, n2_test
#   - Models fit only ONCE per replicate (non-split & split reuse same fit)
#   - Uses unified fit_and_evaluate_model() from 01_model_functions.R
###############################################################################

#' Compute ground truth Delta AUC for nested models
#'
#' Generates fresh train + test data R times, fits each model once per
#' replicate, and averages the split (test-set) AUC differences.
#'
#' @param n1        Number of class-1 training samples
#' @param n2        Number of class-0 training samples
#' @param n1_test   Number of class-1 test samples
#' @param n2_test   Number of class-0 test samples
#' @param R         Number of Monte Carlo replicates
#' @param p         Number of noise features (default 100)
#' @param mu        Mean shift for X1 (default 1)
#' @param sig       SD for noise features (default 10)
#' @param model_types Character vector of models to evaluate
#'                    (default: c("lda", "logistic", "svm", "xgboost"))
#' @param nrounds   XGBoost boosting rounds (default 100)
#' @param reduced_formula Reduced model formula (default "Y ~ X1")
#' @return Named list of ground-truth delta-AUC values per model type
compute_ground_truth <- function(n1, n2,
                                 n1_test = n1, n2_test = n2,
                                 R = 10000,
                                 p = 100, mu = 1, sig = 10,
                                 model_types = c("lda", "logistic",
                                                 "svm", "xgboost"),
                                 nrounds = 100,
                                 reduced_formula = "Y ~ X1") {

  # Storage: one vector per model for split AUC difference
  results <- lapply(model_types, function(m) {
    list(auc_test_full = rep(0, R),
         auc_test_redu = rep(0, R))
  })
  names(results) <- model_types

  for (i in 1:R) {
    if (i %% 500 == 0) cat(sprintf("Ground truth: rep %d / %d\n", i, R))

    # Generate training data
    df_train <- generate_sim_data(n1, n2, p = p, mu = mu, sig = sig)

    # Generate independent testing data (potentially different size)
    df_test <- generate_sim_data(n1_test, n2_test, p = p, mu = mu, sig = sig)
    # Align column names
    colnames(df_test) <- colnames(df_train)
    y_test <- df_test$Y

    # Fit each model ONCE on training data, evaluate on test data
    for (m in model_types) {
      res <- fit_and_evaluate_model(
        model_type       = m,
        df_train         = df_train,
        df_test          = df_test,
        y_test           = y_test,
        reduced_formula  = reduced_formula,
        nrounds          = nrounds,
        direction        = "<",
        xgb_redu_cols    = "X1"
      )
      results[[m]]$auc_test_full[i] <- res$auc_full
      results[[m]]$auc_test_redu[i] <- res$auc_redu
    }
  }

  # Compute ground truth as mean(full) - mean(redu) for each model
  ground_truth <- sapply(model_types, function(m) {
    mean(results[[m]]$auc_test_full) - mean(results[[m]]$auc_test_redu)
  })
  names(ground_truth) <- paste0("delta_auc_", model_types)

  return(list(
    ground_truth = ground_truth,
    details      = results
  ))
}
