###############################################################################
# 06_real_data_ptb.R - Real Data Application: Preterm Birth (PTB) Data
#
# Uses fit_and_evaluate_model() for both non-split and split scenarios.
# Non-split results are computed once (deterministic).
# Split results repeat R times with random train/test splits.
###############################################################################

#' Run preterm birth (PTB) real data analysis
#'
#' Scenario 1: Full model (all features + interactions) vs reduced model.
#' Scenario 2: HC-only features vs base model.
#'
#' @param R   Number of random split repetitions (default 1000)
#' @param model_types Models to evaluate (default: lda, logistic, svm, xgboost)
#' @param nrounds XGBoost rounds (default 100)
#' @return List with results for scenario 1 and scenario 2
run_ptb_analysis <- function(R = 1000,
                             model_types = c("lda", "logistic",
                                             "svm", "xgboost"),
                             nrounds = 100) {

  # --- Data preparation ---
  ptb_update <- ptb[ptb$outcome != "mPTB", ]
  n <- nrow(ptb_update)
  priorPTB <- as.integer(ptb_update$prior_PTB_count > 0)
  prior_ptb_features <- within(ptb_update, rm(outcome))

  # Build interaction features
  df_update_features <- cbind(
    prior_ptb_features, priorPTB,
    ptb_update$ACV1 * ptb_update$GAV1, ptb_update$ACV1 * ptb_update$GAV2,
    ptb_update$ACV2 * ptb_update$GAV1, ptb_update$ACV2 * ptb_update$GAV2,
    ptb_update$CLV1 * ptb_update$GAV1, ptb_update$CLV1 * ptb_update$GAV2,
    ptb_update$CLV2 * ptb_update$GAV1, ptb_update$CLV2 * ptb_update$GAV2,
    ptb_update$InterceptV1 * ptb_update$GAV1,
    ptb_update$InterceptV1 * ptb_update$GAV2,
    ptb_update$InterceptV2 * ptb_update$GAV1,
    ptb_update$InterceptV2 * ptb_update$GAV2,
    ptb_update$MidbandV1 * ptb_update$GAV1,
    ptb_update$MidbandV1 * ptb_update$GAV2,
    ptb_update$MidbandV2 * ptb_update$GAV1,
    ptb_update$MidbandV2 * ptb_update$GAV2,
    ptb_update$SlopeV1 * ptb_update$GAV1, ptb_update$SlopeV1 * ptb_update$GAV2,
    ptb_update$SlopeV2 * ptb_update$GAV1, ptb_update$SlopeV2 * ptb_update$GAV2,
    ptb_update$ACV1 * priorPTB, ptb_update$ACV2 * priorPTB,
    ptb_update$CLV1 * priorPTB, ptb_update$CLV2 * priorPTB,
    ptb_update$InterceptV1 * priorPTB, ptb_update$InterceptV2 * priorPTB,
    ptb_update$MidbandV1 * priorPTB, ptb_update$MidbandV2 * priorPTB,
    ptb_update$SlopeV1 * priorPTB, ptb_update$SlopeV2 * priorPTB,
    ptb_update$ACV1 * (1 - priorPTB), ptb_update$ACV2 * (1 - priorPTB),
    ptb_update$CLV1 * (1 - priorPTB), ptb_update$CLV2 * (1 - priorPTB),
    ptb_update$InterceptV1 * (1 - priorPTB),
    ptb_update$InterceptV2 * (1 - priorPTB),
    ptb_update$MidbandV1 * (1 - priorPTB),
    ptb_update$MidbandV2 * (1 - priorPTB),
    ptb_update$SlopeV1 * (1 - priorPTB), ptb_update$SlopeV2 * (1 - priorPTB)
  )

  Y <- as.integer(factor(ptb_update$outcome)) - 1
  df_update_features <- as.data.frame(scale(df_update_features))
  df_scenario1 <- data.frame(cbind(Y, df_update_features))

  # Scenario 2: HC-only vs base
  X1 <- ptb_update$CLV0
  X_rest <- cbind(ptb_update$CLV2, ptb_update$SlopeV2,
                  ptb_update$SlopeV2 * ptb_update$GAV2)
  df_scenario2 <- data.frame(cbind(Y, X1, X_rest))

  # --- Helper to run one scenario ---
  run_one_scenario <- function(df, reduced_formula, direction_svm = "<",
                               xgb_redu_cols = "X1", label = "") {
    cat(sprintf("\n=== PTB Scenario: %s ===\n", label))

    # Storage
    res_nosplit <- list()
    res_split   <- list()
    for (m in model_types) {
      res_nosplit[[m]] <- list(
        auc_full = 0, auc_redu = 0,
        delong_lower = 0, delong_upper = 0
      )
      res_split[[m]] <- list(
        auc_full = rep(0, R), auc_redu = rep(0, R),
        delong_lower = rep(0, R), delong_upper = rep(0, R)
      )
    }

    # Non-split: fit once (deterministic)
    for (m in model_types) {
      dir <- if (m == "svm") direction_svm else "<"
      res <- fit_and_evaluate_model(
        model_type      = m,
        df_train        = df,
        df_test         = df,
        y_test          = df$Y,
        reduced_formula = reduced_formula,
        nrounds         = nrounds,
        direction       = dir,
        xgb_redu_cols   = xgb_redu_cols
      )
      res_nosplit[[m]]$auc_full     <- res$auc_full
      res_nosplit[[m]]$auc_redu     <- res$auc_redu
      res_nosplit[[m]]$delong_lower <- res$delong_lower
      res_nosplit[[m]]$delong_upper <- res$delong_upper
    }
    cat("  Non-split done.\n")

    # Split: repeat R times
    for (i in 1:R) {
      if (i %% max(1, R %/% 5) == 0)
        cat(sprintf("  Split rep %d / %d\n", i, R))

      test_ix  <- sample(1:n, n %/% 4)
      df_test  <- df[test_ix, ]
      df_train <- df[-test_ix, ]
      y_test   <- df_test$Y

      for (m in model_types) {
        dir <- if (m == "svm") direction_svm else "<"
        res <- fit_and_evaluate_model(
          model_type      = m,
          df_train        = df_train,
          df_test         = df_test,
          y_test          = y_test,
          reduced_formula = reduced_formula,
          nrounds         = nrounds,
          direction       = dir,
          xgb_redu_cols   = xgb_redu_cols
        )
        res_split[[m]]$auc_full[i]     <- res$auc_full
        res_split[[m]]$auc_redu[i]     <- res$auc_redu
        res_split[[m]]$delong_lower[i] <- res$delong_lower
        res_split[[m]]$delong_upper[i] <- res$delong_upper
      }
    }

    return(list(nosplit = res_nosplit, split = res_split))
  }

  # --- Scenario 1: Full interactions vs reduced ---
  reduced_s1 <- "Y ~ prior_PTB_count + CLV2 + SlopeV2 + SlopeV2 * GAV2"
  xgb_redu_s1 <- c("prior_PTB_count", "CLV2", "SlopeV2",
                    grep("SlopeV1.*GAV1", names(df_scenario1), value = TRUE)[1])
  # Fall back if grep doesn't find
  if (is.na(xgb_redu_s1[4])) {
    xgb_redu_s1 <- c("prior_PTB_count", "CLV2", "SlopeV2")
  }

  scenario1 <- run_one_scenario(
    df_scenario1, reduced_s1, direction_svm = ">",
    xgb_redu_cols = xgb_redu_s1, label = "Full interactions vs reduced"
  )

  # --- Scenario 2: HC-only vs base ---
  scenario2 <- run_one_scenario(
    df_scenario2, "Y ~ X1", direction_svm = ">",
    xgb_redu_cols = "X1", label = "HC-only vs base"
  )

  return(list(scenario1 = scenario1, scenario2 = scenario2,
              model_types = model_types, R = R))
}


#' Print PTB results summary
#'
#' @param ptb_result Output from run_ptb_analysis()
print_ptb_summary <- function(ptb_result) {
  for (s_name in c("scenario1", "scenario2")) {
    cat(sprintf("\n====== PTB %s ======\n", s_name))
    sc <- ptb_result[[s_name]]
    for (m in ptb_result$model_types) {
      cat(sprintf("\n--- %s ---\n", toupper(m)))

      # Non-split
      ns <- sc$nosplit[[m]]
      cat(sprintf("  Non-split: AUC_full=%.4f  AUC_redu=%.4f  diff=%.4f\n",
                  ns$auc_full, ns$auc_redu, ns$auc_full - ns$auc_redu))
      cat(sprintf("  DeLong CI: [%.4f, %.4f]\n",
                  ns$delong_lower, ns$delong_upper))

      # Split
      sp <- sc$split[[m]]
      diff_vec <- sp$auc_full - sp$auc_redu
      cat(sprintf("  Split: mean(full)=%.4f sd=%.4f  mean(redu)=%.4f sd=%.4f\n",
                  mean(sp$auc_full), sd(sp$auc_full),
                  mean(sp$auc_redu), sd(sp$auc_redu)))
      cat(sprintf("  Split diff: mean=%.4f sd=%.4f z=%.3f\n",
                  mean(diff_vec), sd(diff_vec),
                  mean(diff_vec) / sd(diff_vec)))
      cat(sprintf("  Split 90%% CI: [%.4f, %.4f]\n",
                  quantile(diff_vec, 0.05), quantile(diff_vec, 0.95)))
      cat(sprintf("  DeLong CI (split mean): [%.4f, %.4f]\n",
                  mean(sp$delong_lower), mean(sp$delong_upper)))
    }
  }
}
