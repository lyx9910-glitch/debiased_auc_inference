###############################################################################
# 07_real_data_heart.R - Real Data Application: Framingham Heart Study
#
# Uses fit_and_evaluate_model() for both non-split and split scenarios.
# Non-split results computed once; split results repeated R times.
# Models: LDA and Logistic only (as in original).
###############################################################################

#' Run Framingham Heart Study data analysis
#'
#' Full model: Y ~ BMI + SYSBP + DIABP + TOTCHOL + AGE
#' Reduced model: Y ~ BMI + SYSBP + DIABP + TOTCHOL  (i.e., Y ~ . - AGE)
#'
#' @param R   Number of random split repetitions (default 1000)
#' @return List with nosplit and split results for LDA and logistic
run_heart_analysis <- function(R = 1000) {

  # --- Data preparation ---
  df_heart <- data.frame(framingham)
  df_heart <- df_heart[complete.cases(df_heart), ]

  # Select participants from last visit for independent samples
  df_last_visit <- df_heart[df_heart$TIME > 4000, ]
  cat(sprintf("Max visits per participant: %d\n",
              max(table(df_last_visit$RANDID))))

  X1     <- cbind(df_last_visit$BMI, df_last_visit$SYSBP,
                  df_last_visit$DIABP, df_last_visit$TOTCHOL)
  X_rest <- df_last_visit$AGE
  Y      <- df_last_visit$DIABETES

  X1     <- as.data.frame(scale(X1))
  X_rest <- as.data.frame(scale(X_rest))

  df <- data.frame(cbind(Y, X1, X_rest))
  colnames(df) <- c("Y", "BMI", "SYSBP", "DIABP", "TOTCHOL", "AGE")

  n <- nrow(df)
  model_types <- c("lda", "logistic")
  reduced_formula <- "Y ~ . - AGE"

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

  # --- Non-split: fit once ---
  for (m in model_types) {
    res <- fit_and_evaluate_model(
      model_type      = m,
      df_train        = df,
      df_test         = df,
      y_test          = df$Y,
      reduced_formula = reduced_formula,
      direction       = "<"
    )
    res_nosplit[[m]]$auc_full     <- res$auc_full
    res_nosplit[[m]]$auc_redu     <- res$auc_redu
    res_nosplit[[m]]$delong_lower <- res$delong_lower
    res_nosplit[[m]]$delong_upper <- res$delong_upper
  }
  cat("Non-split done.\n")

  # --- Split: repeat R times ---
  for (i in 1:R) {
    if (i %% max(1, R %/% 5) == 0) {
      cat(sprintf("  Heart split rep %d / %d\n", i, R))
    }

    test_ix  <- sample(1:n, n %/% 2)
    df_test  <- df[test_ix, ]
    df_train <- df[-test_ix, ]
    y_test   <- df_test$Y

    for (m in model_types) {
      res <- fit_and_evaluate_model(
        model_type      = m,
        df_train        = df_train,
        df_test         = df_test,
        y_test          = y_test,
        reduced_formula = reduced_formula,
        direction       = "<"
      )
      res_split[[m]]$auc_full[i]     <- res$auc_full
      res_split[[m]]$auc_redu[i]     <- res$auc_redu
      res_split[[m]]$delong_lower[i] <- res$delong_lower
      res_split[[m]]$delong_upper[i] <- res$delong_upper
    }
  }

  return(list(
    nosplit     = res_nosplit,
    split       = res_split,
    model_types = model_types,
    R           = R,
    n           = n
  ))
}


#' Print Framingham Heart results summary and plots
#'
#' @param heart_result Output from run_heart_analysis()
print_heart_summary <- function(heart_result) {
  cat("\n====== Framingham Heart Study Results ======\n")
  for (m in heart_result$model_types) {
    cat(sprintf("\n--- %s ---\n", toupper(m)))

    # Non-split
    ns <- heart_result$nosplit[[m]]
    cat(sprintf("  Non-split: AUC_full=%.4f  AUC_redu=%.4f  diff=%.4f\n",
                ns$auc_full, ns$auc_redu, ns$auc_full - ns$auc_redu))
    cat(sprintf("  DeLong CI: [%.4f, %.4f]\n",
                ns$delong_lower, ns$delong_upper))

    # Split
    sp <- heart_result$split[[m]]
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

  # Plot
  par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
  for (m in heart_result$model_types) {
    sp <- heart_result$split[[m]]
    diff_vec <- sp$auc_full - sp$auc_redu
    hist(diff_vec, freq = FALSE, breaks = 20,
         main = paste0(toupper(m), " - Split"),
         xlab = "AUC Difference", xlim = c(-0.1, 0.1))
    x_seq <- seq(-0.08, 0.08, length = 1000)
    y_norm <- dnorm(x_seq, mean = mean(diff_vec), sd = sd(diff_vec))
    lines(x_seq, y_norm, lwd = 1)
  }
}
