###############################################################################
# 05_loo_cv.R - Leave-One-Out Cross Validation Simulation
#
# Key changes from original:
#   - Non-split results computed ONCE per MC replicate (outside B loop).
#   - LOO-CV results also deterministic for fixed data -> computed once.
#   - B loop only needed if resampling; here we keep R x B structure
#     but note LOO on same data is deterministic, so B copies are identical.
#   - Uses pihat_loo() from 01_model_functions.R for logistic LOO.
#   - Uses lda(..., CV=TRUE) for LDA LOO.
###############################################################################

#' Run LOO-CV simulation for LDA and Logistic models
#'
#' @param n1  Number of class-1 samples
#' @param n2  Number of class-0 samples
#' @param R   Number of Monte Carlo replicates (each generates fresh data)
#' @param p   Number of noise features (default 100)
#' @param mu  Mean shift (default 1)
#' @param sig SD of noise features (default 10)
#' @param reduced_formula Formula for reduced model (default "Y ~ X1")
#' @return List with non-split and LOO AUC results, DeLong CIs
run_loo_cv_simulation <- function(n1 = 1000, n2 = 1000,
                                  R = 200,
                                  p = 100, mu = 1, sig = 10,
                                  reduced_formula = "Y ~ X1") {

  # Storage (R replicates, each gives one value for nosplit and one for LOO)
  auc_lda_nosplit_full  <- rep(0, R)
  auc_lda_nosplit_redu  <- rep(0, R)
  auc_lda_loo_full      <- rep(0, R)
  auc_lda_loo_redu      <- rep(0, R)

  auc_logi_nosplit_full <- rep(0, R)
  auc_logi_nosplit_redu <- rep(0, R)
  auc_logi_loo_full     <- rep(0, R)
  auc_logi_loo_redu     <- rep(0, R)

  delong_nosplit_lda    <- matrix(0, R, 2)  # lower, upper
  delong_loo_lda        <- matrix(0, R, 2)
  delong_nosplit_logi   <- matrix(0, R, 2)
  delong_loo_logi       <- matrix(0, R, 2)

  t_start <- proc.time()

  for (i in 1:R) {
    if (i %% max(1, R %/% 10) == 0) {
      cat(sprintf("  LOO-CV: replicate %d / %d\n", i, R))
    }

    # Generate fresh data
    df <- generate_sim_data(n1, n2, p = p, mu = mu, sig = sig)
    Y.true <- df$Y

    # ---- LDA non-split ----
    lda_full <- lda(Y ~ ., data = df)
    pred_full_lda <- as.numeric(predict(lda_full, type = "response")$x)
    auc_lda_nosplit_full[i] <- auc(Y.true, pred_full_lda,
                                   direction = "<", levels = c("0","1"))[1]

    lda_redu <- lda(as.formula(reduced_formula), data = df)
    pred_redu_lda <- as.numeric(predict(lda_redu, type = "response")$x)
    auc_lda_nosplit_redu[i] <- auc(Y.true, pred_redu_lda,
                                   direction = "<", levels = c("0","1"))[1]

    # DeLong for LDA non-split
    roc1 <- roc(Y.true, pred_full_lda, direction = "<", levels = c("0","1"))
    roc2 <- roc(Y.true, pred_redu_lda, direction = "<", levels = c("0","1"))
    dt_lda <- roc.test(roc1, roc2, method = "delong")
    delong_nosplit_lda[i, ] <- dt_lda$conf.int

    # ---- LDA LOO-CV ----
    lda_full_loo <- lda(Y ~ ., data = df, CV = TRUE)
    pred_full_loo_lda <- lda_full_loo$posterior[, 2]
    auc_lda_loo_full[i] <- auc(Y.true, pred_full_loo_lda,
                                direction = "<", levels = c("0","1"))[1]

    lda_redu_loo <- lda(as.formula(reduced_formula), data = df, CV = TRUE)
    pred_redu_loo_lda <- lda_redu_loo$posterior[, 2]
    auc_lda_loo_redu[i] <- auc(Y.true, pred_redu_loo_lda,
                                direction = "<", levels = c("0","1"))[1]

    roc3 <- roc(Y.true, pred_full_loo_lda, direction = "<", levels = c("0","1"))
    roc4 <- roc(Y.true, pred_redu_loo_lda, direction = "<", levels = c("0","1"))
    dt_lda_loo <- roc.test(roc3, roc4, method = "delong")
    delong_loo_lda[i, ] <- dt_lda_loo$conf.int

    # ---- Logistic non-split ----
    logi_full <- glm(Y ~ ., data = df, family = binomial(link = "logit"))
    pred_full_logi <- logi_full$fitted.values
    auc_logi_nosplit_full[i] <- auc(Y.true, pred_full_logi,
                                    direction = "<", levels = c("0","1"))[1]

    logi_redu <- glm(as.formula(reduced_formula), data = df,
                     family = binomial(link = "logit"))
    pred_redu_logi <- logi_redu$fitted.values
    auc_logi_nosplit_redu[i] <- auc(Y.true, pred_redu_logi,
                                    direction = "<", levels = c("0","1"))[1]

    roc5 <- roc(Y.true, pred_full_logi, direction = "<", levels = c("0","1"))
    roc6 <- roc(Y.true, pred_redu_logi, direction = "<", levels = c("0","1"))
    dt_logi <- roc.test(roc5, roc6, method = "delong")
    delong_nosplit_logi[i, ] <- dt_logi$conf.int

    # ---- Logistic LOO-CV ----
    pred_full_loo_logi <- pihat_loo(logi_full, df)
    auc_logi_loo_full[i] <- auc(Y.true, pred_full_loo_logi,
                                 direction = "<", levels = c("0","1"))[1]

    pred_redu_loo_logi <- pihat_loo(logi_redu, df)
    auc_logi_loo_redu[i] <- auc(Y.true, pred_redu_loo_logi,
                                 direction = "<", levels = c("0","1"))[1]

    roc7 <- roc(Y.true, pred_full_loo_logi, direction = "<", levels = c("0","1"))
    roc8 <- roc(Y.true, pred_redu_loo_logi, direction = "<", levels = c("0","1"))
    dt_logi_loo <- roc.test(roc7, roc8, method = "delong")
    delong_loo_logi[i, ] <- dt_logi_loo$conf.int
  }

  elapsed <- (proc.time() - t_start)["elapsed"]
  cat(sprintf("LOO-CV simulation done in %.1f sec.\n", elapsed))

  # Compute differences
  lda_nosplit_diff  <- auc_lda_nosplit_full  - auc_lda_nosplit_redu
  lda_loo_diff      <- auc_lda_loo_full      - auc_lda_loo_redu
  logi_nosplit_diff <- auc_logi_nosplit_full - auc_logi_nosplit_redu
  logi_loo_diff     <- auc_logi_loo_full     - auc_logi_loo_redu

  return(list(
    lda = list(
      nosplit_diff = lda_nosplit_diff,
      loo_diff     = lda_loo_diff,
      nosplit_full = auc_lda_nosplit_full,
      nosplit_redu = auc_lda_nosplit_redu,
      loo_full     = auc_lda_loo_full,
      loo_redu     = auc_lda_loo_redu,
      delong_nosplit = delong_nosplit_lda,
      delong_loo     = delong_loo_lda
    ),
    logistic = list(
      nosplit_diff = logi_nosplit_diff,
      loo_diff     = logi_loo_diff,
      nosplit_full = auc_logi_nosplit_full,
      nosplit_redu = auc_logi_nosplit_redu,
      loo_full     = auc_logi_loo_full,
      loo_redu     = auc_logi_loo_redu,
      delong_nosplit = delong_nosplit_logi,
      delong_loo     = delong_loo_logi
    ),
    params = list(n1 = n1, n2 = n2, R = R, p = p, mu = mu, sig = sig),
    time = elapsed
  ))
}


#' Print LOO-CV results summary and plots
#'
#' @param loo_result  Output from run_loo_cv_simulation()
print_loo_summary <- function(loo_result) {
  for (m in c("lda", "logistic")) {
    info <- loo_result[[m]]
    cat(sprintf("\n--- %s ---\n", toupper(m)))

    cat("  Non-split:\n")
    cat(sprintf("    mean(full)=%.5f  sd=%.5f  mean(redu)=%.5f  sd=%.5f\n",
                mean(info$nosplit_full), sd(info$nosplit_full),
                mean(info$nosplit_redu), sd(info$nosplit_redu)))
    cat(sprintf("    mean(diff)=%.5f  sd=%.5f  z=%.3f\n",
                mean(info$nosplit_diff), sd(info$nosplit_diff),
                mean(info$nosplit_diff) / sd(info$nosplit_diff)))

    cat("  LOO-CV:\n")
    cat(sprintf("    mean(full)=%.5f  sd=%.5f  mean(redu)=%.5f  sd=%.5f\n",
                mean(info$loo_full), sd(info$loo_full),
                mean(info$loo_redu), sd(info$loo_redu)))
    cat(sprintf("    mean(diff)=%.5f  sd=%.5f  z=%.3f\n",
                mean(info$loo_diff), sd(info$loo_diff),
                mean(info$loo_diff) / sd(info$loo_diff)))

    cat("  DeLong CI (nosplit mean):\n")
    cat(sprintf("    [%.5f, %.5f]\n",
                mean(info$delong_nosplit[, 1]),
                mean(info$delong_nosplit[, 2])))

    cat("  DeLong CI (LOO mean):\n")
    cat(sprintf("    [%.5f, %.5f]\n",
                mean(info$delong_loo[, 1]),
                mean(info$delong_loo[, 2])))
  }
}


#' Plot LOO-CV histograms
#'
#' @param loo_result  Output from run_loo_cv_simulation()
#' @param xlim_range  x-axis limits (default c(-0.15, 0.15))
plot_loo_histograms <- function(loo_result, xlim_range = c(-0.15, 0.15)) {
  par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

  for (m in c("lda", "logistic")) {
    info <- loo_result[[m]]
    hist(info$nosplit_diff, freq = FALSE, breaks = 20,
         main = paste0(toupper(m), " - No Split"),
         xlab = "AUC Difference", xlim = xlim_range)
    hist(info$loo_diff, freq = FALSE, breaks = 20,
         main = paste0(toupper(m), " - LOO-CV"),
         xlab = "AUC Difference", xlim = xlim_range)
  }
}
