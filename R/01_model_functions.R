###############################################################################
# 01_model_functions.R - Unified model fitting and evaluation functions
###############################################################################

#' Fit full and reduced models, compute AUC and DeLong test
#'
#' @param model_type One of "lda", "logistic", "svm", "xgboost"
#' @param df_train   Training data (data.frame with column Y)
#' @param df_test    Test data (data.frame, same column names as df_train)
#' @param y_test     True labels for test set
#' @param reduced_formula Character string for the reduced model formula
#'                        (default "Y ~ X1")
#' @param nrounds    Number of boosting rounds for xgboost (default 100)
#' @param direction  Direction for AUC computation (default "<")
#' @param xgb_redu_cols Column names for xgboost reduced model features
#'                      (default "X1"). Pass a character vector for multiple
#'                      columns.
#' @return Named list: auc_full, auc_redu, auc_diff, delong_lower, delong_upper
fit_and_evaluate_model <- function(model_type, df_train, df_test, y_test,
                                   reduced_formula = "Y ~ X1",
                                   nrounds = 100,
                                   direction = "<",
                                   xgb_redu_cols = "X1") {

  if (model_type == "lda") {
    model_full <- suppressWarnings(lda(Y ~ ., data = df_train))
    model_redu <- lda(as.formula(reduced_formula), data = df_train)
    pred_full <- as.numeric(predict(model_full, newdata = df_test,
                                    type = "response")$x)
    pred_redu <- as.numeric(predict(model_redu, newdata = df_test,
                                    type = "response")$x)

  } else if (model_type == "logistic") {
    model_full <- glm(Y ~ ., data = df_train, family = binomial(link = "logit"))
    model_redu <- glm(as.formula(reduced_formula), data = df_train,
                      family = binomial(link = "logit"))
    pred_full <- predict(model_full, newdata = df_test, type = "response")
    pred_redu <- predict(model_redu, newdata = df_test, type = "response")

  } else if (model_type == "svm") {
    model_full <- svm(formula = as.factor(Y) ~ ., data = df_train,
                      probability = FALSE, decision.values = TRUE)
    model_redu <- svm(formula = as.formula(reduced_formula), data = df_train,
                      probability = FALSE, decision.values = TRUE)
    pred_full <- attr(predict(model_full, df_test,
                              decision.values = TRUE), "decision.values")[, 1]
    pred_redu <- attr(predict(model_redu, df_test,
                              decision.values = TRUE), "decision.values")[, 1]

  } else if (model_type == "xgboost") {
    params <- list(objective = "binary:logistic")
    # Full model
    X_train_full <- as.matrix(df_train[, -which(names(df_train) == "Y")])
    dtrain_full  <- xgb.DMatrix(data = X_train_full, label = df_train$Y)
    model_full   <- xgb.train(params = params, data = dtrain_full,
                              nrounds = nrounds, verbose = 0)
    X_test_full  <- as.matrix(df_test[, -which(names(df_test) == "Y")])
    dtest_full   <- xgb.DMatrix(data = X_test_full, label = df_test$Y)
    pred_full    <- predict(model_full, dtest_full)

    # Reduced model
    X_train_redu <- as.matrix(df_train[xgb_redu_cols])
    dtrain_redu  <- xgb.DMatrix(data = X_train_redu, label = df_train$Y)
    model_redu   <- xgb.train(params = params, data = dtrain_redu,
                              nrounds = nrounds, verbose = 0)
    X_test_redu  <- as.matrix(df_test[xgb_redu_cols])
    dtest_redu   <- xgb.DMatrix(data = X_test_redu, label = df_test$Y)
    pred_redu    <- predict(model_redu, dtest_redu)

  } else {
    stop("Unknown model_type. Choose from: lda, logistic, svm, xgboost")
  }

  # AUC
  auc_full <- auc(y_test, pred_full, direction = direction,
                  levels = c("0", "1"))[1]
  auc_redu <- auc(y_test, pred_redu, direction = direction,
                  levels = c("0", "1"))[1]

  # DeLong test
  roc_full <- roc(y_test, pred_full, direction = direction,
                  levels = c("0", "1"))
  roc_redu <- roc(y_test, pred_redu, direction = direction,
                  levels = c("0", "1"))
  delong_test <- suppressWarnings(
    roc.test(roc_full, roc_redu, method = "delong")
  )

  return(list(
    auc_full     = auc_full,
    auc_redu     = auc_redu,
    auc_diff     = auc_full - auc_redu,
    delong_lower = delong_test$conf.int[1],
    delong_upper = delong_test$conf.int[2]
  ))
}


###############################################################################
# Data generation helper for simulations
###############################################################################

#' Generate simulated data for one Monte Carlo replicate
#'
#' @param n1  Number of class-1 samples
#' @param n2  Number of class-0 samples
#' @param p   Number of noise features
#' @param mu  Mean shift for informative feature X1
#' @param sig Standard deviation for noise features
#' @return data.frame with columns Y, X1, V1..Vp
generate_sim_data <- function(n1, n2, p = 100, mu = 1, sig = 10) {
  X1     <- c(runif(n1, 0 + mu, 2 + mu), runif(n2, 0, 2))
  X_rest <- matrix(rnorm((n1 + n2) * p, mean = 0, sd = sig), ncol = p)
  Y      <- c(rep(1, n1), rep(0, n2))
  df     <- data.frame(cbind(Y, X1, X_rest))
  return(df)
}


###############################################################################
# Leave-one-out helper for GLM
###############################################################################

#' Compute leave-one-out predicted probabilities for a GLM model
#'
#' @param mod  A fitted glm object
#' @param df   The data.frame used to fit the model
#' @return Numeric vector of LOO predicted probabilities
pihat_loo <- function(mod, df) {
  fmla <- formula(mod)
  fam  <- family(mod)
  phat <- numeric(nrow(df))
  for (i in 1:nrow(df)) {
    refit   <- glm(fmla, data = df[-i, ], family = fam)
    phat[i] <- predict(refit, newdata = df[i, ], type = "response")
  }
  return(phat)
}


###############################################################################
# Timing helper
###############################################################################

#' Time a single iteration of a function
#'
#' @param fun  A function to call (no arguments)
#' @return Time in seconds
time_single_iter <- function(fun) {
  t0 <- proc.time()
  fun()
  elapsed <- (proc.time() - t0)["elapsed"]
  return(elapsed)
}
