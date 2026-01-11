# Test script for fit_and_evaluate_model function
library("e1071")
library("MASS")
library("xgboost")
library("pROC")

################################################################################
# Function to fit full and reduced models and evaluate with AUC and DeLong test
################################################################################
fit_and_evaluate_model <- function(model_type, df_train, df_test, y_test,
                                    reduced_formula = "Y ~ X1",
                                    nrounds = 100) {

  # Fit full and reduced models based on model type
  if (model_type == "lda") {
    # LDA - works with numeric Y
    model_full = lda(Y ~ ., data = df_train)
    model_redu = lda(as.formula(reduced_formula), data = df_train)

    pred_full = as.numeric(predict(model_full, newdata = df_test, type = "response")$x)
    pred_redu = as.numeric(predict(model_redu, newdata = df_test, type = "response")$x)

  } else if (model_type == "logistic") {
    # Logistic Regression
    model_full = glm(Y ~ ., data = df_train, family = binomial(link = "logit"))
    model_redu = glm(as.formula(reduced_formula), data = df_train, family = binomial(link = "logit"))

    pred_full = predict(model_full, newdata = df_test, type = "response")
    pred_redu = predict(model_redu, newdata = df_test, type = "response")

  } else if (model_type == "svm") {
    # SVM
    model_full = svm(formula = as.factor(Y) ~ ., data = df_train,
                     probability = FALSE, decision.values = TRUE)
    model_redu = svm(formula = as.factor(Y) ~ X1, data = df_train,
                     probability = FALSE, decision.values = TRUE)

    pred_full = attr(predict(model_full, df_test, decision.values = TRUE), "decision.values")[,1]
    pred_redu = attr(predict(model_redu, df_test, decision.values = TRUE), "decision.values")[,1]

  } else if (model_type == "xgboost") {
    # XGBoost
    params = list(objective = "binary:logistic")

    # Full model
    X_train_full = as.matrix(df_train[, -which(names(df_train) == "Y")])
    dtrain_full = xgb.DMatrix(data = X_train_full, label = df_train$Y)
    model_full = xgb.train(params = params, data = dtrain_full, nrounds = nrounds, verbose = 0)
    X_test_full = as.matrix(df_test[, -which(names(df_test) == "Y")])
    dtest_full = xgb.DMatrix(data = X_test_full, label = df_test$Y)
    pred_full = predict(model_full, dtest_full)

    # Reduced model
    X_train_redu = as.matrix(df_train["X1"])
    dtrain_redu = xgb.DMatrix(data = X_train_redu, label = df_train$Y)
    model_redu = xgb.train(params = params, data = dtrain_redu, nrounds = nrounds, verbose = 0)
    X_test_redu = as.matrix(df_test["X1"])
    dtest_redu = xgb.DMatrix(data = X_test_redu, label = df_test$Y)
    pred_redu = predict(model_redu, dtest_redu)

  } else {
    stop("Unknown model type. Choose from: lda, logistic, svm, xgboost")
  }

  # Compute AUCs
  auc_full = auc(y_test, pred_full, direction = "<", levels = c("0", "1"))[1]
  auc_redu = auc(y_test, pred_redu, direction = "<", levels = c("0", "1"))[1]

  # Compute ROC curves
  roc_full = roc(y_test, pred_full, direction = "<", levels = c("0", "1"))
  roc_redu = roc(y_test, pred_redu, direction = "<", levels = c("0", "1"))

  # DeLong test
  delong_test = roc.test(roc_full, roc_redu, method = "delong")

  # Return results
  return(list(
    auc_full = auc_full,
    auc_redu = auc_redu,
    auc_diff = auc_full - auc_redu,
    delong_lower = delong_test$conf.int[1],
    delong_upper = delong_test$conf.int[2]
  ))
}

################################################################################
# Run 100 simulations and average the results
################################################################################
set.seed(123)
n1 = 1000
n2 = 1000
p = 100  # number of non informative features
mu = 1
sig = 10
n_simulations = 20

# Initialize storage for results
results_lda = list(auc_full = numeric(n_simulations), auc_redu = numeric(n_simulations),
                   auc_diff = numeric(n_simulations), delong_lower = numeric(n_simulations),
                   delong_upper = numeric(n_simulations))
results_logistic = list(auc_full = numeric(n_simulations), auc_redu = numeric(n_simulations),
                        auc_diff = numeric(n_simulations), delong_lower = numeric(n_simulations),
                        delong_upper = numeric(n_simulations))
results_svm = list(auc_full = numeric(n_simulations), auc_redu = numeric(n_simulations),
                   auc_diff = numeric(n_simulations), delong_lower = numeric(n_simulations),
                   delong_upper = numeric(n_simulations))
results_xgboost = list(auc_full = numeric(n_simulations), auc_redu = numeric(n_simulations),
                       auc_diff = numeric(n_simulations), delong_lower = numeric(n_simulations),
                       delong_upper = numeric(n_simulations))

cat("Running", n_simulations, "simulations...\n")
for (i in 1:n_simulations) {
  if (i %% 10 == 0) cat("  Simulation", i, "/", n_simulations, "\n")

  # Generating data same way as nested_AUC_diff.01.04.26.R
  X1_ori = c(runif(n1, 0 + mu, 2 + mu), runif(n2, 0, 2))
  X_rest_ori = matrix(rnorm((n1 + n2) * p, mean = 0, sd = sig), ncol = p)
  Y_ori = c(rep(1, n1), rep(0, n2))
  df_ori = data.frame(cbind(Y_ori, X1_ori, X_rest_ori))

  # Resample with replacement
  id = sample(1:(n1 + n2), replace = TRUE)
  X1 = X1_ori[id]
  X_rest = X_rest_ori[id, ]
  Y = Y_ori[id]
  df = data.frame(cbind(Y, X1, X_rest))

  # Split data
  test.ix = sample(1:(n1 + n2), 500)
  df_test = df[test.ix, ]
  df_train = df[-test.ix, ]
  y_test = df_test[,1]

  # Run all models
  res_lda = fit_and_evaluate_model("lda", df_train, df_test, y_test)
  results_lda$auc_full[i] = res_lda$auc_full
  results_lda$auc_redu[i] = res_lda$auc_redu
  results_lda$auc_diff[i] = res_lda$auc_diff
  results_lda$delong_lower[i] = res_lda$delong_lower
  results_lda$delong_upper[i] = res_lda$delong_upper

  res_logistic = fit_and_evaluate_model("logistic", df_train, df_test, y_test)
  results_logistic$auc_full[i] = res_logistic$auc_full
  results_logistic$auc_redu[i] = res_logistic$auc_redu
  results_logistic$auc_diff[i] = res_logistic$auc_diff
  results_logistic$delong_lower[i] = res_logistic$delong_lower
  results_logistic$delong_upper[i] = res_logistic$delong_upper

  res_svm = fit_and_evaluate_model("svm", df_train, df_test, y_test)
  results_svm$auc_full[i] = res_svm$auc_full
  results_svm$auc_redu[i] = res_svm$auc_redu
  results_svm$auc_diff[i] = res_svm$auc_diff
  results_svm$delong_lower[i] = res_svm$delong_lower
  results_svm$delong_upper[i] = res_svm$delong_upper

  res_xgboost = fit_and_evaluate_model("xgboost", df_train, df_test, y_test, nrounds = 100)
  results_xgboost$auc_full[i] = res_xgboost$auc_full
  results_xgboost$auc_redu[i] = res_xgboost$auc_redu
  results_xgboost$auc_diff[i] = res_xgboost$auc_diff
  results_xgboost$delong_lower[i] = res_xgboost$delong_lower
  results_xgboost$delong_upper[i] = res_xgboost$delong_upper
}

################################################################################
# Calculate averages
################################################################################
cat("\n===== Averaged Results Over", n_simulations, "Simulations =====\n\n")

cat("LDA:\n")
cat("  Mean AUC Full:         ", mean(results_lda$auc_full), "\n")
cat("  Mean AUC Reduced:      ", mean(results_lda$auc_redu), "\n")
cat("  Mean AUC Difference:   ", mean(results_lda$auc_diff), "\n")
cat("  Mean DeLong Lower:     ", mean(results_lda$delong_lower), "\n")
cat("  Mean DeLong Upper:     ", mean(results_lda$delong_upper), "\n")

cat("\nLogistic Regression:\n")
cat("  Mean AUC Full:         ", mean(results_logistic$auc_full), "\n")
cat("  Mean AUC Reduced:      ", mean(results_logistic$auc_redu), "\n")
cat("  Mean AUC Difference:   ", mean(results_logistic$auc_diff), "\n")
cat("  Mean DeLong Lower:     ", mean(results_logistic$delong_lower), "\n")
cat("  Mean DeLong Upper:     ", mean(results_logistic$delong_upper), "\n")

cat("\nSVM:\n")
cat("  Mean AUC Full:         ", mean(results_svm$auc_full), "\n")
cat("  Mean AUC Reduced:      ", mean(results_svm$auc_redu), "\n")
cat("  Mean AUC Difference:   ", mean(results_svm$auc_diff), "\n")
cat("  Mean DeLong Lower:     ", mean(results_svm$delong_lower), "\n")
cat("  Mean DeLong Upper:     ", mean(results_svm$delong_upper), "\n")

cat("\nXGBoost:\n")
cat("  Mean AUC Full:         ", mean(results_xgboost$auc_full), "\n")
cat("  Mean AUC Reduced:      ", mean(results_xgboost$auc_redu), "\n")
cat("  Mean AUC Difference:   ", mean(results_xgboost$auc_diff), "\n")
cat("  Mean DeLong Lower:     ", mean(results_xgboost$delong_lower), "\n")
cat("  Mean DeLong Upper:     ", mean(results_xgboost$delong_upper), "\n")

cat("\n===== Summary Table =====\n")
cat(sprintf("%-20s %12s %12s %12s\n", "Model", "AUC Full", "AUC Reduced", "AUC Diff"))
cat(sprintf("%-20s %12s %12s %12s\n", "-----------------", "----------", "----------", "----------"))
cat(sprintf("%-20s %12.6f %12.6f %12.6f\n", "LDA", mean(results_lda$auc_full), mean(results_lda$auc_redu), mean(results_lda$auc_diff)))
cat(sprintf("%-20s %12.6f %12.6f %12.6f\n", "Logistic", mean(results_logistic$auc_full), mean(results_logistic$auc_redu), mean(results_logistic$auc_diff)))
cat(sprintf("%-20s %12.6f %12.6f %12.6f\n", "SVM", mean(results_svm$auc_full), mean(results_svm$auc_redu), mean(results_svm$auc_diff)))
cat(sprintf("%-20s %12.6f %12.6f %12.6f\n", "XGBoost", mean(results_xgboost$auc_full), mean(results_xgboost$auc_redu), mean(results_xgboost$auc_diff)))

cat("\n===== All tests completed successfully! =====\n")
