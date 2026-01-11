# Delta AUC Nested Model
library("e1071") 
library("MASS")
library("randomForest")
library("xgboost")
library("pROC")
library("mvtnorm")
library("SELR")

################################################################################
n1 = 1000
n2 = 1000
R = 1000
B = 1000

p = 100 # number of non informative features. 
mu = 1
sig = 10
# For null hypothesis

snr = (mu - 0)^2/sig^2

auc.lda.full = matrix(0, R, B)
auc.lda.redu = matrix(0, R, B)
auc.lda.test.full = matrix(0, R, B)
auc.lda.test.redu = matrix(0, R, B)


auc.logi.full = matrix(0, R, B)
auc.logi.redu = matrix(0, R, B)
auc.logi.test.full = matrix(0, R, B)
auc.logi.test.redu = matrix(0, R, B)


auc.svm.full = matrix(0, R, B)
auc.svm.redu = matrix(0, R, B)
auc.svm.test.full = matrix(0, R, B)
auc.svm.test.redu = matrix(0, R, B)


auc.xgb.full = matrix(0, R, B)
auc.xgb.redu = matrix(0, R, B)
auc.xgb.test.full = matrix(0, R, B)
auc.xgb.test.redu = matrix(0, R, B)


delong.lower.nosplit.logi = matrix(0, R, B)
delong.upper.nosplit.logi = matrix(0, R, B)
delong.lower.split.logi = matrix(0, R, B)
delong.upper.split.logi = matrix(0, R, B)

delong.lower.nosplit.lda = matrix(0, R, B)
delong.upper.nosplit.lda = matrix(0, R, B)
delong.lower.split.lda = matrix(0, R, B)
delong.upper.split.lda = matrix(0, R, B)


delong.lower.nosplit.svm = matrix(0, R, B)
delong.upper.nosplit.svm = matrix(0, R, B)
delong.lower.split.svm = matrix(0, R, B)
delong.upper.split.svm = matrix(0, R, B)

delong.lower.nosplit.xgb = matrix(0, R, B)
delong.upper.nosplit.xgb = matrix(0, R, B)
delong.lower.split.xgb = matrix(0, R, B)
delong.upper.split.xgb = matrix(0, R, B)



# Simulation for sampling with replacement time as B and Monte Carlo Repetition as R

for(i in 1:R)
{
  # Generating data
  X1_ori = c(runif(n1, 0 + mu, 2 + mu), runif(n2, 0, 2))
  X_rest_ori = matrix(rnorm((n1 + n2)*(p), mean = 0, sd = sig), ncol = p )
  Y_ori = c(rep(1, n1), rep(0, n2))
  ## For simulation setting 1
  df_ori = data.frame(cbind(Y_ori, X1_ori, X_rest_ori))
  ## For simulation setting 2
  # X2_ori = c(rnorm(n1, -1, 1), rnorm(n2, 1, 1))
  # df_ori = data.frame(cbind(Y_ori, X1_ori, X2_ori, X_rest_ori))

  
  for(j in 1:B)
  {
    # resample with replacement
    id = sample(1:(n1+n2), replace = TRUE)
    
    X1 = X1_ori[id]
    X_rest = X_rest_ori[id,]
    Y = Y_ori[id]
    df = data.frame(cbind(Y, X1, X_rest))
    # X2 = X2_ori[id]
    # df = data.frame(cbind(Y, X1, X2, X_rest))
    Y.true = df[,1]
    
    
    # LDA no split
    lda.full = lda(Y ~ ., data = df)
    Y.pred.full = as.numeric(predict(lda.full, type = "response")$x)
    auc.lda.full[i,j] = auc(Y.true, Y.pred.full, direction = "<", levels = c("0", "1"))[1]
    
    
    lda.redu = lda(Y ~ X1, data = df)
    # Y.pred.redu = as.integer(predict(lda.redu)$class) - 1
    Y.pred.redu = as.numeric(predict(lda.redu, type = "response")$x)
    auc.lda.redu[i,j] = auc(Y.true, Y.pred.redu, direction = "<", levels = c("0", "1"))[1]
    
    
    roc_model1 <- roc(Y.true, Y.pred.full, direction = "<", levels = c("0", "1"))
    roc_model2 <- roc(Y.true, Y.pred.redu, direction = "<", levels = c("0", "1"))
    
    # Perform DeLong test to compare the two ROC curves
    delong_test_lda <- roc.test(roc_model1, roc_model2, method = "delong")
    
    delong.lower.nosplit.lda[i,j] = delong_test_lda$conf.int[1]
    delong.upper.nosplit.lda[i,j] = delong_test_lda$conf.int[2]
    
    
    # Logistic Regression no split
    logi.full = glm(Y ~ . , data = df, family = binomial(link = "logit"))
    # Y.pred.full = as.integer(logi.full$fitted.values > 0.5)
    Y.pred.full = logi.full$fitted.values 
    auc.logi.full[i,j] = auc(Y.true, Y.pred.full, direction = "<", levels = c("0", "1"))[1]
    
    
    logi.redu = glm(Y ~ X1 , data = df, family = binomial(link = "logit"))
    # Y.pred.redu = as.integer(logi.redu$fitted.values > 0.5)
    Y.pred.redu = logi.redu$fitted.values 
    auc.logi.redu[i,j] = auc(Y.true, Y.pred.redu, direction = "<", levels = c("0", "1"))[1]
    
    roc_model3 <- roc(Y.true, Y.pred.full, direction = "<", levels = c("0", "1"))
    roc_model4 <- roc(Y.true, Y.pred.redu, direction = "<", levels = c("0", "1"))
    
    # Perform DeLong test to compare the two ROC curves
    delong_test_logi <- roc.test(roc_model3, roc_model4, method = "delong")
    
    delong.lower.nosplit.logi[i,j] = delong_test_logi$conf.int[1]
    delong.upper.nosplit.logi[i,j] = delong_test_logi$conf.int[2]
    
    
    # SVM no split
    svm.full = svm(formula = as.factor(Y) ~ ., data = df, probability = FALSE, decision.values = TRUE)
    Y.pred.full = attr(predict(svm.full, df, decision.values = TRUE), "decision.values")[,1]
    auc.svm.full[i,j] = auc(Y.true, Y.pred.full, direction="<", levels = c("0", "1"))[1]
    
    
    svm.redu = svm(formula = as.factor(Y) ~ X1, data = df, probability = FALSE, decision.values = TRUE)
    Y.pred.redu = attr(predict(svm.redu, df, decision.values = TRUE), "decision.values")[,1]
    auc.svm.redu[i,j] = auc(Y.true, Y.pred.redu, direction="<", levels = c("0", "1"))[1]
    
    roc_model5 <- roc(Y.true, Y.pred.full, direction = "<", levels = c("0", "1"))
    roc_model6 <- roc(Y.true, Y.pred.redu, direction = "<", levels = c("0", "1"))
    
    
    # Perform DeLong test to compare the two ROC curves
    delong_test_svm <- roc.test(roc_model5, roc_model6, method = "delong")
    
    delong.lower.nosplit.svm[i,j] = delong_test_svm$conf.int[1]
    delong.upper.nosplit.svm[i,j] = delong_test_svm$conf.int[2]
    
  
    
    
  
    # XGBoost for classification, 100 boosting rounds
    params = list(objective = "binary:logistic")
    
    X_full = as.matrix(df[, - which(names(df) == "Y")])
    dtrain.full = xgb.DMatrix(data = X_full, label = Y)
    xgb.full = xgb.train(params = params, data = dtrain.full, nrounds = 100)
    Y.pred.full = predict(xgb.full, dtrain.full)
    auc.xgb.full[i,j] = auc(Y, Y.pred.full, direction = "<", levels = c("0", "1"))[1]
    
  
    X_redu = as.matrix(df["X1"])
    dtrain.redu = xgb.DMatrix(data = X_redu, label = Y)
    xgb.redu = xgb.train(params = params, data = dtrain.redu, nrounds = 100)
    Y.pred.redu = predict(xgb.redu, dtrain.redu)
    auc.xgb.redu[i,j] = auc(Y, Y.pred.redu, direction = "<", levels = c("0", "1"))[1]
  
    roc_model7 <- roc(Y.true, Y.pred.full, direction = "<", levels = c("0", "1"))
    roc_model8 <- roc(Y.true, Y.pred.redu, direction = "<", levels = c("0", "1"))
    
    # Perform DeLong test to compare the two ROC curves
    delong_test_xgb <- roc.test(roc_model7, roc_model8, method = "delong")
    
    delong.lower.nosplit.xgb[i,j] = delong_test_xgb$conf.int[1]
    delong.upper.nosplit.xgb[i,j] = delong_test_xgb$conf.int[2]
    
    
    
    
    
    # Sample splitting
    test.ix = sample(1:(n1 + n2), 500)
    df_test = df[test.ix, ]
    df_train = df[-test.ix, ]
    y_train = df_train[,1]
    y_test = df_test[,1]
    
    # LDA split
    lda.full.train = lda(Y ~ . , data = df_train)
    # Y.pred.full.test = as.integer(predict(lda.full.train, newdata = df_test)$class) - 1
    Y.pred.full.test = as.numeric(predict(lda.full.train, newdata = df_test, type = "response")$x)
    auc.lda.test.full[i,j] = auc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))[1]
    
    lda.redu.train = lda(Y ~ X1, data = df_train)
    # Y.pred.redu.test = as.integer(predict(lda.redu.train, newdata = df_test)$class) - 1
    Y.pred.redu.test = as.numeric(predict(lda.redu.train, newdata = df_test, type = "response")$x)
    auc.lda.test.redu[i,j] = auc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))[1]
    
    roc_model1 <- roc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))
    roc_model2 <- roc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))
    
    # Perform DeLong test to compare the two ROC curves
    delong_test_lda <- roc.test(roc_model1, roc_model2, method = "delong")
    
    delong.lower.split.lda[i,j] = delong_test_lda$conf.int[1]
    delong.upper.split.lda[i,j] = delong_test_lda$conf.int[2]
    
    
    
    # Logistic Regression split
    logi.full.train = glm(Y ~ . , data = df_train, family = binomial(link = "logit"))
    # Y.pred.full.test = as.integer(predict(logi.full.train, newdata = df_test, type = "response") > 0.5)
    Y.pred.full.test = predict(logi.full.train, newdata = df_test, type = "response")
    auc.logi.test.full[i,j] = auc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))[1]
    
    logi.redu.train = glm(Y ~ X1, data = df_train, family = binomial(link = "logit"))
    # Y.pred.redu.test = as.integer(predict(logi.redu.train, newdata = df_test, type = "response") > 0.5)
    Y.pred.redu.test = predict(logi.redu.train, newdata = df_test, type = "response")
    auc.logi.test.redu[i,j] = auc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))[1]
    
    
    roc_model3 <- roc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))
    roc_model4 <- roc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))
    
    # Perform DeLong test to compare the two ROC curves
    delong_test_logi <- roc.test(roc_model1, roc_model2, method = "delong")
    
    delong.lower.split.logi[i,j] = delong_test_logi$conf.int[1]
    delong.upper.split.logi[i,j] = delong_test_logi$conf.int[2]
    
    
    
    # SVM split
    svm.full.train = svm(formula = as.factor(Y) ~ ., data = df_train, probability = FALSE, decision.values = TRUE)
    Y.pred.full.test = attr(predict(svm.full.train, df_test, decision.values = TRUE), "decision.values")[,1]
    auc.svm.test.full[i,j] = auc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))[1]
    
    
    svm.redu.train = svm(formula = as.factor(Y) ~ X1, data = df_train, probability = FALSE, decision.values = TRUE)
    Y.pred.redu.test = attr(predict(svm.redu.train, df_test, decision.values = TRUE), "decision.values")[,1]
    auc.svm.test.redu[i,j] = auc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))[1]
    
  
    
    roc_model5 <- roc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))
    roc_model6 <- roc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))
    
    # Perform DeLong test to compare the two ROC curves
    delong_test_svm <- roc.test(roc_model5, roc_model6, method = "delong")
    
    delong.lower.split.svm[i,j] = delong_test_svm$conf.int[1]
    delong.upper.split.svm[i,j] = delong_test_svm$conf.int[2]
    
    
    
  
    # XGBoosting for classification, 100 boosting rounds, split
    params = list(objective = "binary:logistic")
  
    X_train_full = as.matrix(df_train[, - which(names(df_train) == "Y")])
    dtrain.full = xgb.DMatrix(data = X_train_full, label = df_train$Y)
    xgb.full.train = xgb.train(params = params, data = dtrain.full, nrounds = 100)
    X_test_full = as.matrix(df_test[, - which(names(df_test) == "Y")])
    dtest.full = xgb.DMatrix(data = X_test_full, label = df_test$Y)
    Y.pred.full.test = predict(xgb.full.train, dtest.full)
    auc.xgb.test.full[i,j] = auc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))[1]
  
  
  
    X_train_redu = as.matrix(df_train["X1"])
    dtrain.redu = xgb.DMatrix(data = X_train_redu, label = df_train$Y)
    xgb.redu.train = xgb.train(params = params, data = dtrain.redu, nrounds = 100)
    X_test_redu = as.matrix(df_test["X1"])
    dtest.redu = xgb.DMatrix(data = X_test_redu, label = df_test$Y)
    Y.pred.redu.test = predict(xgb.redu.train, dtest.redu)
    auc.xgb.test.redu[i,j] = auc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))[1]
  
    
    roc_model7 <- roc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))
    roc_model8 <- roc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))
    
    # Perform DeLong test to compare the two ROC curves
    delong_test_xgb <- roc.test(roc_model7, roc_model8, method = "delong")
    
    delong.lower.split.xgb[i,j] = delong_test_xgb$conf.int[1]
    delong.upper.split.xgb[i,j] = delong_test_xgb$conf.int[2]
  }
}



auc.lda.nested.diff = auc.lda.full - auc.lda.redu

auc.lda.nested.test.diff = auc.lda.test.full - auc.lda.test.redu

auc.logi.nested.diff = auc.logi.full - auc.logi.redu

auc.logi.nested.test.diff = auc.logi.test.full - auc.logi.test.redu

auc.svm.nested.diff = auc.svm.full - auc.svm.redu

auc.svm.nested.test.diff = auc.svm.test.full - auc.svm.test.redu

auc.xgb.nested.diff = auc.xgb.full - auc.xgb.redu

auc.xgb.nested.test.diff = auc.xgb.test.full - auc.xgb.test.redu


par(mfrow = c(2,2))


hist(auc.lda.nested.diff, freq = FALSE, breaks = 10, main = "No Split", xlab = "LDA AUC Difference", xlim = c(-0.15, 0.15))

hist(auc.lda.nested.test.diff, freq = FALSE, breaks = 20, main = "Split", xlab = "LDA AUC Difference", xlim =c(-0.15, 0.15))


data.frame(mean(auc.lda.full), sd(auc.lda.full),
           mean(auc.lda.redu), sd(auc.lda.redu))
data.frame(mean(auc.lda.nested.diff), sd(auc.lda.nested.diff), 
           z=mean(auc.lda.nested.diff)/sd(auc.lda.nested.diff))

data.frame(mean(auc.lda.test.full), sd(auc.lda.test.full),
           mean(auc.lda.test.redu), sd(auc.lda.test.redu))
data.frame(mean(auc.lda.nested.test.diff), sd(auc.lda.nested.test.diff),
           z=mean(auc.lda.nested.test.diff)/sd(auc.lda.nested.test.diff))






hist(auc.logi.nested.diff, freq = FALSE, breaks = 10, main = "No Split", xlab = "Logistic AUC Difference", xlim =c(-0.15, 0.15))

hist(auc.logi.nested.test.diff, freq = FALSE, breaks = 20, main = "Split", xlab = "Logistic AUC Difference", xlim =c(-0.15, 0.15))



data.frame(mean(auc.logi.full), sd(auc.logi.full),
           mean(auc.logi.redu), sd(auc.logi.redu))
data.frame(mean(auc.logi.nested.diff), sd(auc.logi.nested.diff), 
           z=mean(auc.logi.nested.diff)/sd(auc.logi.nested.diff))

data.frame(mean(auc.logi.test.full), sd(auc.logi.test.full),
           mean(auc.logi.test.redu), sd(auc.logi.test.redu))
data.frame(mean(auc.logi.nested.test.diff), sd(auc.logi.nested.test.diff),
           z=mean(auc.logi.nested.test.diff)/sd(auc.logi.nested.test.diff))





hist(auc.svm.nested.diff, freq = FALSE, breaks = 10, main = "No Split", xlab = "SVM AUC Difference", xlim = c(-0.3, 0.3))

hist(auc.svm.nested.test.diff, freq = FALSE, breaks = 10, main = "Split", xlab = "SVM AUC Difference", xlim =c(-0.3, 0.3))


data.frame(mean(auc.svm.full), sd(auc.svm.full),
           mean(auc.svm.redu), sd(auc.svm.redu))
data.frame(mean(auc.svm.nested.diff), sd(auc.svm.nested.diff),
           z=mean(auc.svm.nested.diff)/sd(auc.svm.nested.diff))

data.frame(mean(auc.svm.test.full), sd(auc.svm.test.full),
           mean(auc.svm.test.redu), sd(auc.svm.test.redu))
data.frame(mean(auc.svm.nested.test.diff), sd(auc.svm.nested.test.diff),
           z=mean(auc.svm.nested.test.diff)/sd(auc.svm.nested.test.diff))



hist(auc.xgb.nested.diff, freq = FALSE, breaks = 10, main = "No Split", xlab = "XGBoost AUC Difference", xlim = c(-0.3, 0.3))

hist(auc.xgb.nested.test.diff, freq = FALSE, breaks = 10, main = "Split", xlab = "XGBoost AUC Difference", xlim =c(-0.3, 0.3))


data.frame(mean(auc.xgb.full), sd(auc.xgb.full),
           mean(auc.xgb.redu), sd(auc.xgb.redu))
data.frame(mean(auc.xgb.nested.diff), sd(auc.xgb.nested.diff),
           z = mean(auc.xgb.nested.diff)/sd(auc.xgb.nested.diff))

data.frame(mean(auc.xgb.test.full), sd(auc.xgb.test.full),
           mean(auc.xgb.test.redu), sd(auc.xgb.test.redu))
data.frame(mean(auc.xgb.nested.test.diff), sd(auc.xgb.nested.test.diff),
           z = mean(auc.xgb.nested.test.diff)/sd(auc.xgb.nested.test.diff))







#########################################################################################################
# Simulate ground truth value
n1 = 1000
n2 = 1000
R = 100000







# Simulation for bootstrap as 1 and Monte Carlo Repetition as R
groud_truth <- function(n1, n2, R){
  p = 100 # number of non informative features. 
  mu = 1
  sig = 10
  # For null hypothesis
  
  snr = (mu - 0)^2/sig^2
  
  auc.lda.full = rep(0, R)
  auc.lda.redu = rep(0, R)
  auc.lda.test.full = rep(0, R)
  auc.lda.test.redu = rep(0, R)
  
  
  auc.logi.full = rep(0, R)
  auc.logi.redu = rep(0, R)
  auc.logi.test.full = rep(0, R)
  auc.logi.test.redu = rep(0, R)
  
  
  auc.svm.full = rep(0, R)
  auc.svm.redu = rep(0, R)
  auc.svm.test.full = rep(0, R)
  auc.svm.test.redu = rep(0, R)
  
  
  auc.xgb.full = rep(0, R)
  auc.xgb.redu = rep(0, R)
  auc.xgb.test.full = rep(0, R)
  auc.xgb.test.redu = rep(0, R)
  
  for(i in 1:R)
  {
    # Generating data
    X1 = c(runif(n1, 0 + mu, 2 + mu), runif(n2, 0, 2))
    X_rest = matrix(rnorm((n1 + n2)*(p), mean = 0, sd = sig), ncol = p )
    Y = c(rep(1, n1), rep(0, n2))
    ## For simulation setting 1
    df = data.frame(cbind(Y, X1, X_rest))
    ## For simulation setting 2
    # X2 = c(rnorm(n1, -1, 1), rnorm(n2, 1, 1))
    # df = data.frame(cbind(Y, X1, X2, X_rest))
    Y.true = df[,1]
    
    # LDA no split
    lda.full = lda(Y ~ ., data = df)
    Y.pred.full = as.numeric(predict(lda.full, type = "response")$x)
    auc.lda.full[i] = auc(Y.true, Y.pred.full, direction = "<", levels = c("0", "1"))[1]
    
    
    lda.redu = lda(Y ~ X1, data = df)
    # Y.pred.redu = as.integer(predict(lda.redu)$class) - 1
    Y.pred.redu = as.numeric(predict(lda.redu, type = "response")$x)
    auc.lda.redu[i] = auc(Y.true, Y.pred.redu, direction = "<", levels = c("0", "1"))[1]
    
    
    
    
    # Logistic Regression no split
    logi.full = glm(Y ~ . , data = df, family = binomial(link = "logit"))
    # Y.pred.full = as.integer(logi.full$fitted.values > 0.5)
    Y.pred.full = logi.full$fitted.values 
    auc.logi.full[i] = auc(Y.true, Y.pred.full, direction = "<", levels = c("0", "1"))[1]
    
    
    logi.redu = glm(Y ~ X1 , data = df, family = binomial(link = "logit"))
    # Y.pred.redu = as.integer(logi.redu$fitted.values > 0.5)
    Y.pred.redu = logi.redu$fitted.values 
    auc.logi.redu[i] = auc(Y.true, Y.pred.redu, direction = "<", levels = c("0", "1"))[1]
    
  
    
    
    # SVM no split
    svm.full = svm(formula = as.factor(Y) ~ ., data = df, probability = FALSE, decision.values = TRUE)
    Y.pred.full = attr(predict(svm.full, df, decision.values = TRUE), "decision.values")[,1]
    auc.svm.full[i] = auc(Y.true, Y.pred.full, direction="<", levels = c("0", "1"))[1]
    
    
    svm.redu = svm(formula = as.factor(Y) ~ X1, data = df, probability = FALSE, decision.values = TRUE)
    Y.pred.redu = attr(predict(svm.redu, df, decision.values = TRUE), "decision.values")[,1]
    auc.svm.redu[i] = auc(Y.true, Y.pred.redu, direction="<", levels = c("0", "1"))[1]
    
  
    
    # XGBoost for classification, 100 boosting rounds
    params = list(objective = "binary:logistic")
    
    X_full = as.matrix(df[, - which(names(df) == "Y")])
    dtrain.full = xgb.DMatrix(data = X_full, label = Y)
    xgb.full = xgb.train(params = params, data = dtrain.full, nrounds = 100)
    Y.pred.full = predict(xgb.full, dtrain.full)
    auc.xgb.full[i] = auc(Y, Y.pred.full, direction = "<", levels = c("0", "1"))[1]
    
    
    X_redu = as.matrix(df["X1"])
    dtrain.redu = xgb.DMatrix(data = X_redu, label = Y)
    xgb.redu = xgb.train(params = params, data = dtrain.redu, nrounds = 100)
    Y.pred.redu = predict(xgb.redu, dtrain.redu)
    auc.xgb.redu[i] = auc(Y, Y.pred.redu, direction = "<", levels = c("0", "1"))[1]
    
  
    
    
    
    # Sample splitting, generate new testing sample
    df_train = df
    y_train = df_train[,1]
  
    # Generating testing data
    X1_test = c(runif(n1, 0 + mu, 2 + mu), runif(n2, 0, 2))
    X_rest_test = matrix(rnorm((n1 + n2)*(p), mean = 0, sd = sig), ncol = p )
    y_test = c(rep(1, n1), rep(0, n2))
    ## For simulation setting 1
    df_test = data.frame(cbind(y_test, X1_test, X_rest_test))
    ## For simulation setting 2
    # X2_test = c(rnorm(n1, -1, 1), rnorm(n2, 1, 1))
    # df_test = data.frame(cbind(y_test, X1_test, X2_test, X_rest_test))
  
    
    
    # LDA split
    lda.full.train = lda(Y ~ . , data = df_train)
    # Y.pred.full.test = as.integer(predict(lda.full.train, newdata = df_test)$class) - 1
    Y.pred.full.test = as.numeric(predict(lda.full.train, newdata = df_test, type = "response")$x)
    auc.lda.test.full[i] = auc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))[1]
    
    lda.redu.train = lda(Y ~ X1, data = df_train)
    # Y.pred.redu.test = as.integer(predict(lda.redu.train, newdata = df_test)$class) - 1
    Y.pred.redu.test = as.numeric(predict(lda.redu.train, newdata = df_test, type = "response")$x)
    auc.lda.test.redu[i] = auc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))[1]
    
  
    
    
    # Logistic Regression split
    logi.full.train = glm(Y ~ . , data = df_train, family = binomial(link = "logit"))
    # Y.pred.full.test = as.integer(predict(logi.full.train, newdata = df_test, type = "response") > 0.5)
    Y.pred.full.test = predict(logi.full.train, newdata = df_test, type = "response")
    auc.logi.test.full[i] = auc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))[1]
    
    logi.redu.train = glm(Y ~ X1, data = df_train, family = binomial(link = "logit"))
    # Y.pred.redu.test = as.integer(predict(logi.redu.train, newdata = df_test, type = "response") > 0.5)
    Y.pred.redu.test = predict(logi.redu.train, newdata = df_test, type = "response")
    auc.logi.test.redu[i] = auc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))[1]
    
    
    roc_model3 <- roc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))
    roc_model4 <- roc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))
  
    
    # SVM split
    svm.full.train = svm(formula = as.factor(Y) ~ ., data = df_train, probability = FALSE, decision.values = TRUE)
    Y.pred.full.test = attr(predict(svm.full.train, df_test, decision.values = TRUE), "decision.values")[,1]
    auc.svm.test.full[i] = auc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))[1]
    
    
    svm.redu.train = svm(formula = as.factor(Y) ~ X1, data = df_train, probability = FALSE, decision.values = TRUE)
    Y.pred.redu.test = attr(predict(svm.redu.train, df_test, decision.values = TRUE), "decision.values")[,1]
    auc.svm.test.redu[i] = auc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))[1]
    
    
    
    
    
    # XGBoosting for classification, 100 boosting rounds, split
    params = list(objective = "binary:logistic")
    
    X_train_full = as.matrix(df_train[, - which(names(df_train) == "Y")])
    dtrain.full = xgb.DMatrix(data = X_train_full, label = df_train$Y)
    xgb.full.train = xgb.train(params = params, data = dtrain.full, nrounds = 100)
    X_test_full = as.matrix(df_test[, - which(names(df_test) == "Y")])
    dtest.full = xgb.DMatrix(data = X_test_full, label = df_test$Y)
    Y.pred.full.test = predict(xgb.full.train, dtest.full)
    auc.xgb.test.full[i] = auc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))[1]
    
    
    
    X_train_redu = as.matrix(df_train["X1"])
    dtrain.redu = xgb.DMatrix(data = X_train_redu, label = df_train$Y)
    xgb.redu.train = xgb.train(params = params, data = dtrain.redu, nrounds = 100)
    X_test_redu = as.matrix(df_test["X1"])
    dtest.redu = xgb.DMatrix(data = X_test_redu, label = df_test$Y)
    Y.pred.redu.test = predict(xgb.redu.train, dtest.redu)
    auc.xgb.test.redu[i] = auc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))[1]
    
    
  }
  
  auc.logi.nested.test.diff.true  = mean(auc.logi.test.full) - mean(auc.logi.test.redu)
  auc.lda.nested.test.diff.true  = mean(auc.lda.test.full) - mean(auc.lda.test.redu)
  auc.svm.nested.test.diff.true  = mean(auc.svm.test.full) - mean(auc.svm.test.redu)
  auc.xgb.nested.test.diff.true = mean(auc.xgb.test.full) - mean(auc.xgb.test.redu)
  
  return(list(auc.logi.nested.test.diff.true, auc.lda.nested.test.diff.true, auc.svm.nested.test.diff.true, auc.xgb.nested.test.diff.true))
}


ground.truth.1 = groud_truth(n1 = 1000, n2 = 1000, R = 10000)
names(ground.truth.1 ) <- c("auc.logi.nested.test.diff.true.1", "auc.lda.nested.test.diff.true.1", "auc.svm.nested.test.diff.true.1", "auc.xgb.nested.test.diff.true.1")

ground.truth.2 = groud_truth(n1 = 750, n2 = 750, R = 10000) 
names(ground.truth.2) <- c("auc.logi.nested.test.diff.true.2", "auc.lda.nested.test.diff.true.2", "auc.svm.nested.test.diff.true.2", "auc.xgb.nested.test.diff.true.2")















# Inference

lda.ci = t(apply(auc.lda.nested.diff, 1, quantile, probs = c(0.05, 0.95)))

lda.test.ci = t(apply(auc.lda.nested.test.diff, 1, quantile, probs = c(0.05, 0.95)))


logi.ci = t(apply(auc.logi.nested.diff, 1, quantile, probs = c(0.05, 0.95)))

logi.test.ci = t(apply(auc.logi.nested.test.diff, 1, quantile, probs = c(0.05, 0.95)))


svm.ci = t(apply(auc.svm.nested.diff, 1, quantile, probs = c(0.05, 0.95)))

svm.test.ci = t(apply(auc.svm.nested.test.diff, 1, quantile, probs = c(0.05, 0.95)))


xgb.ci = t(apply(auc.xgb.nested.diff, 1, quantile, probs = c(0.05, 0.95)))

xgb.test.ci = t(apply(auc.xgb.nested.test.diff, 1, quantile, probs = c(0.05, 0.95)))


# Average mean CI
colMeans(lda.ci)
colMeans(lda.test.ci)
colMeans(logi.ci)
colMeans(logi.test.ci)
colMeans(svm.ci)
colMeans(svm.test.ci)
colMeans(xgb.ci)
colMeans(xgb.test.ci)         

# Coverage rate
cover.rate.nosplit.logi = sum((logi.ci[,2] >= auc.logi.nested.test.diff.true.1) & (logi.ci[,1] <= auc.logi.nested.test.diff.true.1))/R
cover.rate.split.logi = sum((logi.test.ci[,2] >= auc.logi.nested.test.diff.true.2) & (logi.test.ci[,1] <= auc.logi.nested.test.diff.true.2))/R
cover.rate.nosplit.logi 
cover.rate.split.logi

cover.rate.nosplit.lda = sum((lda.ci[,2] >= auc.lda.nested.test.diff.true.1) & (lda.ci[,1] <= auc.lda.nested.test.diff.true.1))/R
cover.rate.split.lda = sum((lda.test.ci[,2] >= auc.lda.nested.test.diff.true.2) & (lda.test.ci[,1] <= auc.lda.nested.test.diff.true.2))/R
cover.rate.nosplit.lda 
cover.rate.split.lda


cover.rate.nosplit.svm = sum((svm.ci[,2] >= auc.svm.nested.test.diff.true.1) & (svm.ci[,1] <= auc.svm.nested.test.diff.true.1))/R
cover.rate.split.svm = sum((svm.test.ci[,2] >= auc.svm.nested.test.diff.true.2) & (svm.test.ci[,1] <= auc.svm.nested.test.diff.true.2))/R
cover.rate.nosplit.svm 
cover.rate.split.svm

cover.rate.nosplit.xgb = sum((xgb.ci[,2] >= auc.xgb.nested.test.diff.true.1) & (xgb.ci[,1] <= auc.xgb.nested.test.diff.true.1))/R
cover.rate.split.xgb = sum((xgb.test.ci[,2] >= auc.xgb.nested.test.diff.true.2) & (xgb.test.ci[,1] <= auc.xgb.nested.test.diff.true.2))/R
cover.rate.nosplit.xgb 
cover.rate.split.xgb




# Delong CI and coverage

ci.mean.nosplit.logi = c(mean(delong.lower.nosplit.logi), mean(delong.upper.nosplit.logi))
ci.mean.split.logi = c(mean(delong.lower.split.logi), mean(delong.upper.split.logi))
ci.mean.nosplit.logi
ci.mean.split.logi

cover.rate.nosplit.logi = sum((delong.upper.nosplit.logi >= auc.logi.nested.test.diff.true.1) & (delong.lower.nosplit.logi <= auc.logi.nested.test.diff.true.1))/R
cover.rate.split.logi = sum((delong.upper.split.logi >= auc.logi.nested.test.diff.true.2) & (delong.lower.split.logi <= auc.logi.nested.test.diff.true.2))/R
cover.rate.nosplit.logi 
cover.rate.split.logi



ci.mean.nosplit.lda = c(mean(delong.lower.nosplit.lda), mean(delong.upper.nosplit.lda))
ci.mean.split.lda = c(mean(delong.lower.split.lda), mean(delong.upper.split.lda))
ci.mean.nosplit.lda 
ci.mean.split.lda


cover.rate.nosplit.lda = sum((delong.upper.nosplit.lda >= auc.lda.nested.test.diff.true.1) & (delong.lower.nosplit.lda <= auc.lda.nested.test.diff.true.1))/R
cover.rate.split.lda = sum((delong.upper.split.lda >= auc.lda.nested.test.diff.true.2) & (delong.lower.split.lda <= auc.lda.nested.test.diff.true.2))/R
cover.rate.nosplit.lda  
cover.rate.split.lda


ci.mean.nosplit.svm = c(mean(delong.lower.nosplit.svm), mean(delong.upper.nosplit.svm))
ci.mean.split.svm = c(mean(delong.lower.split.svm), mean(delong.upper.split.svm))
ci.mean.nosplit.svm
ci.mean.split.svm

cover.rate.nosplit.svm = sum((delong.upper.nosplit.svm >= auc.svm.nested.test.diff.true.1) & (delong.lower.nosplit.svm <= auc.svm.nested.test.diff.true.1))/R
cover.rate.split.svm = sum((delong.upper.split.svm >= auc.svm.nested.test.diff.true.2) & (delong.lower.split.svm <= auc.svm.nested.test.diff.true.2))/R
cover.rate.nosplit.svm  
cover.rate.split.svm



ci.mean.nosplit.xgb = c(mean(delong.lower.nosplit.xgb), mean(delong.upper.nosplit.xgb))
ci.mean.split.xgb = c(mean(delong.lower.split.xgb), mean(delong.upper.split.xgb))
ci.mean.nosplit.xgb 
ci.mean.split.xgb


cover.rate.nosplit.xgb = sum((delong.upper.nosplit.xgb >= auc.xgb.nested.test.diff.true.1) & (delong.lower.nosplit.xgb <= auc.xgb.nested.test.diff.true.1))/R
cover.rate.split.xgb = sum((delong.upper.split.xgb >= auc.xgb.nested.test.diff.true.2) & (delong.lower.split.xgb <= auc.xgb.nested.test.diff.true.2))/R
cover.rate.nosplit.xgb 
cover.rate.split.xgb




################################################################################
# Real Data Application
R = 1000

ptb_update = ptb[ptb$outcome != "mPTB", ]
n = nrow(ptb_update)
prior_ptb = as.integer(ptb_update$prior_PTB_count > 0)

priorPTB = as.integer(ptb_update$prior_PTB_count > 0)
prior_ptb_features = within(ptb_update, rm(outcome))


# Second Visit full vs All interactions 

df_update_features = cbind(prior_ptb_features, priorPTB, 
                           ptb_update$ACV1 * ptb_update$GAV1, ptb_update$ACV1 * ptb_update$GAV2, 
                           ptb_update$ACV2 * ptb_update$GAV1, ptb_update$ACV2 * ptb_update$GAV2, 
                           ptb_update$CLV1 * ptb_update$GAV1, ptb_update$CLV1 * ptb_update$GAV2, 
                           ptb_update$CLV2 * ptb_update$GAV1, ptb_update$CLV2 * ptb_update$GAV2, 
                           ptb_update$InterceptV1 * ptb_update$GAV1, ptb_update$InterceptV1 * ptb_update$GAV2, 
                           ptb_update$InterceptV2 * ptb_update$GAV1, ptb_update$InterceptV2 * ptb_update$GAV2, 
                           ptb_update$MidbandV1 * ptb_update$GAV1, ptb_update$MidbandV1 * ptb_update$GAV2, 
                           ptb_update$MidbandV2 * ptb_update$GAV1, ptb_update$MidbandV2 * ptb_update$GAV2, 
                           ptb_update$SlopeV1 * ptb_update$GAV1, ptb_update$SlopeV1 * ptb_update$GAV2, 
                           ptb_update$SlopeV2 * ptb_update$GAV1, ptb_update$SlopeV2 * ptb_update$GAV2, 
                           
                           ptb_update$ACV1 * priorPTB, ptb_update$ACV2 * priorPTB, 
                           ptb_update$CLV1 * priorPTB, ptb_update$CLV2 * priorPTB, 
                           ptb_update$InterceptV1 * priorPTB, ptb_update$InterceptV2 * priorPTB, 
                           ptb_update$MidbandV1 * priorPTB, ptb_update$MidbandV2 * priorPTB, 
                           ptb_update$SlopeV1 * priorPTB, ptb_update$SlopeV2 * priorPTB, 
                           
                           ptb_update$ACV1 * (1 - priorPTB), ptb_update$ACV2 * (1 - priorPTB), 
                           ptb_update$CLV1 * (1 - priorPTB), ptb_update$CLV2 * (1 - priorPTB), 
                           ptb_update$InterceptV1 * (1 - priorPTB), ptb_update$InterceptV2 * (1 - priorPTB), 
                           ptb_update$MidbandV1 * (1 - priorPTB), ptb_update$MidbandV2 * (1 - priorPTB), 
                           ptb_update$SlopeV1 * (1 - priorPTB), ptb_update$SlopeV2 * (1 - priorPTB))


# Data 
Y = as.integer(factor(ptb_update$outcome)) - 1
# Scale Data
df_update_features = as.data.frame(scale(df_update_features))
df = data.frame(cbind(Y, df_update_features))




auc.lda.full = rep(0, R)
auc.lda.redu = rep(0, R)
auc.lda.test.full = rep(0, R)
auc.lda.test.redu = rep(0, R)


auc.logi.full = rep(0, R)
auc.logi.redu = rep(0, R)
auc.logi.test.full = rep(0, R)
auc.logi.test.redu = rep(0, R)


auc.svm.full = rep(0, R)
auc.svm.redu = rep(0, R)
auc.svm.test.full = rep(0, R)
auc.svm.test.redu = rep(0, R)


auc.xgb.full = rep(0, R)
auc.xgb.redu = rep(0, R)
auc.xgb.test.full = rep(0, R)
auc.xgb.test.redu = rep(0, R)



delong.lower.nosplit.lda = rep(0, R)
delong.upper.nosplit.lda = rep(0, R)
delong.lower.split.lda = rep(0, R)
delong.upper.split.lda = rep(0, R)


delong.lower.nosplit.logi = rep(0, R)
delong.upper.nosplit.logi = rep(0, R)
delong.lower.split.logi = rep(0, R)
delong.upper.split.logi = rep(0, R)

delong.lower.nosplit.svm = rep(0, R)
delong.upper.nosplit.svm = rep(0, R)
delong.lower.split.svm = rep(0, R)
delong.upper.split.svm = rep(0, R)

delong.lower.nosplit.xgb = rep(0, R)
delong.upper.nosplit.xgb = rep(0, R)
delong.lower.split.xgb = rep(0, R)
delong.upper.split.xgb = rep(0, R)


for(i in 1:R)
{
  # Logistic
  logi.full = glm(Y ~ . , data = df, family = binomial(link = "logit"))
  Y.true = df[,1]
  Y.logi.pred.full = logi.full$fitted.values 
  auc.logi.full[i] = auc(Y.true, Y.logi.pred.full, direction = "<", levels = c("0", "1"))[1]
  
  
  logi.redu = glm(Y ~ prior_PTB_count + CLV2 + SlopeV2 + SlopeV2 * GAV2, data = df, family = binomial(link = "logit"))
  Y.true = df[,1]
  Y.logi.pred.redu = logi.redu$fitted.values 
  auc.logi.redu[i] = auc(Y.true, Y.logi.pred.redu, direction = "<", levels = c("0", "1"))[1]
  
  roc_model1 <- roc(Y.true, Y.logi.pred.full, direction = "<", levels = c("0", "1"))
  roc_model2 <- roc(Y.true, Y.logi.pred.redu, direction = "<", levels = c("0", "1"))
  
  # Perform DeLong test to compare the two ROC curves
  delong_test_logi <- roc.test(roc_model1, roc_model2, method = "delong")
  
  delong.lower.nosplit.logi[i] = delong_test_logi$conf.int[1]
  delong.upper.nosplit.logi[i] = delong_test_logi$conf.int[2]
  
  
  
  # LDA
  lda.full = suppressWarnings(lda(Y ~ ., data = df))
  Y.true = df[,1]
  Y.pred.full = as.numeric(predict(lda.full, type = "response")$x)
  auc.lda.full[i] = auc(Y.true, Y.pred.full, direction = "<", levels = c("0", "1"))[1]
  
  
  lda.redu = lda(Y ~ prior_PTB_count + CLV2 + SlopeV2 + SlopeV2 * GAV2, data = df)
  Y.true = df[,1]
  Y.pred.redu = as.numeric(predict(lda.redu, type = "response")$x)
  auc.lda.redu[i] = auc(Y.true, Y.pred.redu, direction = "<", levels = c("0", "1"))[1]
  
  
  roc_model3 <- roc(Y.true, Y.pred.full, direction = "<", levels = c("0", "1"))
  roc_model4 <- roc(Y.true, Y.pred.redu, direction = "<", levels = c("0", "1"))
  
  # Perform DeLong test to compare the two ROC curves
  delong_test_lda <- roc.test(roc_model3, roc_model4, method = "delong")
  
  delong.lower.nosplit.lda[i] = delong_test_lda$conf.int[1]
  delong.upper.nosplit.lda[i] = delong_test_lda$conf.int[2]


  
  
  # SVM no split
  svm.full = svm(formula = as.factor(Y) ~ ., data = df, probability = FALSE, decision.values = TRUE)
  Y.true = df[,1]
  Y.pred.full = attr(predict(svm.full, df, decision.values = TRUE), "decision.values")[,1]
  auc.svm.full[i] = auc(Y.true, Y.pred.full, direction=">", levels = c("0", "1"))[1]
  
  
  svm.redu = svm(formula = as.factor(Y) ~ prior_PTB_count + CLV2 + SlopeV2 + SlopeV2 * GAV2, data = df, probability = FALSE, decision.values = TRUE)
  Y.true = df[,1]
  Y.pred.redu = attr(predict(svm.redu, df, decision.values = TRUE), "decision.values")[,1]
  auc.svm.redu[i] = auc(Y.true, Y.pred.redu, direction=">", levels = c("0", "1"))[1]
  
  roc_model5 <- roc(Y.true, Y.pred.full, direction = ">", levels = c("0", "1"))
  roc_model6 <- roc(Y.true, Y.pred.redu, direction = ">", levels = c("0", "1"))
  
  # Perform DeLong test to compare the two ROC curves
  delong_test_svm <- roc.test(roc_model5, roc_model6, method = "delong")
  
  delong.lower.nosplit.svm[i] = delong_test_svm$conf.int[1]
  delong.upper.nosplit.svm[i] = delong_test_svm$conf.int[2]
  
  
  
  
  # XGBoost for classification, 100 boosting rounds
  params = list(objective = "binary:logistic")
  
  X_full = as.matrix(df[, - which(names(df) == "Y")])
  dtrain.full = xgb.DMatrix(data = X_full, label = Y)
  xgb.full = xgb.train(params = params, data = dtrain.full, nrounds = 100)
  Y.pred.full = predict(xgb.full, dtrain.full)
  auc.xgb.full[i] = auc(Y, Y.pred.full, direction = "<", levels = c("0", "1"))[1]
  
  
  X_redu = as.matrix(df[c("prior_PTB_count", "CLV2", "SlopeV2", "ptb_update.SlopeV1...ptb_update.GAV1")])
  dtrain.redu = xgb.DMatrix(data = X_redu, label = Y)
  xgb.redu = xgb.train(params = params, data = dtrain.redu, nrounds = 100)
  Y.pred.redu = predict(xgb.redu, dtrain.redu)
  auc.xgb.redu[i] = auc(Y, Y.pred.redu, direction = "<", levels = c("0", "1"))[1]
  
  roc_model7 <- roc(Y.true, Y.pred.full, direction = "<", levels = c("0", "1"))
  roc_model8 <- roc(Y.true, Y.pred.redu, direction = "<", levels = c("0", "1"))
  
  # Perform DeLong test to compare the two ROC curves
  delong_test_xgb <- suppressWarnings(roc.test(roc_model7, roc_model8, method = "delong"))
  
  delong.lower.nosplit.xgb[i] = delong_test_xgb$conf.int[1]
  delong.upper.nosplit.xgb[i] = delong_test_xgb$conf.int[2]
  
  
  
  
  
  
  
  # Sample splitting
  test.ix = sample(1:n, n/4)
  df_test = df[test.ix, ]
  df_train = df[-test.ix, ]
  y_train = df_train[,1]
  y_test = df_test[,1]
  
  
  # Logistic
  logi.full.train = glm(Y ~ . , data = df_train, family = binomial(link = "logit"))
  Y.pred.full.test = predict(logi.full.train, newdata = df_test, type = "response") 
  auc.logi.test.full[i] = auc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))[1]
  
  
  logi.redu.train = glm(Y ~ prior_PTB_count + CLV2 + SlopeV2 + SlopeV2 * GAV2, data = df_train, family = binomial(link = "logit"))           
  Y.pred.redu.test = predict(logi.redu.train, newdata = df_test, type = "response")
  auc.logi.test.redu[i] = auc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))[1]
  
  roc_model1 <- roc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))
  roc_model2 <- roc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))
  
  # Perform DeLong test to compare the two ROC curves
  delong_test_logi <- roc.test(roc_model1, roc_model2, method = "delong")
  
  delong.lower.split.logi[i] = delong_test_logi$conf.int[1]
  delong.upper.split.logi[i] = delong_test_logi$conf.int[2]
  
  
  
  
  # LDA
  lda.full.train = suppressWarnings(lda(Y ~ . , data = df_train))
  Y.pred.full.test = as.numeric(predict(lda.full.train, newdata = df_test, type = "response")$x)
  auc.lda.test.full[i] = auc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))[1]
  
  lda.redu.train = lda(Y ~ prior_PTB_count + CLV2 + SlopeV2 + SlopeV2 * GAV2, data = df_train)
  Y.pred.redu.test = as.numeric(predict(lda.redu.train, newdata = df_test, type = "response")$x)
  auc.lda.test.redu[i] = auc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))[1]
  
  roc_model3 <- roc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))
  roc_model4 <- roc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))
  
  # Perform DeLong test to compare the two ROC curves
  delong_test_lda <- roc.test(roc_model3, roc_model4, method = "delong")
  
  delong.lower.split.lda[i] = delong_test_lda$conf.int[1]
  delong.upper.split.lda[i] = delong_test_lda$conf.int[2]

  
  

  # SVM split
  svm.full.train = svm(formula = as.factor(Y) ~ ., data = df_train, probability = FALSE, decision.values = TRUE)
  Y.pred.full.test = attr(predict(svm.full.train, df_test, decision.values = TRUE), "decision.values")[,1]
  auc.svm.test.full[i] = auc(y_test, Y.pred.full.test, direction=">", levels = c("0", "1"))[1]
  
  
  svm.redu.train = svm(formula = as.factor(Y) ~ prior_PTB_count + CLV2 + SlopeV2 + SlopeV2 * GAV2, data = df_train, probability = FALSE, decision.values = TRUE)
  Y.pred.redu.test = attr(predict(svm.redu.train, df_test, decision.values = TRUE), "decision.values")[,1]
  auc.svm.test.redu[i] = auc(y_test, Y.pred.redu.test, direction = ">", levels = c("0", "1"))[1]
  
  
  roc_model5 <- roc(y_test, Y.pred.full.test, direction = ">", levels = c("0", "1"))
  roc_model6 <- roc(y_test, Y.pred.redu.test, direction = ">", levels = c("0", "1"))
  
  # Perform DeLong test to compare the two ROC curves
  delong_test_svm <- roc.test(roc_model5, roc_model6, method = "delong")
  
  delong.lower.split.svm[i] = delong_test_svm$conf.int[1]
  delong.upper.split.svm[i] = delong_test_svm$conf.int[2]
  
  
  
  
  # XGBoosting for classification, 100 boosting rounds, split
  params = list(objective = "binary:logistic")
  
  X_train_full = as.matrix(df_train[, - which(names(df_train) == "Y")])
  dtrain.full = xgb.DMatrix(data = X_train_full, label = df_train$Y)
  xgb.full.train = xgb.train(params = params, data = dtrain.full, nrounds = 100)
  X_test_full = as.matrix(df_test[, - which(names(df_test) == "Y")])
  dtest.full = xgb.DMatrix(data = X_test_full, label = df_test$Y)
  Y.pred.full.test = predict(xgb.full.train, dtest.full)
  auc.xgb.test.full[i] = auc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))[1]
  
  
  
  X_train_redu = as.matrix(df_train[c("prior_PTB_count", "CLV2", "SlopeV2", "ptb_update.SlopeV1...ptb_update.GAV1")])
  dtrain.redu = xgb.DMatrix(data = X_train_redu, label = df_train$Y)
  xgb.redu.train = xgb.train(params = params, data = dtrain.redu, nrounds = 100)
  X_test_redu = as.matrix(df_test[c("prior_PTB_count", "CLV2", "SlopeV2", "ptb_update.SlopeV1...ptb_update.GAV1")])
  dtest.redu = xgb.DMatrix(data = X_test_redu, label = df_test$Y)
  Y.pred.redu.test = predict(xgb.redu.train, dtest.redu)
  auc.xgb.test.redu[i] = auc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))[1]
  
  
  roc_model7 <- roc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))
  roc_model8 <- roc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))
  
  # Perform DeLong test to compare the two ROC curves
  delong_test_xgb <- roc.test(roc_model7, roc_model8, method = "delong")
  
  delong.lower.split.xgb[i] = delong_test_xgb$conf.int[1]
  delong.upper.split.xgb[i] = delong_test_xgb$conf.int[2]
}


auc.logi.nested.diff = auc.logi.full - auc.logi.redu

auc.logi.nested.test.diff = auc.logi.test.full - auc.logi.test.redu

auc.lda.nested.diff = auc.lda.full - auc.lda.redu

auc.lda.nested.test.diff = auc.lda.test.full - auc.lda.test.redu

auc.svm.nested.diff = auc.svm.full - auc.svm.redu

auc.svm.nested.test.diff = auc.svm.test.full - auc.svm.test.redu

auc.xgb.nested.diff = auc.xgb.full - auc.xgb.redu

auc.xgb.nested.test.diff = auc.xgb.test.full - auc.xgb.test.redu




hist(auc.logi.nested.test.diff, freq = FALSE, breaks = 20, main = "Split", xlab = "Logistic AUC Difference", xlim =c(-1, 1))

data.frame(mean(auc.logi.full), sd(auc.logi.full),
           mean(auc.logi.redu), sd(auc.logi.redu))
data.frame(mean(auc.logi.nested.diff), sd(auc.logi.nested.diff), 
           z=mean(auc.logi.nested.diff)/sd(auc.logi.nested.diff))

data.frame(mean(auc.logi.test.full), sd(auc.logi.test.full),
           mean(auc.logi.test.redu), sd(auc.logi.test.redu))
data.frame(mean(auc.logi.nested.test.diff), sd(auc.logi.nested.test.diff),
           z=mean(auc.logi.nested.test.diff)/sd(auc.logi.nested.test.diff))

quantile(auc.logi.test.full, c(0.05,0.95))

quantile(auc.logi.test.redu, c(0.05,0.95))

quantile(auc.logi.nested.test.diff, c(0.05,0.95))








hist(auc.lda.nested.test.diff, freq = FALSE, breaks = 20, main = "Split", xlab = "LDA AUC Difference", xlim =c(-1, 1))


data.frame(mean(auc.lda.full), sd(auc.lda.full),
           mean(auc.lda.redu), sd(auc.lda.redu))
data.frame(mean(auc.lda.nested.diff), sd(auc.lda.nested.diff), 
           z=mean(auc.lda.nested.diff)/sd(auc.lda.nested.diff))

data.frame(mean(auc.lda.test.full), sd(auc.lda.test.full),
           mean(auc.lda.test.redu), sd(auc.lda.test.redu))
data.frame(mean(auc.lda.nested.test.diff), sd(auc.lda.nested.test.diff),
           z=mean(auc.lda.nested.test.diff)/sd(auc.lda.nested.test.diff))

quantile(auc.lda.test.full, c(0.05,0.95))

quantile(auc.lda.test.redu, c(0.05,0.95))

quantile(auc.lda.nested.test.diff, c(0.05,0.95))





hist(auc.svm.nested.diff, freq = FALSE, breaks = 10, main = "No Split", xlab = "SVM AUC Difference", xlim = c(-0.3, 0.3))

hist(auc.svm.nested.test.diff, freq = FALSE, breaks = 10, main = "Split", xlab = "SVM AUC Difference", xlim =c(-0.3, 0.3))


data.frame(mean(auc.svm.full), sd(auc.svm.full),
           mean(auc.svm.redu), sd(auc.svm.redu))
data.frame(mean(auc.svm.nested.diff), sd(auc.svm.nested.diff),
           z=mean(auc.svm.nested.diff)/sd(auc.svm.nested.diff))

data.frame(mean(auc.svm.test.full), sd(auc.svm.test.full),
           mean(auc.svm.test.redu), sd(auc.svm.test.redu))
data.frame(mean(auc.svm.nested.test.diff), sd(auc.svm.nested.test.diff),
           z=mean(auc.svm.nested.test.diff)/sd(auc.svm.nested.test.diff))

quantile(auc.svm.full, c(0.05,0.95))

quantile(auc.svm.redu, c(0.05,0.95))

quantile(auc.svm.nested.diff, c(0.05,0.95))


quantile(auc.svm.test.full, c(0.05,0.95))

quantile(auc.svm.test.redu, c(0.05,0.95))

quantile(auc.svm.nested.test.diff, c(0.05,0.95))




hist(auc.xgb.nested.diff, freq = FALSE, breaks = 10, main = "No Split", xlab = "XGBoost AUC Difference", xlim = c(-0.3, 0.3))

hist(auc.xgb.nested.test.diff, freq = FALSE, breaks = 10, main = "Split", xlab = "XGBoost AUC Difference", xlim =c(-0.3, 0.3))


data.frame(mean(auc.xgb.full), sd(auc.xgb.full),
           mean(auc.xgb.redu), sd(auc.xgb.redu))
data.frame(mean(auc.xgb.nested.diff), sd(auc.xgb.nested.diff),
           z = mean(auc.xgb.nested.diff)/sd(auc.xgb.nested.diff))

data.frame(mean(auc.xgb.test.full), sd(auc.xgb.test.full),
           mean(auc.xgb.test.redu), sd(auc.xgb.test.redu))
data.frame(mean(auc.xgb.nested.test.diff), sd(auc.xgb.nested.test.diff),
           z = mean(auc.xgb.nested.test.diff)/sd(auc.xgb.nested.test.diff))

quantile(auc.xgb.full, c(0.05,0.95))

quantile(auc.xgb.redu, c(0.05,0.95))

quantile(auc.xgb.nested.diff, c(0.05,0.95))


quantile(auc.xgb.test.full, c(0.05,0.95))

quantile(auc.xgb.test.redu, c(0.05,0.95))

quantile(auc.xgb.nested.test.diff, c(0.05,0.95))



# Delong CI

ci.mean.nosplit.logi = c(mean(delong.lower.nosplit.logi), mean(delong.upper.nosplit.logi))
ci.mean.split.logi = c(mean(delong.lower.split.logi), mean(delong.upper.split.logi))
ci.mean.nosplit.logi
ci.mean.split.logi



ci.mean.nosplit.lda = c(mean(delong.lower.nosplit.lda), mean(delong.upper.nosplit.lda))
ci.mean.split.lda = c(mean(delong.lower.split.lda), mean(delong.upper.split.lda))
ci.mean.nosplit.lda 
ci.mean.split.lda



ci.mean.nosplit.svm = c(mean(delong.lower.nosplit.svm), mean(delong.upper.nosplit.svm))
ci.mean.split.svm = c(mean(delong.lower.split.svm), mean(delong.upper.split.svm))
ci.mean.nosplit.svm
ci.mean.split.svm



ci.mean.nosplit.xgb = c(mean(delong.lower.nosplit.xgb), mean(delong.upper.nosplit.xgb))
ci.mean.split.xgb = c(mean(delong.lower.split.xgb), mean(delong.upper.split.xgb))
ci.mean.nosplit.xgb 
ci.mean.split.xgb







################################################################################
# To detect the significant difference, try to use HC only vs Base model.

auc.lda.full = rep(0, R)
auc.lda.redu = rep(0, R)
auc.lda.test.full = rep(0, R)
auc.lda.test.redu = rep(0, R)


auc.logi.full = rep(0, R)
auc.logi.redu = rep(0, R)
auc.logi.test.full = rep(0, R)
auc.logi.test.redu = rep(0, R)


auc.svm.full = rep(0, R)
auc.svm.redu = rep(0, R)
auc.svm.test.full = rep(0, R)
auc.svm.test.redu = rep(0, R)


auc.xgb.full = rep(0, R)
auc.xgb.redu = rep(0, R)
auc.xgb.test.full = rep(0, R)
auc.xgb.test.redu = rep(0, R)



delong.lower.nosplit.lda = rep(0, R)
delong.upper.nosplit.lda = rep(0, R)
delong.lower.split.lda = rep(0, R)
delong.upper.split.lda = rep(0, R)


delong.lower.nosplit.logi = rep(0, R)
delong.upper.nosplit.logi = rep(0, R)
delong.lower.split.logi = rep(0, R)
delong.upper.split.logi = rep(0, R)

delong.lower.nosplit.svm = rep(0, R)
delong.upper.nosplit.svm = rep(0, R)
delong.lower.split.svm = rep(0, R)
delong.upper.split.svm = rep(0, R)

delong.lower.nosplit.xgb = rep(0, R)
delong.upper.nosplit.xgb = rep(0, R)
delong.lower.split.xgb = rep(0, R)
delong.upper.split.xgb = rep(0, R)


for(i in 1:R)
{
  # Data 
  X1 = ptb_update$CLV0
  # QUS for Visit 2
  X_rest = cbind(ptb_update$CLV2, ptb_update$SlopeV2, ptb_update$SlopeV2 * ptb_update$GAV2)
  Y = as.integer(factor(ptb_update$outcome)) - 1
  df = data.frame(cbind(Y, X1, X_rest))
  
  
  # Logistic
  logi.full = glm(Y ~ . , data = df, family = binomial(link = "logit"))
  Y.true = df[,1]
  Y.logi.pred.full = logi.full$fitted.values 
  auc.logi.full[i] = auc(Y.true, Y.logi.pred.full, direction = "<", levels = c("0", "1"))[1]
  
  
  logi.redu = glm(Y ~ X1 , data = df, family = binomial(link = "logit"))
  Y.true = df[,1]
  Y.logi.pred.redu = logi.redu$fitted.values 
  auc.logi.redu[i] = auc(Y.true, Y.logi.pred.redu, direction = "<", levels = c("0", "1"))[1]
  
  roc_model1 <- roc(Y.true, Y.logi.pred.full, direction = "<", levels = c("0", "1"))
  roc_model2 <- roc(Y.true, Y.logi.pred.redu, direction = "<", levels = c("0", "1"))
  
  # Perform DeLong test to compare the two ROC curves
  delong_test_logi <- roc.test(roc_model1, roc_model2, method = "delong")
  
  delong.lower.nosplit.logi[i] = delong_test_logi$conf.int[1]
  delong.upper.nosplit.logi[i] = delong_test_logi$conf.int[2]
  
  
  
  
  
  # LDA
  lda.full = suppressWarnings(lda(Y ~ . , data = df))
  Y.true = df[,1]
  Y.pred.full = as.numeric(predict(lda.full, type = "response")$x)
  auc.lda.full[i] = auc(Y.true, Y.pred.full, direction = "<", levels = c("0", "1"))[1]
  
  
  lda.redu = lda(Y ~ X1 , data = df)
  Y.true = df[,1]
  Y.pred.redu = as.numeric(predict(lda.redu, type = "response")$x)
  auc.lda.redu[i] = auc(Y.true, Y.pred.redu, direction = "<", levels = c("0", "1"))[1]
  
  roc_model3 <- roc(Y.true, Y.pred.full, direction = "<", levels = c("0", "1"))
  roc_model4 <- roc(Y.true, Y.pred.redu, direction = "<", levels = c("0", "1"))
  
  # Perform DeLong test to compare the two ROC curves
  delong_test_lda <- roc.test(roc_model3, roc_model4, method = "delong")
  
  delong.lower.nosplit.lda[i] = delong_test_lda$conf.int[1]
  delong.upper.nosplit.lda[i] = delong_test_lda$conf.int[2]
  
  # SVM no split
  svm.full = svm(formula = as.factor(Y) ~ ., data = df, probability = FALSE, decision.values = TRUE)
  Y.true = df[,1]
  Y.pred.full = attr(predict(svm.full, df, decision.values = TRUE), "decision.values")[,1]
  auc.svm.full[i] = auc(Y.true, Y.pred.full, direction=">", levels = c("0", "1"))[1]
  
  
  svm.redu = svm(formula = as.factor(Y) ~ X1, data = df, probability = FALSE, decision.values = TRUE)
  Y.true = df[,1]
  Y.pred.redu = attr(predict(svm.redu, df, decision.values = TRUE), "decision.values")[,1]
  auc.svm.redu[i] = auc(Y.true, Y.pred.redu, direction=">", levels = c("0", "1"))[1]
  
  roc_model5 <- roc(Y.true, Y.pred.full, direction = ">", levels = c("0", "1"))
  roc_model6 <- roc(Y.true, Y.pred.redu, direction = ">", levels = c("0", "1"))
  
  
  # Perform DeLong test to compare the two ROC curves
  delong_test_svm <- roc.test(roc_model5, roc_model6, method = "delong")
  
  delong.lower.nosplit.svm[i] = delong_test_svm$conf.int[1]
  delong.upper.nosplit.svm[i] = delong_test_svm$conf.int[2]
  
  

  
  # XGBoost for classification, 100 boosting rounds
  params = list(objective = "binary:logistic")
  
  X_full = as.matrix(df[, - which(names(df) == "Y")])
  dtrain.full = xgb.DMatrix(data = X_full, label = Y)
  xgb.full = xgb.train(params = params, data = dtrain.full, nrounds = 100)
  Y.pred.full = predict(xgb.full, dtrain.full)
  auc.xgb.full[i] = auc(Y, Y.pred.full, direction = "<", levels = c("0", "1"))[1]
  
  
  X_redu = as.matrix(df["X1"])
  dtrain.redu = xgb.DMatrix(data = X_redu, label = Y)
  xgb.redu = xgb.train(params = params, data = dtrain.redu, nrounds = 100)
  Y.pred.redu = predict(xgb.redu, dtrain.redu)
  auc.xgb.redu[i] = auc(Y, Y.pred.redu, direction = "<", levels = c("0", "1"))[1]
  
  roc_model7 <- roc(Y.true, Y.pred.full, direction = "<", levels = c("0", "1"))
  roc_model8 <- roc(Y.true, Y.pred.redu, direction = "<", levels = c("0", "1"))
  
  # Perform DeLong test to compare the two ROC curves
  delong_test_xgb <- roc.test(roc_model7, roc_model8, method = "delong")
  
  delong.lower.nosplit.xgb[i] = delong_test_xgb$conf.int[1]
  delong.upper.nosplit.xgb[i] = delong_test_xgb$conf.int[2]
  
  
  
  
  
  
  # Sample splitting
  test.ix = sample(1:n, n/4)
  df_test = df[test.ix, ]
  df_train = df[-test.ix, ]
  y_train = df_train[,1]
  y_test = df_test[,1]
  
  
  # Logistic
  logi.full.train = glm(Y ~ . , data = df_train, family = binomial(link = "logit"))
  Y.pred.full.test = predict(logi.full.train, newdata = df_test, type = "response") 
  auc.logi.test.full[i] = auc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))[1]
  
  
  logi.redu.train = glm(Y ~ X1, data = df_train, family = binomial(link = "logit"))
  Y.pred.redu.test = predict(logi.redu.train, newdata = df_test, type = "response")
  auc.logi.test.redu[i] = auc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))[1]
  
  roc_model1 <- roc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))
  roc_model2 <- roc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))
  
  # Perform DeLong test to compare the two ROC curves
  delong_test_logi <- roc.test(roc_model1, roc_model2, method = "delong")
  
  delong.lower.split.logi[i] = delong_test_logi$conf.int[1]
  delong.upper.split.logi[i] = delong_test_logi$conf.int[2]
  
  
  # LDA
  lda.full.train = suppressWarnings(lda(Y ~ . , data = df_train))
  Y.pred.full.test = as.numeric(predict(lda.full.train, newdata = df_test, type = "response")$x)
  auc.lda.test.full[i] = auc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))[1]
  
  lda.redu.train = lda(Y ~ X1, data = df_train)
  Y.pred.redu.test = as.numeric(predict(lda.redu.train, newdata = df_test, type = "response")$x)
  auc.lda.test.redu[i] = auc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))[1]
  
  
  roc_model3 <- roc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))
  roc_model4 <- roc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))
  
  # Perform DeLong test to compare the two ROC curves
  delong_test_lda <- roc.test(roc_model3, roc_model4, method = "delong")
  
  delong.lower.split.lda[i] = delong_test_lda$conf.int[1]
  delong.upper.split.lda[i] = delong_test_lda$conf.int[2]
 

  
  # SVM split
  svm.full.train = svm(formula = as.factor(Y) ~ ., data = df_train, probability = FALSE, decision.values = TRUE)
  Y.pred.full.test = attr(predict(svm.full.train, df_test, decision.values = TRUE), "decision.values")[,1]
  auc.svm.test.full[i] = auc(y_test, Y.pred.full.test, direction=">", levels = c("0", "1"))[1]
  
  
  svm.redu.train = svm(formula = as.factor(Y) ~ X1, data = df_train, probability = FALSE, decision.values = TRUE)
  Y.pred.redu.test = attr(predict(svm.redu.train, df_test, decision.values = TRUE), "decision.values")[,1]
  auc.svm.test.redu[i] = auc(y_test, Y.pred.redu.test, direction = ">", levels = c("0", "1"))[1]
  
  
  roc_model5 <- roc(y_test, Y.pred.full.test, direction = ">", levels = c("0", "1"))
  roc_model6 <- roc(y_test, Y.pred.redu.test, direction = ">", levels = c("0", "1"))
  
  # Perform DeLong test to compare the two ROC curves
  delong_test_svm <- roc.test(roc_model5, roc_model6, method = "delong")
  
  delong.lower.split.svm[i] = delong_test_svm$conf.int[1]
  delong.upper.split.svm[i] = delong_test_svm$conf.int[2]
  
  
  
  
  
  
  # XGBoosting for classification, 100 boosting rounds, split
  params = list(objective = "binary:logistic")
  
  X_train_full = as.matrix(df_train[, - which(names(df_train) == "Y")])
  dtrain.full = xgb.DMatrix(data = X_train_full, label = df_train$Y)
  xgb.full.train = xgb.train(params = params, data = dtrain.full, nrounds = 100)
  X_test_full = as.matrix(df_test[, - which(names(df_test) == "Y")])
  dtest.full = xgb.DMatrix(data = X_test_full, label = df_test$Y)
  Y.pred.full.test = predict(xgb.full.train, dtest.full)
  auc.xgb.test.full[i] = auc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))[1]
  
  
  
  X_train_redu = as.matrix(df_train["X1"])
  dtrain.redu = xgb.DMatrix(data = X_train_redu, label = df_train$Y)
  xgb.redu.train = xgb.train(params = params, data = dtrain.redu, nrounds = 100)
  X_test_redu = as.matrix(df_test["X1"])
  dtest.redu = xgb.DMatrix(data = X_test_redu, label = df_test$Y)
  Y.pred.redu.test = predict(xgb.redu.train, dtest.redu)
  auc.xgb.test.redu[i] = auc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))[1]
  
  
  roc_model7 <- roc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))
  roc_model8 <- roc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))
  
  # Perform DeLong test to compare the two ROC curves
  delong_test_xgb <- roc.test(roc_model7, roc_model8, method = "delong")
  
  delong.lower.split.xgb[i] = delong_test_xgb$conf.int[1]
  delong.upper.split.xgb[i] = delong_test_xgb$conf.int[2]
  
  
}


auc.logi.nested.diff = auc.logi.full - auc.logi.redu

auc.logi.nested.test.diff = auc.logi.test.full - auc.logi.test.redu

auc.lda.nested.diff = auc.lda.full - auc.lda.redu

auc.lda.nested.test.diff = auc.lda.test.full - auc.lda.test.redu

auc.svm.nested.diff = auc.svm.full - auc.svm.redu

auc.svm.nested.test.diff = auc.svm.test.full - auc.svm.test.redu

auc.xgb.nested.diff = auc.xgb.full - auc.xgb.redu

auc.xgb.nested.test.diff = auc.xgb.test.full - auc.xgb.test.redu




hist(auc.logi.nested.test.diff, freq = FALSE, breaks = 20, main = "Split", xlab = "Logistic AUC Difference", xlim =c(-1, 1))


data.frame(mean(auc.logi.full), sd(auc.logi.full),
           mean(auc.logi.redu), sd(auc.logi.redu))
data.frame(mean(auc.logi.nested.diff), sd(auc.logi.nested.diff), 
           z=mean(auc.logi.nested.diff)/sd(auc.logi.nested.diff))

data.frame(mean(auc.logi.test.full), sd(auc.logi.test.full),
           mean(auc.logi.test.redu), sd(auc.logi.test.redu))
data.frame(mean(auc.logi.nested.test.diff), sd(auc.logi.nested.test.diff),
           z=mean(auc.logi.nested.test.diff)/sd(auc.logi.nested.test.diff))

quantile(auc.logi.test.full, c(0.05,0.95))

quantile(auc.logi.test.redu, c(0.05,0.95))

quantile(auc.logi.nested.test.diff, c(0.05,0.95))






hist(auc.lda.nested.test.diff, freq = FALSE, breaks = 20, main = "Split", xlab = "LDA AUC Difference", xlim =c(-1, 1))


data.frame(mean(auc.lda.full), sd(auc.lda.full),
           mean(auc.lda.redu), sd(auc.lda.redu))
data.frame(mean(auc.lda.nested.diff), sd(auc.lda.nested.diff), 
           z=mean(auc.lda.nested.diff)/sd(auc.lda.nested.diff))

data.frame(mean(auc.lda.test.full), sd(auc.lda.test.full),
           mean(auc.lda.test.redu), sd(auc.lda.test.redu))
data.frame(mean(auc.lda.nested.test.diff), sd(auc.lda.nested.test.diff),
           z=mean(auc.lda.nested.test.diff)/sd(auc.lda.nested.test.diff))

quantile(auc.lda.test.full, c(0.05,0.95))

quantile(auc.lda.test.redu, c(0.05,0.95))

quantile(auc.lda.nested.test.diff, c(0.05,0.95))



hist(auc.svm.nested.diff, freq = FALSE, breaks = 20, main = "No Split", xlab = "SVM AUC Difference", xlim = c(-1, 1))

hist(auc.svm.nested.test.diff, freq = FALSE, breaks = 20, main = "Split", xlab = "SVM AUC Difference", xlim =c(-1, 1))


data.frame(mean(auc.svm.full), sd(auc.svm.full),
           mean(auc.svm.redu), sd(auc.svm.redu))
data.frame(mean(auc.svm.nested.diff), sd(auc.svm.nested.diff),
           z=mean(auc.svm.nested.diff)/sd(auc.svm.nested.diff))

data.frame(mean(auc.svm.test.full), sd(auc.svm.test.full),
           mean(auc.svm.test.redu), sd(auc.svm.test.redu))
data.frame(mean(auc.svm.nested.test.diff), sd(auc.svm.nested.test.diff),
           z=mean(auc.svm.nested.test.diff)/sd(auc.svm.nested.test.diff))

quantile(auc.svm.full, c(0.05,0.95))

quantile(auc.svm.redu, c(0.05,0.95))

quantile(auc.svm.nested.diff, c(0.05,0.95))


quantile(auc.svm.test.full, c(0.05,0.95))

quantile(auc.svm.test.redu, c(0.05,0.95))

quantile(auc.svm.nested.test.diff, c(0.05,0.95))




hist(auc.xgb.nested.diff, freq = FALSE, breaks = 20, main = "No Split", xlab = "XGBoost AUC Difference", xlim = c(-1, 1))

hist(auc.xgb.nested.test.diff, freq = FALSE, breaks = 20, main = "Split", xlab = "XGBoost AUC Difference", xlim =c(-1, 1))


data.frame(mean(auc.xgb.full), sd(auc.xgb.full),
           mean(auc.xgb.redu), sd(auc.xgb.redu))
data.frame(mean(auc.xgb.nested.diff), sd(auc.xgb.nested.diff),
           z = mean(auc.xgb.nested.diff)/sd(auc.xgb.nested.diff))

data.frame(mean(auc.xgb.test.full), sd(auc.xgb.test.full),
           mean(auc.xgb.test.redu), sd(auc.xgb.test.redu))
data.frame(mean(auc.xgb.nested.test.diff), sd(auc.xgb.nested.test.diff),
           z = mean(auc.xgb.nested.test.diff)/sd(auc.xgb.nested.test.diff))

quantile(auc.xgb.full, c(0.05,0.95))

quantile(auc.xgb.redu, c(0.05,0.95))

quantile(auc.xgb.nested.diff, c(0.05,0.95))


quantile(auc.xgb.test.full, c(0.05,0.95))

quantile(auc.xgb.test.redu, c(0.05,0.95))

quantile(auc.xgb.nested.test.diff, c(0.05,0.95))


# Delong CI

ci.mean.nosplit.logi = c(mean(delong.lower.nosplit.logi), mean(delong.upper.nosplit.logi))
ci.mean.split.logi = c(mean(delong.lower.split.logi), mean(delong.upper.split.logi))
ci.mean.nosplit.logi
ci.mean.split.logi




ci.mean.nosplit.lda = c(mean(delong.lower.nosplit.lda), mean(delong.upper.nosplit.lda))
ci.mean.split.lda = c(mean(delong.lower.split.lda), mean(delong.upper.split.lda))
ci.mean.nosplit.lda 
ci.mean.split.lda



ci.mean.nosplit.svm = c(mean(delong.lower.nosplit.svm), mean(delong.upper.nosplit.svm))
ci.mean.split.svm = c(mean(delong.lower.split.svm), mean(delong.upper.split.svm))
ci.mean.nosplit.svm
ci.mean.split.svm





ci.mean.nosplit.xgb = c(mean(delong.lower.nosplit.xgb), mean(delong.upper.nosplit.xgb))
ci.mean.split.xgb = c(mean(delong.lower.split.xgb), mean(delong.upper.split.xgb))
ci.mean.nosplit.xgb 
ci.mean.split.xgb




################################################################################
# Firmingham Heart Data
# Delta AUC Nested Model
library("e1071") 
library("MASS")
library("randomForest")
library("xgboost")
library("pROC")
library("mvtnorm")
library("riskCommunicator")


# Data 
df_heart = data.frame(framingham)

# Drop missing data
df_heart = df_heart[complete.cases(df_heart),]


# Select participant for independent samples
df_last_visit = df_heart[df_heart$TIME > 4000,]

# To check no participant has more than one sample
max(table(df_last_visit$RANDID))



# X1 = cbind(df_heart$AGE, df_heart$SYSBP, df_heart$HYPERTEN, df_heart$SEX) # df_heart$TOTCHOL, df_heart$HDLC not included here. too many missing values. 
X1 = cbind(df_last_visit$BMI, df_last_visit$SYSBP, df_last_visit$DIABP, df_last_visit$TOTCHOL)
# QUS for Visit 2
X_rest = df_last_visit$AGE
Y = df_last_visit$DIABETES


# Scale Data 
X1 = as.data.frame(scale(X1))
X_rest = as.data.frame(scale(X_rest))


df_select = data.frame(cbind(Y, X1, X_rest))
colnames(df_select) = c("Y", "BMI", "SYSBP", "DIABP", "TOTCHOL", "AGE")


# # Drop missing data

df = df_select


R = 1000
n = nrow(df_heart)

auc.lda.full = rep(0, R)
auc.lda.redu = rep(0, R)
auc.lda.test.full = rep(0, R)
auc.lda.test.redu = rep(0, R)

auc.logi.full = rep(0, R)
auc.logi.redu = rep(0, R)
auc.logi.test.full = rep(0, R)
auc.logi.test.redu = rep(0, R)


delong.lower.nosplit.logi = rep(0, R)
delong.upper.nosplit.logi = rep(0, R)
delong.lower.split.logi = rep(0, R)
delong.upper.split.logi = rep(0, R)

delong.lower.nosplit.lda = rep(0, R)
delong.upper.nosplit.lda = rep(0, R)
delong.lower.split.lda = rep(0, R)
delong.upper.split.lda = rep(0, R)




for(i in 1:R)
{
  # Logistic
  logi.full = glm(Y ~ . , data = df, family = binomial(link = "logit"))
  Y.true = df[,1]
  Y.logi.pred.full = logi.full$fitted.values 
  auc.logi.full[i] = auc(Y.true, Y.logi.pred.full, direction = "<", levels = c("0", "1"))[1]
  
  
  logi.redu = glm(Y ~ . - AGE, data = df, family = binomial(link = "logit"))
  Y.true = df[,1]
  Y.logi.pred.redu = logi.redu$fitted.values 
  auc.logi.redu[i] = auc(Y.true, Y.logi.pred.redu, direction = "<", levels = c("0", "1"))[1]
  
  roc_model1 <- roc(Y.true, Y.logi.pred.full, direction = "<", levels = c("0", "1"))
  roc_model2 <- roc(Y.true, Y.logi.pred.redu, direction = "<", levels = c("0", "1"))
  
  # Perform DeLong test to compare the two ROC curves
  delong_test_logi <- roc.test(roc_model1, roc_model2, method = "delong")
  
  delong.lower.nosplit.logi[i] = delong_test_logi$conf.int[1]
  delong.upper.nosplit.logi[i] = delong_test_logi$conf.int[2]
  
  # LDA
  lda.full = lda(Y ~ . , data = df)
  Y.true = df[,1]
  Y.pred.full = as.numeric(predict(lda.full, type = "response")$x)
  auc.lda.full[i] = auc(Y.true, Y.pred.full, direction = "<", levels = c("0", "1"))[1]
  
  
  lda.redu = lda(Y ~ . - AGE, data = df)
  Y.true = df[,1]
  Y.pred.redu = as.numeric(predict(lda.redu, type = "response")$x)
  auc.lda.redu[i] = auc(Y.true, Y.pred.redu, direction = "<", levels = c("0", "1"))[1]
  
  

  roc_model3 <- roc(Y.true, Y.pred.full, direction = "<", levels = c("0", "1"))
  roc_model4 <- roc(Y.true, Y.pred.redu, direction = "<", levels = c("0", "1"))
  
  # Perform DeLong test to compare the two ROC curves
  delong_test_lda <- roc.test(roc_model3, roc_model4, method = "delong")
  
  delong.lower.nosplit.lda[i] = delong_test_lda$conf.int[1]
  delong.upper.nosplit.lda[i] = delong_test_lda$conf.int[2]
  


  
  
  
  # Sample splitting
  test.ix = sample(1:n, n/2)
  df_test = df[test.ix, ]
  df_train = df[-test.ix, ]
  y_train = df_train[,1]
  y_test = df_test[,1]
  
  
  # Logistic
  logi.full.train = glm(Y ~ . , data = df_train, family = binomial(link = "logit"))
  Y.pred.full.test = predict(logi.full.train, newdata = df_test, type = "response") 
  auc.logi.test.full[i] = auc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))[1]
  
  
  logi.redu.train = glm(Y ~ . - AGE, data = df_train, family = binomial(link = "logit"))
  Y.pred.redu.test = predict(logi.redu.train, newdata = df_test, type = "response")
  auc.logi.test.redu[i] = auc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))[1]
  
  
  roc_model1 <- roc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))
  roc_model2 <- roc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))
  
  # Perform DeLong test to compare the two ROC curves
  delong_test_logi <- roc.test(roc_model1, roc_model2, method = "delong")
  
  delong.lower.split.logi[i] = delong_test_logi$conf.int[1]
  delong.upper.split.logi[i] = delong_test_logi$conf.int[2]
  
  
  
  # LDA
  lda.full.train = lda(Y ~ . , data = df_train)
  Y.pred.full.test = as.numeric(predict(lda.full.train, newdata = df_test, type = "response")$x)
  auc.lda.test.full[i] = auc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))[1]
  
  lda.redu.train = lda(Y ~ . - AGE, data = df_train)
  Y.pred.redu.test = as.numeric(predict(lda.redu.train, newdata = df_test, type = "response")$x)
  auc.lda.test.redu[i] = auc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))[1]
  
  
  roc_model3 <- roc(y_test, Y.pred.full.test, direction = "<", levels = c("0", "1"))
  roc_model4 <- roc(y_test, Y.pred.redu.test, direction = "<", levels = c("0", "1"))
  
  # Perform DeLong test to compare the two ROC curves
  delong_test_lda <- roc.test(roc_model3, roc_model4, method = "delong")
  
  delong.lower.split.lda[i] = delong_test_lda$conf.int[1]
  delong.upper.split.lda[i] = delong_test_lda$conf.int[2]
  
}


auc.logi.nested.diff = auc.logi.full - auc.logi.redu

auc.logi.nested.test.diff = auc.logi.test.full - auc.logi.test.redu

auc.lda.nested.diff = auc.lda.full - auc.lda.redu

auc.lda.nested.test.diff = auc.lda.test.full - auc.lda.test.redu





hist(auc.logi.nested.test.diff, freq = FALSE, breaks = 20, main = "Split", xlab = "Logistic AUC Difference", xlim =c(-0.1, 0.1))
# lines(density(auc.logi.nested.test.diff, bw = 0.001))
x.logi = seq(-0.08, 0.08, length=1000)
y.logi <- dnorm(x.logi, mean = mean(auc.logi.nested.test.diff), sd = sd(auc.logi.nested.test.diff))
lines(x.logi, y.logi, type="l", lwd=1)


data.frame(mean(auc.logi.full), sd(auc.logi.full),
           mean(auc.logi.redu), sd(auc.logi.redu))
data.frame(mean(auc.logi.nested.diff), sd(auc.logi.nested.diff), 
           z=mean(auc.logi.nested.diff)/sd(auc.logi.nested.diff))

data.frame(mean(auc.logi.test.full), sd(auc.logi.test.full),
           mean(auc.logi.test.redu), sd(auc.logi.test.redu))
data.frame(mean(auc.logi.nested.test.diff), sd(auc.logi.nested.test.diff),
           z=mean(auc.logi.nested.test.diff)/sd(auc.logi.nested.test.diff))

quantile(auc.logi.test.full, c(0.05,0.95))

quantile(auc.logi.test.redu, c(0.05,0.95))

quantile(auc.logi.nested.test.diff, c(0.05,0.95))







hist(auc.lda.nested.test.diff, freq = FALSE, breaks = 20, main = "Split", xlab = "LDA AUC Difference", xlim =c(-0.1, 0.1))

data.frame(mean(auc.lda.full), sd(auc.lda.full),
           mean(auc.lda.redu), sd(auc.lda.redu))
data.frame(mean(auc.lda.nested.diff), sd(auc.lda.nested.diff), 
           z=mean(auc.lda.nested.diff)/sd(auc.lda.nested.diff))

data.frame(mean(auc.lda.test.full), sd(auc.lda.test.full),
           mean(auc.lda.test.redu), sd(auc.lda.test.redu))
data.frame(mean(auc.lda.nested.test.diff), sd(auc.lda.nested.test.diff),
           z=mean(auc.lda.nested.test.diff)/sd(auc.lda.nested.test.diff))

quantile(auc.lda.test.full, c(0.05,0.95))

quantile(auc.lda.test.redu, c(0.05,0.95))

quantile(auc.lda.nested.test.diff, c(0.05,0.95))








# Delong CI


ci.mean.nosplit.logi = c(mean(delong.lower.nosplit.logi), mean(delong.upper.nosplit.logi))
ci.mean.split.logi = c(mean(delong.lower.split.logi), mean(delong.upper.split.logi))
ci.mean.nosplit.logi
ci.mean.split.logi





ci.mean.nosplit.lda = c(mean(delong.lower.nosplit.lda), mean(delong.upper.nosplit.lda))
ci.mean.split.lda = c(mean(delong.lower.split.lda), mean(delong.upper.split.lda))
ci.mean.nosplit.lda 
ci.mean.split.lda







################################################################################
# Leave one out simulation for first setting
n1 = 1000
n2 = 1000
R = 200
B = 200

p = 100 # number of features. 
mu = 1
sig = 10
# For null hypothesis

snr = (mu - 0)^2/sig^2

auc.lda.full = matrix(0, R, B)
auc.lda.redu = matrix(0, R, B)
auc.lda.test.full = matrix(0, R, B)
auc.lda.test.redu = matrix(0, R, B)


auc.logi.full = matrix(0, R, B)
auc.logi.redu = matrix(0, R, B)
auc.logi.test.full = matrix(0, R, B)
auc.logi.test.redu = matrix(0, R, B)



delong.lower.nosplit.logi = matrix(0, R, B)
delong.upper.nosplit.logi = matrix(0, R, B)
delong.lower.split.logi = matrix(0, R, B)
delong.upper.split.logi = matrix(0, R, B)

delong.lower.nosplit.lda = matrix(0, R, B)
delong.upper.nosplit.lda = matrix(0, R, B)
delong.lower.split.lda = matrix(0, R, B)
delong.upper.split.lda = matrix(0, R, B)

## Function to compute leave-one-out predictions. Used for glm.

pihat <- function(mod, df) {
  pihat <- numeric(nrow(df))
  for(i in 1:nrow(df))
    pihat[i] <- predict(update(mod, subset=-i),
                        newdata=df[i,],type="response")
  pihat
}




# Simulation for resampling as B and Monte Carlo Repetition as R
for(i in 1:R)
{
  # Generating data
  X1 = c(runif(n1, 0 + mu, 2 + mu), runif(n2, 0, 2))
  # X2 = c(rnorm(n1, -1, 1), rnorm(n2, 1, 1))
  X_rest = matrix(rnorm((n1 + n2) * p, mean = 0, sd = sig), nrow = (n1 + n2), ncol = p)
  Y = c(rep(1, n1), rep(0, n2))
  df = data.frame(cbind(Y, X1, X_rest))
  # df = data.frame(cbind(Y, X1, X2, X_rest))
  
  for(j in 1:B)
  {
    lda.full = lda(Y ~ ., data = df)
    Y.true = df[,1]
    Y.pred.full = as.numeric(predict(lda.full, type = "response")$x)
    auc.lda.full[i,j] = auc(Y.true, Y.pred.full, direction = "<", levels = c("0", "1"))[1]
    
    
    lda.redu = lda(Y ~ X1, data = df)
    Y.true = df[,1]
    Y.pred.redu = as.numeric(predict(lda.redu, type = "response")$x)
    auc.lda.redu[i,j] = auc(Y.true, Y.pred.redu, direction = "<", levels = c("0", "1"))[1]
    
    
    roc_model1 <- roc(Y.true, Y.pred.full, direction = "<", levels = c("0", "1"))
    roc_model2 <- roc(Y.true, Y.pred.redu, direction = "<", levels = c("0", "1"))
    
    # Perform DeLong test to compare the two ROC curves
    delong_test_lda <- roc.test(roc_model1, roc_model2, method = "delong")
    
    delong.lower.nosplit.lda[i,j] = delong_test_lda$conf.int[1]
    delong.upper.nosplit.lda[i,j] = delong_test_lda$conf.int[2]
    
    
    
    
    logi.full = glm(Y ~ . , data = df, family = binomial(link = "logit"))
    Y.true = df[,1]
    Y.logi.pred.full = logi.full$fitted.values 
    auc.logi.full[i,j] = auc(Y.true, Y.logi.pred.full, direction = "<", levels = c("0", "1"))[1]
    
    
    logi.redu = glm(Y ~ X1 , data = df, family = binomial(link = "logit"))
    Y.true = df[,1]
    Y.logi.pred.redu = logi.redu$fitted.values 
    auc.logi.redu[i,j] = auc(Y.true, Y.logi.pred.redu, direction = "<", levels = c("0", "1"))[1]
    
    
    roc_model3 <- roc(Y.true, Y.pred.full, direction = "<", levels = c("0", "1"))
    roc_model4 <- roc(Y.true, Y.pred.redu, direction = "<", levels = c("0", "1"))
    
    # Perform DeLong test to compare the two ROC curves
    delong_test_logi <- roc.test(roc_model3, roc_model4, method = "delong")
    
    delong.lower.nosplit.logi[i,j] = delong_test_logi$conf.int[1]
    delong.upper.nosplit.logi[i,j] = delong_test_logi$conf.int[2]
    
    
    
    
    
    
    
    
    
    
    # Leave One Out Sample splitting
    
    # LDA
    lda.full.loo = lda(Y ~ . , data = df, CV = TRUE)
    # Y.pred.full.test = as.integer(predict(lda.full.train, newdata = df_test)$class) - 1
    Y.pred.full.test = lda.full.loo$posterior[,2]
    auc.lda.test.full[i,j] = auc(Y.true, Y.pred.full.test, direction = "<", levels = c("0", "1"))[1]
    
    lda.redu.loo = lda(Y ~ X1, data = df, CV = TRUE)
    # Y.pred.redu.test = as.integer(predict(lda.redu.train, newdata = df_test)$class) - 1
    Y.pred.redu.test = lda.redu.loo$posterior[,2]
    auc.lda.test.redu[i,j] = auc(Y.true, Y.pred.redu.test, direction = "<", levels = c("0", "1"))[1]
    
    roc_model1 <- roc(Y.true, Y.pred.full.test, direction = "<", levels = c("0", "1"))
    roc_model2 <- roc(Y.true, Y.pred.redu.test, direction = "<", levels = c("0", "1"))
    
    # Perform DeLong test to compare the two ROC curves
    delong_test_logi <- roc.test(roc_model1, roc_model2, method = "delong")
    
    delong.lower.split.logi[i,j] = delong_test_logi$conf.int[1]
    delong.upper.split.logi[i,j] = delong_test_logi$conf.int[2]
    
    
    
    # Logistic 
    Y.pred.full.test = pihat(logi.full, df)
    auc.logi.test.full[i,j] = auc(Y.true, Y.pred.full.test, direction = "<", levels = c("0", "1"))[1]
    
    Y.pred.redu.test = pihat(logi.redu, df)
    auc.logi.test.redu[i,j] = auc(Y.true, Y.pred.redu.test, direction = "<", levels = c("0", "1"))[1]
    
    roc_model3 <- roc(Y.true, Y.pred.full.test, direction = "<", levels = c("0", "1"))
    roc_model4 <- roc(Y.true, Y.pred.redu.test, direction = "<", levels = c("0", "1"))
    
    # Perform DeLong test to compare the two ROC curves
    delong_test_lda <- roc.test(roc_model3, roc_model4, method = "delong")
    
    delong.lower.split.lda[i,j] = delong_test_lda$conf.int[1]
    delong.upper.split.lda[i,j] = delong_test_lda$conf.int[2]
  }
}



auc.lda.nested.diff = auc.lda.full - auc.lda.redu

auc.lda.nested.test.diff = auc.lda.test.full - auc.lda.test.redu

auc.logi.nested.diff = auc.logi.full - auc.logi.redu

auc.logi.nested.test.diff = auc.logi.test.full - auc.logi.test.redu



par(mfrow = c(2,2))

hist(auc.logi.nested.diff, freq = FALSE, breaks = 10, main = "No Split", xlab = "Logistic AUC Difference", xlim =c(-0.15, 0.15))

hist(auc.logi.nested.test.diff, freq = FALSE, breaks = 20, main = "Split", xlab = "Logistic AUC Difference", xlim =c(-0.15, 0.15))




data.frame(mean(auc.logi.full), sd(auc.logi.full),
           mean(auc.logi.redu), sd(auc.logi.redu))
data.frame(mean(auc.logi.nested.diff), sd(auc.logi.nested.diff), 
           z=mean(auc.logi.nested.diff)/sd(auc.logi.nested.diff))

data.frame(mean(auc.logi.test.full), sd(auc.logi.test.full),
           mean(auc.logi.test.redu), sd(auc.logi.test.redu))
data.frame(mean(auc.logi.nested.test.diff), sd(auc.logi.nested.test.diff),
           z=mean(auc.logi.nested.test.diff)/sd(auc.logi.nested.test.diff))



hist(auc.lda.nested.diff, freq = FALSE, breaks = 10, main = "No Split", xlab = "LDA AUC Difference", xlim = c(-0.15, 0.15))

hist(auc.lda.nested.test.diff, freq = FALSE, breaks = 20, main = "Split", xlab = "LDA AUC Difference", xlim =c(-0.15, 0.15))



data.frame(mean(auc.lda.full), sd(auc.lda.full),
           mean(auc.lda.redu), sd(auc.lda.redu))
data.frame(mean(auc.lda.nested.diff), sd(auc.lda.nested.diff), 
           z=mean(auc.lda.nested.diff)/sd(auc.lda.nested.diff))

data.frame(mean(auc.lda.test.full), sd(auc.lda.test.full),
           mean(auc.lda.test.redu), sd(auc.lda.test.redu))
data.frame(mean(auc.lda.nested.test.diff), sd(auc.lda.nested.test.diff),
           z=mean(auc.lda.nested.test.diff)/sd(auc.lda.nested.test.diff))






# Inference

lda.ci = t(apply(auc.lda.nested.diff, 1, quantile, probs = c(0.05, 0.95)))

lda.test.ci = t(apply(auc.lda.nested.test.diff, 1, quantile, probs = c(0.05, 0.95)))


logi.ci = t(apply(auc.logi.nested.diff, 1, quantile, probs = c(0.05, 0.95)))

logi.test.ci = t(apply(auc.logi.nested.test.diff, 1, quantile, probs = c(0.05, 0.95)))





# Delong CI results

ci.mean.nosplit.logi = c(mean(delong.lower.nosplit.logi), mean(delong.upper.nosplit.logi))
ci.mean.split.logi = c(mean(delong.lower.split.logi), mean(delong.upper.split.logi))
ci.mean.nosplit.logi
ci.mean.split.logi



ci.mean.nosplit.lda = c(mean(delong.lower.nosplit.lda), mean(delong.upper.nosplit.lda))
ci.mean.split.lda = c(mean(delong.lower.split.lda), mean(delong.upper.split.lda))
ci.mean.nosplit.lda 
ci.mean.split.lda



















