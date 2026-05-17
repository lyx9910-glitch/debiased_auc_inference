"""
model_functions.py
==================
Python translation of R/01_model_functions.R

Provides:
  - generate_sim_data()       : simulate binary classification data
  - fit_and_evaluate_model()  : fit full/reduced model, return AUC + DeLong CI
  - delong_ci()               : paired DeLong confidence interval (95 %)
"""

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from scipy import stats

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_sim_data(n1: int, n2: int, p: int = 100,
                      mu: float = 1.0, sig: float = 10.0):
    """
    Generate simulated binary classification data.

    Informative feature X1:
        class-1 (Y=1): Uniform(mu, 2+mu)
        class-0 (Y=0): Uniform(0, 2)
    Noise features V1..Vp: i.i.d. Normal(0, sig^2)

    Parameters
    ----------
    n1  : number of class-1 samples
    n2  : number of class-0 samples
    p   : number of noise features
    mu  : mean shift for X1
    sig : std dev for noise features

    Returns
    -------
    X : np.ndarray of shape (n1+n2, 1+p)
    y : np.ndarray of shape (n1+n2,)  — integer labels 1/0
    feature_names : list of str
    """
    rng = np.random.default_rng()
    X1_pos = rng.uniform(0 + mu, 2 + mu, size=n1)
    X1_neg = rng.uniform(0, 2, size=n2)
    X1 = np.concatenate([X1_pos, X1_neg])

    X_noise = rng.normal(0, sig, size=(n1 + n2, p))

    X = np.column_stack([X1, X_noise])
    y = np.concatenate([np.ones(n1, dtype=int), np.zeros(n2, dtype=int)])

    feature_names = ["X1"] + [f"V{i+1}" for i in range(p)]
    return X, y, feature_names


# ---------------------------------------------------------------------------
# DeLong paired CI
# ---------------------------------------------------------------------------

def _auc_var_delong(y_true, scores_a, scores_b):
    """
    Variance and covariance of two AUCs using the DeLong (1988) structural
    component method. Returns (var_a, var_b, cov_ab).
    """
    pos = scores_a[y_true == 1]
    neg = scores_a[y_true == 0]
    pos_b = scores_b[y_true == 1]
    neg_b = scores_b[y_true == 0]
    n1, n0 = len(pos), len(neg)

    def _kernel(x, y):
        if x > y:
            return 1.0
        elif x == y:
            return 0.5
        return 0.0

    # Structural components for model A
    V10_a = np.array([np.mean([_kernel(p, q) for q in neg]) for p in pos])
    V01_a = np.array([np.mean([_kernel(p, q) for p in pos]) for q in neg])
    # Structural components for model B
    V10_b = np.array([np.mean([_kernel(p, q) for q in neg_b]) for p in pos_b])
    V01_b = np.array([np.mean([_kernel(p, q) for p in pos_b]) for q in neg_b])

    var_a  = (np.var(V10_a, ddof=1) / n1 + np.var(V01_a, ddof=1) / n0)
    var_b  = (np.var(V10_b, ddof=1) / n1 + np.var(V01_b, ddof=1) / n0)
    cov_ab = (np.cov(V10_a, V10_b, ddof=1)[0, 1] / n1 +
              np.cov(V01_a, V01_b, ddof=1)[0, 1] / n0)
    return var_a, var_b, cov_ab


def delong_ci(y_true, scores_full, scores_redu, alpha: float = 0.05):
    """
    95 % DeLong confidence interval for AUC(full) − AUC(reduced).

    Returns (lower, upper).
    """
    auc_full = roc_auc_score(y_true, scores_full)
    auc_redu = roc_auc_score(y_true, scores_redu)
    diff = auc_full - auc_redu

    var_f, var_r, cov_fr = _auc_var_delong(y_true, scores_full, scores_redu)
    se_diff = np.sqrt(var_f + var_r - 2 * cov_fr)

    z = stats.norm.ppf(1 - alpha / 2)
    return diff - z * se_diff, diff + z * se_diff


# ---------------------------------------------------------------------------
# Model fitting and evaluation
# ---------------------------------------------------------------------------

def fit_and_evaluate_model(model_type: str,
                           X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           redu_col: int = 0,
                           nrounds: int = 100):
    """
    Fit full and reduced models of the given type, compute AUC and DeLong CI.

    Parameters
    ----------
    model_type : one of "lda", "logistic", "svm", "xgboost"
    X_train    : (n_train, n_features) array — full feature set
    y_train    : (n_train,) binary labels
    X_test     : (n_test, n_features) array
    y_test     : (n_test,) binary labels
    redu_col   : column index(es) for the reduced model (default 0 = X1)
    nrounds    : XGBoost boosting rounds

    Returns
    -------
    dict with keys: auc_full, auc_redu, auc_diff, delong_lower, delong_upper
    """
    redu_cols = [redu_col] if isinstance(redu_col, int) else list(redu_col)
    X_train_r = X_train[:, redu_cols]
    X_test_r  = X_test[:, redu_cols]

    if model_type == "lda":
        clf_full = LinearDiscriminantAnalysis()
        clf_redu = LinearDiscriminantAnalysis()
        clf_full.fit(X_train, y_train)
        clf_redu.fit(X_train_r, y_train)
        sc_full = clf_full.decision_function(X_test)
        sc_redu = clf_redu.decision_function(X_test_r)

    elif model_type == "logistic":
        clf_full = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf_redu = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf_full.fit(X_train, y_train)
        clf_redu.fit(X_train_r, y_train)
        sc_full = clf_full.predict_proba(X_test)[:, 1]
        sc_redu = clf_redu.predict_proba(X_test_r)[:, 1]

    elif model_type == "svm":
        clf_full = SVC(kernel="rbf", decision_function_shape="ovr")
        clf_redu = SVC(kernel="rbf", decision_function_shape="ovr")
        clf_full.fit(X_train, y_train)
        clf_redu.fit(X_train_r, y_train)
        sc_full = clf_full.decision_function(X_test)
        sc_redu = clf_redu.decision_function(X_test_r)

    elif model_type == "xgboost":
        dtrain_full = xgb.DMatrix(X_train, label=y_train)
        dtest_full  = xgb.DMatrix(X_test,  label=y_test)
        dtrain_redu = xgb.DMatrix(X_train_r, label=y_train)
        dtest_redu  = xgb.DMatrix(X_test_r,  label=y_test)

        params = {"objective": "binary:logistic", "verbosity": 0}
        model_full = xgb.train(params, dtrain_full, num_boost_round=nrounds,
                               verbose_eval=False)
        model_redu = xgb.train(params, dtrain_redu, num_boost_round=nrounds,
                               verbose_eval=False)
        sc_full = model_full.predict(dtest_full)
        sc_redu = model_redu.predict(dtest_redu)

    else:
        raise ValueError(f"Unknown model_type '{model_type}'. "
                         "Choose from: lda, logistic, svm, xgboost")

    auc_full = roc_auc_score(y_test, sc_full)
    auc_redu = roc_auc_score(y_test, sc_redu)
    dl_lower, dl_upper = delong_ci(y_test, sc_full, sc_redu)

    return {
        "auc_full":     auc_full,
        "auc_redu":     auc_redu,
        "auc_diff":     auc_full - auc_redu,
        "delong_lower": dl_lower,
        "delong_upper": dl_upper,
    }
