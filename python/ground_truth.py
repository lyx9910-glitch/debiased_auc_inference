"""
ground_truth.py
===============
Python translation of R/02_ground_truth.R

Computes the ground-truth Δ AUC via large-scale Monte Carlo:
  - Generates fresh train + test sets for each replicate
  - Fits models once per replicate on training data
  - Averages split (held-out) AUC differences as the ground truth
"""

import numpy as np
from typing import List, Optional

from .model_functions import generate_sim_data, fit_and_evaluate_model


def compute_ground_truth(
    n1: int, n2: int,
    n1_test: Optional[int] = None,
    n2_test: Optional[int] = None,
    R: int = 10_000,
    p: int = 100,
    mu: float = 1.0,
    sig: float = 10.0,
    model_types: List[str] = ("lda", "logistic"),
    nrounds: int = 100,
    seed: Optional[int] = None,
) -> dict:
    """
    Estimate ground-truth Δ AUC for each model via Monte Carlo.

    For each replicate i = 1..R:
      1. Generate independent train and test data.
      2. Fit full and reduced models on train.
      3. Evaluate AUC on held-out test set.
    Ground truth = mean(AUC_full) − mean(AUC_redu) across all replicates.

    Parameters
    ----------
    n1, n2        : class-1 / class-0 training sample sizes
    n1_test, n2_test : test sample sizes (defaults to n1, n2)
    R             : number of Monte Carlo replicates
    p             : number of noise features
    mu            : mean shift for informative feature X1
    sig           : std dev for noise features
    model_types   : which models to evaluate
    nrounds       : XGBoost boosting rounds
    seed          : random seed for reproducibility

    Returns
    -------
    dict with:
        "ground_truth"  : dict {delta_auc_<model>: float}
        "auc_full_all"  : dict {model: np.ndarray of shape (R,)}
        "auc_redu_all"  : dict {model: np.ndarray of shape (R,)}
    """
    if n1_test is None:
        n1_test = n1
    if n2_test is None:
        n2_test = n2

    if seed is not None:
        np.random.seed(seed)

    model_types = list(model_types)

    auc_full_all = {m: np.zeros(R) for m in model_types}
    auc_redu_all = {m: np.zeros(R) for m in model_types}

    for i in range(R):
        if (i + 1) % 100 == 0:
            print(f"Ground truth: rep {i+1} / {R}")

        X_train, y_train, _ = generate_sim_data(n1, n2, p=p, mu=mu, sig=sig)
        X_test,  y_test,  _ = generate_sim_data(n1_test, n2_test, p=p, mu=mu, sig=sig)

        for m in model_types:
            res = fit_and_evaluate_model(
                model_type=m,
                X_train=X_train, y_train=y_train,
                X_test=X_test,   y_test=y_test,
                nrounds=nrounds,
            )
            auc_full_all[m][i] = res["auc_full"]
            auc_redu_all[m][i] = res["auc_redu"]

    ground_truth = {
        f"delta_auc_{m}": float(np.mean(auc_full_all[m]) - np.mean(auc_redu_all[m]))
        for m in model_types
    }

    return {
        "ground_truth": ground_truth,
        "auc_full_all": auc_full_all,
        "auc_redu_all": auc_redu_all,
    }
