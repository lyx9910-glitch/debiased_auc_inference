"""
simulation_rxb.py
=================
Python translation of R/03_simulation_RxB.R

Runs the R × B simulation:
  - R outer Monte Carlo replicates
  - B resample-split iterations per replicate
  - Computes both "no-split" (biased) and "split" (debiased) AUC differences
"""

import time
import numpy as np
from typing import List, Optional

from .model_functions import generate_sim_data, fit_and_evaluate_model


def run_simulation_RxB(
    n1: int = 250, n2: int = 250,
    R: int = 30, B: int = 30,
    p: int = 100, mu: float = 1.0, sig: float = 10.0,
    test_frac: float = 0.25,
    model_types: List[str] = ("lda", "logistic"),
    nrounds: int = 100,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """
    R × B bootstrap simulation for nested AUC difference inference.

    For each MC replicate i = 1..R:
      1. Generate fresh data (n1 + n2 observations) → df_ori
      2. No-split: fit models on full data, evaluate on same data (biased).
      3. For j = 1..B:
           a. Resample df_ori with replacement → bootstrap sample
           b. Split into train (75%) / test (25%)
           c. Fit models on train, evaluate on test → unbiased split estimate

    Parameters
    ----------
    n1, n2      : class-1 / class-0 sample sizes
    R           : Monte Carlo replicates
    B           : resample-split iterations per replicate
    p           : noise features
    mu          : X1 mean shift
    sig         : noise std dev
    test_frac   : fraction used as test set in each split
    model_types : list of model keys
    nrounds     : XGBoost rounds
    seed        : global random seed
    verbose     : print progress

    Returns
    -------
    dict with keys:
        "nosplit"  : {model: {"auc_full": (R,), "auc_redu": (R,),
                              "delong_lower": (R,), "delong_upper": (R,)}}
        "split"    : {model: {"auc_full": (R,B), "auc_redu": (R,B),
                              "delong_lower": (R,B), "delong_upper": (R,B)}}
        "params"   : dict of run parameters
        "time_single" : float (seconds for one R×B cell)
        "time_total"  : float (total seconds)
    """
    if seed is not None:
        np.random.seed(seed)

    model_types = list(model_types)
    n_total = n1 + n2
    n_test  = max(1, round(n_total * test_frac))

    # ---- Storage ---------------------------------------------------------
    nosplit = {
        m: {
            "auc_full":     np.zeros(R),
            "auc_redu":     np.zeros(R),
            "delong_lower": np.zeros(R),
            "delong_upper": np.zeros(R),
        }
        for m in model_types
    }
    split_res = {
        m: {
            "auc_full":     np.zeros((R, B)),
            "auc_redu":     np.zeros((R, B)),
            "delong_lower": np.zeros((R, B)),
            "delong_upper": np.zeros((R, B)),
        }
        for m in model_types
    }

    # ---- Timing one iteration --------------------------------------------
    t0 = time.time()
    _X, _y, _ = generate_sim_data(n1, n2, p, mu, sig)
    for m in model_types:
        fit_and_evaluate_model(m, _X, _y, _X, _y, nrounds=nrounds)
    _idx = np.random.choice(n_total, n_test, replace=False)
    _mask = np.ones(n_total, dtype=bool); _mask[_idx] = False
    for m in model_types:
        fit_and_evaluate_model(m, _X[_mask], _y[_mask],
                               _X[_idx], _y[_idx], nrounds=nrounds)
    t_single = time.time() - t0
    est_total = t_single * R * B
    if verbose:
        print(f"  Single iteration: {t_single:.2f} sec")
        print(f"  Estimated total  (R={R}, B={B}): "
              f"{est_total:.1f} sec ({est_total/60:.1f} min)")

    # ---- Main loop -------------------------------------------------------
    t_start = time.time()

    for i in range(R):
        if verbose and (i + 1) % max(1, R // 5) == 0:
            print(f"  MC replicate {i+1} / {R}")

        X_ori, y_ori, _ = generate_sim_data(n1, n2, p, mu, sig)

        # No-split: evaluate on training data (biased)
        for m in model_types:
            res_ns = fit_and_evaluate_model(
                m, X_ori, y_ori, X_ori, y_ori, nrounds=nrounds)
            nosplit[m]["auc_full"][i]     = res_ns["auc_full"]
            nosplit[m]["auc_redu"][i]     = res_ns["auc_redu"]
            nosplit[m]["delong_lower"][i] = res_ns["delong_lower"]
            nosplit[m]["delong_upper"][i] = res_ns["delong_upper"]

        # Split: resample + train/test split B times
        for j in range(B):
            boot_idx = np.random.choice(n_total, n_total, replace=True)
            X_boot   = X_ori[boot_idx]
            y_boot   = y_ori[boot_idx]

            test_idx  = np.random.choice(n_total, n_test, replace=False)
            train_mask = np.ones(n_total, dtype=bool)
            train_mask[test_idx] = False

            X_tr, y_tr = X_boot[train_mask], y_boot[train_mask]
            X_te, y_te = X_boot[test_idx],   y_boot[test_idx]

            for m in model_types:
                res_sp = fit_and_evaluate_model(
                    m, X_tr, y_tr, X_te, y_te, nrounds=nrounds)
                split_res[m]["auc_full"][i, j]     = res_sp["auc_full"]
                split_res[m]["auc_redu"][i, j]     = res_sp["auc_redu"]
                split_res[m]["delong_lower"][i, j] = res_sp["delong_lower"]
                split_res[m]["delong_upper"][i, j] = res_sp["delong_upper"]

    t_total = time.time() - t_start
    if verbose:
        print(f"Simulation completed in {t_total:.1f} sec ({t_total/60:.1f} min).")

    return {
        "nosplit":     nosplit,
        "split":       split_res,
        "params":      dict(n1=n1, n2=n2, R=R, B=B, p=p, mu=mu, sig=sig,
                            test_frac=test_frac, model_types=model_types),
        "time_single": t_single,
        "time_total":  t_total,
    }
