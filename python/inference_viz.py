"""
inference_viz.py
================
Python translation of R/04_inference_viz.R

Provides inference summaries and visualizations for R×B simulation output.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional


def compute_inference(sim_result: dict,
                      ground_truth: Optional[dict] = None) -> dict:
    """
    Compute inference statistics from run_simulation_RxB() output.

    Parameters
    ----------
    sim_result    : output of run_simulation_RxB()
    ground_truth  : optional dict {delta_auc_<model>: float}
                    (e.g. from compute_ground_truth()["ground_truth"])

    Returns
    -------
    dict {model: {nosplit stats, split stats, coverage}}
    """
    model_types = sim_result["params"]["model_types"]
    R = sim_result["params"]["R"]
    B = sim_result["params"]["B"]
    N_total = R * B

    out = {}

    for m in model_types:
        ns = sim_result["nosplit"][m]
        sp = sim_result["split"][m]

        # --- No-split differences (R values) ---
        ns_diff = ns["auc_full"] - ns["auc_redu"]

        # --- Split differences (R x B matrix) ---
        sp_diff_mat = sp["auc_full"] - sp["auc_redu"]
        sp_diff_vec = sp_diff_mat.ravel()

        # Per-replicate bootstrap CI (quantiles over B columns)
        row_ci = np.quantile(sp_diff_mat, [0.025, 0.975], axis=1).T  # (R, 2)

        # Pooled CI from all R*B values
        pooled_ci = np.quantile(sp_diff_vec, [0.025, 0.975])

        # Mean DeLong CI (split)
        delong_ci_mean_sp = [
            float(np.mean(sp["delong_lower"])),
            float(np.mean(sp["delong_upper"])),
        ]

        # Mean DeLong CI (no-split)
        ns_delong_ci = [
            float(np.mean(ns["delong_lower"])),
            float(np.mean(ns["delong_upper"])),
        ]

        # --- Coverage rates ---
        cover_boot_split      = np.nan
        cover_delong_split    = np.nan
        cover_delong_nosplit  = np.nan

        if ground_truth is not None:
            gt_key = f"delta_auc_{m}"
            if gt_key in ground_truth:
                gt_val = ground_truth[gt_key]

                cover_boot_split = float(np.mean(
                    (row_ci[:, 0] <= gt_val) & (row_ci[:, 1] >= gt_val)
                ))
                cover_delong_split = float(np.mean(
                    (sp["delong_lower"] <= gt_val) &
                    (sp["delong_upper"] >= gt_val)
                ))
                cover_delong_nosplit = float(np.mean(
                    (ns["delong_lower"] <= gt_val) &
                    (ns["delong_upper"] >= gt_val)
                ))

        out[m] = {
            # No-split
            "nosplit_mean_diff": float(np.mean(ns_diff)),
            "nosplit_sd_diff":   float(np.std(ns_diff, ddof=1)),
            "nosplit_z":         float(np.mean(ns_diff) / np.std(ns_diff, ddof=1)),
            "nosplit_delong_ci": ns_delong_ci,
            # Split pooled
            "split_mean_diff":   float(np.mean(sp_diff_vec)),
            "split_sd_diff":     float(np.std(sp_diff_vec, ddof=1)),
            "split_z":           float(np.mean(sp_diff_vec) /
                                        np.std(sp_diff_vec, ddof=1)),
            "split_pooled_ci":   pooled_ci.tolist(),
            "split_row_ci_mean": row_ci.mean(axis=0).tolist(),
            # DeLong
            "split_delong_ci_mean": delong_ci_mean_sp,
            # Coverage
            "cover_boot_split":      cover_boot_split,
            "cover_delong_split":    cover_delong_split,
            "cover_delong_nosplit":  cover_delong_nosplit,
            "N_total": N_total,
        }

    return out


def print_inference_summary(inference: dict) -> None:
    """Print a formatted summary table of inference results."""
    print("\n====== Inference Summary (R*B pooled) ======\n")
    for m, info in inference.items():
        print(f"--- {m.upper()} ---")
        print(f"  No-split  : mean={info['nosplit_mean_diff']:.5f}  "
              f"sd={info['nosplit_sd_diff']:.5f}  z={info['nosplit_z']:.3f}")
        print(f"  No-split DeLong CI : "
              f"[{info['nosplit_delong_ci'][0]:.5f}, "
              f"{info['nosplit_delong_ci'][1]:.5f}]")
        print(f"  Split (pooled {info['N_total']}) : "
              f"mean={info['split_mean_diff']:.5f}  "
              f"sd={info['split_sd_diff']:.5f}  z={info['split_z']:.3f}")
        print(f"  Split pooled CI    : "
              f"[{info['split_pooled_ci'][0]:.5f}, "
              f"{info['split_pooled_ci'][1]:.5f}]")
        print(f"  Split DeLong CI (mean): "
              f"[{info['split_delong_ci_mean'][0]:.5f}, "
              f"{info['split_delong_ci_mean'][1]:.5f}]")
        if not np.isnan(info["cover_boot_split"]):
            print(f"  Coverage (boot split)    : {info['cover_boot_split']:.3f}")
            print(f"  Coverage (DeLong split)  : {info['cover_delong_split']:.3f}")
            print(f"  Coverage (DeLong nosplit): {info['cover_delong_nosplit']:.3f}")
        print()


def plot_auc_histograms(sim_result: dict,
                        ground_truth: Optional[dict] = None,
                        xlim: tuple = (-0.15, 0.15),
                        figsize_per_row: tuple = (12, 3)) -> plt.Figure:
    """
    Plot histograms of AUC differences for each model.

    Two columns per model:
      Left : No-split (biased) AUC differences
      Right: Split (debiased) AUC differences (R*B pooled)

    Parameters
    ----------
    sim_result    : output of run_simulation_RxB()
    ground_truth  : optional dict for vertical reference lines
    xlim          : x-axis range for all histograms
    figsize_per_row : figure size scaling per model row

    Returns
    -------
    matplotlib Figure
    """
    model_types = sim_result["params"]["model_types"]
    n_rows = len(model_types)

    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(figsize_per_row[0], figsize_per_row[1] * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, m in enumerate(model_types):
        ns = sim_result["nosplit"][m]
        sp = sim_result["split"][m]

        ns_diff = ns["auc_full"] - ns["auc_redu"]
        sp_diff = (sp["auc_full"] - sp["auc_redu"]).ravel()

        gt_val = None
        if ground_truth is not None:
            gt_val = ground_truth.get(f"delta_auc_{m}")

        # --- No-split ---
        ax = axes[row, 0]
        ax.hist(ns_diff, bins=25, density=True, color="#4C72B0", alpha=0.75,
                edgecolor="white", linewidth=0.4)
        ax.set_xlim(xlim)
        ax.set_xlabel("AUC Difference", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{m.upper()} — No-split (biased)", fontsize=11, fontweight="bold")
        ax.axvline(np.mean(ns_diff), color="red", linestyle="--",
                   linewidth=1.2, label=f"mean={np.mean(ns_diff):.4f}")
        if gt_val is not None:
            ax.axvline(gt_val, color="green", linestyle="-",
                       linewidth=1.2, label=f"truth={gt_val:.4f}")
        ax.legend(fontsize=8)

        # --- Split ---
        ax = axes[row, 1]
        ax.hist(sp_diff, bins=40, density=True, color="#DD8452", alpha=0.75,
                edgecolor="white", linewidth=0.4)
        ax.set_xlim(xlim)
        ax.set_xlabel("AUC Difference", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{m.upper()} — Split (debiased, R×B pooled)", fontsize=11,
                     fontweight="bold")
        ax.axvline(np.mean(sp_diff), color="red", linestyle="--",
                   linewidth=1.2, label=f"mean={np.mean(sp_diff):.4f}")
        if gt_val is not None:
            ax.axvline(gt_val, color="green", linestyle="-",
                       linewidth=1.2, label=f"truth={gt_val:.4f}")
        ax.legend(fontsize=8)

    fig.suptitle("Δ AUC Distributions: No-split vs Split", fontsize=13,
                 fontweight="bold", y=1.01)
    return fig


def plot_coverage_comparison(inference: dict,
                              ground_truth: Optional[dict] = None) -> plt.Figure:
    """
    Bar chart comparing coverage rates for bootstrap and DeLong CIs.
    """
    models = list(inference.keys())
    labels = ["Boot (split)", "DeLong (split)", "DeLong (no-split)"]

    cover_data = np.array([
        [inference[m]["cover_boot_split"],
         inference[m]["cover_delong_split"],
         inference[m]["cover_delong_nosplit"]]
        for m in models
    ])  # shape (n_models, 3)

    x = np.arange(len(models))
    width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(models)), 4),
                           constrained_layout=True)

    for k, (label, color) in enumerate(zip(labels, colors)):
        vals = cover_data[:, k]
        bars = ax.bar(x + k * width, vals, width, label=label, color=color,
                      alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0.95, color="black", linestyle="--", linewidth=1.2,
               label="Nominal 95 %")
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Coverage Rate")
    ax.set_title("CI Coverage Rates by Model and Method", fontweight="bold")
    ax.legend(fontsize=9)

    return fig
