###############################################################################
# 04_inference_viz.R - Inference and Visualization for R x B simulation
#
# Key changes from original:
#   - Inference now uses ALL R*B observations (pooled) rather than per-row CI.
#   - Also provides per-replicate (row) bootstrap CI for coverage analysis.
#   - Functions for summary tables, histograms, and coverage rates.
###############################################################################

#' Compute inference results from R x B simulation output
#'
#' @param sim_result  Output from run_simulation_RxB()
#' @param ground_truth Named vector of true delta-AUC values (optional)
#' @return List with summary statistics, CIs, and coverage rates
compute_inference <- function(sim_result, ground_truth = NULL) {

  model_types <- sim_result$params$model_types
  R <- sim_result$params$R
  B <- sim_result$params$B
  N_total <- R * B  # total split samples

  out <- list()

  for (m in model_types) {
    # --- Non-split: R values ---
    ns_diff <- sim_result$nosplit[[m]]$auc_full -
               sim_result$nosplit[[m]]$auc_redu

    # --- Split: R x B matrix -> pooled R*B values ---
    sp_diff_mat <- sim_result$split[[m]]$auc_full -
                   sim_result$split[[m]]$auc_redu
    sp_diff_vec <- as.vector(sp_diff_mat)  # pool all R*B

    # Per-replicate bootstrap CI (each row = B bootstrap samples)
    row_ci <- t(apply(sp_diff_mat, 1, quantile,
                      probs = c(0.025, 0.975), na.rm = TRUE))

    # Pooled CI from all R*B split differences
    pooled_ci <- quantile(sp_diff_vec, probs = c(0.025, 0.975), na.rm = TRUE)

    # DeLong CI (split): per-replicate means
    delong_lower_mat <- sim_result$split[[m]]$delong_lower
    delong_upper_mat <- sim_result$split[[m]]$delong_upper
    # Average DeLong CI across all R*B
    delong_ci_mean <- c(mean(delong_lower_mat), mean(delong_upper_mat))

    # Non-split DeLong CI
    ns_delong_ci <- c(mean(sim_result$nosplit[[m]]$delong_lower),
                      mean(sim_result$nosplit[[m]]$delong_upper))

    # Coverage rates (if ground truth provided)
    cover_boot_split <- NA
    cover_delong_split <- NA
    cover_delong_nosplit <- NA
    if (!is.null(ground_truth)) {
      gt_key <- paste0("delta_auc_", m)
      if (gt_key %in% names(ground_truth)) {
        gt_val <- ground_truth[[gt_key]]

        # Bootstrap CI coverage (per-replicate)
        cover_boot_split <- mean(
          (row_ci[, 1] <= gt_val) & (row_ci[, 2] >= gt_val)
        )

        # DeLong CI coverage (per R*B entry)
        cover_delong_split <- mean(
          (delong_lower_mat <= gt_val) & (delong_upper_mat >= gt_val)
        )

        # Non-split DeLong coverage
        cover_delong_nosplit <- mean(
          (sim_result$nosplit[[m]]$delong_lower <= gt_val) &
          (sim_result$nosplit[[m]]$delong_upper >= gt_val)
        )
      }
    }

    out[[m]] <- list(
      # Non-split summary
      nosplit_mean_diff = mean(ns_diff),
      nosplit_sd_diff   = sd(ns_diff),
      nosplit_z         = mean(ns_diff) / sd(ns_diff),
      nosplit_delong_ci = ns_delong_ci,

      # Split summary (pooled R*B)
      split_mean_diff     = mean(sp_diff_vec),
      split_sd_diff       = sd(sp_diff_vec),
      split_z             = mean(sp_diff_vec) / sd(sp_diff_vec),
      split_pooled_ci     = pooled_ci,

      # Per-replicate bootstrap CI
      split_row_ci_mean   = colMeans(row_ci),

      # DeLong
      split_delong_ci_mean = delong_ci_mean,

      # Coverage
      cover_boot_split     = cover_boot_split,
      cover_delong_split   = cover_delong_split,
      cover_delong_nosplit = cover_delong_nosplit,

      # Total number of inference samples
      N_total = N_total
    )
  }

  return(out)
}


#' Print a summary table of inference results
#'
#' @param inference  Output from compute_inference()
print_inference_summary <- function(inference) {
  cat("\n====== Inference Summary (R*B pooled) ======\n\n")
  for (m in names(inference)) {
    info <- inference[[m]]
    cat(sprintf("--- %s ---\n", toupper(m)))
    cat(sprintf("  Non-split  : mean=%.5f  sd=%.5f  z=%.3f\n",
                info$nosplit_mean_diff, info$nosplit_sd_diff, info$nosplit_z))
    cat(sprintf("  Non-split DeLong CI: [%.5f, %.5f]\n",
                info$nosplit_delong_ci[1], info$nosplit_delong_ci[2]))
    cat(sprintf("  Split (pooled %d) : mean=%.5f  sd=%.5f  z=%.3f\n",
                info$N_total, info$split_mean_diff,
                info$split_sd_diff, info$split_z))
    cat(sprintf("  Split pooled CI : [%.5f, %.5f]\n",
                info$split_pooled_ci[1], info$split_pooled_ci[2]))
    cat(sprintf("  Split DeLong CI (mean): [%.5f, %.5f]\n",
                info$split_delong_ci_mean[1], info$split_delong_ci_mean[2]))
    if (!is.na(info$cover_boot_split)) {
      cat(sprintf("  Coverage (boot split)   : %.3f\n",
                  info$cover_boot_split))
      cat(sprintf("  Coverage (DeLong split) : %.3f\n",
                  info$cover_delong_split))
      cat(sprintf("  Coverage (DeLong nosplit): %.3f\n",
                  info$cover_delong_nosplit))
    }
    cat("\n")
  }
}


#' Plot histograms of AUC differences for all models
#'
#' @param sim_result  Output from run_simulation_RxB()
#' @param xlim_range  x-axis limits for histograms (default c(-0.15, 0.15))
plot_auc_histograms <- function(sim_result, xlim_range = c(-0.15, 0.15)) {
  model_types <- sim_result$params$model_types
  n_models    <- length(model_types)

  par(mfrow = c(n_models, 2), mar = c(4, 4, 3, 1))

  for (m in model_types) {
    # Non-split difference
    ns_diff <- sim_result$nosplit[[m]]$auc_full -
               sim_result$nosplit[[m]]$auc_redu
    hist(ns_diff, freq = FALSE, breaks = 20,
         main = paste0(toupper(m), " - No Split"),
         xlab = "AUC Difference", xlim = xlim_range)

    # Split difference (pooled)
    sp_diff <- as.vector(
      sim_result$split[[m]]$auc_full - sim_result$split[[m]]$auc_redu
    )
    hist(sp_diff, freq = FALSE, breaks = 30,
         main = paste0(toupper(m), " - Split"),
         xlab = "AUC Difference", xlim = xlim_range)
  }
}
