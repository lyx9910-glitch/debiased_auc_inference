# Inference function
compute_inference <- function(delta_auc_matrix, delta_auc_true, 
                             delong_lower, delong_upper, 
                             probs = c(0.05, 0.95)) {
  # Get number of repetitions
  R <- nrow(delta_auc_matrix)
  
  # Compute quantile-based CI for each repetition
  quantile_ci <- t(apply(delta_auc_matrix, 1, quantile, probs = probs))
  
  # Compute DeLong CI means
  delong_ci_mean <- c(mean(delong_lower), mean(delong_upper))
  
  # Compute quantile CI means
  quantile_ci_mean <- colMeans(quantile_ci)
  
  # Compute coverage rate for quantile method
  coverage_quantile <- mean((quantile_ci[, 2] >= delta_auc_true) & 
                          (quantile_ci[, 1] <= delta_auc_true))
  
  # Compute coverage rate for DeLong method
  coverage_delong_RB <- mean((delong_upper >= delta_auc_true) & 
                        (delong_lower <= delta_auc_true))
  
  # Compute coverage rate for DeLong method. Average over bootstrap
  delong_lower_B_avg <- RowMeans(delong_lower)
  delong_upper_B_avg <- RowMeans(delong_upper)
  coverage_delong_B <- mean((delong_lower_B_avg <= delta_auc_true) & 
                          (delong_upper_B_avg >= delta_auc_true))
  
  # Return results as a list
  return(list(
    coverage_quantile = coverage_quantile,
    coverage_delong_RB = coverage_delong_RB,
    coverage_delong_B = coverage_delong_B,
    ci_mean_quantile = quantile_ci_mean,
    ci_mean_delong = delong_ci_mean
  ))
}

# lda["non-split"]["AUC.diff"]
# lda["non-split"]["delta.AUC.true"]

# auc.matrix["lda"]["non-split"]

# Apply inference to all model combinations
# Define model names and scenarios
models <- c("lda", "logi", "svm", "xgb")
scenarios <- c("nosplit", "split")

# Create results storage
results <- list()

# Loop over all combinations
for (model in models) {
  for (scenario in scenarios) {
    # Construct variable names
    if (scenario == "nosplit") {
      delta_auc_var <- paste0("auc.", model, ".nested.diff")
      delta_true_var <- paste0("auc.", model, ".nested.test.diff.true.1")
    } else {
      delta_auc_var <- paste0("auc.", model, ".nested.test.diff")
      delta_true_var <- paste0("auc.", model, ".nested.test.diff.true.2")
    }
    
    delong_lower_var <- paste0("delong.lower.", scenario, ".", model)
    delong_upper_var <- paste0("delong.upper.", scenario, ".", model)
    
    # Get the actual data
    delta_auc_matrix <- get(delta_auc_var)
    delta_auc_true <- get(delta_true_var)
    delong_lower <- get(delong_lower_var)
    delong_upper <- get(delong_upper_var)
    
    # Compute inference
    result_name <- paste0(model, ".", scenario)
    results[[result_name]] <- compute_inference(
      delta_auc_matrix = delta_auc_matrix,
      delta_auc_true = delta_auc_true,
      delong_lower = delong_lower,
      delong_upper = delong_upper
    )
  }
}


# Print results in organized format
cat("\n=== INFERENCE RESULTS ===\n\n")

for (model in models) {
  cat("\n", toupper(model), "Model:\n", sep = "")
  cat(rep("-", 60), "\n", sep = "")
  
  for (scenario in scenarios) {
    result_name <- paste0(model, ".", scenario)
    res <- results[[result_name]]
    
    cat("\n", toupper(scenario), " Scenario:\n", sep = "")
    cat("  Quantile Method:\n")
    cat("    Coverage Rate: ", round(res$coverage_quantile, 4), "\n")
    cat("    Mean CI: [", round(res$ci_mean_quantile[1], 4), ", ", 
        round(res$ci_mean_quantile[2], 4), "]\n", sep = "")
    
    cat("  DeLong Method:\n")
    cat("    Coverage Rate: ", round(res$coverage_delong, 4), "\n")
    cat("    Mean CI: [", round(res$ci_mean_delong[1], 4), ", ", 
        round(res$ci_mean_delong[2], 4), "]\n", sep = "")
  }
  cat("\n")
}


# Optionally, create a summary table
create_summary_table <- function(results, models, scenarios) {
  n_rows <- length(models) * length(scenarios)
  
  summary_df <- data.frame(
    Model = character(n_rows),
    Scenario = character(n_rows),
    Coverage_Quantile = numeric(n_rows),
    Coverage_DeLong = numeric(n_rows),
    CI_Lower_Quantile = numeric(n_rows),
    CI_Upper_Quantile = numeric(n_rows),
    CI_Lower_DeLong = numeric(n_rows),
    CI_Upper_DeLong = numeric(n_rows),
    stringsAsFactors = FALSE
  )
  
  idx <- 1
  for (model in models) {
    for (scenario in scenarios) {
      result_name <- paste0(model, ".", scenario)
      res <- results[[result_name]]
      
      summary_df[idx, ] <- list(
        Model = model,
        Scenario = scenario,
        Coverage_Quantile = res$coverage_quantile,
        Coverage_DeLong = res$coverage_delong,
        CI_Lower_Quantile = res$ci_mean_quantile[1],
        CI_Upper_Quantile = res$ci_mean_quantile[2],
        CI_Lower_DeLong = res$ci_mean_delong[1],
        CI_Upper_DeLong = res$ci_mean_delong[2]
      )
      idx <- idx + 1
    }
  }
  
  return(summary_df)
}

# Create and display summary table
summary_table <- create_summary_table(results, models, scenarios)
print(summary_table)