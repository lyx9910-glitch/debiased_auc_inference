###############################################################################
# demo_run_all.R - Demo runner to execute all analyses and produce output
#
# This file sources all R/ modules and runs each analysis section.
# Adjust R and B values below based on your timing budget.
###############################################################################

cat("==============================================================\n")
cat("  Delta AUC Nested Model - Complete Analysis Demo\n")
cat("==============================================================\n\n")

# Get the directory where this script lives
script_dir <- dirname(sys.frame(1)$ofile)
# If running interactively, set manually:
# script_dir <- "/Users/yuxuanliu/Desktop/Classification ROC, Statistical Tests and Cross Validation/debiased_auc_inference"

# Source all R modules
source(file.path(script_dir, "R", "00_packages.R"))
source(file.path(script_dir, "R", "01_model_functions.R"))
source(file.path(script_dir, "R", "02_ground_truth.R"))
source(file.path(script_dir, "R", "03_simulation_RxB.R"))
source(file.path(script_dir, "R", "04_inference_viz.R"))
source(file.path(script_dir, "R", "05_loo_cv.R"))
source(file.path(script_dir, "R", "06_real_data_ptb.R"))
source(file.path(script_dir, "R", "07_real_data_heart.R"))


###############################################################################
# 0. Simulation parameters
#    - Start with small R, B. Time one iteration, then scale up.
#    - The timing printout helps you decide reasonable R and B.
###############################################################################
p  <- 100    # number of noise features
mu <- 1      # signal mean shift
sig <- 10    # noise SD
snr <- (mu - 0)^2 / sig^2
cat(sprintf("SNR = %.4f\n\n", snr))


###############################################################################
# 1. Ground Truth
###############################################################################
cat("\n==================== 1. GROUND TRUTH ====================\n")

# Setting 1: n_train = n_test = 2000
ground_truth_1 <- compute_ground_truth(
  n1 = 1000, n2 = 1000,
  n1_test = 1000, n2_test = 1000,
  R = 10000,                        # large R for accurate ground truth
  p = p, mu = mu, sig = sig,
  model_types = c("lda", "logistic", "svm", "xgboost")
)
cat("Ground truth setting 1 (n_train=2000, n_test=2000):\n")
print(ground_truth_1$ground_truth)

# Setting 2: n_train = 1500, n_test = 2000
ground_truth_2 <- compute_ground_truth(
  n1 = 750, n2 = 750,
  n1_test = 1000, n2_test = 1000,
  R = 10000,
  p = p, mu = mu, sig = sig,
  model_types = c("lda", "logistic", "svm", "xgboost")
)
cat("\nGround truth setting 2 (n_train=1500, n_test=2000):\n")
print(ground_truth_2$ground_truth)


###############################################################################
# 2. R x B Simulation
###############################################################################
cat("\n==================== 2. R x B SIMULATION ====================\n")

# Start small to time. Adjust R and B based on the timing output.
# A single iter is timed automatically; estimated total is printed.
sim_result <- run_simulation_RxB(
  n1 = 1000, n2 = 1000,
  R = 50, B = 20,                   # small for demo; increase as needed
  p = p, mu = mu, sig = sig,
  test_frac = 0.25,
  model_types = c("lda", "logistic", "svm", "xgboost"),
  nrounds = 100,
  reduced_formula = "Y ~ X1"
)


###############################################################################
# 3. Inference & Visualization (R*B total)
###############################################################################
cat("\n==================== 3. INFERENCE & VIZ ====================\n")

# Compute inference using ground truth from setting 1
inference <- compute_inference(sim_result,
                               ground_truth = ground_truth_1$ground_truth)
print_inference_summary(inference)

# Histograms
plot_auc_histograms(sim_result, xlim_range = c(-0.15, 0.15))


###############################################################################
# 4. Leave-One-Out Cross Validation
###############################################################################
cat("\n==================== 4. LOO-CV ====================\n")

loo_result <- run_loo_cv_simulation(
  n1 = 1000, n2 = 1000,
  R = 50,                            # small for demo
  p = p, mu = mu, sig = sig,
  reduced_formula = "Y ~ X1"
)
print_loo_summary(loo_result)
plot_loo_histograms(loo_result)


###############################################################################
# 5a. Real Data Application - Preterm Birth
###############################################################################
cat("\n==================== 5a. PTB DATA ====================\n")

# NOTE: Requires SELR package with ptb dataset loaded
if (exists("ptb")) {
  ptb_result <- run_ptb_analysis(
    R = 200,                          # small for demo
    model_types = c("lda", "logistic", "svm", "xgboost"),
    nrounds = 100
  )
  print_ptb_summary(ptb_result)
} else {
  cat("PTB dataset not found. Skipping preterm birth analysis.\n")
  cat("Make sure the SELR package is loaded and 'ptb' is available.\n")
}


###############################################################################
# 5b. Real Data Application - Framingham Heart
###############################################################################
cat("\n==================== 5b. HEART DATA ====================\n")

# NOTE: Requires riskCommunicator package with framingham dataset
if (exists("framingham")) {
  heart_result <- run_heart_analysis(R = 200)  # small for demo
  print_heart_summary(heart_result)
} else {
  cat("Framingham dataset not found. Skipping heart data analysis.\n")
  cat("Make sure the riskCommunicator package is loaded.\n")
}


cat("\n==============================================================\n")
cat("  Demo complete.\n")
cat("==============================================================\n")
