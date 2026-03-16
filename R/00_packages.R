###############################################################################
# 00_packages.R - Load all required libraries
###############################################################################

required_packages <- c(
  "e1071",
  "MASS",
  "randomForest",
  "xgboost",
  "pROC",
  "mvtnorm",
  "SELR",
  "riskCommunicator"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("Package '%s' not installed. Please install it.", pkg))
  } else {
    suppressPackageStartupMessages(library(pkg, character.only = TRUE))
  }
}

cat("All packages loaded.\n")
