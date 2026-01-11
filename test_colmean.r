x = matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3, byrow = TRUE)
print(x)
print(colMeans(x))
print(rowMeans(x))


quantile_ci <- t(apply(x, 1, quantile, probs = c(0.05, 0.95)))
print(quantile_ci)