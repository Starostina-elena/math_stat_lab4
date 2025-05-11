import numpy as np
import scipy.stats as stats

variables = []
prices = []

with open('cars93.csv') as f:
    lines = f.readlines()
    for i in lines[1:]:
        line = i.strip().split(',')
        variables.append([float(line[6]), float(line[7]), float(line[12])])
        prices.append(float(line[4]))

X = np.array(variables)
y = np.array(prices)

X = np.hstack([np.ones((X.shape[0], 1)), X])

XtX = X.T @ X
Xty = X.T @ y
beta = np.linalg.solve(XtX, Xty)

y_pred = X @ beta
residuals = y - y_pred
sigma_squared = (residuals.T @ residuals) / (len(y) - X.shape[1])

total_variance = ((y - y.mean()).T @ (y - y.mean()))
r_squared = 1 - (residuals.T @ residuals) / total_variance

XtX_inv = np.linalg.inv(XtX)
se_beta = np.sqrt(np.diag(sigma_squared * XtX_inv))

n, p = X.shape
t_value = stats.t.ppf(1 - 0.05 / 2, df=n - p)

ci_beta = [(beta[i] - t_value * se_beta[i], beta[i] + t_value * se_beta[i]) for i in range(len(beta))]

chi2_lower = stats.chi2.ppf(0.05 / 2, df=n - p)
chi2_upper = stats.chi2.ppf(1 - 0.05 / 2, df=n - p)
ci_sigma_squared = ((n - p) * sigma_squared / chi2_upper, (n - p) * sigma_squared / chi2_lower)

print("Коэффициенты модели:", beta)
print("Доверительные интервалы для коэффициентов:")
for i, ci in enumerate(ci_beta):
    print(f"b{i}: {ci}")
print("Остаточная дисперсия:", sigma_squared)
print("Доверительный интервал для остаточной дисперсии:", ci_sigma_squared)
print("Коэффициент детерминации (R^2):", r_squared)
