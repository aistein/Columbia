from numpy.linalg import inv
import numpy as np
from scipy.stats import t

X = np.array(  [[1, 25, 162],
                [1, 25, 184],
                [1, 42, 166],
                [1, 55, 150],
                [1, 30, 192],
                [1, 40, 155],
                [1, 66, 184],
                [1, 60, 202],
                [1, 38, 174]])
Y = np.array([112, 144, 138, 145, 152, 110, 118, 160, 108])

n = X.shape[0]
k = X.shape[1] - 1
Y.shape = (n,1)
X_sq_inv = inv(np.matmul(X.T, X))
B = np.matmul( X_sq_inv, np.matmul(X.T, Y))
SS_R = np.matmul(Y.T, Y) - np.matmul(np.matmul(B.T, X.T), Y)

# part (b)
# calculate the t-statistic that we may test the second hypothesis with
# H0: When weight is known, Age gives no information in predicting blood pressure
# - i.e. B1 = 0
df = n - k - 1
T_crit = t.ppf(1-0.025, df)
T_val = np.sqrt(df/SS_R) * B[1] / np.sqrt(X_sq_inv[k,k])
p_value = 2*(1 - t.cdf(T_val, df)[0,0])
print("H0: When weight is known, Age doesn't affect blood pressure --> p-value = ", p_value)
print("--> T_obs = ", T_val)
print("--> T_crit = ", T_crit)
if T_val > T_crit:
    print("--Reject H0 at 5%--")
else:
    print("--Cannot Reject H0 at 5%--")

# part (a)
# H0: Age gives no information in predicting blood pressure
X1_sq_inv = inv(np.matmul(X[:, 0:2].T, X[:, 0:2]))
B = np.matmul( X1_sq_inv, np.matmul(X[:, 0:2].T, Y))
SS_R = np.matmul(Y.T, Y) - np.matmul(np.matmul(B.T, X[:, 0:2].T), Y)
df = n - k - 2
T_crit = t.ppf(1-0.025, df)
T_val = np.sqrt(df/SS_R) * B[1] / np.sqrt(X1_sq_inv[k-1, k-1])
p_value = 2*(1 - t.cdf(T_val, df)[0,0])
print("H0: Age doesn't affect blood pressure --> p-value = ", p_value)
print("--> T_obs = ", T_val)
print("--> T_crit = ", T_crit)
if T_val > T_crit:
    print("--Reject H0 at 5%--")
else:
    print("--Cannot Reject H0 at 5%--")

# The O.G. Way
n = X.shape[0]
x_hat = np.mean(X[:,1])
y_hat = np.mean(Y)

# calculate the linear model parameters SxY, Sxx, SYY, and use them to
# find the estimators A and B, the correlation, and the residuals
S_xY = S_xx = S_YY = 0.0
for i in range(0,n-1):
    S_xY += (X[i,1] - x_hat)*(Y[i] - y_hat)
    S_xx += (X[i,1] - x_hat)**2
    S_YY += (Y[i] - y_hat)**2
B = S_xY / S_xx
A = y_hat - B * x_hat
SS_R = (S_xx * S_YY - S_xY**2) / S_xx
R_sq = 1 - SS_R / S_YY
r = S_xY / (np.sqrt(S_xx * S_YY))
T_obs = np.sqrt((n-2)*S_xx/SS_R)*np.abs(B)
T_crit = t.ppf(1-0.025, n-2)
p_value = 2*(1 - t.cdf(T_obs, n-2))
print("---THE O.G. Method--")
print("H0: Age doesn't affect blood pressure --> p-value = ", p_value)
print("--> T_obs = ", T_obs)
print("--> T_crit = ", T_crit)
if T_val > T_crit:
    print("--Reject H0 at 5%--")
else:
    print("--Cannot Reject H0 at 5%--")
