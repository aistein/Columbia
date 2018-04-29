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
# calculate the t-statistic that we may test the first hypothesis with
# H0: When weight is known, Age gives no information in predicting blood pressure
# - i.e. B1 = 0
df = n - k - 1
T_val = np.sqrt( (df/SS_R * B[1]) / np.sqrt(X_sq_inv[k,k]) )
p_value = t.cdf(T_val, df)[0,0]
print("H0: When weight is known, Age doesn't affect blood pressure --> p-value = ", p_value)

# part (a)
# H0: Age gives no information in predicting blood pressure
X1_sq_inv = inv(np.matmul(X[:, 0:2].T, X[:, 0:2]))
B = np.matmul( X1_sq_inv, np.matmul(X[:, 0:2].T, Y))
SS_R = np.matmul(Y.T, Y) - np.matmul(np.matmul(B.T, X[:, 0:2].T), Y)
df = n - k - 2
T_val = np.sqrt( (df/SS_R * B[1]) / np.sqrt(X1_sq_inv[k-1, k-1]) )
p_value = t.cdf(T_val, df)[0,0]
print("H0: Age doesn't affect blood pressure --> p-value = ", p_value)
