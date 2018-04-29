import numpy as np
from scipy.stats import t

x = [-3, -2, -1, 0, 1, 2, 3]
sims = np.zeros(1000)

n = 7
x_hat = np.mean(x)
S_xx = 0.0
for i in range(0,n):
    S_xx += (x[i] - x_hat)**2

for i in range(0,1000):
    S_xY = S_YY = SS_R = 0.0
    y = np.zeros(n)
    for j in range(0,n):
        mu = 0.0
        sigma = 0.3
        epsilon = np.random.normal(mu,sigma,1)
        y[j] = 5 * x[j] + epsilon[0]

    y_hat = np.mean(y)
    for j in range (0,n):
        S_xY += (x[j] - x_hat)*(y[j] - y_hat)
        S_YY += (y[j] - y_hat)**2

    B = S_xY / S_xx
    A = y_hat - B * x_hat
    # for j in range(0,n):
    #     SS_R += (y[j] - A - B * x[j])**2
    SS_R = (S_xx * S_YY - S_xY**2) / S_xx

    SE = np.sqrt(SS_R/((n-2)*S_xx))
    t_val = t.ppf(1-0.025, n-2)
    interval_min = B - SE * t_val
    interval_max = B + SE * t_val

    print("interval: ", interval_min, ", ", interval_max)
    if (interval_min <= 5 and interval_max >= 5):
        sims[i] = 1

count = 0
for i in range (0,1000):
    count += sims[i]
if count/1000.0 >= 0.95:
    print("We captured the true mean at least 95% of the time (",count/10,"%)")
else:
    print("we did NOT capture the true mean at least 95% of the time (",count/10,"%)")
