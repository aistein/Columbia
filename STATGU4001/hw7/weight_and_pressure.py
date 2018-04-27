import csv
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

pressure_data = np.genfromtxt('weight_and_pressure.csv', delimiter=',')

n = pressure_data.shape[0]
x_hat = np.mean(pressure_data[:,0])
s_x = np.std(pressure_data[:,0])
y_hat = np.mean(pressure_data[:,1])
s_y = np.std(pressure_data[:,1])

# calculate the linear model parameters SxY, Sxx, SYY, and use them to
# find the estimators A and B, the correlation, and the residuals
S_xY = S_xx = S_YY = 0.0
for i in range(0,n-1):
    S_xY += (pressure_data[i,0] - x_hat)*(pressure_data[i,1] - y_hat)
    S_xx += (pressure_data[i,0] - x_hat)**2
    S_YY += (pressure_data[i,1] - y_hat)**2
B = S_xY / S_xx
A = y_hat - B * x_hat
SS_R = (S_xx * S_YY - S_xY**2) / S_xx
R_sq = 1 - SS_R / S_YY
r = S_xY / (np.sqrt(S_xx * S_YY))
T_obs = np.sqrt((n-2)*S_xx/SS_R)*np.abs(B)
T_crit = t.ppf(1-0.05, n-2)
p_value = t.cdf(T_obs, n-2)

print("x_hat =", x_hat)
print("s_x =", s_x)
print("y_hat =", y_hat)
print("s_y =", s_y)
print("R_sq =", R_sq)
print("r =",r)
print("S_YY =", S_YY)
print("SS_R =", SS_R)

print("Linear model Y = A + Bx, with A=", A, " and B=", B)
print("\nNull Hypotheseis: weight does not affect systolic blood pressure")
print("--> T_obs = ", T_obs)
print("--> T_crit = ", T_crit)
print("--> P_value = ", p_value)
if (T_obs > T_crit):
    print(" -- Reject the Null -- ")
else:
    print(" -- Do not Recejt the Null -- \n")

# SE of response estimator at the 182lb level, 95% confidence interval
SE_182 = np.sqrt(1/n + (185-x_hat)**2/S_xx) * np.sqrt(SS_R/(n-2))
T_val = t.ppf((1-0.025), n-2)
print("\nWe are 95% confident that for a sample of a large number of 182lb males")
print("Systolic blood pressure will be: ", A + B*182," +/- ", T_val * SE_182)


# calculate and plot the normalized residuals
residuals = np.zeros((n),dtype=np.float32)
for i in range(0,n-1):
    Yi = pressure_data[i,1]
    Xi = pressure_data[i,0]
    residuals[i] = (Yi - A - B * Xi) / (SS_R / (n - 2))

plt.subplot(2,1,1)
plt.plot(pressure_data[:,0], np.zeros((n)), color='red')
plt.scatter(pressure_data[:,0], residuals, color='green')
plt.ylabel("normalized residual")
plt.title("Linear Regression: Weight vs. Blood Pressure")

plt.subplot(2,1,2)
plt.plot(pressure_data[:,0], A + B * (pressure_data[:,0]), color='red')
plt.scatter(pressure_data[:,0], pressure_data[:,1], color='blue')
plt.ylabel("systolic blood pressure")
plt.xlabel("weight (lbs)")

plt.savefig('weight_vs_pressure.png')
plt.show()
