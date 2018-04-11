import csv
import numpy as np
import matplotlib.pyplot as plt

# 2. The included data is earnings as a function of height.
#
# a. transforms earnings to log earnings and build a linear model.
#
# b. what is the % increase in earnings per extra inch in height.
#
# c. cut your data set into two data sets, one for men and one for women and repeat (a) on each data set.
#
# d. what is the % increase in earnings per extra inch in heights by sex.
#
# e. can you combine your two linear models from (d) into one model. Hint use an indicator function
#
# f(sex) = 1 if sex = woman and 0 otherwise

earnings_data = np.genfromtxt('heights.csv', delimiter=',', dtype=np.float32)

# part a: transform earnings into log-earnings
for person in earnings_data:
    person[1] = np.log2(person[1])
    person[2] = 12 * person[2] + person[3] # convert to inches
    # print("height: ", person[2], "\" log-earn: ", person[1])

# x - height, y - log-earnings
n = earnings_data.shape[0]
x_hat = np.mean(earnings_data[:,2])
s_x = np.std(earnings_data[:,2])
y_hat = np.mean(earnings_data[:,1])
s_y = np.std(earnings_data[:,1])

# calculate the linear model parameters SxY, Sxx, SYY, and use them to
# find the estimators A and B, the correlation, and the residuals
S_xY = S_xx = S_YY = 0.0
for i in range(0,n-1):
    S_xY += (earnings_data[i,2] - x_hat)*(earnings_data[i,1] - y_hat)
    S_xx += (earnings_data[i,2] - x_hat)**2
    S_YY += (earnings_data[i,1] - y_hat)**2
B = S_xY / S_xx
A = y_hat - B * x_hat
SS_R = (S_xx * S_YY - S_xY**2) / S_xx
R_sq = 1 - SS_R / S_YY
r = S_xY / (np.sqrt(S_xx * S_YY))

print("x_hat =", x_hat)
print("s_x =", s_x)
print("y_hat =", y_hat)
print("s_y =", s_y)
print("R_sq =", R_sq)
print("r =",r)
print("S_YY =", S_YY)
print("SS_R =", SS_R)

print("Linear model Y = A + Bx, with A=", A, " and B=", B)

# calculate and plot the normalized residuals
residuals = np.zeros((n),dtype=np.float32)
for i in range(0,n-1):
    Yi = earnings_data[i,1]
    Xi = earnings_data[i,2]
    residuals[i] = (Yi - A - B * Xi) / (SS_R / (n - 2))

plt.subplot(2,1,1)
plt.plot(earnings_data[:,2], np.zeros((n)), color='red')
plt.scatter(earnings_data[:,2], residuals, color='green')
plt.ylabel("normalized residual")
plt.title("Linear Regression: Height vs. Earnings")

plt.subplot(2,1,2)
plt.plot(earnings_data[:,2], A + B * (earnings_data[:,2]), color='red')
plt.scatter(earnings_data[:,2], earnings_data[:,1], color='blue')
plt.ylabel("earnings (log-dollars)")
plt.xlabel("height (in)")

plt.savefig('residuals_2a.png')
plt.show()
