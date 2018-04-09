import csv
import numpy as np
import matplotlib.pyplot as plt

# 1. Included is data on animal weight and brain weight.
#
# a) calculate the mean and standard deviation of the brain weight and the body weight of the animals.
# Calculate the correlation as well.  Build a linear model to explain brain weight
# (unobservable without killing the animal) as a function of body weight (observable).
#
# b) Plot a histogram of the residuals.  Plot the residuals as a function of body weight.
#
# c) Add a brotonsaurs to your data (google brain size and body weight) and rebuild your model.
# How did it change? Which has better R^2 ?
#
# d) Transform brain weight to log brain weight and repeat a & b.  Which is a better model
#
# e) Transform body weight to log body weight and repeat d

animal_data1 = np.genfromtxt('AnimalWeights.csv', delimiter=',')

# x - body_weight, y - brain_weight
n = animal_data1.shape[0]
x_hat = np.mean(animal_data1[:,0])
s_x = np.std(animal_data1[:,0])
y_hat = np.mean(animal_data1[:,1])
s_y = np.std(animal_data1[:,1])

# calculate the linear model parameters SxY, Sxx, SYY, and use them to
# find the estimators A and B, the correlation, and the residuals
S_xY = S_xx = S_YY = 0.0
for i in range(0,n-1):
    S_xY += (animal_data1[i,0] - x_hat)*(animal_data1[i,1] - y_hat)
    S_xx += (animal_data1[i,0] - x_hat)**2
    S_YY += (animal_data1[i,1] - y_hat)**2
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
print("r = ",r)
print("S_YY =", S_YY)
print("SS_R =", SS_R)

print("Linear model Y = A + Bx, with A=", A, " and B=", B)

# calculate and plot the normalized residuals
residuals = np.zeros((n),dtype=np.float32)
for i in range(0,n-1):
    Yi = animal_data1[i,1]
    Xi = animal_data1[i,0]
    residuals[i] = (Yi - A - B * Xi) / (SS_R / (n - 2))
