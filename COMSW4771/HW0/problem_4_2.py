import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import linalg as LA

L = np.matrix('1.25 -1.5; -1.5 5')

print L

# 500 2-vectors from Gaussian distribution, normalized
R = np.random.normal(0,1,(500,2))
for i, row in enumerate(R):
    R[i,:] = row / np.linalg.norm(row)

# Distorted Vectors
R_hat = np.dot(R,L)

# Eigenvalues of L
e_vals, e_vecs = LA.eig(L)
lambda_max = e_vals.max()
lambda_min = e_vals.min()
v_max = e_vecs[1];
print "Lambda Max: ", lambda_max
print "V_max: ", v_max
print "Lambda Min: ", lambda_min

# Magnitudes of R_hat
R_hat_mag = np.array([np.linalg.norm(row) for row in R_hat])

# Histogram of Distorted Vectors' Magnitudes
print "histogramming Distorted Vectors Magnitudes"
plt.hist(R_hat_mag, rwidth=0.1, bins=np.arange(0.6, 5.7, 0.1))
plt.title("Histogram of Distorted Vectors")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# Plot of All the Distorted Vectors and compare to lambda_max
print "plotting Distorted Vectors vs. V_max"
rows,cols = R_hat.shape

for i,l in enumerate(range(0,rows)):
    plt.axes().arrow(0,0,R_hat[i,0],R_hat[i,1],head_width=0.05,head_length=0.1,color = 'b')

l_v_max = np.dot(L,v_max)
plt.axes().arrow(0,0,l_v_max[0,0],l_v_max[0,1],head_width=0.05,head_length=0.1,color = 'r')

plt.plot(0,0,'ok') #<-- plot a black point at the origin
plt.axis('equal')  #<-- set the axes to the same scale
plt.xlim([-8,8]) #<-- set the x axis limits
plt.ylim([-8,8]) #<-- set the y axis limits
plt.grid(b=True, which='major') #<-- plot grid lines
plt.show()
