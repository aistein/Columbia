import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import linalg as LA

mat_dict = {}
mat_dict.update(loadmat('hw0data.mat'))

M = mat_dict['M']

print "The dimensions of matrix M: ", M.shape
print "The 4th row of M: ", M[3,:]
print "The 5th column of M: ", M[:,4]
print "Mean value of the 5th column of M: ", np.mean(M[:,4])

print "histogramming the 4th row..."
plt.hist(M[3,:], rwidth=0.5)
plt.title("Histogram of M[3,:]")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

M_square = np.dot(np.transpose(M), M)
e_vals, e_vecs = LA.eig(M_square)

print "Top three eigenvalues of M_transpose * M: ", e_vals[0:3]
