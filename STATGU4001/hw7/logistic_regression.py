import numpy as np
import matplotlib.pyplot as plt

def inv_logit(x, b0, b1):
    return np.exp(b0 + b1*x)/(1 + np.exp(b0 + b1 * x))

data = np.array([[60, 1],
                 [50, 1],
                 [46, 0],
                 [40, 1],
                 [35, 0],
                 [30, 0],
                 [25, 1]])

# plot the input data
plt.scatter(data[:,0], data[:,1], color='green')

# calculate the inverse logits based on chosen parameters
mu = np.mean(data[:,0])
sigma = np.std(data[:,0])
b0, b1 = -mu, 1 # best
# b0, b1 = -mu, 1.1
# b0, b1 = mu, -1
# b0, b1 = mu + sigma, -1.5
x = np.linspace(-50, 100, 1000)
y = inv_logit(x, b0, b1)
plt.plot(x, y, color='red')
plt.savefig('logistic_regression.png')
plt.show()
