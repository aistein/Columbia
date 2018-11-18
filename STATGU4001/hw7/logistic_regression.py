import numpy as np
import matplotlib.pyplot as plt

def inv_logit(x, b0, b1):
    return np.exp(b0 + b1*x)/(1 + np.exp(b0 + b1 * x))

def logistic_p(x, data, b0, b1):
    y = inv_logit(x, b0, b1)
    p = 1
    for i in range(0, data.shape[0]):
        if data[i,1] == 1:
            p *= (1-y[i])
        else:
            p *= y[i]
    print("with b0: ", b0, ", b1: ", b1,", p-value = ",p)
    return p

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

x = np.linspace(-50, 100, 1000)

b0, b1 = -mu, 1 # best
p1 = logistic_p(x, data, b0, b1)

b0, b1 = -mu, 1.1
p2 = logistic_p(x, data, b0, b1)

b0, b1 = mu, -1
p3 = logistic_p(x, data, b0, b1)

b0, b1 = mu + sigma, -1.5
p4 = logistic_p(x, data, b0, b1)

b0, b1 = mu - 3.2*sigma, -0.1
p5 = logistic_p(x, data, b0, b1)

b0, b1 = mu -2*sigma, -0.4
p5 = logistic_p(x, data, b0, b1)
y = inv_logit(x, b0, b1)


plt.plot(x, y, color='red')
plt.savefig('logistic_regression.png')
plt.show()
