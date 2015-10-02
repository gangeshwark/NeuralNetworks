# 2 Layer Neural Network:
import numpy as np


# sigmoid function
def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# input dataset
x = np.array([[0, 0, 1],
              [1, 0, 1],
              [1, 1, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 1, 0],
              [1, 0, 0],
              [0, 0, 0]])
# output dataset 
y = np.array([[0, 1, 1, 0, 0, 1, 1, 0]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2 * np.random.random((3, 1)) - 1

for i in xrange(100000):  # increase the range value for more precision
    # forward propagation
    l0 = x
    l1 = nonlin(np.dot(l0, syn0))

    # how much did we miss?
    l1_error = y - l1
    if (i % 10000) == 0:
        print "Error:" + str(np.mean(np.abs(l1_error)))
    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1, True)

    # update weights
    syn0 += np.dot(l0.T, l1_delta)

print "Output After Training:"
print l1
for i in xrange(len(l1)):
    print round(l1[i], 3), " -> ", round(l1[i])
