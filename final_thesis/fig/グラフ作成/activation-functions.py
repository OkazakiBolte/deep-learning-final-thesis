import matplotlib.pyplot as plt
import numpy as np
# import math


def relu(x):
  return np.maximum(0, x)
def sigmoid(x):
  return 1 / (1+np.exp(-x))
def softmax(a):
  exp_a = np.exp(a)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a
  z = 100 * y
  return z

def drelu(x):
  y = 1 * (x > 0)
  return y
def dsigmoid(x):
    y = (1 - sigmoid(x)) * sigmoid(x)
    return y

x = np.linspace(-5, 5, 500)

plt.figure(figsize=(10, 4))
plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.subplot(1, 2, 1)
plt.plot(x, relu(x), label='ReLU')
plt.plot(x, sigmoid(x), color='orange', label='sigmoid')
# plt.plot(x, softmax(x), color='yellowgreen', label=r'$100\times$softmax')
# plt.title('$H(p) = - p \log p - (1 - p) \log (1 - p)$')
plt.xlabel('$x$')
plt.legend(loc='upper left')
plt.ylim(-0.1, 1.1)
plt.grid(True)
plt.title('Activation functions')

# """
plt.subplot(1, 2, 2)
plt.plot(x, drelu(x))
plt.plot(x, dsigmoid(x), color='orange')
# plt.plot(x, softmax(x), color='yellowgreen', label='100$\cdot$softmax')
# plt.title('$H(p) = - p \log p - (1 - p) \log (1 - p)$')
plt.xlabel('$x$')
# plt.legend(loc='upper left')
plt.ylim(-0.1, 1.1)
plt.grid(True)
plt.title('Derivatives')
# """
plt.show()

# plt.plot(x, dsigmoid(x))
# plt.show()

"""
x = np.linspace(-5, 5, 500)
plt.plot(x, relu(x), label='ReLU')
plt.plot(x, sigmoid(x), color='orange', label='sigmoid')
plt.plot(x, softmax(x), color='yellowgreen', label='100$\cdot$softmax')
# plt.title('$H(p) = - p \log p - (1 - p) \log (1 - p)$')
plt.xlabel('$x$')
plt.legend(loc='upper left')
plt.ylim(-0.1, 1)
plt.grid(True)
# plt.ylabel('Shannon entropy $H(p)$ in nats')
plt.show()
# plot
"""
# matplotlib.colors.cnames
