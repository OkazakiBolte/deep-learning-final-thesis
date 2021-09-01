import matplotlib.pyplot as plt
import numpy as np
import math

# print(type(3.14))

num_hidden_layers = 6
num_classes = 10
a = 2
num_first_hidden_units = num_classes * a ** num_hidden_layers
for k in range(2, num_hidden_layers):
    num_hidden_units = num_classes * a ** (num_hidden_layers + 1 - k)
    print(k, num_hidden_units)


"""
x = np.linspace(-3, 3, 20)
y1 = x
y2 = x ** 2

# figure は 1 つ
plt.figure(figsize=(3, 4))

plt.subplot(2,1,1)
plt.plot(x, y1)

plt.subplot(2,1,2)
plt.plot(x, y2)

plt.show()
"""

"""
def f(x):
    # return (x - 2) * x * (x + 2)
    return - x * np.log(x) - (1 - x) * np.log(1 - x)
# print(f(0.2))

# x = np.arange(-3, 3.5, 0.5)
x = np.linspace(0, 1, 100)
# print( np.round(x, 3) )
plt.plot(x, f(x))
# plt.legend(loc='upper left')
# plt.ylim(0, 1)
plt.title('$H(p) = - p \log p - (1 - p) \log (1 - p)$')
plt.xlabel('probability $p$')
plt.ylabel('Shannon entropy $H(p)$ in nats')
plt.show()
"""




"""
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
plot
"""
# matplotlib.colors.cnames
