import matplotlib.pyplot as plt
import numpy as np
# import math


a = 0.084
b = 0.1
c = 1.4

def f(x):
    y = a * x ** 2 + b * x + c * np.sin(x)
    return y
plt.figure(1, figsize=(15, 15))
plt.xticks(color="None")
plt.yticks(color="None")
plt.tick_params(length=0)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$J(\theta)$')
plt.plot(-7.05544, f(-7.05544), marker = '.', color=[0, 0.447, 0.741],
         markersize = 20)
plt.plot(-1.5, f(-1.5), marker = '.', color=[0, 0.447, 0.741],
         markersize = 20)
plt.plot(4.12198, f(4.12198), marker = '.', color=[0, 0.447, 0.741],
         markersize = 20)

x = np.linspace(-9, 6, 500)
plt.plot(x, f(x))
plt.show()
