import numpy as np
from scipy import optimize
import random
import matplotlib.pyplot as plt

# coef = [-2,  2,  3,  4, 1] # np.random.randint(-5, 5, 5)
# print(coef)

# f = np.poly1d(coef)
# x = np.linspace(0, 2, 20)
# y = f(x) + 1.0*np.random.normal(size=len(x))
# x = x[y > 0]
# y = y[y > 0]
x = [0, 100, 200, 300, 400]
y = [10, 5, 6, 2, 10]
print(np.stack([x, y], 1))
xn = np.linspace(0, 400, 200)

p_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
for p in p_list:
    popt = np.polyfit(x, y, p)
    print(popt)

    yn = np.polyval(popt, xn)

    plt.plot(x, y, 'or')
    plt.plot(xn, yn)
    plt.show()