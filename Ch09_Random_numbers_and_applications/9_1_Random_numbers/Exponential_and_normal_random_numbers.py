import time
import numpy as np
from matplotlib import pyplot as plt

def Minimal_standard_random_number_generator(x):
    a = int(pow(7, 5))
    m = int(pow(2, 31)) - 1
    x = a * x % m
    u = x/m
    return u, x

def exponential_random_numbers(a, u):
    return -np.log(1 - u)/ a

print('-'*100)
print("Exponential random numbers")
print('-'*100)
a = 5
x = time.time()
for i in range(10):
    r, x = Minimal_standard_random_number_generator(x)
    er = exponential_random_numbers(a, r)
    print(er)

print('-' * 100)
print("Generating normal random numbers")
print('-' * 100)
def standard_normal_random_numbers(U, n):
    x = time.time()
    r = []
    while True:
        x1, x = U(x)
        x2, x = U(x)
        if len(r) < n:
            r.append(np.sqrt(-2 * np.log(1 - x1))*np.cos(2*np.pi*x2))
        else:
            break
        if len(r) < n:
            r.append(np.sqrt(-2 * np.log(1 - x1)) * np.sin(2 * np.pi * x2))
        else:
            break

    return r

n = 10000
r = standard_normal_random_numbers(Minimal_standard_random_number_generator, n)
plt.hist(r, bins = 100, density = True)
plt.show()

print('-' * 100)
print("The Box–Muller Method for generating normal random numbers")
print('-' * 100)

def Box_Muller_standard_normal_random_numbers(U, n):
    x = time.time()
    r = []
    while True:
        x1, x = U(x)
        x2, x = U(x)
        u1 = x1*x1 + x2*x2
        print(x1, ' ', x2)
        while u1 > 1.:
            x1, x = U(x)
            x2, x = U(x)
            u1 = x1*x1 + x2*x2

        if len(r) < n:
            r.append(x1 * np.sqrt(-2 * np.log(u1) / u1))
            if len(r) < n:
                r.append(-x1 * np.sqrt(-2 * np.log(u1) / u1))
            else:
                break
        else:
            break
        if len(r) < n:
            r.append(x2 * np.sqrt(-2 * np.log(u1) / u1))
            if len(r) < n:
                r.append(-x2 * np.sqrt(-2 * np.log(u1) / u1))
            else:
                break
        else:
            break

    return np.sort(r)

n = 10000
r = Box_Muller_standard_normal_random_numbers(Minimal_standard_random_number_generator, n)
plt.hist(r, bins = 100, density = True)
plt.show()