import math
import time
import numpy as np
from matplotlib import pyplot as plt

def Minimal_standard_random_number_generator(x):
    a = int(pow(7, 5))
    m = int(pow(2, 31)) - 1
    x = a * x % m
    u = x/m
    return u, x

def halton(p, n):
    b = np.zeros(math.ceil(math.log(n + 1) / math.log(p)))
    u = np.zeros(n)
    for j in range(n):
        i = 0
        b[0] = b[0] + 1
        while b[i] > p - 1 + np.finfo(float).eps:
            b[i] = 0
            i += 1
            b[i] += 1
        u[j] = 0
        for k in range(1, b.size + 1):
            u[j] = u[j] + b[k-1] * pow(p, -k)
    return u

print(halton(2, 8))
print(halton(3, 8))

def Type_1_Monte_Carlo_pseudo_random_numbers(n):
    sum_v = 0.0
    x = time.time()
    for i in range(n):
        ui, x = Minimal_standard_random_number_generator(x)
        sum_v += ui ** 2

    return sum_v/n

def Type_1_Monte_Carlo_halton(p, n):
    u = halton(p, n)
    sum_v = 0.0
    for i in range(len(u)):
        sum_v += u[i] ** 2

    return sum_v/n

print('-'*100)
print("Find Type 1 Monte Carlo estimates, using pseudo-random numbers and quasi-random number for the area\n\
under the curve of y = x**2 in [0, 1]")
print('-'*100)
num_tail = 50
true_area = 1/3
p = 2

n_points = [100, 200, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400]

type_1_pseudo = []
type_1_quasi = []
for i in range(len(n_points)):
    n = n_points[i]
    e1 = 0
    e2 = 0
    for _ in range(num_tail):
        e1 += abs(Type_1_Monte_Carlo_pseudo_random_numbers(n) - true_area)
        e2 += abs(Type_1_Monte_Carlo_halton(p, n) - true_area)

    type_1_pseudo.append(e1 / num_tail)
    type_1_quasi.append(e2 / num_tail)
    print(i)

plt.figure()
plt.loglog(n_points, type_1_pseudo, 'o-', label='pseudo random numbers')
plt.loglog(n_points, type_1_quasi, 'o--', label='quasi random number')
plt.xlabel('Number of points n')
plt.ylabel('Error')
plt.legend()
plt.title('Mean error of Type 1 Monte Carlo estimate')
plt.show()

print('-'*100)
print("pseudo-random vs quasi-random")
print('-'*100)
pair_count = 2000

pr_xdata = np.array([])
pr_ydata = np.array([])
qr_xdata = np.array([])
qr_ydata = np.array([])

qrx_seq = halton(2, pair_count)
qry_seq = halton(3, pair_count)

x = time.time()

for idx in range(pair_count):
    ux, x = Minimal_standard_random_number_generator(x)
    uy, x = Minimal_standard_random_number_generator(x)
    pr_xdata = np.append(pr_xdata, ux)
    pr_ydata = np.append(pr_ydata, uy)
    qr_xdata = np.append(qr_xdata, qrx_seq[idx])
    qr_ydata = np.append(qr_ydata, qry_seq[idx])


fig, axs = plt.subplots(1, 2, figsize=(16,6))
for ax in axs.flat:
    ax.set(xlim=(0, 1), ylim=(0, 1))

fig.suptitle("pseudo-random vs quasi-random")
axs[0].plot(pr_xdata, pr_ydata, 'o', markersize=1)
axs[1].plot(qr_xdata, qr_ydata, 'o', markersize=1)
plt.show()