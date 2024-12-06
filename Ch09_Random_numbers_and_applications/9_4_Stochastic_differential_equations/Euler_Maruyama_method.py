from math import sqrt
from scipy.stats import norm
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(132)

def brownian(x0, n, dt, delta, out=None):
    """
        Generate an instance of Brownian motion (i.e. the Wiener process):

            X(t) = X(0) + N(0, delta**2 * t; 0, t)

        where N(a,b; t0, t1) is a normally distributed random variable with mean a and
        variance b.  The parameters t0 and t1 make explicit the statistical
        independence of N on different time intervals; that is, if [t0, t1) and
        [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
        are independent.

        Written as an iteration scheme,

            X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)
    """
    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples.
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out

print('-'*100)
print("Solve the stochastic differential equation dy(t) = r*dt + σ*dBt with initial condition y(0) = y0 .")
print('-'*100)
N = 500
dt = 0.1
delta = 0.1
xlimit = 2
y0 = 1
r = 0.1
sigma = 0.3
X = np.linspace(0, xlimit, N)

# For Brownian motion
B = brownian(0, N, dt, delta)

Y = y0 * np.exp((r - 0.5 * pow(sigma, 2)) * X + sigma * B) # The solution (9.18)

#For Euler-Maruyama Method
times = np.arange(0, xlimit + dt, dt)

dB = np.random.standard_normal(times.size) * np.sqrt(dt)
ws = np.empty(times.size)
ws[0] = y0

for i in range(times.size - 1):
    ws[i + 1] = ws[i] + r * ws[i] * dt + sigma * ws[i] * dB[i]

# Plot the chart
plt.plot(X, Y, label='The solution (9.18)')
plt.plot(X, B, linestyle = '--', label='Brownian motion path')
plt.scatter(times, ws, s=10, c='m', label='Euler–Maruyama approximation')
plt.axhline(y=0, color='black')
plt.axvline(x=0, color='black')
plt.grid(True, which='both')
plt.legend()
plt.show()

print('-'*100)
print("Numerically solve the Langevin equation")
print('-'*100)

dt = 0.1
xlimit = 50
y0 = 0
r = 10
sigma = 1
delta = 0.5
times = np.arange(0, xlimit + dt, dt)
dB = np.random.standard_normal(times.size) * np.sqrt(dt)
ws = np.empty(times.size)
ws[0] = y0

for i in range(times.size - 1):
    ws[i + 1] = ws[i] - r * ws[i] * dt + sigma * dB[i]

# For Brownian motion realization
BM = brownian(0, times.size, dt, delta)

# Plot the chart
plt.plot(times, ws, label='Langevin equation')
plt.plot(times, BM, linestyle = '--', label='Brownian motion')
plt.axhline(y=0, color='black')
plt.axvline(x=0, color='black')
plt.grid(True, which='both')
plt.legend()
plt.show()