from math import sqrt
from scipy.stats import norm
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(42)

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
xlim = 2.0

# For SDE
sigma = 0.3
r = 1
y0 = 0
X = np.linspace(0, xlim, N)

# For Brownian motion
dt = 0.1
delta = 0.3
B1 = brownian(y0, N, dt, delta)
B2 = brownian(y0, N, dt, delta)

# Process
Y = y0 + r * X
Y1 = y0 + r * X + sigma * B1
Y2 = y0 + r * X + sigma * B2
plt.xlim(0, 2)
plt.plot(X, Y1)
plt.plot(X, Y2)
plt.plot(X, Y, color='black')
plt.show()


print('-'*100)
print("Ito formula")
print('-'*100)
N = 500
xlim = 2.0
r = 0.1
sigma = 0.3
delta = 0.1
dt = 0.2
y0 = 1
X = np.linspace(0, xlim, N)

# For Brownian motion
B = brownian(0, N, dt, delta)

# Process
Y = y0 * np.exp((r - 0.5 * pow(sigma, 2)) * X + sigma * B)
plt.plot(X, Y, label='The solution (9.18)')
plt.plot(X, B, linestyle = '--', label='Brownian motion path')
plt.grid(True)
plt.legend()
plt.show()
