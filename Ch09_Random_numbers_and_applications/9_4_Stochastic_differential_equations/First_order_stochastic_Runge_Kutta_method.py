import numpy as np
from matplotlib import pyplot as plt

np.random.seed(132)

print('-'*100)
print("Example 9.15 use the Euler–Maruyama Method, the Milstein Method, and the First-Order\n\
Stochastic Runge-Kutta Method to solve the SDE:")
print('-'*100)
dt = 0.1
xlimit = 4
y0 = 2
times = np.arange(0, xlimit + dt, dt)
dB = np.random.standard_normal(times.size) * np.sqrt(dt)

ws_em = np.empty(times.size)
ws_em[0] = y0  # For Euler-Maruyama Method
ws_m = np.empty(times.size)
ws_m[0] = y0  # For Milstein Method
ws_rk = np.empty(times.size)
ws_rk[0] = y0  # For First-Order Stochastic Runge-Kutta Method

# Calculate y(T)
tmp = dB
tmp[-1] = 0
B = np.cumsum(np.roll(tmp, 1))
f = lambda y0, B: np.log(2*B + np.exp(y0) )
Y = f(y0, B)

for i in range(times.size - 1):
    # Euler-Maruyama Method
    ws_em[i + 1] = ws_em[i] - 2 * np.exp(-2 * ws_em[i]) * dt + 2 * np.exp(-ws_em[i]) * dB[i]

    # Milstein Method
    ws_m[i + 1] = ws_m[i] - 2 * np.exp(-2 * ws_m[i]) * dt + 2 * np.exp(-ws_m[i]) * dB[i] - \
                  2 * np.exp(-2 * ws_m[i]) * (np.power(dB[i], 2) - dt)

    # First-Order Stochastic Runge-Kutta Method
    ws_rk[i + 1] = ws_rk[i] - 2 * np.exp(-2 * ws_rk[i]) * dt + 2 * np.exp(-ws_rk[i]) * dB[i] + \
                   (2 * np.exp(-(ws_rk[i] + 2 * np.exp(-ws_rk[i]) * np.sqrt(dt))) - 2 * np.exp(-ws_rk[i])) * (
                               np.power(dB[i], 2) - dt) / (2 * np.sqrt(dt))

# Plot the chart
plt.figure(figsize=(10, 8))
plt.plot(times, ws_em, linestyle = '-.', lw=2, label='Euler-Maruyama Method')
plt.plot(times, ws_m, linestyle = '--', lw = 2, label='Milstein Method')
plt.plot(times, ws_rk, linestyle = ':', lw=3, label='First-Order Stochastic Runge-Kutta Method')
plt.plot(times, Y, 'o', label='Correct solution')
plt.legend()
plt.show()
