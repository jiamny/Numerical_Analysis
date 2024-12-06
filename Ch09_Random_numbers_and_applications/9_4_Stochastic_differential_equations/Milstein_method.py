import numpy as np
from matplotlib import pyplot as plt

np.random.seed(132)

print('-'*100)
print("Apply the Milstein Method to geometric Brownian motion")
print('-'*100)
dt = 0.1
xlimit = 4
y0 = 1e-2
r = 0.1
sigma = 0.3
times = np.arange(0, xlimit + dt, dt)
dB = np.random.standard_normal(times.size) * np.sqrt(dt)

ws = np.empty(times.size)
ws[0] = y0  # For Euler-Maruyama Method
wms = np.empty(times.size)
wms[0] = y0  # For Milstein Method

for i in range(times.size - 1):
    # Euler-Maruyama
    ws[i + 1] = ws[i] + r * ws[i] * dt \
                + sigma * ws[i] * dB[i]
    # Milstein
    wms[i + 1] = wms[i] + r * wms[i] * dt \
                 + sigma * wms[i] * dB[i] \
                 + 0.5 * pow(sigma, 2) * wms[i] * (pow(dB[i], 2) - dt)

# Calculate y(T)
tmp = dB
tmp[-1] = 0
B = np.cumsum(np.roll(tmp, 1))
f = lambda y0, sigma, t, B: y0 * np.exp((r - 0.5 * np.power(sigma, 2)) * t + sigma * B)
Y = f(y0, sigma, times, B)

# Plot the chart
plt.plot(times, ws, label='w(t) by Euler-Maruyama Method')
plt.plot(times, wms, label='w(t) by Milstein Method')
plt.plot(times, Y, label='Y(T)')
plt.grid(True, which='both')
plt.legend()
plt.show()

# Plot the chart
plt.ylabel('|y(T)-w(T)|')
plt.plot(times, np.abs(Y - ws), label='Euler-Maruyama Method')
plt.plot(times, np.abs(Y - wms), label='Milstein Method')
plt.grid(True, which='both')
plt.legend()
plt.show()

print('-'*100)
print("Applying the Euler–Maruyama Method and the Milstein Method with decreasing step sizes delta_t")
print('-'*100)
dts = np.array([
    pow(2, -1), pow(2, -2), pow(2, -3), pow(2, -4), pow(2, -5),
    pow(2, -6), pow(2, -7), pow(2, -8), pow(2, -9), pow(2, -10)
])
# sort dts in ascending order
dts = np.sort(dts)
errs_em = np.empty(dts.size)
errs_m = np.empty(dts.size)
xlimit = 4
y0 = 1e-2
r = 0.1
sigma = 0.3
num_trail = 100
# For each dt
for i in range(dts.size):
    dt = dts[i]
    e_em = 0
    e_mm = 0
    for _ in range(num_trail):
        times = np.arange(0, xlimit + dt, dt)
        dB = np.random.standard_normal(times.size) * np.sqrt(dt)
        ws = np.empty(times.size)
        ws[0] = y0  # For Euler-Maruyama Method
        wms = np.empty(times.size)
        wms[0] = y0  # For Milstein Method

        for j in range(times.size - 1):
            # Euler-Maruyama
            ws[j + 1] = ws[j] + r * ws[j] * dt \
                        + sigma * ws[j] * dB[j]
            # Milstein
            wms[j + 1] = wms[j] + r * wms[j] * dt \
                         + sigma * wms[j] * dB[j] \
                         + 0.5 * pow(sigma, 2) * wms[j] * (pow(dB[j], 2) - dt)
        # Calculate y(T)
        tmp = dB
        tmp[-1] = 0
        B = np.cumsum(np.roll(tmp, 1))
        f = lambda y0, sigma, t, B: y0 * np.exp((r - 0.5 * np.power(sigma, 2)) * t + sigma * B)
        Y = f(y0, sigma, times, B)

        e_em += abs(Y[-1] - ws[-1])
        e_mm += abs(Y[-1] - wms[-1])

    errs_em[i] = e_em/num_trail
    errs_m[i] = e_mm/num_trail
# Plot the chart
fig, ax = plt.subplots()
plt.xlabel('Step size h')
plt.ylabel('Mean error')
plt.loglog(dts, errs_em, 'o', label='Euler-Maruyama Method')
plt.loglog(dts, errs_m, 'x', label='Milstein Method')
plt.legend()
fig.autofmt_xdate()
plt.show()