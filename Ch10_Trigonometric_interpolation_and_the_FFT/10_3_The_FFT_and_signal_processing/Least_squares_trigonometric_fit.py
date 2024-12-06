import numpy as np
from matplotlib import pyplot as plt

def least_squares_even_trigonmetric(m, t, a, b, c, d):
    return a[0] / np.sqrt(a.size) + 2 / np.sqrt(a.size) * \
        np.sum([a[k] * np.cos(2 * k * np.pi * (t - c) / (d - c)) - \
                b[k] * np.sin(2 * k * np.pi * (t - c) / (d - c)) for k in range(1, int(m / 2))]) + \
        a[int(m / 2)] / np.sqrt(a.size) * np.cos(a.size * np.pi * (t - c) / (d - c))


def dftfilter(inter, x, m, n, p):
    c = inter[0]
    d = inter[1]
    t = []
    tp = []
    for i in range(n):
        t.append(c + (d - c) * i / n)               # time points for data (n)
    for i in range(p):
        tp.append(c + (d - c) * i / p)              # time points for interpolant (p)

    y = np.fft.fft(x) / np.sqrt(x.size)             # compute interpolation coefficients
    yp = []                                         # yp will hold coefficients for ifft
    for i in range(p):
        yp.append( 0 + 0j )

    yp[0:int(m/2)] = y[0: int(m/2)]                 # keep only first m frequencies
    yp[int(m/2) + 1] = np.real(y[int(m/2)+ 1])      # since m is even, keep cos term only
    if m < n:                                       # unless at the maximum frequency,
        yp[int(p - m / 2) + 1] = yp[int(m / 2) + 1] # add complex conjugate to
                                                    # corresponding place in upper tier
    yp[int(p-m/2) + 2:p] = y[int(n-m/2) + 2:n]      # more conjugates for upper tier
    xp = np.real(np.fft.ifft(yp) * np.sqrt(x.size)) * (p / n)
    return tp, xp #t, tp, xp, y

print('-'*100)
print("Fit the temperature data from Example 10.3 by least squares trigonometric functions of\
orders 4, 6 and 8:")
print('-'*100)

m1, m2, m3 = 4, 6, 8
c, d = 0, 1
n = 8
xt = np.linspace(c, d, n, False)
x = np.array([-2.2, -2.8, -6.1, -3.9, 0, 1.1, -0.6, -1.1])
y = np.fft.fft(x) / np.sqrt(x.size)
a = np.array(y.real)
b = np.array(y.imag)
p = 40
t = np.linspace(c, d, p)
data1 = [least_squares_even_trigonmetric(m1, i, a, b, c, d) for i in t]
data2 = [least_squares_even_trigonmetric(m2, i, a, b, c, d) for i in t]
data3 = [least_squares_even_trigonmetric(m3, i, a, b, c, d) for i in t]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.xticks(xt, ['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7'])
plt.plot(t, data1, label='order 4')
plt.plot(t, data2, label='order 6')
plt.plot(t, data3, label='order 8')
plt.plot(xt, x, 'o', color='m', markersize=10)
plt.title('Use least_squares_even_trigonmetric()')
plt.legend()
plt.grid(True)

tp, xp1 = dftfilter([0, 1], x, m1, n, p)
plt.subplot(1, 2, 2)
plt.xticks(xt, ['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7'])
plt.plot(tp, xp1, label='order 4')
tp2, xp2 = dftfilter([0, 1], x, m2, n, p)
plt.plot(tp2, xp2, label='order 6')
tp3, xp3 = dftfilter([0, 1], x, m3,n, p)
plt.plot(tp3, xp3, label='order 8')
plt.plot(xt, x, 'o', color='m', markersize=10)
plt.title('Use dftfilter()')
plt.legend()
plt.grid(True)
plt.show()