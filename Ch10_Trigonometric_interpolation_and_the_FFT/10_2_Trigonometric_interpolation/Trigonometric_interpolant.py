import numpy as np
import matplotlib.pyplot as plt

# DFT Interpolation Theorem Corollary for even integer
def even_trigonometric_interpolant(t, a, b, c, d):
    return a[0] / np.sqrt(a.size) + 2 / np.sqrt(a.size) * \
        np.sum([a[k] * np.cos(2 * k * np.pi * (t - c) / (d - c)) - \
        b[k] * np.sin(2 * k * np.pi * (t - c) / (d - c)) for k in range(1, int(a.size / 2))]) + \
        a[int(a.size / 2)] / np.sqrt(a.size) * np.cos(a.size * np.pi * (t - c) / (d - c))

print('-'*100)
print("Find the trigonometric interpolant for Example 10.1 (vector x = [1, 0, −1, 0].T):")
print('-'*100)
c, d = 0, 1
x = np.array([1, 0, -1, 0]).T
s = ['x1', 'x2', 'x3', 'x4']
y = np.fft.fft(x) / np.sqrt(x.size)
a = np.array(y.real)
b = np.array(y.imag)

t = np.linspace(c, d, 40)
xt = np.linspace(0, 1, 4, False)
data = []
for i in t:
    data.append(even_trigonometric_interpolant(i, a, b, c, d))
print(s)
plt.figure(figsize=(8, 6))
plt.plot(t, data)
plt.plot(xt, x, 'o', color='m')
for i in range(len(s)):
    plt.text(xt[i]+0.01, x[i]+0.01, s[i])
plt.axhline(y=0, color='black')
plt.show()

print('-'*100)
print("Find the trigonometric interpolant for the temperature data from Example 4.6: x =\n\
[−2.2, −2.8, −6.1, −3.9, 0.0, 1.1, −0.6, −1.1] on the interval [0, 1]:")
print('-'*100)
c, d = 0, 1
n, p = 8, 100
xt = np.linspace(c, d, n, False)
x = np.array([-2.2, -2.8, -6.1, -3.9, 0, 1.1, -0.6, -1.1])
y = np.fft.fft(x) / np.sqrt(x.size)

print("The Fourier transform output, accurate to four decimal places:")
print('%10s %10s' % ('real', 'imag'))
for i in y:
    print("%10.4f %10.4f" % (i.real, i.imag))

a = np.array(y.real)
b = np.array(y.imag)
t = np.linspace(c, d, p)
data = [even_trigonometric_interpolant(i, a, b, c, d) for i in t]
print('n = %d p = %d' %(n, p))
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)

plt.xticks(xt, ['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7'])
plt.plot(t, data, label ='Interpolant at p (>=n) evenly spaced points')
plt.plot(xt, x, 'o', color='m', label ='Interpolate 8 points on [0,1] with trig function P(t)')
plt.xlabel('t')
plt.ylabel('Interpolant')
plt.legend()
plt.title('Use even_trigonometric_interpolant()')

def dftinterp(inter,x,n,p):
    c = inter[0]
    d = inter[1]
    t = []
    tp = []
    for i in range(n):
        t.append(c + (d - c) * i / n)
    for i in range(p):
        tp.append(c + (d - c) * i / p)

    y = np.fft.fft(x) / np.sqrt(x.size)             # apply DFT
    yp = []                                         # yp will hold coefficients for ifft
    for i in range(p):
        yp.append( 0 + 0j )
    yp[0: int(n / 2 + 1)] = y[0:int(n / 2 + 1)]     # move n frequencies from n to p
    yp[int(p - n / 2 + 2): p] = y[int(n / 2 + 2): n]
    xp = np.real(np.fft.ifft(yp) * np.sqrt(x.size)) * (p / n)
    return t, tp, xp, y

print('-'*100)
print("Interpolate n data points on [c,d] with trig function P(t)\n\
and plot interpolant at p (>=n) evenly spaced points.:")
print('-'*100)
t, tp, xp, y = dftinterp([0, 1], np.array([-2.2, -2.8, -6.1, -3.9, 0, 1.1, -0.6, -1.1]),n,p)

print('dftinterp: n = %d p = %d' %(n, p))
print("The Fourier transform output, accurate to four decimal places:")
print('%10s %10s' % ('real', 'imag'))
for i in y:
    print("%10.4f %10.4f" % (i.real, i.imag))

plt.subplot(1, 2, 2)
plt.xticks(t, ['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7'])
plt.plot(tp, xp, label ='Interpolant at p (>=n) evenly spaced points')
plt.plot(t, x, 'o', color='m', label ='Interpolate 8 points on [0,1] with trig function P(t)')
plt.xlabel('t')
plt.ylabel('Interpolant')
plt.legend()
plt.title('Use dftinterp()')
plt.show()

