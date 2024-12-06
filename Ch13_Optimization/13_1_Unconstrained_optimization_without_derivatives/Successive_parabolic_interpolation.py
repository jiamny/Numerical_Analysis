import matplotlib.pyplot as plt
import numpy as np
from sympy.printing.pretty.pretty_symbology import line_width


def spi(f, r, s, t, k):
    x = np.zeros(k+3)
    fx = np.zeros(k+3)
    x[2] = r
    x[1] = s
    x[0] = t
    fr = f(r)
    fx[2] = fr
    fs = f(s)
    fx[1] = fs
    ft = f(t)
    fx[0] = ft
    for i in range(3, k + 3, 1):
        x[i] = (r + s) / 2 - (fs - fr) * (t - r) * (t - s) / (2 * ((s - r) * (ft - fs)\
                                                                   - (fs - fr) * (t - s)))
        t = s
        s = r
        r = x[i]
        ft = fs
        fs = fr
        fr = f(r)
        fx[i] = fr
    return x, fx

print('-'*100)
print("Use Successive Parabolic Interpolation to ﬁnd the minimum of f(x) = x**6 − 11x**3 + 17x**2\n\
 − 7x + 1 on the interval [0, 1]. Using starting points r = 0, s = 0.7, t = 1,")
print('-'*100)

a, b = 0, 1
r = 0
s = 0.7
t = 1
k = 15
f = lambda x: x ** 6 - 11 * x ** 3 + 17 * x ** 2 - 7 * x + 1
z = np.linspace(0, 1, 50)
print(f(z))
x, fx = spi(f, r, s, t, k)

print('%5s %10s %10s' %('step', 'x', 'f(x)'))
for i in range(k+3):
    if i < 3:
        print('%5d %10.5f %10.5f' % (0, x[i], fx[i]))
    else:
        print('%5d %10.5f %10.5f' % (i - 2, x[i], fx[i]))

plt.figure(figsize=(8, 6))
plt.plot([r, s, t], f(np.array([r, s, t])), 'o', c='k', markersize=10, label='current points r, s, t')
plt.plot(z, f(z), '-', label='the parabola')
plt.plot(x[3:], fx[3:], 'm--', linewidth=3, label='step is repeated with the new r, s, t')
plt.annotate(text='the minimum x of the parabola', xy=(x[-1], fx[-1]), xytext=(x[-1]+ 0.15, fx[-1] + 0.05), arrowprops={'arrowstyle':'->'})
plt.show()