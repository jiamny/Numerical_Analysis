import numpy as np


def Golden_section_search(f, a, b, k):
    g = (np.sqrt(5) - 1) /2
    x1 = a + (1 - g) * (b - a)
    x2 = a + g * (b - a)
    f1 = f(x1)
    f2 = f(x2)
    As, Bs, X1s, X2s = [], [], [], []
    As.append(a)
    Bs.append(b)
    X1s.append(x1)
    X2s.append(x2)
    for i in range(1, k+1, 1):
        if f1 < f2:                     # if f(x1) < f(x2), replace b with x2
            b = x2
            x2 = x1
            x1 = a + (1 - g) * (b - a)
            f2 = f1                     # single function evaluation
            f1 = f(x1)
        else:                           # otherwise, replace a with x1
            a = x1
            x1 = x2
            x2 = a + g * (b - a)
            f1 = f2                     # single function evaluation
            f2 = f(x2)

        As.append(a)
        Bs.append(b)
        X1s.append(x1)
        X2s.append(x2)
    y = (a + b)/2
    return y,  As, Bs, X1s, X2s


print('-'*100)
print("Use Golden Section Search to ﬁnd the minimum of f(x) = x**6 − 11x**3 + 17x**2 − 7x + 1 on the interval [0, 1]:")
print('-'*100)

a, b = 0, 1
k = 15
y, As, Bs, X1s, X2s = Golden_section_search(lambda x: x ** 6 - 11 * x ** 3 + 17 * x ** 2 - 7 * x + 1, a, b, k)
print("y = %.5f" % (y))

print('%5s %10s %10s %10s %10s' %('step', 'a', 'x1', 'x2', 'b'))
for i in range(k+1):
    print('%5d %10.5f %10.5f %10.5f %10.5f' % (i, As[i], X1s[i], X2s[i], Bs[i]))

