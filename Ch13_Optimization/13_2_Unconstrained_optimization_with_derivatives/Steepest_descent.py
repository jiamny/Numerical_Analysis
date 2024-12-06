import numpy as np


def Steepest_descent(f, df, x, eta=0.01, iter_num=100):
    Xs = np.zeros(iter_num + 1)
    Ys = np.zeros(iter_num + 1)
    fs = np.zeros(iter_num + 1)
    Xs[0] = x[0]
    Ys[0] = x[1]
    fs[0] = f(x)
    y_pre = f(x)
    for i in range(iter_num):
        v = df(x)
        x = x - eta * np.array(v)

        Xs[i + 1] = x[0]
        Ys[i + 1] = x[1]
        fs[i + 1] = f(x)

    return x, Xs, Ys, fs

print('-'*100)
print("Locate the minimum of the function f (x, y) = 5x**4 + 4x**2*y − xy**3 + 4y**4 − x, using Steepest Descent:")
print('-'*100)

def f(x):
    return 5*x[0]**4 + 4*x[0]**2*x[1] - x[0]*x[1]**3 + 4*x[1]**4 - x[0]

def df(x):
    return  [20*x[0]**3 + 8*x[0]*x[1] - x[1]**3 - 1, 4*x[0]**2 - 3*x[0]*x[1]**2 + 16*x[1]**3]

x0 = np.array([1.0, -1.0])
eta = 0.01
x2, Xs2, Ys2, fs2 = Steepest_descent(f, df, x0, eta)
print('the minimum value near (x, y) = ', x2)
its = [0, 5, 10, 15, 20, 25]
print('%5s %12s %12s %12s' %('step', 'x', 'y', 'f(x, y)'))
for i in its:
    print('%5d %12.8f %12.8f  %12.8f' % (i, Xs2.tolist()[i], Ys2.tolist()[i], fs2.tolist()[i]))