import numpy as np


def Conjugate_gradient_search(f, df, d2f, x, iter_num=100):

    Xs, Ys, fs = [], [], []
    d = r = -np.array(df(x))
    y_pre = f(x)
    Xs.append(x[0])
    Ys.append(x[1])
    fs.append(y_pre)
    for k in range(iter_num):
        if r.any() == 0:
            break

        A = np.array(d2f(x))
        dt_A_d = np.dot(np.matmul(d.T, A), d)
        alpha = np.dot(r.T, r)/dt_A_d

        # update
        x = x + np.dot(alpha, d)
        Xs.append(x[0])
        Ys.append(x[1])
        fs.append(f(x))

        pre_r = np.copy(r)
        r = - np.array(df(x))
        beta = r.T.dot(r)/pre_r.T.dot(pre_r)
        d = r + np.dot(beta, d)

    return x, Xs, Ys, fs

print('-'*100)
print("Locate the minimum of the function f (x, y) = 5x**4 + 4x**2*y − xy**3 + 4y**4 − x, Conjugate Gradient Search:")
print('-'*100)

def f(x):
    return 5*x[0]**4 + 4*x[0]**2*x[1] - x[0]*x[1]**3 + 4*x[1]**4 - x[0]

def df(x):
    return  [20*x[0]**3 + 8*x[0]*x[1] - x[1]**3 - 1, 4*x[0]**2 - 3*x[0]*x[1]**2 + 16*x[1]**3]

def d2f(x):
    return [[60*x[0]**2 + 8*x[1], 8*x[0] - 3*x[1]**2], [8*x[0] - 3*x[1]**2, -6*x[0]*x[1] + 48*x[1]**2]]

x0 = [1.0, -1.0]
x, Xs, Ys, fs = Conjugate_gradient_search(f, df, d2f, x0)
print('the minimum value near (x, y) = ', x)
print(len(Xs))
its = [0, 5, 10, 15, 20, 25]
print('%5s %12s %12s %12s' %('step', 'x', 'y', 'f(x, y)'))
for i in its:
    print('%5d %12.8f %12.8f  %12.8f' % (i, Xs[i], Ys[i], fs[i]))