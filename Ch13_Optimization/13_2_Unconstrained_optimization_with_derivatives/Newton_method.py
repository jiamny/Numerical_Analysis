import numpy as np
from scipy import optimize

def Gaussion_elimination(A, b, eps = 1e-6):
    # ----------------------------------
    # eliminating each column
    # ----------------------------------
    assert(A.shape[0] == A.shape[1])
    for j in range(0, A.shape[1] - 1):
        if abs(A[j, j]) < eps:
            raise ValueError("zero pivot encountered!")
            return

        for i in range(j+1, A.shape[0]):
            mult = A[i, j]/A[j, j]
            for k in range(j+1, A.shape[1]):
                A[i, k] = A[i, k] - mult * A[j, k]
            b[i] = b[i] - mult * b[j]

    # ----------------------------------
    # the back-substitution step
    # ----------------------------------
    x = np.zeros((A.shape[0], 1))
    for i in reversed(range(A.shape[0])):
        for j in range(i+1, A.shape[1]):
            b[i] = b[i] - A[i, j] * x[j].item()
        x[i] = b[i] / A[i, i]

    return x

def Newton_Method(f, dF, d2F, x, iter_num=10, use_numpy=False,):
    Xs = np.zeros(iter_num + 1)
    Ys = np.zeros(iter_num + 1)
    fs = np.zeros(iter_num + 1)
    Xs[0] = x[0]
    Ys[0] = x[1]
    fs[0] = f(x)
    for i in range(iter_num):
        if use_numpy:
            s = np.linalg.inv(np.array(d2f(x))).dot(np.array(df(x)))
            x -= s
            Xs[i + 1] = x[0]
            Ys[i + 1] = x[1]
            fs[i + 1] = f(x)
        else:
            A = d2f(x)
            b = df(x)
            s = Gaussion_elimination(np.array(A), -np.array(b))
            x = x + s.squeeze()
            Xs[i + 1] = x[0]
            Ys[i + 1] = x[1]
            fs[i + 1] = f(x)

    return x, Xs, Ys, fs


print('-'*100)
print("Locate the minimum of the function f (x, y) = 5x**4 + 4x**2*y − xy**3 + 4y**4 − x, using the Newton’s Method:")
print('-'*100)

def f(x):
    return 5*x[0]**4 + 4*x[0]**2*x[1] - x[0]*x[1]**3 + 4*x[1]**4 - x[0]

def df(x):
    return  [20*x[0]**3 + 8*x[0]*x[1] - x[1]**3 - 1, 4*x[0]**2 - 3*x[0]*x[1]**2 + 16*x[1]**3]

def d2f(x):
    return [[60*x[0]**2 + 8*x[1], 8*x[0] - 3*x[1]**2], [8*x[0] - 3*x[1]**2, -6*x[0]*x[1] + 48*x[1]**2]]

print('-'*100)
print("Use numpy np.linalg.inv():")
print('-'*100)
x0 = [1, 1]
iter_num = 10
x, Xs, Ys, fs = Newton_Method(f, df, d2f, x0, iter_num, use_numpy=True)
print(x)
print(Xs)

print('-'*100)
print("Use Gaussion_elimination():")
print('-'*100)
x2, Xs2, Ys2, fs2 = Newton_Method(f, df, d2f, x0, iter_num, use_numpy=False)
print('the minimum value near (x, y) = ', x2)
print('%5s %12s %12s %12s' %('step', 'x', 'y', 'f(x, y)'))
for i in range(iter_num+1):
    print('%5d %12.8f %12.8f  %12.8f' % (i, Xs2.tolist()[i], Ys2.tolist()[i], fs2.tolist()[i]))

print('-'*100)
print("Use scipy optimize API:")
print('-'*100)
res = optimize.minimize(f, np.array([1, 1]), method='Newton-CG', jac=df, hess=d2f)

print(res)
