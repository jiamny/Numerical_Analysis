import numpy as np

def Broyden_Method_I(F, x, A, iter_num=10 ** 4):
    x = np.array(x, dtype=float)
    n = len(x)

    for _ in range(iter_num):
        x_1 = x - np.linalg.inv(A).dot(F(x))
        d_1 = x_1 - x
        D_1 = F(x_1) - F(x)
        A += (D_1 - A.dot(d_1)).reshape(n, 1).dot(d_1.reshape(1, n)) / (d_1.T).dot(d_1)
        x += d_1

    return x


def Broyden_Method_II(F, x, B, iter_num=10 ** 4):
    x = np.array(x, dtype=float)
    n = len(x)

    for _ in range(iter_num):
        x_1 = x - B.dot(F(x))
        d_1 = x_1 - x
        D_1 = F(x_1) - F(x)
        B += (d_1 - B.dot(D_1)).reshape(n, 1).dot(d_1.reshape(1, n)).dot(B) / d_1.dot(B).dot(D_1)
        x += d_1

    return x


def F(x):
    u, v = x
    f1 = u ** 3 - v ** 3 + u
    f2 = u ** 2 + v ** 2 - 1

    return np.array([f1, f2])

print('-'*70)
print('Use Broyden’s Method I to find the solution of the inconsistent system')
print('-'*70)
x_0 = (1, 1)
A = np.identity(2).astype(float)
print("A: \n", A)
required_step = 13
x = Broyden_Method_I(F, x_0, A, required_step)
print("Step: %d Broyden_Method_I(F, x_0, A) = (u, v):" % required_step, tuple(x))

print('-'*70)
print('Use Broyden’s Method II to find the solution of the inconsistent system')
print('-'*70)
B = np.identity(2).astype(float)
print("B: \n", B)
required_step = 13
x = Broyden_Method_II(F, x_0, B, required_step)
print("Step: %d Broyden_Method_II(F, x_0, B) = (u, v):" % required_step, tuple(x))