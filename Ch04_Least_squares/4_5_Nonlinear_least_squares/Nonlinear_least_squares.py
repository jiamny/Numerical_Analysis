import numpy as np
import matplotlib.pyplot as plt

def Newton_Method(F, DF, x, iter_num=10 ** 4):
    for _ in range(iter_num):
        s = np.linalg.inv(DF(x)).dot(F(x))
        x -= s
    return x


def Gauss_Newton(r, Dr, x, iter_num=10 ** 4):
    for _ in range(iter_num):
        A = Dr(x)
        v = -np.linalg.inv(A.T.dot(A)).dot(A.T).dot(r(x))
        x += v

    return x


def Levenberg_Marquardt(r, Dr, x, lamb=1, iter_num=10 ** 4):
    for _ in range(iter_num):
        A = Dr(x)
        ATA = A.T.dot(A)
        v = -np.linalg.inv(ATA + lamb * np.diag(np.diag(ATA))).dot(A.T).dot(r(x))
        x += v
    return x

print('-'*70)
print('Consider the three circles in the plane with centers (x1 , y1 ) = (−1, 0),\n\
(x2 , y2 ) = (1, 1/2), (x3 , y3 ) = (1, −1/2) and radii R1 = 1, R2 = 1/2, R3 = 1/2, respectively.')
print('-'*70)

def r(x):
    c1, c2, c3 = ((-1, 0), (1, 1/2), (1, -1/2))
    r1 = np.sqrt(np.sum((c1 - x) ** 2)) - 1
    r2 = np.sqrt(np.sum((c2 - x) ** 2)) - 1/2
    r3 = np.sqrt(np.sum((c3 - x) ** 2)) - 1/2

    return np.array([r1, r2, r3])


def Dr(x):
    c1, c2, c3 = ((-1, 0), (1, 1/2), (1, -1/2))
    dr1 = (x - c1) / np.sqrt(np.sum((c1 - x) ** 2))
    dr2 = (x - c2) / np.sqrt(np.sum((c2 - x) ** 2))
    dr3 = (x - c3) / np.sqrt(np.sum((c3 - x) ** 2))

    return np.array([dr1, dr2, dr3])

initial_guess = np.zeros(2, dtype=float)
(x, y) = Gauss_Newton(r, Dr, initial_guess)
print("The Gauss–Newton iteration with initial vector (x0 , y0 ) = (0, 0) converges to (x, y) = (%.6f, %.6f)" % (x, y))


def F(X):
    x = X[:2]
    K = X[2]

    c1, c2, c3 = ((-1, 0), (1, 1/2), (1, -1/2))
    f1 = np.sqrt(np.sum((c1 - x) ** 2)) - (1 + K)
    f2 = np.sqrt(np.sum((c2 - x) ** 2)) - (1/2 + K)
    f3 = np.sqrt(np.sum((c3 - x) ** 2)) - (1/2 + K)

    return np.array([f1, f2, f3])


def DF(X):
    x = X[:2]
    c1, c2, c3 = ((-1, 0), (1, 1/2), (1, -1/2))
    n = 3

    dr1 = (x - c1) / np.sqrt(np.sum((c1 - x) ** 2))
    dr2 = (x - c2) / np.sqrt(np.sum((c2 - x) ** 2))
    dr3 = (x - c3) / np.sqrt(np.sum((c3 - x) ** 2))

    df = np.array([dr1, dr2, dr3])
    return np.concatenate((df, -np.ones((n, 1))), axis=1)

initial_guess = np.zeros(3, dtype=float)
(x, y, K) = Newton_Method(F, DF, initial_guess)

print("Newton’s Method yields the solution (x, y, K) = (%.6f, %.6f, %.6f)" % (x, y, K))

print('-'*70)
print('Consider the four circles with centers (−1, 0), (1, 1/2), (1, −1/2),\n\
(0, 1) and radii 1, 1/2, 1/2, 1/2, respectively.')
print('-'*70)
def r(X):
    x = X[:2]
    k = X[2]
    c1, c2, c3, c4 = ((-1, 0), (1, 1/2), (1, -1/2), (0, 1))

    r1 = np.sqrt(np.sum((c1 - x) ** 2)) - (1 + k)
    r2 = np.sqrt(np.sum((c2 - x) ** 2)) - (1/2 + k)
    r3 = np.sqrt(np.sum((c3 - x) ** 2)) - (1/2 + k)
    r4 = np.sqrt(np.sum((c4 - x) ** 2)) - (1/2 + k)

    return np.array([r1, r2, r3, r4])


def Dr(X):
    x = X[:2]
    n = 4

    c1, c2, c3, c4 = ((-1, 0), (1, 1/2), (1, -1/2), (0, 1))
    dr1 = (x - c1) / np.sqrt(np.sum((c1 - x) ** 2))
    dr2 = (x - c2) / np.sqrt(np.sum((c2 - x) ** 2))
    dr3 = (x - c3) / np.sqrt(np.sum((c3 - x) ** 2))
    dr4 = (x - c4) / np.sqrt(np.sum((c4 - x) ** 2))
    dr = np.array([dr1, dr2, dr3, dr4])

    return np.concatenate((dr, -np.ones((n, 1))), axis=1)

initial_guess = np.zeros(3, dtype=float)
(x, y, K) = Gauss_Newton(r, Dr, initial_guess)
print("The Gauss–Newton Method provides the solution (x, y) = (%.6f, %.6f) with K = %.6f" % (x, y, K))

print('-'*70)
print('Use Levenberg–Marquardt to ﬁt the model y = c1*e**−c2*(t−c3)**2 to the \n\
data points (ti , yi ) = {(1, 3), (2, 5), (2, 7), (3, 5), (4, 1)}')
print('-'*70)
x = np.array([1, 2, 2, 3, 4])
y = np.array([3, 5, 7, 5, 1])

def model(c1, c2, c3, t):
    return c1*np.exp(-c2 * (t - c3)**2)

def r(Z):
    c1, c2, c3 = Z
    return model(c1, c2, c3, x) - y


def Dr(Z):
    c1, c2, c3 = Z
    dr1 = np.exp(-c2*(x - c3)**2)
    dr2 = -c1*(x - c3)**2*np.exp(-c2*(x -c3)**2)
    dr3 = 2*c1*c2*(x - c3 )*np.exp(-c2*(x - c3)**2)

    return np.array([dr1, dr2, dr3]).T


initial_guess = np.array([1, 1, 1], dtype=float)
lamb = 50
c1, c2, c3 = Levenberg_Marquardt(r, Dr, initial_guess, lamb)
RMSE = np.sqrt(np.mean((model(c1, c2, c3, x) - y)**2))

print("The Levenberg–Marquardt provides the solution (c1, c2, c3): (%.6f, %.6f, %.6f)" % (c1, c2, c3))
print("RMSE: %f" % RMSE)
print("The best least squares model y = %.4f*EXP(−%.4f*(x−%.4f)**2)" %(c1, c2, c3))
x_range = np.arange(0., 5., 0.1, dtype=float)
y_fited = model(c1, c2, c3, x_range)

plt.scatter(x, y, s=50, c='r')
plt.plot(x_range, y_fited)
plt.show()