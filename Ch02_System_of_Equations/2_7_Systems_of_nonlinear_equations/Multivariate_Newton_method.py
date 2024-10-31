import numpy as np

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

def Newton_Method(F, DF, x, use_np=False, iter_num=10 ** 4):
    for _ in range(iter_num):
        if use_np :
            s = np.linalg.inv(DF(x)).dot(F(x))
            x -= s
        else:
            A = DF(x)
            b = F(x)
            s = Gaussion_elimination(A, -b)
            x = x + s.squeeze()
    return x

def F(x):
    u, v = x
    f1 = v - u ** 3
    f2 = u ** 2 + v ** 2 - 1
    return np.array([f1, f2])

# get the Jacobian matrix
def DF(x):
    u, v = x
    df1 = (-3 * u ** 2, 1)
    df2 = (2 * u, 2 * v)
    return np.array([df1, df2])

print('-'*70)
print('Use Newton’s Method to find the solution of the inconsistent system')
print('-'*70)
x_true = [(0.8260, 0.5636), (-0.8260, -0.5636)]
starting_points = [(1, 2), (-1, -2)]
for i, x_0 in enumerate(starting_points):
    x = Newton_Method(F, DF, x_0)
    print("(u, v):", tuple(x), "Ans:", tuple(x_true[i]))