import numpy as np

def Gaussion_elimination(A, b, eps):
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

def BVP_finite_element_method(a, b, ya, yb, n):
    h = (b - a) / (n + 1)
    alpha = (8 / 3) * h + 2 / h
    beta = (2 / 3) * h - 1 / h
    A = np.zeros(n * n).reshape(n, n)
    np.fill_diagonal(A, alpha)
    dia_range = np.arange(n - 1)
    A[dia_range, dia_range + 1] = beta
    A[dia_range + 1, dia_range] = beta
    b = np.zeros(n)
    b[0] = -ya * beta
    b[-1] = -yb * beta
    #x = np.linalg.solve(A, b)
    x = Gaussion_elimination(A, b, eps = 1e-6)
    return x

print('-'*100)
print("Apply the Finite Element Method to the BVP y'' = 4y, y(0) = 1, y(1) = 3:")
print('-'*100)

w = BVP_finite_element_method(0, 1, 1, 3, 3)
print('The approximate solution w:\n', w)
