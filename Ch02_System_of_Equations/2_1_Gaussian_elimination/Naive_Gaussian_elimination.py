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

print('-'*70)
print('Find the solution of linear equations with Gaussian elimination')
print('-'*70)
eps = 1e-6
A = np.array([[1, 2, -1],
              [2, 1, -2],
              [-3, 1, 1]], dtype=np.float64)

b = np.array([[3],
              [3],
              [-6]], dtype=np.float64)

print('A=\n', A)
print('b=\n', b)
x = Gaussion_elimination(A, b, eps)
print('Gaussion_elimination(A, b, eps) =\n', x)