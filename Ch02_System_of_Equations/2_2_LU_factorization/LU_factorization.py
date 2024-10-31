import numpy as np

def LU_decomposition(A):
    """LU implementation."""

    if not A.shape[0] == A.shape[1]:
        raise ValueError("Input matrix must be square.")

    n = A.shape[0]

    L = np.zeros((n, n), dtype=np.float64)
    U = np.zeros((n, n), dtype=np.float64)

    U[:] = A

    np.fill_diagonal(L, 1)

    for col in range(n-1):
        for row in range(col + 1, n):
            L[row, col] = U[row, col] / U[col, col]
            U[row, col:] = U[row, col:] - L[row, col] * U[col, col:]
            U[row, col] = 0

    return (L, U)

def back_substitution_U(U, b):
    x = np.zeros((U.shape[0], 1), dtype=U.dtype)
    for i in reversed(range(U.shape[0])):
        for j in range(i+1, U.shape[1]):
            b[i] = b[i] - U[i, j] * x[j]
        x[i] = b[i] / U[i, i]

    return x

def back_substitution_L(L, b):
    x = np.zeros((L.shape[0], 1), dtype=L.dtype)
    for i in range(L.shape[0]):
        for j in range(i):
            b[i] = b[i] - L[i, j] * x[j].item()

        x[i] = b[i] / L[i, i]

    return x

def LU_back_substitution(L, U, b):
    # set vector c = U x
    # (a) Solve Lc = b for c.
    c = back_substitution_L(L, b)  # lower triangular
    print(c)
    # (b) Solve U x = c for x.
    x = back_substitution_U(U, c)  # upper triangular
    return x

print('-'*70)
print('Find the solution of linear equations with LU decomposition')
print('-'*70)
A = np.array([[1, 2, -1],
              [2, 1, -2],
              [-3, 1, 1]], dtype=np.float64)

b = np.array([[3],
              [3],
              [-6]], dtype=np.float64)

print('A=\n', A)
print('b=\n', b)
L, U = LU_decomposition(A)  # A=LU
print('L=\n', L)
print('U=\n', U)
# back substitution
x = LU_back_substitution(L, U, b)
print('LU_back_substitution(L, U, b) =\n', x)