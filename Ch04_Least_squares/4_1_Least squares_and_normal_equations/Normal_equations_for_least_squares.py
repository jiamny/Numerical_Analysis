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

def Normal_equations_for_least_squares(A, b):
    A_transpose = np.transpose(A)
    A_transpose_A = np.dot(A_transpose, A)
    A_transpose_b = np.dot(A_transpose, b)

    # solved by Gaussian elimination
    x = Gaussion_elimination(A_transpose_A, A_transpose_b)
    return x

print('-'*70)
print('Use the normal equations to ﬁnd the least squares solution of the inconsistent system')
print('-'*70)

A = np.array([[1, 1],
              [1, -1],
              [1, 1]], dtype=np.float64)

b = np.array([[2],
              [1],
              [3]], dtype=np.float64)

print('A\n:', A)
print('b\n:', b)
x = Normal_equations_for_least_squares(A, b)
print('Normal_equations_for_least_squares(A, b): ', x)
r = b - np.matmul(A, x)
print('The residual of the least squares solution r = ', r)

print('-'*70)
print('To measure our success at ﬁtting the data')
print('-'*70)
EL = np.sqrt(np.sum(np.power(r, 2)))
print('Euclidean norm = ', EL)
SE = np.sum(np.power(r, 2))
print('The squared error = ', SE)
RMSE = np.sqrt(SE/len(r))
print('Root mean squared error = ', RMSE)

A = np.array([[1, -4],
              [2, 3],
              [2, 2]], dtype=np.float64)

b = np.array([[-3],
              [15],
              [9]], dtype=np.float64)

print('A\n:', A)
print('b\n:', b)
x = Normal_equations_for_least_squares(A, b)
print('The solution of the normal equations: ', x)
r = b - np.matmul(A, x)
print('The residual of the least squares solution r = ', r)
EL = np.sqrt(np.sum(np.power(r, 2)))
print('Euclidean norm = ', EL)
SE = np.sum(np.power(r, 2))
print('The squared error = ', SE)
RMSE = np.sqrt(SE/len(r))
print('Root mean squared error = ', RMSE)