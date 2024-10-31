import numpy as np

def Cholesky_factorization(A):
    n = A.shape[0]
    R = np.zeros(A.shape)
    for k in range(n):
        if A[k,k] < 0:
            break
        R[k,k] = np.sqrt(A[k,k])
        uT = np.copy(A[k,k+1:n]/R[k,k]).reshape(1, -1)
        R[k, k+1:n] = uT
        A[k+1:n, k+1:n] = A[k+1:n, k+1:n] - uT.T.dot(uT)
    return R

print('-' * 70)
print('Apply Cholesky Factorization:')
print('-' * 70)
A = np.array([
        [4, -2, 2],
        [-2, 2, -4],
        [2, -4, 11]
    ], dtype=np.float32)
print('A:\n', A)

R = Cholesky_factorization(A)
print('Cholesky_factorization(A):\n', R)
print('R.T*R:\n', R.T.dot(R))