import numpy as np


def normalized_simultaneous_iteration(A, k=10):

    m, n = A.shape
    Q = np.eye(m)
    for _ in range(k):
        Q, R = np.linalg.qr(np.matmul(A, Q))
    eigval = np.diag(np.matmul(np.matmul(Q.T, A), Q))
    eigvec = Q
    return eigval, eigvec

print('-'*100)
print("Example:")
print('-'*100)
A = np.array([
    [1, 3],
    [2, 2]
])

eigval, eigvec = normalized_simultaneous_iteration(A)
print('eigval: ', eigval)
print('eigvec: ', eigvec)