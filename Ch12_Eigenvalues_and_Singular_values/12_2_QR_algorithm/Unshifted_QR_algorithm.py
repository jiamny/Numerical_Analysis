import numpy as np


def unshifted_qr(A, k=10):
    m, n = A.shape
    Q = np.eye(m)
    Qbar = Q
    R = A
    for _ in range(k):
        Q, R = np.linalg.qr(np.matmul(R, Q))
        Qbar = np.matmul(Qbar, Q)
    eigval = np.diag(np.matmul(R, Q))
    eigvec = Qbar
    return eigval, eigvec

print('-'*100)
print("Example:")
print('-'*100)
A = np.array([
    [1, 3],
    [2, 2]
])

eigval, eigvec = unshifted_qr(A)
print('eigval: ', eigval)
print('eigvec: ', eigvec)