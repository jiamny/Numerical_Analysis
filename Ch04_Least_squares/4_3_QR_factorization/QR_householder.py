import numpy as np

def norm(x):
    return np.sqrt(sum(x**2))


def QR_Householder_reﬂectors(A):
    m, n = np.shape(A)
    H = np.tile(np.identity(m), (n, 1, 1))
    R = A.copy()

    for i in range(n):
        A_i = R[i:, i]
        w = np.zeros(m - i)
        w[0] = norm(A_i) * (2 * (A_i[0] < 0) - 1)

        v = (A_i - w).reshape(-1, 1)
        P = np.dot(v, v.T) / norm(v) ** 2

        H[i, i:, i:] -= 2 * P
        R = np.dot(H[i], R)
    Q = np.linalg.multi_dot(H)

    return Q, R

print('-'*70)
print('Use Householder reﬂectors to ﬁnd the QR factorization of:')
print('-'*70)
A = np.array([[1, -4], [2, 3], [2, 2]])
print('A:\n', A)

Q, R = QR_Householder_reﬂectors(A)
print('Q:\n', Q.round(10))
print('R:\n', R.round(10))

print('-'*70)
print('Apply Householder reﬂectors to ﬁnd the full QR factorization of the following matrix:')
print('-'*70)
A = np.array([[4, 8, 1], [0, 2, -2], [3, 6, 7]])
print('A:\n', A)

Q, R = QR_Householder_reﬂectors(A)
print('Q:\n', Q.round(10))
print('R:\n', R.round(10))