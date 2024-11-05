import numpy as np

def norm(x):
    return np.sqrt(sum(x**2))

def QR_modiﬁed_Gram_Schmidt_orthogonalization(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    k = n

    r = norm(A[:, 0])
    Q[:, 0] = A[:, 0] / r
    R[0, 0] = r

    for i in range(1, k):
        y = A[:, i]
        for j in range(i):
            r = np.dot(Q[:, j], y)
            y = y - r*Q[:, j]
            try:
                R[j, i] = r
            except:
                pass
        r = norm(y)
        Q[:, i] = y / r
        try:
            R[i, i] = r
        except:
            pass

    return Q, R

print('-'*70)
print('Use modiﬁed Gram Schmidt orthogonalization to find the full QR factorization of:')
print('-'*70)
A = np.array([[1, -4], [2, 3], [2, 2]])
print('A:\n', A)

Q, R = QR_modiﬁed_Gram_Schmidt_orthogonalization(A)
print('Q:\n', Q)
print('R:\n', R)

print('-'*70)
print('Apply modiﬁed Gram–Schmidt orthogonalization to ﬁnd the full QR factorization of the following matrix:')
print('-'*70)
A = np.array([[4, 8, 1], [0, 2, -2], [3, 6, 7]])
print('A:\n', A)

Q, R = QR_modiﬁed_Gram_Schmidt_orthogonalization(A)
print('Q:\n', Q)
print('R:\n', R)