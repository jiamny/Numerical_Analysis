import numpy as np

def Jocobi_sparse(A, b, max_iter=200):
    x = np.zeros(b.shape, dtype=np.float32)
    d = np.diag(A)
    R = A - np.diag(d)

    for t in range(max_iter):
        x = (b - R.dot(x))/d
    return x

print('-'*70)
print('Sparse matrix computations')
print('-'*70)
n = 10
A = np.zeros((n, n))
for i in range(n):
    A[i, i] = 3
    if i < n - 1:
        A[i, i + 1] = -1
        A[i + 1, i] = -1
print('A:\n', A)

b = np.ones(n)
b[0], b[-1] = 2, 2
x_true = np.ones(n)
print('b:\n', b)

print('Jocobi_sparse(A, b):\n', Jocobi_sparse(A, b))
