import numpy as np

def shifted_qr(A, tol=1e-14, max_count=1000):

    m = A.shape[0] # row size
    eigval = np.zeros(m)
    n = m
    while n > 1:
        count = 0
        while np.max(A[n-1, 0:n-1]) > tol and count < max_count:
            count += 1
            shift = A[n-1, n-1]
            Q, R = np.linalg.qr(A - shift * np.eye(n))
            A = np.matmul(R, Q) + shift * np.eye(n)
        if count < max_count:
            eigval[n-1] = A[n-1, n-1]
            n -= 1
            A = A[0:n, 0:n]
        else:
            disc = (A[n-2, n-2] - A[n-1, n-1])^2 + 4 * A[n-1, n-2] * A[n-2, n-1]
            eigval[n-1] = (A[n-2, n-2] + A[n-1,n-1] + np.sqrt(disc)) / 2
            eigval[n-2] = (A[n-2, n-2] + A[n-1,n-1] - np.sqrt(disc)) / 2
            n -= 2
            A = A[0:n, 0:n]
    if n > 0:
        eigval[0] = A[0, 0]
    return eigval

print('-'*100)
print("Example:")
print('-'*100)
A = np.array([
    [1, 3],
    [2, 2]
])

eigval = shifted_qr(A)
print('eigval: ', eigval)
