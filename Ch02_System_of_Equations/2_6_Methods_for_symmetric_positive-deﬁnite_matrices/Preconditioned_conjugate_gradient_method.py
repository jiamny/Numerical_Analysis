import numpy as np

def Preconditioned_conjugate_gradient_method(A, b):
    assert(A.shape[0] == A.shape[1])
    n = A.shape[0]
    x = np.zeros(A.shape[0]) #initial guess
    M = np.diag(A)
    r = b - np.dot(A, x)
    z = r/M
    d = np.copy(z)
    for k in range(n):
        if r.any() == 0:
            break
        alpha = np.dot(r.T, z)/np.dot(np.dot(d.T, A), d)
        pre_r = np.copy(r)
        x = x + np.dot(alpha, d)
        r = pre_r - np.dot(np.dot(alpha, A), d)
        pre_z = np.copy(z)
        z = r / M
        beta = np.dot(r, z) / np.dot(pre_r.T, pre_z)
        d = z + np.dot(beta, d)
    return x


print('-' * 70)
print('Apply Conjugate Gradient Method:')
print('-' * 70)
A = np.array([
            [2, 2],
            [2, 5]], dtype=np.float32)

b = np.array([6, 3], dtype=np.float32)
print('A:\n', A)
print('b:\n', b)

x = Preconditioned_conjugate_gradient_method(A, b)
print('Preconditioned_conjugate_gradient_method(A, b): ', x)