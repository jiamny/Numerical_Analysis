import numpy as np

def Conjugate_gradient_method(A, b):
    assert(A.shape[0] == A.shape[1])
    n = A.shape[0]
    x = np.zeros(A.shape[0]) #initial guess
    d = r = b - A.dot(x)

    for k in range(n):
        if r.any() == 0:
            break
        dt_A_d = np.dot(np.matmul(d.T, A), d)
        alpha = np.dot(r.T, r)/dt_A_d

        # update
        x = x + np.dot(alpha, d)
        pre_r = np.copy(r)
        r = r - np.dot(np.dot(alpha, A), d)
        beta = r.T.dot(r)/pre_r.T.dot(pre_r)
        d = r + np.dot(beta, d)

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

x = Conjugate_gradient_method(A, b)
print('Conjugate_gradient_method(A, b): ', x)