import numpy as np


# Help function
def is_zero_vector(v):
    if v.ndim != 1 and not np.isscalar(v):
        raise TypeError('v is not a vector')
    return not np.any(v)

def power_iteration(A, x, k=10):
    if is_zero_vector(x):
        raise ValueError('x is a zero vector')
    for _ in range(k):
        eigvec = x / np.linalg.norm(x)
        x = np.matmul(A, eigvec)
        eigval = np.matmul(np.matmul(eigvec.T, A), eigvec)
    eigvec = x / np.linalg.norm(x)
    return eigval, eigvec

print('-'*100)
print("Example:")
print('-'*100)
A = np.array([
    [1, 3],
    [2, 2]
])

x = np.random.rand(2)
eigval, eigvec = power_iteration(A, x, 25)
print('eigval: ', eigval)
print('eigvec: ', eigvec)