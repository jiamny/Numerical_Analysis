import numpy as np

def inverse_power_iteration(A, x, s, k=10):
    As = A - s * np.eye(A.shape[0])

    for _ in range(k):
        eigvec = x / np.linalg.norm(x)
        x = np.linalg.solve(As, eigvec)
        eigval = np.matmul(eigvec.T, x)

    eigvec = x / np.linalg.norm(x)
    eigval = 1 / eigval + s
    return eigval, eigvec

print('-'*100)
print("Example 1:")
print('-'*100)
A = np.array([
    [1, 3],
    [2, 2]
])

x = np.random.rand(2)

shift = -1.1
print('shift: ', shift)
eigval, eigvec = inverse_power_iteration(A, x, shift)
print('eigval: ', eigval)
print('eigvec: ', eigvec)
shift = 0
print('shift: ', shift)
eigval, eigvec = inverse_power_iteration(A, x, shift)
print('eigval: ', eigval)
print('eigvec: ', eigvec)
shift = 8
print('shift: ', shift)
eigval, eigvec = inverse_power_iteration(A, x, shift)
print('eigval: ', eigval)
print('eigvec: ', eigvec)

print('-'*100)
print("Example 2:")
print('-'*100)
A = np.array([
    [3,   2,  4],
    [2,   1,  2],
    [4,   2,  3]
])

x = np.random.rand(3)

shift = -1.1
print('shift: ', shift)
eigval, eigvec = inverse_power_iteration(A, x, shift)
print('eigval: ', eigval)
print('eigvec: ', eigvec)
shift = 0
print('shift: ', shift)
eigval, eigvec = inverse_power_iteration(A, x, shift)
print('eigval: ', eigval)
print('eigvec: ', eigvec)
shift = 8
print('shift: ', shift)
eigval, eigvec = inverse_power_iteration(A, x, shift)
print('eigval: ', eigval)
print('eigvec: ', eigvec)

