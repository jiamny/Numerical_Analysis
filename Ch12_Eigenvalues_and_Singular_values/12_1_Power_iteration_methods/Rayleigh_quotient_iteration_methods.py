import numpy as np

def rayleigh_quotient_iteration(A, x, k=10):
    for _ in range(k):
        eigvec = x / np.linalg.norm(x)
        eigval = np.matmul(np.matmul(eigvec.T, A), eigvec)
        # stop if Singular matrix
        try:
            x = np.linalg.solve(A - eigval * np.eye(A.shape[0]), eigvec)
        except  np.linalg.LinAlgError as e:
            print("stop if singular matrix!")
            break
    eigvec = x / np.linalg.norm(x)
    eigval = np.matmul(np.matmul(eigvec.T, A), eigvec)
    return eigval, eigvec

print('-'*100)
print("Example:")
print('-'*100)
A = np.array([
    [1, 3],
    [2, 2]
])

x = np.random.rand(2)
eigval, eigvec = rayleigh_quotient_iteration(A, x, 25)
print('eigval: ', eigval)
print('eigvec: ', eigvec)