import numpy as np

def Gauss_Seidel_iteration_method(A, b, max_iter=200, eps=1e-8):
    assert (A.shape[0] == A.shape[1] == b.shape[0])
    n = A.shape[0]
    x0 = 0.01 * np.ones(n)
    x_iter = x0  # x_next表示x(k+1)
    for iteration in range(max_iter):
        for j in range(n):
            sum_g = np.dot(A[j, :j], x_iter[:j]) + np.dot(A[j, j + 1:], x_iter[j + 1:])
            x_iter[j] = (b[j] - sum_g) / A[j, j]
        precision = np.linalg.norm(b - np.dot(A, x_iter))
        if precision <= eps:
            break
    return x_iter

def Gauss_Seidel_ﬁxed_point_iteration(A, b, max_iter=200):
    assert(A.shape[0] == A.shape[1] == b.shape[0])
    n, m = A.shape[0], A.shape[1]
    x = np.zeros(b.shape, dtype=np.float32)
    D = np.diag(A)
    # R = A - np.diag(D)
    # np.triu(R)
    # np.tril(R)
    U = np.zeros((n, m), dtype=np.float32)
    L = np.zeros((n, m), dtype=np.float32)
    for r in range(n):
        for c in range(m):
            if r == c:
                U[r, c] = 0
                L[r, c] = 0
            else:
                if r < c:
                    L[r, c] = A[r, c]
                else:
                    U[r, c] = A[r, c]

    x_next = np.copy(x)
    for t in range(max_iter):
        print('t: ', t, ' ', x_next)
        x_before = np.copy(x_next)
        for j in range(n):
            x_next[j] = (b[j] - np.dot(U[j], x_before) - np.dot(L[j], x_next))/D[j]

    return x_next


print('-'*70)
print('Apply the Gauss-Seidel Method to the following system:')
print('-'*70)
A = np.array([
             [3, 1, -1],
             [2, 4, 1],
             [-1, 2, 5]
         ],dtype=np.float32
)

print('A:\n', A)
b = np.array(
[4, 1, 1],dtype=np.float32
)

print('b:\n', b)

x = Gauss_Seidel_iteration_method(A, b)
print("Gauss_Seidel_iteration_method(A, b) = ", x)

x = Gauss_Seidel_ﬁxed_point_iteration(A, b, 2)
print("Gauss_Seidel_ﬁxed_point_iteration(A, b) = ", x)