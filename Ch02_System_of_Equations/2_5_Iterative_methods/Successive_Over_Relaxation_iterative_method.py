import numpy as np

def SOR_iteration_method(A, b, max_iter=200, omega=1.5, eps=1e-8):
    assert (A.shape[0] == A.shape[1] == b.shape[0])
    n = A.shape[0]
    x0 = 0.01 * np.ones(n)
    x_next = x0  # x_next represents x(k+1)
    for iteration in range(max_iter):
        x_before = np.copy(x_next)
        for j in range(n):
            sum_g = np.dot(A[j, :j], x_next[:j]) + np.dot(A[j, j + 1:], x_before[j + 1:])
            x_next[j] = (b[j] - sum_g) / A[j, j]

            x_next[j] = x_before[j] + omega * (x_next[j] - x_before[j])

        precision = np.linalg.norm(b - np.dot(A, x_next))
        if precision <= eps:
            break
    return x_next


def SOR_ﬁxed_point_iteration(A, b, max_iter=200, omega=1.5):
    assert(A.shape[0] == A.shape[1] == b.shape[0])
    n, m = A.shape[0], A.shape[1]
    x = np.zeros(b.shape, dtype=np.float32)
    D = np.diag(A)
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
        x_before = np.copy(x_next)
        for j in range(n):
            x_next[j] = (1 - omega) * x_before[j] + omega * (b[j] - np.dot(U[j], x_before) - np.dot(L[j], x_next)) / D[j]

    return x_next

if __name__ == '__main__':
    print('-' * 70)
    print('Apply the Successive Over-Relaxation (SOR) Method to the following system:')
    print('-' * 70)
    A = np.array([
        [3, 1, -1],
        [2, 4, 1],
        [-1, 2, 5]
    ], dtype=np.float32)

    print('A:\n', A)
    b = np.array(
        [4, 1, 1], dtype=np.float32)
    print('b:\n', b)

    x = SOR_iteration_method(A, b)
    print("SOR_iteration_method(A, b) = ", x)

    x = SOR_ﬁxed_point_iteration(A, b, max_iter=200, omega=1.25)
    print("SOR_ﬁxed_point_iteration(A, b) = ", x)