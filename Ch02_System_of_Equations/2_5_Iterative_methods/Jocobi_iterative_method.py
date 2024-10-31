import numpy as np

# check the given matrix is Diagonally Dominant Matrix or not.
def isDDM(m, n):
    # for each row
    for i in range(0, n):
        # for each column, finding sum of each row.
        sum = 0
        for j in range(0, n):
            sum = sum + abs(m[i][j])
        # removing the diagonal element.
        sum = sum - abs(m[i][i])
        # checking if diagonal element is less than sum of non-diagonal element.
        if (abs(m[i][i]) < sum):
            return False
    return True

def JacobIterativeMethod(A, b, max_iter=200, eps=1e-8):
    assert (A.shape[0] == A.shape[1] == b.shape[0])
    n = A.shape[0]
    x0 = 0.01 * np.ones(n)
    x_next = np.copy(x0)  # x_next represents x(k+1)， x_before represent x(k)
    iteration = 0
    for iteration in range(max_iter):
        x_before = np.copy(x_next)
        for i in range(n):
            sum_j = np.dot(A[i, :i], x_before[:i]) + np.dot(A[i, i + 1:], x_before[i + 1:])
            x_next[i] = (b[i] - sum_j) / A[i, i]
        precision = np.linalg.norm(b - np.dot(A, x_next))

        if precision <= eps:
            break
    if iteration >= max_iter - 1:
        if not isDDM(A, n):
            print('The Jacobi Method not converge')
    return x_next

def Jacobi_ﬁxed_point_iteration(A, b, max_iter=200):
    assert(A.shape[0] == A.shape[1] == b.shape[0])
    x = np.zeros(b.shape, dtype=np.float32)
    D = np.diag(A)
    R = A - np.diag(D)
    for t in range(max_iter):
        x = (b - np.matmul(R, x))/D

    return x

print('-'*70)
print('Apply the Jacobi Method to the system 3u + v = 5, u + 2v = 5')
print('-'*70)

A = np.array([
             [3, 1],
             [1, 2],
         ],dtype=np.float32
)

b = np.array(
[5, 5],dtype=np.float32
)

x = JacobIterativeMethod(A, b)
print("JacobIterativeMethod(A, b) = ", x)

x = Jacobi_ﬁxed_point_iteration(A, b)
print("Jacobi_ﬁxed_point_iteration(A, b) = ", x)

print('-'*70)
print('Apply the Jacobi Method to the following system:')
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

x = JacobIterativeMethod(A, b)
print("JacobIterativeMethod(A, b) = ", x)

x = Jacobi_ﬁxed_point_iteration(A, b)
print("Jacobi_ﬁxed_point_iteration(A, b) = ", x)
