import numpy as np

def PA_LU_decomposition(A):
    assert(A.shape[0] == A.shape[1])
    U = np.copy(A)
    L = np.zeros(A.shape, dtype=np.float32)
    P = np.eye(A.shape[0], dtype=np.float32)

    for j in range(U.shape[1]): #消去第j列的数
        # abs(U[j ,j]) as eliminate pivoting
        print("Before exchanged U：\n", U)
        max_row = j
        max_val = U[j, j]
        # cumulative permutation matrix
        P_plus = np.eye(U.shape[0], dtype=np.float32)
        for i in range(j + 1, U.shape[0]):
            if abs(U[i, j]) > abs(max_val):
                max_val = A[i, j]
                max_row = i

        # row exchange is required before eliminating
        L_commutator = np.copy(L[j, :])
        L[j, :] = np.copy(L[max_row, :])
        L[max_row, :] = np.copy(L_commutator)

        U_commutator = np.copy(U[j, :])
        U[j, :] = np.copy(U[max_row, :])
        U[max_row, :] = np.copy(U_commutator)

        # max_row = j，
        P_plus[j, j], P_plus[max_row, max_row] = 0, 0
        P_plus[j, max_row], P_plus[max_row, j] = 1, 1

        print("Current permutation matrix:\n", P_plus)
        P = np.matmul(P_plus, P)

        L[j, j] = 1
        # eliminate U[i, j]
        for i in range(j+1, U.shape[0]):
            mult_coeff = U[i, j]/U[j, j]
            L[i, j] = mult_coeff
            # Update
            for k in range(j, U.shape[1]):
                U[i, k] = U[i, k] - mult_coeff * U[j, k]
    print("After exchanged U:\n", U)
    return P, L, U


def back_substitution_U(U, b):
    x = np.zeros((U.shape[0], 1), dtype=U.dtype)
    for i in reversed(range(U.shape[0])):
        for j in range(i+1, U.shape[1]):
            b[i] = b[i] - U[i, j] * x[j]
        x[i] = b[i] / U[i, i]
    return x

def back_substitution_L(L, b):
    x = np.zeros((L.shape[0], 1), dtype=L.dtype)
    for i in range(L.shape[0]):
        for j in range(i):
            b[i] = b[i] - L[i, j] * x[j].item()

        x[i] = b[i] / L[i, i]
    return x


def PA_LU_factorization(P, L, U, b):
    # LUx = Pb ，set Ux = c
    # solve Lc = Pb for c
    # permute b
    b = np.matmul(P, b)
    c = back_substitution_L(L, b)
    print('c:\n', c)

    # solve Ux=c for x
    x = back_substitution_U(U, c)
    return x

print('-'*70)
print('Use the PA = LU factorization to solve the system Ax = b')
print('-'*70)
A = np.array([[2, 1, 5],
            [4, 4, -4],
            [1, 3, 1]],
        dtype=np.float32)

b = np.array([[5],
            [0],
            [6]],
        dtype=np.float32)

print('A:\n', A)
P, L, U = PA_LU_decomposition(A)
print('P:\n', P)
print('L:\n', L)
print('U:\n', U)
print('b:\n', b)
print('PA_LU_factorization(P, L, U, b) = \n', PA_LU_factorization(P, L, U, b))

A = np.array([[2, 3],
            [3, 2]],
        dtype=np.float32)
b = np.array([[4],
            [1]],
        dtype=np.float32)

print('A:\n', A)
P, L, U = PA_LU_decomposition(A)
print('P:\n', P)
print('L:\n', L)
print('U:\n', U)
print('b:\n', b)
print('PA_LU_factorization(P, L, U, b) = \n', PA_LU_factorization(P, L, U, b))