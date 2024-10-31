import numpy as np

def partial_pivoting_gaussion_elimination(A, b):
    assert(A.shape[0] == A.shape[1])
    for j in range(0, A.shape[1] - 1):
        max_row = j
        max_val = A[j, j]

        # partial pivoting asks that we select the pth row, where |Ap1 | ≥ |Ai1 |
        for i in range(j + 1, A.shape[0]):
            if abs(A[i, j]) > abs(max_val):
                max_val = A[i, j]
                max_row = i

        # row is swapped with the pivot row
        A_commutator = np.copy(A[j, :])
        A[j, :] = np.copy(A[max_row, :])
        A[max_row, :] = np.copy(A_commutator)
        b_commutator = np.copy(b[j, :])
        b[j, :] = np.copy(b[max_row, :])
        b[max_row, :] = np.copy(b_commutator)

        # eliminate A[i, j]
        for i in range(j+1, A.shape[0]):
            mult_coeff = A[i, j]/A[j, j] 
            # Update
            for k in range(j+1, A.shape[1]):
                A[i, k] = A[i, k] - mult_coeff * A[j, k]
            b[i] = b[i] - mult_coeff * b[j] 

    # ----------------------------------
    # the back-substitution step
    # ----------------------------------
    x = np.zeros((A.shape[0], 1))
    for i in reversed(range(A.shape[0])):
        for j in range(i+1, A.shape[1]):
            b[i] = b[i] - A[i, j] * x[j].item()
        x[i] = b[i] / A[i, i]

    return x


print('-'*70)
print('Find the solution of linear equations with partial pivoting Gaussian elimination')
print('-'*70)
A = np.array([[1, -1, 3],
            [-1, 0, -2],
            [2, 2, 4]],
        dtype=np.float32)
b = np.array([[-3],
            [1],
            [0]],
        dtype=np.float32)

x = partial_pivoting_gaussion_elimination(A, b)
print('Partial pivoting Gaussian elimination(A, b) = \n', x)