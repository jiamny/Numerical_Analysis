import numpy as np

def SVD(A):
    """
    :param A: traget matrix
    :return:
        m×r decomposed front r columns
        r rank dialog matrix
        n×r decomposed front r rows
    """

    # W = A^T*A
    W = np.dot(A.T, A)

    # eigh to get eigen value and eigen vector
    l, V = np.linalg.eigh(W)

    # sort eigen value
    l = np.sqrt(l[::-1])

    # ---------- 2. get r rank dialog matrix ----------
    # S remove v <= 0
    S = np.diag([v for v in l if v > 0])

    # get S's rank
    r = S.shape[0]

    # ---------- 3. get V matrix ----------
    V = V[:, -1:-1 - r:-1]

    # ---------- 4. get U matrix ----------
    U = np.hstack([(np.dot(A, V[:, i]) / l[i])[:, np.newaxis] for i in range(r)])

    return U, S, V.T

print('-'*100)
print("Example 1:")
print('-'*100)
A = np.array(
        [
            [0, 1],
            [1, 3/2]
        ])

U, S, V = SVD(A)
print("SVD() results：")
print(U, "\n", S, "\n", V)
print(np.dot(np.dot(U, S), V))
print("\n")

print("numpy api results：")
U2, S2, V2 = np.linalg.svd(A)
print(U2, "\n", S2, "\n", V2)
print(np.dot(np.dot(U2, np.diag(S2)), V2))

print('-'*100)
print("Example 2:")
print('-'*100)
A = np.array([[1, 1],
              [2, 2],
              [0, 0]])

print("Example 2 SVD() results：")
U, S, VT = SVD(A)
print('U:\n', U)
print('S:\n', S)
print('VT:\n', VT)
print(np.dot(np.dot(U, S), VT))
print("\n")

print("Example 2 numpy api results：")
U2, S2, V2 = np.linalg.svd(A)
print(U2, "\n", S2, "\n", V2)
if S2.ndim < 2:
    S2 = np.vstack((np.diag(S2), np.array([0, 0])))
print(np.dot(np.dot(U2, S2), V2))
