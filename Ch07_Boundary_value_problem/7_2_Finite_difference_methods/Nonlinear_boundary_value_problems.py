import numpy as np
import matplotlib.pyplot as plt

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

def jac(w,inter,bv,n):
    a = np.zeros((n,n))
    h=(inter[1] -inter[0])/(n+1)

    for j in range(n):
        a[j,j] =- 2+h*h*np.sin(w[j][0])

    for j in range(n-1):
        a[j,j+1] = 1-h/2
        a[j+1,j] = 1+h/2

    return a

def f(w,inter,bv,n):
    y=np.zeros((n,1))
    h=(inter[1] - inter[0])/(n+1)
    y[0] = bv[0]-(2+h**2)*w[0]+h**2*w[0]**2+w[1]
    y[n-1] = w[n-2] -(2+h**2)*w[n-1]+h**2*w[n-1]**2+bv[1]
    for i in range(1, n-1, 1): # i=2:n-1
        y[i] = w[i-1] -(2+h**2)*w[i]+h**2*w[i]**2+w[i+1]

    return y

# Nonlinear Finite Difference Method for BVP
def nlbvpfd(inter, bv, n):
    a = inter[0]
    b = inter[1]
    ya = bv[0]
    yb = bv[1]
    h = (b - a) / (n + 1)
    w = np.zeros((n, 1))

    for i in range(20):
        A = jac(w, inter, bv, n)
        B = f(w, inter, bv, n)
        print(B.shape)
        #C = np.matmul(np.linalg.inv(A), B)
        C = partial_pivoting_gaussion_elimination(A, B)
        w = w - C

    return w

print('-'*100)
print("Solve the BVP (7.9) y'' = y - y^2, y(0) = 1, y(1) = 4:")
print('-'*100)
n = 40
inter = [0, 1]
bv = [1, 4]

h = (inter[1] - inter[0]) / (n + 1)
X = []
X.append(inter[0])
for i in range(1, n-1, 1):
    X.append(inter[0]+h*i)
X.append(inter[1])

w = nlbvpfd(inter, bv, n)
print('Solutions of Nonlinear BVP:\n', w)
Y = []
Y.append(bv[0])
for i in range(1, n-1, 1):
    Y.append(w[i-1][0])
Y.append(bv[1])

plt.plot(list(X), list(Y))
plt.xlabel('x')
plt.ylabel('y')
plt.show()


