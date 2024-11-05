import numpy as np
import matplotlib.pyplot as plt


def Conjugate_gradient_method(A, b, initial_guess=0, iter_num=0):
    n = len(A)
    x = initial_guess * np.ones(n)
    r = b - A.dot(x)
    d = r

    if iter_num == 0:
        iter_num = n

    for _ in range(iter_num):
        alpha = r.dot(r) / d.dot(A).dot(d)
        x = x + alpha * d
        beta = np.dot(r - alpha * A.dot(d), r - alpha * A.dot(d)) / r.dot(r)
        r = r - alpha * A.dot(d)
        d = r + beta * d

        if r.max() == 0:
            break
    return x


def Preconditioned_GMRES(A, b, precondition, initial_guess=0, iter_num=0):
    n = len(A)

    if precondition == "Jacobi":
        M = np.diag(A)
        M_inv = np.diag(1 / M)

    elif precondition == "Gauss_Seidel":
        D = np.diag(A)
        M = (np.tril(A) / D).dot(np.triu(A))
        M_inv = np.linalg.inv(M)

    else:
        raise ValueError("Please give the right preconditioner!")

    x = initial_guess * np.ones(n)
    r = b - A.dot(x)
    d = r.dot(M_inv)
    z = d

    if iter_num == 0:
        iter_num = n

    for _ in range(iter_num):
        alpha = r.dot(z) / d.dot(A).dot(d)
        x = x + alpha * d
        beta = np.dot(r - alpha * A.dot(d), (r - alpha * A.dot(d)).dot(M_inv)) / r.dot(z)
        r = r - alpha * A.dot(d)
        z = r.dot(M_inv)
        d = z + beta * d

        if r.max() == 0:
            break
    return x

def creat_A(n):
    A = np.zeros((n, n))

    for i in range(n):
        A[i, i] = i + 1
        try:
            A[i, i + 1] = 0.5
            A[i + 1, i] = 0.5
            A[i, i + 2] = 0.5
            A[i + 2, i] = 0.5
        except:
            pass
    return A

print('-'*70)
print('Let A be the n × n matrix with n = 1000 and entries A(i, i) = i, A(i, i + 1) = A(i + 1, i) = 1/2, \
A(i, i + 2) = A(i + 2, i) = 1/2 for all i that ﬁt within the matrix.')
print('-'*70)
n = 1000
A = creat_A(n)
print('A:\n', A[0:10, 0:10])

print('-'*70)
print('Print the nonzero structure of A:')
print('-'*70)
plt.matshow(A != 0, cmap='binary')
plt.show()

print('-'*70)
print('Apply the Conjugate Gradient Method, without preconditioner, with the Jacobi preconditioner, and with the Gauss–Seidel preconditioner:')
print('-'*70)
x_e = np.ones(n)
b = A.dot(x_e)
tol = 1e-10
max_iter_n = 30

error_matrix = np.zeros((3, max_iter_n))
error_matrix[:, 0] = 1

print('-'*70)
print('Conjugate Gradient Method, without preconditioner:')
print('-'*70)
iter_n = 0
x = np.zeros(n)

while True:
    x = Conjugate_gradient_method(A, b, initial_guess=x, iter_num=1)
    error = abs(x - x_e).max()
    iter_n += 1

    if iter_n < max_iter_n:
        error_matrix[0, iter_n] = error

    if error < tol:
        break

backward_error = (A.dot(x) - b).max()
print("Iter: %d / Backward error: %g" % (iter_n, backward_error))

print('-'*70)
print('With the Jacobi preconditioner:')
print('-'*70)
iter_n = 0
x = np.zeros(n)

while True:
    x = Preconditioned_GMRES(A, b, "Jacobi", initial_guess=x, iter_num=1)
    error = abs(x - x_e).max()
    iter_n += 1

    if iter_n < 30:
        error_matrix[1, iter_n] = error

    if error < tol:
        break

backward_error = (A.dot(x) - b).max()
print("Iter: %d / Backward error: %g" % (iter_n, backward_error))

print('-' * 70)
print('With the Gauss–Seidel preconditioner:')
print('-' * 70)
iter_n = 0
x = np.zeros(n)

while True:
    x = Preconditioned_GMRES(A, b, "Gauss_Seidel", initial_guess=x, iter_num=1)
    error = abs(x - x_e).max()
    iter_n += 1

    if iter_n < 30:
        error_matrix[2, iter_n] = error

    if error < tol:
        break

backward_error = (A.dot(x) - b).max()
print("Iter: %d / Backward error: %g" % (iter_n, backward_error))

plt.plot(np.arange(30), error_matrix[0], label="No precond.")
plt.plot(np.arange(30), error_matrix[1], label="Jacobi precond.")
plt.plot(np.arange(10), error_matrix[2, :10], label="Gauss-Seidel precond.")
plt.yscale("log")
plt.legend()
plt.show()