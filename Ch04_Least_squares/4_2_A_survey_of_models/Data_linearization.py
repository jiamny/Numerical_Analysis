import numpy as np
import matplotlib.pyplot as plt

def Gaussion_elimination(A, b, eps = 1e-6):
    # ----------------------------------
    # eliminating each column
    # ----------------------------------
    assert(A.shape[0] == A.shape[1])
    for j in range(0, A.shape[1] - 1):
        if abs(A[j, j]) < eps:
            raise ValueError("zero pivot encountered!")
            return

        for i in range(j+1, A.shape[0]):
            mult = A[i, j]/A[j, j]
            for k in range(j+1, A.shape[1]):
                A[i, k] = A[i, k] - mult * A[j, k]
            b[i] = b[i] - mult * b[j]

    # ----------------------------------
    # the back-substitution step
    # ----------------------------------
    x = np.zeros((A.shape[0], 1))
    for i in reversed(range(A.shape[0])):
        for j in range(i+1, A.shape[1]):
            b[i] = b[i] - A[i, j] * x[j].item()
        x[i] = b[i] / A[i, i]

    return x

def Normal_equations_for_least_squares(A, b):
    A_transpose = np.transpose(A)
    A_transpose_A = np.dot(A_transpose, A)
    A_transpose_b = np.dot(A_transpose, b)
    print('The components of the normal equations A.T * A:\n', A_transpose_A)
    # solved by Gaussian elimination
    x = Gaussion_elimination(A_transpose_A, A_transpose_b)
    return x

def exponential_model(x, c):
    return np.exp(c[0])*np.exp(c[1]*x)

def log_linearized_least_squares_error(x, y, c):
    return np.sum(((c[0] + c[1]*x) - np.log(y))**2)

print('-'*70)
print('Use model linearization to ﬁt y = c1*e**(c2*t) to the following world automobile supply data:')
print('-'*70)
years = np.array([1950, 1955, 1960, 1965, 1970, 1975, 1980])
cars = np.array([53.05e6, 73.04e6, 98.31e6, 139.78e6, 193.48e6, 260.20e6, 320.39e6])

A = np.ones((7, 2))
A[:,1] = years

b = np.log(cars).reshape(-1,1)
n = len(b)
coef = Normal_equations_for_least_squares(A, b).squeeze()
print('coef:\n', coef)
x_range = np.arange(1950, 1980, 1, dtype=float)
y_fited = exponential_model(x_range, coef)

plt.scatter(years, cars, s=50, c='g')
plt.plot(years, cars, label="cars = c1*exp(c2*years)" )
plt.plot( x_range, y_fited, 'm--',
         label="Fitted y = %e*exp(%e*t)" % (np.exp(coef[0]), coef[1]))
plt.legend()
plt.show()
error = np.sqrt(log_linearized_least_squares_error(years, cars, coef)/ n)
print("RMSE: %f" % error.item())

print('-'*70)
print('The time course of drug concentration y in the bloodstream: y = c1*t*e**(c2*t)')
print('-'*70)
def f(x, c):
    return np.exp(c[0])*x*np.exp(c[1]*x)

hour = np.array([1, 2, 3, 4, 5, 6, 7, 8])
concentration = np.array([8.0, 12.3, 15.5, 16.8, 17.1, 15.8, 15.2, 14.0])
A = np.ones((8, 2))
A[:, 1] = hour
b = (np.log(concentration) - np.log(hour)).reshape(-1,1)
n = len(b)

coef = Normal_equations_for_least_squares(A, b).squeeze()
print('coef:\n', coef)
x_range = np.linspace(1, 10, 100)
y_fited = f(x_range, coef)

plt.scatter(hour, concentration, s=50, c='g')
plt.scatter(-1/coef[1], f(-1/coef[1], coef), s=80, c='r', label="Maximum: %f" % f(-1/coef[1], coef))
plt.plot(hour, concentration, label="concentration = c1*hour*e**(c2*hour)" )
plt.plot(x_range, y_fited, 'm--',
         label="Fitted y = %e*t*exp(%e*t)" % (np.exp(coef[0]), coef[1]))
plt.legend()
plt.show()
error = np.sqrt(np.sum((concentration - f(hour, coef))**2)/ n)
print("RMSE: %f" % error.item())