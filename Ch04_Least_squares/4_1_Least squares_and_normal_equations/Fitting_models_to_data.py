import numpy as np
import matplotlib.pyplot as plt

#def least_square(A, b):
#    return np.linalg.inv(np.dot(A.T, A)).dot(A.T).dot(b)
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

    # solved by Gaussian elimination
    x = Gaussion_elimination(A_transpose_A, A_transpose_b)
    return x

def coef(x, y, degree):
    n = len(x)

    A = np.zeros((degree, n), dtype=x.dtype)
    for i in range(degree):
        A[i] = x ** i
    A = A.T
    b = y.reshape(n, 1)
    return Normal_equations_for_least_squares(A, b)

def poly_fit(x, y, degree):
    c = coef(x, y, degree)

    def f(x):
        ans = 0
        for i in range(degree):
            ans += c[i] * (x ** i)
        return ans

    return f

print('-'*70)
print('Find the line model y = c1 + c2 * t that best ﬁts the three data points')
print('-'*70)
point_x = [1, -1, 1]
point_y = [2, 1, 3]

print('-'*70)
print('Substitution of the data points into the model yields A and b')
print('-'*70)
A = np.array([[1,1], [1,-1], [1,1]], dtype=float)
b = np.array([[2], [1], [3]], dtype=float)

x_ = Normal_equations_for_least_squares(A, b)
error = np.sqrt(np.sum((b - np.dot(A, x_))**2))
print("x_: ", x_)
print("2-norm error: %f" % error)
x = np.arange(-2, 2, 0.1, dtype=float)
y = x_[0,0] + x_[1,0] * x
plt.scatter(point_x, point_y, s=40, c='b')
plt.plot(x, y, c='m')
plt.show()

print('-'*70)
print('Fitting data by least squares - example 1')
print('-'*70)
point_x = np.array([-1, 0, 1, 2], dtype=float)
point_y = np.array([1, 0, 0, -2], dtype=float)
degree = 2

c = coef(point_x, point_y, 2)
print("S = %fP + %f" % (c[1].item(), c[0].item()))

n = len(point_y)
f = poly_fit(point_x, point_y, degree)

x_range = np.arange(-2, 2, 0.1, dtype=float)
plt.scatter(point_x, point_y, s=40, c='b')
plt.plot(x_range, f(x_range), c='m')
plt.show()
error = np.sqrt(sum((point_y - f(point_x))**2) / n)
print("RMSE: %f" % error)

print('-'*70)
print('Fitting data by least squares - example 2')
print('-'*70)
year = np.array([1800, 1850, 1900, 2000], dtype=float)
co2 = np.array([280, 283, 391, 370], dtype=float)
n = len(year)
degrees = [2, 3, 4]
x_range = np.linspace(1800, 2000, 1000)

plt.rcParams['figure.figsize'] = [18, 6]
fig, axs = plt.subplots(1, 3, sharey=True)

for i, degree in enumerate(degrees):
    f = poly_fit(year, co2, degree)
    axs[i].scatter(year, co2, s=40, c='b')
    axs[i].plot(x_range, f(x_range), c='m')
    axs[i].set_title("Degree %d" % degree)

    error = np.sqrt(sum((co2 - f(year)) ** 2) / n)
    estimate = f(1950)
    print("Degree %d RMSE: %f" % (degree, error))
    print("Degree %d Estimate in 1950: %f" % (degree, estimate.item()))
plt.show()


