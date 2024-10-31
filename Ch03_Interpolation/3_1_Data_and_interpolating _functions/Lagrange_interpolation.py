import matplotlib.pyplot as plt
import numpy as np
import sympy

def L_k(x: float, X: list[float], k:int) -> float:
    """
    x: Evaluation point
    X: Set of nodes of length n
    k: Index of k-th node
    Return the n + 1 lagrange polynomial evaluated in x
    """
    L = 1
    for j in range(len(X)):
        if j != k:
            L *= (x - X[j])/(X[k] - X[j])
    return L

def Lagrange_interpolating_polynomial(x:float, X: list[float], Y: list[float]) -> float:
    """
    x: Evaluation point
    X: X-coordinates of interpolation points
    Y: Y-coordinates of interpolation points
    Return the interpolation formula evaluated in x
    """
    P_n_1 = 0
    for k in range(len(X)):
        P_n_1 += Y[k] * L_k(x, X, k)
    return P_n_1

def Lagrange_interpolating_polynomial_symbol(x, y):
    n = len(x)
    t = sympy.Symbol("x")
    polynomial =  0.0
    for i in range(n):
        base_fun = 1.0  # i != j
        for j in range(i):
            base_fun = base_fun * (t - x[j]) / (x[i] - x[j])
        for j in range(i + 1, n):
            base_fun = base_fun * (t - x[j]) / (x[i] - x[j])
        #self.interp_base_fun.append(sympy.expand(base_fun))  # 存储插值基函数
        polynomial += base_fun * y[i]
    polynomial = sympy.expand(polynomial)
    polynomial = sympy.Poly(polynomial, t)
    #poly_coefficient = polynomial.coeffs()
    #poly_degree = polynomial.degree()
    #coefficient_order = polynomial.monoms()
    return polynomial

print('-' * 70)
print('Apply Lagrange interpolating to f(x) = 1/2 *x** 2 − 1/2*x + 1:')
print('-' * 70)

# Interval of x
xmin,xmax = -2,4

# Set of points
n = 3
X = [0, 2, 3]
Y = [1, 2, 4]

# Interpolation
sub_n = 20
subX = np.linspace(xmin,xmax,sub_n)
I = [Lagrange_interpolating_polynomial(x, X, Y) for x in subX]

# Real values
realY = 1/2*subX**2 - 1/2*subX + 1

# Plot
plt.close()
fig, ax = plt.subplots()

nodes = ax.scatter(X, Y, s=80, c='b')
interpolation, = ax.plot(subX, I, "r", linewidth=1)
real, = ax.plot(subX, realY, linestyle=':', linewidth=4, color='green')

fig.legend((nodes, interpolation, real), ("Given points", "Interpolation", "Real curve"))
plt.title("Lagrange Interpolation")
plt.grid(True)
plt.show()

polynomial = Lagrange_interpolating_polynomial_symbol(X, Y)
print('Lagrange_interpolating_polynomial_symbol(X, Y): ', polynomial)

print('-' * 70)
print('Apply Lagrange interpolating to the points (0, 2), (1, 1), (2, 0), and (3, −1)')
print('-' * 70)
X = [0, 1, 2, 3]
Y = [2, 1, 0, -1]
polynomial = Lagrange_interpolating_polynomial_symbol(X, Y)
print('Lagrange_interpolating_polynomial_symbol(X, Y): ', polynomial)