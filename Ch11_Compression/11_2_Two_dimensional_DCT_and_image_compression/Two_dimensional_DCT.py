import numpy as np

def generate_dct_matrix(n):
    C = np.ones((n, n)) * (1/np.sqrt(2))
    for i in range(1, n):
        for j in range(n):
            C[i, j] = np.cos( (np.pi * i + i * j * 2 * np.pi) / (2 * n) )
    C *= np.sqrt(2 / n)
    return C

print('-'*100)
print("Find the 2D Discrete Cosine Transform of the data in Figure 11.4(a):")
print('-'*100)

X = np.array([
    [1, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 1]
])
C = generate_dct_matrix(X.shape[0])
Y = np.matmul(C, np.matmul(X, C.T))
Y = np.round(Y, 4)
print(Y)