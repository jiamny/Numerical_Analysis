import numpy as np
import matplotlib.pyplot as plt

def approximation(A, p):
    U, s, V_T = np.linalg.svd(A)
    B = np.zeros(A.shape)
    for i in range(p):
        B += s[i]*U[:,i].reshape(-1, 1).dot(V_T[i, :].reshape(1, -1))
    return B

print('-'*100)
print("Find the best one-dimensional subspace ﬁtting the data vectors [3, 2], [2, 4], [−2, −1], [−3, −5]:")
print('-'*100)

A = np.array(
         [
             [3, 2, -2, -3],
             [2, 4, -1, -5]
         ])
print(A.shape)
A1 =  approximation(A, 1)
print('A1:\n', A1)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for i in range(A.shape[1]):
    plt.arrow(0, 0, A[0, i], A[1,i], head_width=0.15, head_length=0.1, fc='r', ec='b')
plt.title('data vectors to be projected')
plt.axhline(y=0, xmin=0., xmax=1.0, c='k')
plt.axvline(x=0, ymin=0., ymax=1.0, c='k')

plt.subplot(1, 2, 2)
for i in range(A.shape[1]):
    plt.arrow(0, 0, A[0, i], A[1,i], head_width=0.15, head_length=0.1, fc='r', ec='b')
for i in range(A1.shape[1]):
    plt.arrow(0, 0, A1[0, i], A1[1,i], head_width=0.15, head_length=0.1, fc='r', ec='m', linestyle=':')
plt.title('the orthogonal projections down to the subspace')
plt.axhline(y=0, xmin=0., xmax=1.0, c='k')
plt.axvline(x=0, ymin=0., ymax=1.0, c='k')
plt.show()


