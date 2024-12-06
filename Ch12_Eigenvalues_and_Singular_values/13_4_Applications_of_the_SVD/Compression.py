import cv2
import numpy as np
from matplotlib import pyplot as plt

def approximation(A, p):
    U, s, V_T = np.linalg.svd(A)
    B = np.zeros(A.shape)
    for i in range(p):
        B += s[i]*U[:,i].reshape(-1, 1).dot(V_T[i, :].reshape(1, -1))
    return B

print('-'*100)
print("Result of compression and decompression image by SVD:")
print('-'*100)
image = cv2.imread('bRpvP3A.png', cv2.IMREAD_GRAYSCALE)
print("image shape: ", image.shape)
plt.imshow(image, cmap='gray')
plt.show()

num_ps = [8, 16, 32]
X = image.copy().astype(np.int16)
X -= 128

plt.figure(figsize=(16, 5))
for i, p in enumerate(num_ps):
    X1 =  approximation(X, p)
    X1 += 128
    plt.subplot(1, 3, i+1)
    plt.imshow(X1.astype(np.uint8), cmap='gray')
    plt.title('Number of singular values retained = %d' % (p))
    print('Compression and decompression image by SVD with p = ', p)
plt.show()