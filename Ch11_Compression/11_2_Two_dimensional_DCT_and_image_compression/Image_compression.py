import cv2
import numpy as np
from matplotlib import pyplot as plt

def generate_dct_matrix(n):
    C = np.ones((n, n)) * (1/np.sqrt(2))
    for i in range(1, n):
        for j in range(n):
            C[i, j] = np.cos( (np.pi * i + i * j * 2 * np.pi) / (2 * n) )
    C *= np.sqrt(2 / n)
    return C

# 2D-DCT Interpolation Theorem
def two_dim_dct_interpolation(Y, s, t):
    val = 0
    n = Y.shape[0]
    for k in range(n):
        for l in range(n):
            ak = 1 if k > 0 else (1 / np.sqrt(2))
            al = 1 if l > 0 else (1 / np.sqrt(2))
            val += Y[k,l] * ak * al \
                * np.cos(k * (2 * s + 1) * np.pi / (2 * n)) \
                * np.cos(l * (2 * t + 1) * np.pi / (2 * n))
    return 2 * val / n

print('-'*100)
print("Read image:")
print('-'*100)
image = cv2.imread('bRpvP3A.png', cv2.IMREAD_GRAYSCALE)
print(image.shape)
plt.imshow(image, cmap='gray')
plt.show()

print('-'*100)
print("Crude compression—each 8 × 8 square of pixels is colored by its average grayscale value.:")
print('-'*100)
poor_quality_image = np.zeros(image.shape)
block_size = 16
print('block_size: ', block_size)
# Do average for each (block_size x block_size) block
for i in range(0, poor_quality_image.shape[0], block_size):
    for j in range(0, poor_quality_image.shape[1], block_size):
        block = image[i:i + block_size, j:j + block_size]
        poor_quality_image[i:i + block_size, j:j + block_size] = block.mean()

plt.imshow(poor_quality_image, cmap='gray')
plt.show()

print('-'*100)
print("2D-DCT Interpolation to process image compression:")
print('-'*100)

print('subtracted 256/2 = 128 from the pixel numbers to make them approximately centered around zero.')
X = image.copy().astype(np.int16)
two_dim_dct_image = np.zeros(image.shape, dtype = np.int16)

X -= 128
C = generate_dct_matrix(two_dim_dct_image.shape[0])

print('D-DCT) of the n × n matrix X is the matrix Y = C*X*C.T')
Y = np.matmul(C, np.matmul(X, C.T))

print('round to nearest integer for simplicity')
Y = np.round(Y)

print('create low-pass filter')
lowpass_mask = np.empty(Y.shape)
for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
        lowpass_mask[i, j] = 1 if i + j <= Y.shape[0] - 1 else 0

print('process with low-pass filter')
Y = Y * lowpass_mask

print('Reconstruct the image by the inverse 2D-DCT as C.T*Y_low*C, and adding back the 128.')
two_dim_dct_image = np.matmul(C.T, np.matmul(Y, C))
two_dim_dct_image += 128

print('Show image')
plt.imshow(two_dim_dct_image.astype(np.uint8), cmap='gray')
plt.show()