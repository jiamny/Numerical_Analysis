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

# linear quantization, constant p, called the loss parameter
def linear_quantization(p, m, n):
    Q = np.empty((m, n))
    for k in range(m):
        for l in range(n):
            Q[k, l] = 8 * p * (k + l + 1)
    return Q

print('-'*100)
print('Result of linear quantization for loss parameter p = 1, 2, and 4:')
print('-'*100)

image = cv2.imread('bRpvP3A.png', cv2.IMREAD_GRAYSCALE)
loss_ps = [1, 2, 4]
plt.figure(figsize=(16, 5))
for k in range(len(loss_ps)):
    print('--------------------------------- loss parameter = ', loss_ps[k])
    print('Add quantization into image compression')
    X = image.copy().astype(np.int16)
    two_dim_dct_quantization_image = np.zeros(image.shape, dtype=np.int16)

    print('approximately centered around zero')
    Xd = X.astype(np.float64)
    Xd -= 128

    C = generate_dct_matrix(two_dim_dct_quantization_image.shape[0])
    Y = np.matmul(C, np.matmul(Xd, C.T))

    print('linear quantization quant_size = 8')
    quant_size = 8
    Q = linear_quantization(loss_ps[k], quant_size, quant_size)
    Yq = Y.copy()
    for i in range(0, Y.shape[0], quant_size):
        for j in range(0, Y.shape[1], quant_size):
            block = Y[i:i + quant_size, j:j + quant_size]
            Yq[i:i + quant_size, j:j + quant_size] = np.round(block / Q) # Quantization: z = round(y/q)

    print('Dequantization: to recover the image')
    for i in range(0, Y.shape[0], quant_size):
        for j in range(0, Y.shape[1], quant_size):
            block = Yq[i:i + quant_size, j:j + quant_size]
            Yq[i:i + quant_size, j:j + quant_size] = block * Q          # Dequantization Ydq= Yq * Q

    print('Reconstruct the image')
    two_dim_dct_quantization_image = np.matmul(C.T, np.matmul(Yq, C))   # Xdq = C.T*Ydq*C
    two_dim_dct_quantization_image += 128

    # Show image
    plt.subplot(1, 3, k + 1)
    plt.imshow(two_dim_dct_quantization_image.astype(np.uint8), cmap='gray')
    plt.title('loss parameter p = %d' %( loss_ps[k] ))
plt.show()