import numpy as np


# The Discrete Cosine Transform (version 4)
def dct_v4(n):
    E = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            E[i, j] = np.cos((i + 0.5) * (j + 0.5) * np.pi / n)
    E *= np.sqrt(2/n)
    return E

# The Modiﬁed Discrete Cosine Transform (MDCT)
def mdct(n):
    E = np.empty((n, 2 * n))
    for i in range(n):
        for j in range(2 * n):
            E[i, j] = np.cos((i + 0.5) * (j + 0.5 + n / 2) * np.pi / n)
    E *= np.sqrt(2/n)
    return E

print('-'*100)
print("Use the overlapped MDCT to transform the signal x = [1, 2, 3, 4, 5, 6]. Then invert the\n\
transform to reconstruct the middle section [3, 4]:")
print('-'*100)

print('encode the singal')
M = mdct(2)
v1 = np.matmul(M, np.array([1,2,3,4]))
v2 = np.matmul(M, np.array([3,4,5,6]))
signal = np.vstack((v1, v2)).T
print('The transformed signal is : \n')
print(signal)

print('decode the singal')
N = M.T
w1 = np.matmul(N, v1)
w2 = np.matmul(N, v2)
u2 = (w1[-2:] + w2[:2]) / 2
print('\nThe orignal of overlapped signal is : \n')
print(u2)

# Bit quantization
def signal_quantize(y, L, b):
    q = 2 * L / (pow(2, b) - 1)
    z = np.round(y / q)
    return z

def signal_dequantize(z, L, b):
    q = 2 * L / (pow(2, b) - 1)
    y = q * z
    return y

print('-'*100)
print("Quantize the MDCT output of Example 11.9 to 4-bit integers. Then dequantize, invert the\n\
MDCT, and ﬁnd the quantization error.:")
print('-'*100)
print('quantize')
vq1 = signal_quantize(v1, 12, 4)
print('vq1 :', vq1)
vq2 = signal_quantize(v2, 12, 4)
print('vq2 :', vq2)

print('dequantize')
lossy_v1 = signal_dequantize(vq1, 12, 4)
lossy_v2 = signal_dequantize(vq2, 12, 4)

print('decode the signal')
lossy_w1 = np.matmul(N, lossy_v1)
lossy_w2 = np.matmul(N, lossy_v2)
lossy_u2 = (lossy_w1[-2:] + lossy_w2[:2]) / 2
print('lossy_u2 :', lossy_u2)
print('quantization error :', np.abs(lossy_u2 - u2))