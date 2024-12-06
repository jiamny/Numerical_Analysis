import time
import numpy as np
from matplotlib import pyplot as plt

print('-'*100)
print("Linear congruential generator")
print('-'*100)
def linear_congruential_generator(x, a, b, m):
    x = (a * x + b) % m
    u = x / m
    return u, x, a, b, m

x0 = 3
args = (x0, 13, 0, 31)
for i in range(10):
    u, *args = linear_congruential_generator(*args)
    print('idx_%02d x:%02d, u:%.4f' %(i + 1, args[0], u))

print('-'*100)
print("Minimal standard random number generator")
print('-'*100)
def Minimal_standard_random_number_generator(x):
    a = int(pow(7, 5))
    m = int(pow(2, 31)) - 1
    x = a * x % m
    u = x/m
    return u, x

x = time.time()

for i in range(10):
    r, x = Minimal_standard_random_number_generator(x)
    print(r)

print('-'*100)
print("The randu generator")
print('-'*100)
def randu_generator(x):
    a = int(pow(2, 16) + 3)
    m = int(pow(2, 31))
    x = a * x % m
    u = x/m
    return u, x

x = time.time()
for i in range(10):
    r, x = randu_generator(x)
    print(r)

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection = '3d')
ax.view_init(azim=225)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

datax = np.array([])
datay = np.array([])
dataz = np.array([])
x = 3
total_iterations = 20000

# Process
for i in range(total_iterations):
    u1, x = randu_generator(x)
    u2, x = randu_generator(x)
    u3, x = randu_generator(x)
    datax = np.append(datax, u1)
    datay = np.append(datay, u2)
    dataz = np.append(dataz, u3)

ax.scatter(datax, datay, dataz, zdir='z', s=2)
plt.title("The randu generator")

ax = fig.add_subplot(122, projection = '3d')
ax.view_init(azim=225)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

datax = np.array([])
datay = np.array([])
dataz = np.array([])
a = int(pow(7, 5))
m = int(pow(2, 31))-1
x = time.time()
total_iterations = 20000

for i in range(total_iterations):
    u1, x = Minimal_standard_random_number_generator(x)
    u2, x = Minimal_standard_random_number_generator(x)
    u3, x = Minimal_standard_random_number_generator(x)
    datax = np.append(datax, u1)
    datay = np.append(datay, u2)
    dataz = np.append(dataz, u3)

ax.scatter(datax, datay, dataz, zdir='z', s=2)
plt.title("Minimal standard random number generator")
plt.show()

