import time

from matplotlib import pyplot as plt

def Minimal_standard_random_number_generator(x):
    a = int(pow(7, 5))
    m = int(pow(2, 31)) - 1
    x = a * x % m
    u = x/m
    return u, x

print('-'*100)
print("Monte Carlo Type 1 problem: approximate the area under the curve y = x 2 in [0, 1]")
print('-'*100)

sum_v = 0.0
x = time.time()
n = 10
for i in range(n):
    ui, x = Minimal_standard_random_number_generator(x)
    sum_v += ui**2
print("Approximate area:", sum_v/n)

print('-'*100)
print("Monte Carlo Type 2 problem: find the area of the set of points (x, y) that satisfy \n\
      4(2x − 1)**4 + 8(2y − 1)**8 < 1 + 2(2y − 1)**3 * (3x − 2)**2")
print('-'*100)

n= 10000
sum_v = 0
s = time.time()
vx = []
vy = []
for i in range(n):
    x, s = Minimal_standard_random_number_generator(s)
    y, s = Minimal_standard_random_number_generator(s)
    if 4*(2*x-1)**4+8*(2*y-1)**8 < 1 + 2*(2*y-1)**3*(3*x-2)**2:
        sum_v += 1
        vx.append(x)
        vy.append(y)

print("Approximate area:", sum_v/n)
plt.scatter(vx, vy, 1)
plt.show()