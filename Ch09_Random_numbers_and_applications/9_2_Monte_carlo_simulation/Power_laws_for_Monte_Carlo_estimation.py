import time
from matplotlib import pyplot as plt

def Minimal_standard_random_number_generator(x):
    a = int(pow(7, 5))
    m = int(pow(2, 31)) - 1
    x = a * x % m
    u = x/m
    return u, x

def Type_1_Monte_Carlo(n):
    sum_v = 0.0
    x = time.time()
    for i in range(n):
        ui, x = Minimal_standard_random_number_generator(x)
        sum_v += ui ** 2

    return sum_v/n

def Type_2_Monte_Carlo(n):
    sum_v = 0
    s = time.time()
    for i in range(n):
        x, s = Minimal_standard_random_number_generator(s)
        y, s = Minimal_standard_random_number_generator(s)
        if y < x**2:
            sum_v += 1
    return sum_v/n

print('-'*100)
print("Find Type 1 and Type 2 Monte Carlo estimates, using pseudo-random numbers for the area\n\
under the curve of y = x**2 in [0, 1]")
print('-'*100)
num_tail = 500
true_area = 1/3

n_points = [100, 200, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400]

type_1_errors = []
type_2_errors = []
for i in range(len(n_points)):
    n = n_points[i]
    tp1_e = 0
    tp2_e = 0
    for _ in range(num_tail):
        tp1_e += abs(Type_1_Monte_Carlo(n) - true_area)
        tp2_e += abs(Type_2_Monte_Carlo(n) - true_area)

    type_1_errors.append(tp1_e/num_tail)
    type_2_errors.append(tp2_e / num_tail)
    print(i)

plt.figure()
plt.loglog(n_points, type_1_errors, 'o-', label='Type 1 Monte Carlo')
plt.loglog(n_points, type_2_errors, 'o--', label='Type 2 Monte Carlo')
plt.xlabel('Number of points n')
plt.ylabel('Error')
plt.legend()
plt.title('Mean error of Type 1 Monte Carlo estimate')
plt.show()