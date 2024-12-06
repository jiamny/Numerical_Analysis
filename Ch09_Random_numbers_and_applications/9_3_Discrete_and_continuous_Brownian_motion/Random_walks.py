import time
from matplotlib import pyplot as plt

def Minimal_standard_random_number_generator(x):
    a = int(pow(7, 5))
    m = int(pow(2, 31)) - 1
    x = a * x % m
    u = x/m
    return u, x

print('-'*100)
print("A random walk of 10 steps")
print('-'*100)
t = 10
w = 0
x = time.time()

for i in range(t):
    r, x = Minimal_standard_random_number_generator(x)
    if r > 0.5:
        w += 1
    else:
        w -= 1
print(w)

def random_walk(n, interval):
    lowerbound = interval[0]
    upperbound = interval[1]
    top_exits = 0
    avg_esc_time = 0
    x = time.time()
    for _ in range(n):
        w = 0
        l = 0
        while(True):
            r, x = Minimal_standard_random_number_generator(x)
            if r > 0.5: # random.random() > 0.5:
                w += 1
            else:
                w -= 1
            l += 1
            if w == lowerbound:
                pass
                break
            elif w == upperbound:
                top_exits += 1
                break
        avg_esc_time += l
    return top_exits, avg_esc_time / n

print('-'*100)
print("Use a Monte Carlo simulation to approximate the probability that the random walk exits\n\
the interval [−3, 6] through the top boundary 6")
print('-'*100)

interval = (-3, 6)

num_ns = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
print("%12s %12s %12s %12s" %('n', 'top_exit', 'prob', 'error'))
for i in range(len(num_ns)):
    n = num_ns[i]
    top_exit, _ = random_walk(n, interval)
    print('%12d %12d %12.3f %12.5f' % (n, top_exit, top_exit / n, abs(1 / 3 - top_exit / n)))

print('-'*100)
print("Use a Monte Carlo simulation to estimate the escape time for a random walk escaping the\n\
interval [-3, 6]. The expected value of the escape time is ab = 18")
print('-'*100)

ab = 18
print("%12s %18s %12s" %('n', 'avg esc. time', 'error'))
for i in range(len(num_ns)):
    n = num_ns[i]
    _, avg_esc_time = random_walk(n, interval)
    print('%12d %18.3f %12.5f' % (n, avg_esc_time, abs(ab - avg_esc_time)))

print('-'*100)
print("Error of Monte Carlo estimation for escape problem")
print('-'*100)
num_tail = 50
true_escape = 18
true_prob = 1/3

num_ns = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400]
prob_errors = []
escape_errors = []
for i in range(len(num_ns)):
    n = num_ns[i]
    prob_e = 0
    escape_e = 0
    for _ in range(num_tail):
        top_exit, avg_esc_time = random_walk(n, interval)
        prob_e += abs(true_prob - top_exit / n)
        escape_e += abs(true_escape - avg_esc_time)
    print(n)
    prob_errors.append(prob_e/num_tail)
    escape_errors.append(escape_e/num_tail)

plt.figure()
plt.loglog(num_ns, prob_errors, 'o-', label='probability of escaping [−3, 6] by hitting 6')
plt.semilogx(num_ns, escape_errors, 'o--', label='the escape time of the same problem')
plt.xlabel('Number of random walks')
plt.ylabel('Error')
plt.legend()
plt.title('Estimation error versus number of random walks')
plt.tight_layout()
plt.show()