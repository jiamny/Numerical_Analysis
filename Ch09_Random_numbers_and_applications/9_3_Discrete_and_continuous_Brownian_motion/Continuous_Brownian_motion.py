import time, math, numpy as np
from matplotlib import pyplot as plt

print('-'*100)
print("An approximation to Brownian motion")
print('-'*100)
k = 250
sqdelt = math.sqrt(1/25)
print('sqdelt: ', sqdelt)
s = []
w = []
b=0
for i in range(k):
    r = np.random.randn()
    b += sqdelt*r
    s.append(i)
    w.append(b)

plt.figure()
plt.plot(s, w)
plt.show()

print('-'*100)
print("Design a Monte Carlo simulation to estimate the probability that Brownian motion escapes\n\
through the top of the given interval [−b, a]. Use n = 1000 Brownian motion paths of step size delta_t\n\
 = 0.01. Calculate the error by comparing with the correct answer b/(a + b), interval [−b=-2, a=5] ")
print('-'*100)
n = 1000
dt = 0.01
step_height = 0.1
interval = [-2, 5]

b = -interval[0]
a = interval[1]
print('b = ', b)
correct_ans = b / (a + b)
count_top = 0

for _ in range(n):
    W_t = 0
    while (W_t < a) & (W_t > -b):
        s_i = step_height * np.random.randn()
        W_t += s_i

    if W_t >= a:
        count_top += 1

estimate = count_top / n
error = abs(estimate - correct_ans)
print("Correct Answer: %f / Estimate: %f / Error: %f" % (correct_ans, estimate, error))