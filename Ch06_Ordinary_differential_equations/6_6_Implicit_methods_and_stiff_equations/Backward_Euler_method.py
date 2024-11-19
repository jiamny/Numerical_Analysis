import matplotlib.pyplot as plt
import numpy as np

print('-'*70)
print("Apply theuses the backward Euler method to solve y'=-a*y+0.25*t^2 from t=0 to t=5 with a=2:")
print('-'*70)
a=2
y0=2
h=0.1
n=int(5/h)
t = np.linspace(0, 5, n+1)
y=np.zeros(n+1)
y[0]=y0

for j in range(n):
  y[j+1]=(y[j]+h*0.25*t[j+1]**2)/(1+h*a)

plt.figure()
plt.plot(t, y,'--', lw=2)
plt.xlabel('t')
plt.ylabel('y')
plt.title('Backward Euler method')
plt.show()