import numpy as np

'''
Computes approximate solution of g(x)=x
Input: function handle g, starting guess x0,
number of iteration steps k
Output: Approximate solution xc
'''
def fpi(g, x0, k):
    x = []
    x.append(x0)
    for i in range(k):
        x.append( g(x[i]) )
    xc = x[k]
    return xc

print('------------------------------------------------------')
print('Find a root of the function f(x) = cos(x)')
print('------------------------------------------------------')
g = lambda x: np.cos(x)
Xc =fpi(g, 0, 10)
print('fpi(lambda x: cos(x), 0, 10) = ', Xc)

print('------------------------------------------------------')
print('Find the ﬁxed points of g(x) = (1 + 2*x**3 )/(1 + 3*x**2 )')
print('------------------------------------------------------')
g = lambda x: (1 + 2*x**3 )/(1 + 3*x**2 )
Xc =fpi(g, 0.5, 10)
print('fpi(lambda x: (1 + 2*x**3 )/(1 + 3*x**2), 0.5, 10) = ', Xc)

print('------------------------------------------------------')
print('Find the ﬁxed points of g(x) = 2.8x − x**2')
print('------------------------------------------------------')
g = lambda x: 2.8*x - x**2
Xc =fpi(g, 0.1, 10)
print('fpi(lambda x: 2.8*x - x**2, 0.1, 10) = ', Xc)

print('------------------------------------------------------')
print('Calculate sqrt(2) by using FPI')
print('------------------------------------------------------')
g = lambda x: (x + 2/x) / 2
Xc =fpi(g, 1.0, 10)
print('fpi(lambda x: (x + 2/x) / 2, 1., 10) = ', Xc)