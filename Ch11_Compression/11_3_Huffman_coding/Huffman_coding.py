import numpy as np

print('-'*100)
print("Find the Shannon information of the string ABAACDAB:")
print('-'*100)

a = 4 / 8
b = 2 / 8
c = 1 / 8
d = 1 / 8
ps = np.array([a, b, c, d])
shannon = - np.sum([ps[i] * np.log2(ps[i]) for i in range(ps.size)]) # bits/symbol
print('The Shannon information, or Shannon entropy = ', shannon)

print('-'*100)
print("Huffman coding of the 8 × 8 matrix, the so-called AC components: \n\
y be an integer. The size of y is deﬁned to be L = 0 if y =0, L = ﬂoor(log2 |y|) + 1 if y != 0")
print('-'*100)

print('%5s %115s' % ('L', 'entry'))
pre_y = 0
for n in range(7):
    y = 2**n - 1
    s = ''
    if y == 0:
        L = 0
        s = '0,'
    else:
        L = np.floor(np.log2(np.abs(y))) + 1
        s = ''
        for i in range(-y, y+1, 1):
            if abs(i) > pre_y:
                s += str(i) + ','
    pre_y = y
    t = '%5d %' + str(115 + 2**n + int(2**n/2)) + 's'
    print( t % (L, s[0:len(s)-1]))