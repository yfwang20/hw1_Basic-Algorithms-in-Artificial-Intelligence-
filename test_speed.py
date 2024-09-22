import numpy as np
num = 0
a = np.ones((100000, 10000))
b = np.ones((100000, 10000))
c = np.sqrt(np.sum(np.square(a + b)))
print(c)
#for i in range(300000):
#    print(i)
#    c = np.sqrt(np.sum(np.square(a + b)))
#    print(c)
