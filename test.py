import numpy as np

temp = 0
def calculate_Euclidean_distance(images_first, images_second):
    '''calculate Euclidean distance between two points'''
    difference = np.abs(images_first - images_second)
    global temp
    temp += 1
    print(temp)
    print(np.sqrt(np.sum(np.square(difference))))
    return np.sqrt(np.sum(np.square(difference)))

A = np.zeros((2, 2, 2))
B = np.zeros((2, 2, 2))
A[:] = ([1,1],
        [2,2])
B[:] = ([2,2],
        [1,1])
distance = np.zeros((2, 2))
ufunc = np.frompyfunc(calculate_Euclidean_distance, 2, 1)
distance = ufunc(A[:, np.newaxis], B)
#vectorized_func = np.vectorize(calculate_Euclidean_distance)
#distance = vectorized_func(A[:, np.newaxis], B)
lable = np.argmin(distance, axis=0)
print(distance)