import os, sys
import numpy as np
import matplotlib.pyplot as plt

import mnist_dataloader
import mnist_viewer

import time

mnist_dataset = mnist_dataloader.read_data_sets("./MNIST_dataset/")
dataset_A, dataset_B, dataset_C = mnist_dataset.train, mnist_dataset.test, mnist_dataset.multi

train_size = dataset_A.num_examples
test_size = dataset_B.num_examples
print('Dataset size: ', '(train, test) =', (train_size, test_size))

# you can modify two parameters to choose any image in dataset to visualize
# idx: 0 ~ 59999, switch: {'on', 'off'} (you should close the image page to continue processing)
mnist_viewer.view(dataset_A, idx=1, switch='off')

# you can use index to get specific item (e.g. image_A[0])
images_A, images_B, images_C = dataset_A.images, dataset_B.images, dataset_C.images
labels_A, labels_B, labels_C = dataset_A.labels, dataset_B.labels, dataset_C.labels

#print(images_A[3])
#print(labels_A[3])

# TODO: create your kNN classifier

temp = 0
def calculate_Euclidean_distance(images_first, images_second):
    '''calculate Euclidean distance between two points'''
    difference = images_first - images_second
    difference_abs = np.abs(difference)
    return np.sqrt(np.sum(np.square(difference_abs), axis=(2)))

def calculate_Manhattan_distance(images_first, images_second):
    '''calculate Manhattan distance between two points'''
    difference = images_first - images_second
    difference_abs = np.abs(difference)
    return np.sum(difference_abs, axis=(2))

def calculate_L_infinity_distance(images_first, images_second):
    '''calculate L_infinity distance between two points'''
    difference = images_first - images_second
    difference_abs = np.abs(difference)
    return np.max(difference_abs, axis=(2))

def calculate_p_four_distance(images_first, images_second):
    '''calculate p_fou distance between two points'''
    difference = images_first - images_second
    difference_abs = np.abs(difference)
    return np.power(np.sum(np.power(difference_abs, 4), axis=(2)), 1/4)

#a = calculate_Euclidean_distance(images_A[1], images_A[2])
#print(a)

def find_min_distance(train, test, flag):
    '''
    to find nearest point
    flag 1-Euclidean 2-Manhattan 3-L_infinity 4-p=4
    '''
    train_expanded = train[:, np.newaxis, :]
    test_expanded = test[np.newaxis, :, :]
    if flag == 1:
        distance = calculate_Euclidean_distance(train_expanded, test_expanded)
    else:
        if flag == 2:
            distance = calculate_Manhattan_distance(train_expanded, test_expanded)
        else:
            if flag == 3:
                distance = calculate_L_infinity_distance(train_expanded, test_expanded)
            else:
                if flag == 4:
                    distance = calculate_p_four_distance(train_expanded, test_expanded)

    #print(3)
    lable = np.argmin(distance, axis=0)
    return lable

start_time = time.perf_counter()    
# task 1
num = 0
print(np.shape(images_A))
result_1 = np.zeros(30000, dtype=int)
print(result_1[1])

for i in range(30000):
    result_1[(i * 1) : ((i + 1) * 1)] = find_min_distance(images_A[0:30000, :], images_B[(i * 1):((i + 1) * 1), :], 1)
    #for j in range(100):
    #    if labels_B[(i * 100) + j] == labels_A[location[j]]:
    #        num += 1
    print(i)
for i in range(30000):
    if labels_B[i] == labels_A[result_1[i]]:
        num += 1
print(num)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"任务1运行时间: {elapsed_time} 秒")