import os, sys
import numpy as np
import matplotlib.pyplot as plt

import mnist_dataloader
import mnist_viewer

import time
import threading
from scipy.interpolate import RectBivariateSpline

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


# TODO: create your kNN classifier

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
    '''calculate p_four distance between two points'''
    difference = images_first - images_second
    difference_abs = np.abs(difference)
    return np.power(np.sum(np.power(difference_abs, 4), axis=(2)), 1/4)

def find_min_distance(train, test, flag):
    '''
    to find nearest point
    flag 1-Euclidean 2-Manhattan 3-L_infinity 4-p=4
    flag 5~9-KNN 3,5,7,11,31
    '''
    train_expanded = train[:, np.newaxis, :]
    test_expanded = test[np.newaxis, :, :]
    if flag <= 4:
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
        location = np.argmin(distance, axis=0)
    if flag >= 5 and flag <= 9:
        if flag == 5:
            NN = 3
        else:
            if flag == 6:
                NN = 5
            else:
                if flag == 7:
                    NN = 7
                else:
                    if flag == 8:
                        NN = 11
                    else:
                        if flag == 9:
                            NN = 31
        distance = calculate_Euclidean_distance(train_expanded, test_expanded)
        sorted_indices = np.argsort(distance, axis=0)
        location = sorted_indices[0:NN, :]
    if flag == 10:
        distance = calculate_Euclidean_distance(train_expanded, test_expanded)
        sorted_indices = np.argsort(distance, axis=0)
        location = sorted_indices[1, :]
    return location

def calculate_and_compare(start_index, end_index, flag, location):
    for i in range(start_index, end_index):
        location[(i * 1) : ((i + 1) * 1)] = find_min_distance(images_A[:, :], images_B[(i * 1):((i + 1) * 1), :], flag)

def calculate_and_compare_KNN(start_index, end_index, flag, location):
    for i in range(start_index, end_index):
        location[:, (i * 1) : ((i + 1) * 1)] = find_min_distance(images_A[: :], images_B[(i * 1):((i + 1) * 1), :], flag)

def calculate_and_compare_C(start_index, end_index, flag, location):
    for i in range(start_index, end_index):
        location[(i * 1) : ((i + 1) * 1)] = find_min_distance(images_C[: :], images_C[(i * 1):((i + 1) * 1), :], flag)

def calculate_and_compare_task5(start_index, end_index, flag, location):
    global images_A_maxdown_2
    global images_B_maxdown_2
    global images_A_maxdown_4
    global images_B_maxdown_4
    global images_A_linearup_2
    global images_B_linearup_2
    if flag == 11:
        for i in range(start_index, end_index):
            location[(i * 1) : ((i + 1) * 1)] = find_min_distance(images_A_maxdown_2[:, :], images_B_maxdown_2[(i * 1):((i + 1) * 1), :], 1)
    else:
        if flag == 12:
            for i in range(start_index, end_index):
                location[(i * 1) : ((i + 1) * 1)] = find_min_distance(images_A_maxdown_4[:, :], images_B_maxdown_4[(i * 1):((i + 1) * 1), :], 1)
        else:
            if flag == 13:
                for i in range(start_index, end_index):
                    location[(i * 1) : ((i + 1) * 1)] = find_min_distance(images_A_linearup_2[:, :], images_B_linearup_2[(i * 1):((i + 1) * 1), :], 1)

def downsampling_2_max(image):
    A = image[:, ::2]
    B = image[:, 1::2]
    return np.maximum(A, B)

def downsampling_4_max(image):
    image_2d = image.reshape(30000, 28, 28)
    A = image_2d[:, ::2, ::2]
    B = image_2d[:, 1::2, ::2]
    C = image_2d[:, ::2, 1::2]
    D = image_2d[:, 1::2, 1::2]
    new = np.maximum.reduce([A, B, C, D])
    new_1d = new.reshape(30000, 196)
    return new_1d
    
def interpolation_2_bilinearity(image):
    original_dim = 28
    x_new_dim = 56
    y_new_dim = 56
    upsampled_matrices = np.empty((30000, x_new_dim * y_new_dim))

    x = np.linspace(0, original_dim - 1, original_dim)
    y = np.linspace(0, original_dim - 1, original_dim)


    x_new = np.linspace(0, original_dim - 1, x_new_dim)
    y_new = np.linspace(0, original_dim - 1, y_new_dim)
    Y, X = np.meshgrid(y_new, x_new)
    for i in range(30000):
        matrix = image[i].reshape(original_dim, original_dim)
    
        spline = RectBivariateSpline(x, y, matrix, kx=1, ky=1)
        upsampled_matrix = spline.ev(X.ravel(), Y.ravel()).reshape(x_new_dim, y_new_dim)
        upsampled_matrices[i] = upsampled_matrix.ravel()
    return upsampled_matrices



# # task 1
# start_time = time.perf_counter()
# num = 0
# location_1 = np.zeros(30000, dtype=int)
# threads = []
# chunk_size = 7500 # 每个线程处理的块大小
# for i in range(0, 30000, chunk_size):
#     t = threading.Thread(target=calculate_and_compare, args=(i, min(i + chunk_size, 30000), 1, location_1))
#     threads.append(t)
#     t.start()
# for t in threads:
#     t.join()
# for i in range(30000):
#     if labels_B[i] == labels_A[location_1[i]]:
#         num += 1
# end_time = time.perf_counter()
# elapsed_time = end_time - start_time
# print(f"task1运行时间：{elapsed_time} 秒")
# print(f"task1符合个数：{num}个")
# print(f"task1正确率：{num / 30000}")

# # task 2 曼哈顿距离
# start_time = time.perf_counter()
# num = 0
# location_2_1 = np.zeros(30000, dtype=int)
# threads = []
# chunk_size = 7500 # 每个线程处理的块大小
# for i in range(0, 30000, chunk_size):
#     t = threading.Thread(target=calculate_and_compare, args=(i, min(i + chunk_size, 30000), 2, location_2_1))
#     threads.append(t)
#     t.start()
# for t in threads:
#     t.join()
# for i in range(30000):
#     if labels_B[i] == labels_A[location_2_1[i]]:
#         num += 1
# end_time = time.perf_counter()
# elapsed_time = end_time - start_time
# print(f"task2（曼哈顿距离）运行时间：{elapsed_time} 秒")
# print(f"task2（曼哈顿距离）符合个数：{num}个")
# print(f"task2（曼哈顿距离）正确率：{num / 30000}")

# # task 2 L_infinity
# start_time = time.perf_counter()
# num = 0
# location_2_2 = np.zeros(30000, dtype=int)
# threads = []
# chunk_size = 7500 # 每个线程处理的块大小
# for i in range(0, 30000, chunk_size):
#     t = threading.Thread(target=calculate_and_compare, args=(i, min(i + chunk_size, 30000), 3, location_2_2))
#     threads.append(t)
#     t.start()
# for t in threads:
#     t.join()
# for i in range(30000):
#     if labels_B[i] == labels_A[location_2_2[i]]:
#         num += 1
# end_time = time.perf_counter()
# elapsed_time = end_time - start_time
# print(f"task2（L_infinity）运行时间：{elapsed_time} 秒")
# print(f"task2（L_infinity）符合个数：{num}个")
# print(f"task2（L_infinity）正确率：{num / 30000}")

# # task 2 p=4
# start_time = time.perf_counter()
# num = 0
# location_2_3 = np.zeros(30000, dtype=int)
# threads = []
# chunk_size = 7500 # 每个线程处理的块大小
# for i in range(0, 30000, chunk_size):
#     t = threading.Thread(target=calculate_and_compare, args=(i, min(i + chunk_size, 30000), 4, location_2_3))
#     threads.append(t)
#     t.start()
# for t in threads:
#     t.join()
# for i in range(30000):
#     if labels_B[i] == labels_A[location_2_3[i]]:
#         num += 1
# end_time = time.perf_counter()
# elapsed_time = end_time - start_time
# print(f"task2（p=4）运行时间：{elapsed_time} 秒")
# print(f"task2（p=4）符合个数：{num}个")
# print(f"task2（p=4）正确率：{num / 30000}")

# # task 3 NN=3
# start_time = time.perf_counter()
# num = 0
# NN = 3
# location_3_1 = np.zeros((NN, 30000), dtype=int) 
# threads = []
# chunk_size = 7500 # 每个线程处理的块大小
# for i in range(0, 30000, chunk_size):
#     t = threading.Thread(target=calculate_and_compare_KNN, args=(i, min(i + chunk_size, 30000), 5, location_3_1))
#     threads.append(t)
#     t.start()
# for t in threads:
#     t.join()
# labels_count_3_1 = np.zeros((10, 30000), dtype=int)
# for i in range(NN):
#     for j in range(30000):
#         labels_count_3_1[labels_A[location_3_1[i, j]], j] += 1
# labels_3_1 = np.argmax(labels_count_3_1, axis=0)
# for i in range(30000):
#     if labels_B[i] == labels_3_1[i]:
#         num += 1
# end_time = time.perf_counter()
# elapsed_time = end_time - start_time
# print(f"task3（NN=3）运行时间：{elapsed_time} 秒")
# print(f"task3（NN=3）符合个数：{num}个")
# print(f"task3（NN=3）正确率：{num / 30000}")

# # task 3 NN=5
# start_time = time.perf_counter()
# num = 0
# NN = 5
# location_3_2 = np.zeros((NN, 30000), dtype=int) 
# threads = []
# chunk_size = 7500 # 每个线程处理的块大小
# for i in range(0, 30000, chunk_size):
#     t = threading.Thread(target=calculate_and_compare_KNN, args=(i, min(i + chunk_size, 30000), 6, location_3_2))
#     threads.append(t)
#     t.start()
# for t in threads:
#     t.join()
# labels_count_3_2 = np.zeros((10, 30000), dtype=int)
# for i in range(NN):
#     for j in range(30000):
#         labels_count_3_2[labels_A[location_3_2[i, j]], j] += 1
# labels_3_2 = np.argmax(labels_count_3_2, axis=0)
# for i in range(30000):
#     if labels_B[i] == labels_3_2[i]:
#         num += 1
# end_time = time.perf_counter()
# elapsed_time = end_time - start_time
# print(f"task3（NN=5）运行时间：{elapsed_time} 秒")
# print(f"task3（NN=5）符合个数：{num}个")
# print(f"task3（NN=5）正确率：{num / 30000}")

# # task 3 NN=7
# start_time = time.perf_counter()
# num = 0
# NN = 7
# location_3_3 = np.zeros((NN, 30000), dtype=int) 
# threads = []
# chunk_size = 7500 # 每个线程处理的块大小
# for i in range(0, 30000, chunk_size):
#     t = threading.Thread(target=calculate_and_compare_KNN, args=(i, min(i + chunk_size, 30000), 7, location_3_3))
#     threads.append(t)
#     t.start()
# for t in threads:
#     t.join()
# labels_count_3_3 = np.zeros((10, 30000), dtype=int)
# for i in range(NN):
#     for j in range(30000):
#         labels_count_3_3[labels_A[location_3_3[i, j]], j] += 1
# labels_3_3 = np.argmax(labels_count_3_3, axis=0)
# for i in range(30000):
#     if labels_B[i] == labels_3_3[i]:
#         num += 1
# end_time = time.perf_counter()
# elapsed_time = end_time - start_time
# print(f"task3（NN=7）运行时间：{elapsed_time} 秒")
# print(f"task3（NN=7）符合个数：{num}个")
# print(f"task3（NN=7）正确率：{num / 30000}")

# # task 3 NN=11
# start_time = time.perf_counter()
# num = 0
# NN = 11
# location_3_4 = np.zeros((NN, 30000), dtype=int) 
# threads = []
# chunk_size = 7500 # 每个线程处理的块大小
# for i in range(0, 30000, chunk_size):
#     t = threading.Thread(target=calculate_and_compare_KNN, args=(i, min(i + chunk_size, 30000), 8, location_3_4))
#     threads.append(t)
#     t.start()
# for t in threads:
#     t.join()
# labels_count_3_4 = np.zeros((10, 30000), dtype=int)
# for i in range(NN):
#     for j in range(30000):
#         labels_count_3_4[labels_A[location_3_4[i, j]], j] += 1
# labels_3_4 = np.argmax(labels_count_3_4, axis=0)
# for i in range(30000):
#     if labels_B[i] == labels_3_4[i]:
#         num += 1
# end_time = time.perf_counter()
# elapsed_time = end_time - start_time
# print(f"task3（NN=11）运行时间：{elapsed_time} 秒")
# print(f"task3（NN=11）符合个数：{num}个")
# print(f"task3（NN=11）正确率：{num / 30000}")

# # task 3 NN=31
# start_time = time.perf_counter()
# num = 0
# NN =31
# location_3_5 = np.zeros((NN, 30000), dtype=int) 
# threads = []
# chunk_size = 7500 # 每个线程处理的块大小
# for i in range(0, 30000, chunk_size):
#     t = threading.Thread(target=calculate_and_compare_KNN, args=(i, min(i + chunk_size, 30000), 9, location_3_5))
#     threads.append(t)
#     t.start()
# for t in threads:
#     t.join()
# labels_count_3_5 = np.zeros((10, 30000), dtype=int)
# for i in range(NN):
#     for j in range(30000):
#         labels_count_3_5[labels_A[location_3_5[i, j]], j] += 1
# labels_3_5 = np.argmax(labels_count_3_5, axis=0)
# for i in range(30000):
#     if labels_B[i] == labels_3_5[i]:
#         num += 1
# end_time = time.perf_counter()
# elapsed_time = end_time - start_time
# print(f"task3（NN=31）运行时间：{elapsed_time} 秒")
# print(f"task3（NN=31）符合个数：{num}个")
# print(f"task3（NN=31）正确率：{num / 30000}")

# # task4
# start_time = time.perf_counter()
# num = 0
# location_4 = np.zeros(60000, dtype=int)
# threads = []
# chunk_size = 30000 # 每个线程处理的块大小
# for i in range(0, 60000, chunk_size):
#     t = threading.Thread(target=calculate_and_compare_C, args=(i, min(i + chunk_size, 60000), 10, location_4))
#     threads.append(t)
#     t.start()
# for t in threads:
#     t.join()
# for i in range(60000):
#     if labels_C[i] == labels_C[location_4[i]]:
#         num += 1
# end_time = time.perf_counter()
# elapsed_time = end_time - start_time
# print(f"task4运行时间：{elapsed_time} 秒")
# print(f"task4符合个数：{num}个")
# print(f"task4正确率：{num / 60000}")

# task5
# 降采样
images_A_maxdown_2 = downsampling_2_max(images_A)
images_B_maxdown_2 = downsampling_2_max(images_B)
images_A_maxdown_4 = downsampling_4_max(images_A)
images_B_maxdown_4 = downsampling_4_max(images_B)
print(np.shape(images_A_maxdown_2))
images_A_linearup_2 = interpolation_2_bilinearity(images_A)
images_B_linearup_2 = interpolation_2_bilinearity(images_B)
print(np.shape(images_A_linearup_2))

# # 5_1
# start_time = time.perf_counter()
# num = 0
# location_5_1 = np.zeros(30000, dtype=int)
# threads = []
# chunk_size = 7500 # 每个线程处理的块大小
# for i in range(0, 30000, chunk_size):
#     t = threading.Thread(target=calculate_and_compare_task5, args=(i, min(i + chunk_size, 30000), 11, location_5_1))
#     threads.append(t)
#     t.start()
# for t in threads:
#     t.join()
# for i in range(30000):
#     if labels_B[i] == labels_A[location_5_1[i]]:
#         num += 1
# end_time = time.perf_counter()
# elapsed_time = end_time - start_time
# print(f"task5_1运行时间：{elapsed_time} 秒")
# print(f"task5_1符合个数：{num}个")
# print(f"task5_1正确率：{num / 30000}")

# # 5_2
# start_time = time.perf_counter()
# num = 0
# location_5_2 = np.zeros(30000, dtype=int)
# threads = []
# chunk_size = 7500 # 每个线程处理的块大小
# for i in range(0, 30000, chunk_size):
#     t = threading.Thread(target=calculate_and_compare_task5, args=(i, min(i + chunk_size, 30000), 12, location_5_2))
#     threads.append(t)
#     t.start()
# for t in threads:
#     t.join()
# for i in range(30000):
#     if labels_B[i] == labels_A[location_5_2[i]]:
#         num += 1
# end_time = time.perf_counter()
# elapsed_time = end_time - start_time
# print(f"task5_2运行时间：{elapsed_time} 秒")
# print(f"task5_2符合个数：{num}个")
# print(f"task5_2正确率：{num / 30000}")

# 5_3
start_time = time.perf_counter()
num = 0
location_5_3 = np.zeros(30000, dtype=int)
threads = []
chunk_size = 15000 # 每个线程处理的块大小
for i in range(0, 30000, chunk_size):
    t = threading.Thread(target=calculate_and_compare_task5, args=(i, min(i + chunk_size, 30000), 13, location_5_3))
    threads.append(t)
    t.start()
for t in threads:
    t.join()
for i in range(30000):
    if labels_B[i] == labels_A[location_5_3[i]]:
        num += 1
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"task5_3运行时间：{elapsed_time} 秒")
print(f"task5_3符合个数：{num}个")
print(f"task5_3正确率：{num / 30000}")