import numpy as np

# 假设这是您的自定义函数，它接受两个二维矩阵并返回一个值
# 这里将其改写为向量化的形式
def custom_function(images_first, images_second):
    difference = np.abs(images_first - images_second)
    return np.sqrt(np.sum(np.square(difference), axis=(2)))

# 假设以下为两个数组的示例，每个元素都是一个二维矩阵
array1 = np.array([[[1, 2, 3, 4]], [[5, 6, 7, 8]]])
array2 = np.array([[[9, 10, 11, 12]], [[13, 14, 15, 16]]])
print(np.shape(array1))
# 调整数组形状以进行广播
array1_expanded = array1[:, np.newaxis, :]
array2_expanded = array2[np.newaxis, :, :]
print(np.shape(array1_expanded))
# 使用numpy的广播机制来计算所有矩阵对的函数结果
result_matrix = custom_function(array1_expanded, array2_expanded)

print(result_matrix)
