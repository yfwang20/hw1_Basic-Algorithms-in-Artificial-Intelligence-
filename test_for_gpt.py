import numpy as np

# 定义自定义的计算函数，返回一个标量
def custom_matrix_operation(a, b):
    return np.sum(a - b)  # 示例：计算矩阵元素之差的总和

# 定义两个数组，每个元素是一个二维矩阵
A = np.array([np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])])
B = np.array([np.array([[1, 1], [1, 1]]), np.array([[2, 2], [2, 2]])])

# 使用 numpy 的 frompyfunc，将自定义函数应用于每个元素对
ufunc = np.frompyfunc(custom_matrix_operation, 2, 1)

# 计算 A 和 B 中每对元素的自定义计算结果
result_matrix = ufunc(A[:, np.newaxis], B)

# 转换为浮点数矩阵并打印
result_matrix = np.array(result_matrix, dtype=float)
print(result_matrix)
