import numpy as np
from scipy.interpolate import RectBivariateSpline

def interpolation_2_bilinearity(image):
    original_dim = 2
    x_new_dim = 2
    y_new_dim = 4
    upsampled_matrices = np.empty((1, x_new_dim * y_new_dim))

    x = np.linspace(0, original_dim - 1, original_dim)
    y = np.linspace(0, original_dim - 1, original_dim)


    x_new = np.linspace(0, original_dim - 1, x_new_dim)
    y_new = np.linspace(0, original_dim - 1, y_new_dim)
    Y, X = np.meshgrid(y_new, x_new)
    for i in range(1):
        matrix = image[i].reshape(original_dim, original_dim)
    
        spline = RectBivariateSpline(x, y, matrix, kx=1, ky=1)
        upsampled_matrix = spline.ev(X.ravel(), Y.ravel()).reshape(x_new_dim, y_new_dim)
        upsampled_matrices[i] = upsampled_matrix.ravel()
    return upsampled_matrices

a = np.zeros((1, 2, 2))
a[0, 0, 0] = 2
a[0, 0, 1] = 4
a[0, 1, 0] = 6
a[0, 1, 1] = 8
b = np.zeros((1, 4))
b[0, :] = [2, 4, 6, 8]
print(a.reshape(1,4))
print(b.reshape(1,2,2))
print(interpolation_2_bilinearity(b).reshape(1, 2, 4))