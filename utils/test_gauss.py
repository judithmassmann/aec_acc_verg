from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

#texture = '../data/texture1.jpg'
#Image.open(texture).show()
#ndimage.gaussian_filter
#plt.imshow(texture)
#plt.show()


nx, ny = (3, 2)
x = np.linspace(1.5, 3, nx)
y = np.linspace(1, 2, ny)
xv, yv = np.meshgrid(x, y)


x1 = np.arange(9.0).reshape((3, 3))
print(x1)
x2 = np.arange(9.0).reshape((3, 3))
print(x2)
o = x1 + x2
print(o)
print(np.sqrt(o))
