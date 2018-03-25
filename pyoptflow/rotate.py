from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import rotate
U = np.load("U.npy")
V = np.load("V.npy")
U_sum = np.sum(V,axis=0)
plt.imshow(U_sum)
plt.show()