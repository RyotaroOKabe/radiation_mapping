
#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap

# create yellow colormaps
N = 256
# yellow = np.ones((N, 4))
# yellow[:, 0] = np.linspace(255/256, 1, N) # R = 255
# yellow[:, 1] = np.linspace(232/256, 1, N) # G = 232
# yellow[:, 2] = np.linspace(11/256, 1, N)  # B = 11
# yellow_cmp = ListedColormap(yellow)
yellow = np.ones((N, 3))
yellow[:, 0] = np.linspace(1, 255/256, N) # R = 255
yellow[:, 1] = np.linspace(1, 232/256, N) # G = 232
yellow[:, 2] = np.linspace(1, 11/256, N)  # B = 11
yellow_cmp = ListedColormap(yellow)

red = np.ones((N, 3))
red[:, 0] = np.linspace(1, 255/256, N) # R = 255
red[:, 1] = np.linspace(1, 0/256, N) # G = 232
red[:, 2] = np.linspace(1, 65/256, N)  # B = 11
red_cmp = ListedColormap(red)

blue = np.ones((N, 3))
blue[:, 0] = np.linspace(200/256, 0, N) # R = 255
blue[:, 1] = np.linspace(200/256, 0, N) # G = 232
blue[:, 2] = np.linspace(256/256, 139/256, N)  # B = 11
blue_cmp = ListedColormap(blue)

orange = np.ones((N, 3))
orange[:, 0] = np.linspace(256/256, 255/256, N) # R = 255
orange[:, 1] = np.linspace(240/256, 100/256, N) # G = 232
orange[:, 2] = np.linspace(200/256, 0/256, N)  # B = 11
orange_cmp = ListedColormap(orange)
# %%
matrix = np.array([[1,2,3],[4,5,6]])
plt.imshow(matrix, cmap=yellow_cmp)
plt.colorbar()
#%%
plt.imshow(matrix, cmap=red_cmp)
plt.colorbar()

#%%
plt.imshow(matrix, cmap=blue_cmp)
plt.colorbar()

#%%
plt.imshow(matrix, cmap=orange_cmp)
plt.colorbar()
# %%
