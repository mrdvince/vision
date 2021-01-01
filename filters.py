# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
# %%
image = mpimg.imread('images/bridge_trees_example.jpg')
plt.imshow(image)
# %%
# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap='gray')
# %%
# custom kernel
sobel_y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

sobel_x = sobel_y.T
# filter image
filtered_img = cv2.filter2D(gray, -1, sobel_y)
plt.imshow(filtered_img, cmap='gray')
# %%
filtered_img_x = cv2.filter2D(gray, -1, sobel_x)
plt.imshow(filtered_img_x, cmap='gray')
# %%
# odd filter
plt.imshow(cv2.filter2D(gray, -1, np.array([
    [-1, -2, -1, -2],
    [0, 0, 0, 0],
    [1, 2, 1, 2],
    [1, 2, 1, 1]
])), cmap='gray')
