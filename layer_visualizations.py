# %%
import matplotlib.pyplot as plt
import cv2
# %%
# visualize 2 filtered outputs (a.k.a activation maps)
image = cv2.imread('images/udacity_sdc.png')
gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# %%
# normalize
gray_img = gray_img.astype('float16')/255
plt.imsh
