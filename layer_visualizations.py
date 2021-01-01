# %%
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
# %%
# visualize 2 filtered outputs (a.k.a activation maps)
image = cv2.imread('images/udacity_sdc.png')
gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# %%
# normalize
gray_img = gray_img.astype('float32')/255
plt.imshow(gray_img, cmap='gray')
# plt.show()
# %%

filter_vals = np.array([[-1, -1, 1, 1], [-1, -1, 1, 1],
                        [-1, -1, 1, 1], [-1, -1, 1, 1]])
print('Filter shape: ', filter_vals.shape)
# %%
# define four filters
filter_1 = filter_vals
filter_2 = -filter_1
filter_3 = filter_1.T
filter_4 = -filter_3
filters = np.array([filter_1, filter_2, filter_3, filter_4])
# %%
# visualize the filters
fig = plt.figure(figsize=(12, 6))

for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title(f'Filter {i+1}')

    width, height = filters[i].shape
    for x in range(width):
        for y in range(height):
            ax.annotate(
                str(filters[i][x][y]), xy=(y, x),
                horizontalalignment='center',
                verticalalignment='center',
                color='white' if filters[i][x][y] < 0 else 'black'
            )
# %%
# define a neural network with a single convolutional layer with four filters


class Net(nn.Module):

    def __init__(self, weight):
        super(Net, self).__init__()
        # initializes the weights of the convolutional layer to be the weights of the 4 defined filters
        k_height, k_width = weight.shape[2:]
        # defines the convolutional layer, assumes there are 4 grayscale filters
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv = nn.Conv2d(1, 4, kernel_size=(
            k_height, k_width), bias=False)
        self.conv.weight = torch.nn.Parameter(weight)

    def forward(self, x):
        # calculates the output of a convolutional layer
        # pre- and post-activation
        conv_x = self.conv(x)
        activated_x = F.relu(conv_x)

        # returns both layers
        return conv_x, activated_x


# instantiate the model and set the weights
weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
model = Net(weight)

# print out the layer in the network
print(model)  # %%
# visualize each ouput


def viz_layer(layer, n_filters=4):
    fig = plt.figure(figsize=(20, 20))

    for i in range(n_filters):
        ax = fig.add_subplot(1, n_filters, i+1, xticks=[], yticks=[])
        # layer ouputs
        ax.imshow(np.squeeze(layer[0, i].data.numpy()), cmap='gray')
        ax.set_title(f'Output {i+1}')


# %%
# visualize output of conv before and after relu
plt.imshow(gray_img, cmap='gray')
# visualize all filters
fig = plt.figure(figsize=(12, 6))
fig.subplots_adjust(left=0, right=1.5, bottom=0.8,
                    top=1, hspace=0.05, wspace=0.05)
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title(f'Filter {i+1}')


# convert the image into an input Tensor
gray_img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(1)

# get the convolutional layer (pre and post activation)
conv_layer, activated_layer = model(gray_img_tensor)

# visualize the output of a conv layer
viz_layer(conv_layer)
# %%
"""
ReLU activation
In this model, we've used an activation function that scales the output of the convolutional layer. We've chose a ReLU function to do this, and this function simply turns all negative pixel values in 0's (black). See the equation pictured below for input pixel values, x.
"""
viz_layer(activated_layer)
# %%
# maxpooling layer


class PooledNet(nn.Module):
    def __init__(self, weight):
        super(PooledNet, self).__init__()
        # initializes the weights of the convolutional layer to be the weights of the 4 defined filters
        k_height, k_width = weight.shape[2:]
        # defines the convolutional layer, assumes there are 4 grayscale filters
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv = nn.Conv2d(1, 4, kernel_size=(
            k_height, k_width), bias=False)
        self.conv.weight = torch.nn.Parameter(weight)
        # define a pooling layer
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        activated_x = F.relu(x)
        pooled_x = self.pool(activated_x)
        return x, activated_x, pooled_x


weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
p_model = PooledNet(weight)
print(p_model)
print('Filter 1: \n', filter_1)
# %%
x, ac_x, pooled = p_model(gray_img_tensor)
viz_layer(pooled)
viz_layer(ac_x)
viz_layer(x)
