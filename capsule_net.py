# %%
'''
https://github.com/cezannec/capsule_net_pytorch/blob/master/Capsule_Network.ipynb
'''
from typing import Any
from torch import tensor
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn.modules import padding
from torchvision import datasets, transforms
import numpy as np
import torch

# %%
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
train_loader = torch.utils.data.DataLoader(datasets.MNIST(
    'data', train=True, download=True, transform=transforms.ToTensor()), batch_size=20, shuffle=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(datasets.MNIST(
    'data', train=False, download=True, transform=transforms.ToTensor()), batch_size=20, num_workers=2)

# %%
# visualize data
images, labels = iter(train_loader).next()
images = images.numpy()

fig = plt.figure(figsize=(25, 4))
for idx in range(images.shape[0]):
    ax = fig.add_subplot(2, 10, idx+1,  xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]))
    ax.set_title(str(labels[idx].item()))
# %%
# Encoder
# 1st layer


class ConvLayer(nn.Module):
    def __init__(self) -> None:
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=256,
                              kernel_size=9, stride=1, padding=0)

    def forward(self, x):
        features = F.relu(self.conv(x))
        return features
# 2nd layer
# 8 primary capsules


class PrimaryCapsules(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32):
        super(PrimaryCapsules, self).__init__()
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=9, stride=2, padding=0) for _ in range(num_capsules)
        ])

    def forward(self, x):
        batch_size = x.shape[0]
        u = [capsule(x).view(batch_size, 32*6*6, 1)
             for capsule in self.capsules]
        # stack outputs, u
        u = torch.cat(u, dim=-1)
        # squash stack
        u_squash = self.squash(u)
        return u_squash

    def squash(self, input_tensor):
        # input tensor -> magnitude between 0-1
        squared_norm = (input_tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm/(1 + squared_norm)  # norm coeff
        output_tensor = scale * input_tensor / torch.sqrt(squared_norm)
        return output_tensor

# 3rd layer
# Digit capsules


def softmax(input_tensor: Any, dim=1) -> tensor:
    # transpose input
    transposed_input = input_tensor.transpose(dim, len(input_tensor.size())-1)
    softmaxed_output = F.softmax(
        transposed_input.contiguos().view(-1, transposed_input.size(-1)), dim=-1)
    # untranspose
    return softmaxed_output.view(
        *transposed_input.size()).transpose(dim, len(input_tensor.size())-1)


def dynamic_routing(b_ij, u_hat, squash, routing_iteratins=3):
    '''Performs dynamic routing between two capsule layers.
       param b_ij: initial log probabilities that capsule i should be coupled to capsule j
       param u_hat: input, weighted capsule vectors, W u
       param squash: given, normalizing squash function
       param routing_iterations: number of times to update coupling coefficients
       return: v_j, output capsule vectors
       '''
    v_j = None  # unbound variable error warning
    for iteration in range(routing_iteratins):
        c_ij = softmax(b_ij, dim=2)  # softmax calc of coupling coefficients
        s_j = (c_ij * u_hat).sum(dim=2, keepdim=True)
        v_j = squash(s_j)

        if iteration < routing_iteratins - 1:
            a_ij = (u_hat * v_j).sum(dim=-1, keepdim=True)
            # new b_ij
            b_ij = b_ij + a_ij
    return v_j


# %%
device = torch.device('cuda' if torch.cuda.is_available() else ('cpu'))

print(device)
# %%


class DigitCaps(nn.Module):
    def __init__(
        self,
        num_capsules=10,
        previous_layer_nodes=32*6*6,
        in_channels=8,
        out_channels=16
    ):
        super(DigitCaps, self).__init__()

        self.num_capsules = num_capsules
        self.previous_layer_nodes = previous_layer_nodes  # 1152
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = nn.Parameter(torch.randn(
            num_capsules, previous_layer_nodes, in_channels, out_channels
        ))

    def forward(self, x):
        '''
        Defines the feedforward behavior.
        param u: the input; vectors from the previous PrimaryCaps layer
        return: a set of normalized, capsule output vectors
        '''
        u = x[None, :, :, None, :]
        # 4d weight matrix
        W = self.W[:, None, :, :, :]
        # u_hat = u*w
        u_hat = torch.matmul(u, W)
        b_ij = torch.zeros(*u_hat.size())
        b_ij = b_ij.to(device)
        v_j = dynamic_routing(b_ij, u_hat, self.squash, routing_iteratins=3)

        return v_j

    def squash(self, x: tensor):
        '''
        Squashes an input Tensor so it has a magnitude between 0-1.
        param input_tensor: a stack of capsule inputs, s_j
        return: a stack of normalized, capsule output vectors, v_j
        '''
        squared_norm = (x**2).sum(dim=-1, keepdim=True)
        scale = squared_norm/(1+squared_norm)
        out_tensor = scale * x / torch.sqrt(squared_norm)
        return out_tensor

# %%
