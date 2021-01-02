# %%
'''
https://github.com/cezannec/capsule_net_pytorch/blob/master/Capsule_Network.ipynb
'''
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np
import torch
from models import CapsuleNetwork

device = torch.device('cuda' if torch.cuda.is_available() else ('cpu'))
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
# model instance
model = CapsuleNetwork().to(device)
print(model)
# %%
# custom loss


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(
            reduction='sum')  # cumulative loss

    def forward(self, x, labels, images, reconstructions):
        batch_size = x.shape[0]

        # calculate margin loss
        v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))

        # correct and incorrrect loss
        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        # calculate the reconstruction
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        # weighted, summed loss averaged over a batch_size
        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)
# %%
