# %%
'''
https://github.com/cezannec/capsule_net_pytorch/blob/master/Capsule_Network.ipynb
'''
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
