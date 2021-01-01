from tqdm.auto import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        'data', download=True, train=True,
        transform=transforms.ToTensor()),
    batch_size=32, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        'data', train=False,
        download=True,
        transform=transforms.ToTensor()),
    batch_size=32, shuffle=True
)

images, labels = iter(train_loader).next()

images = images.numpy()

# fig = plt.figure(figsize=(25, 4))
# for idx in range(20):
#    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
#    ax.imshow(np.squeeze(images[idx]), cmap='gray')
#    ax.set_title(str(labels[idx].item()))
# plt.savefig('output.png')


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()

        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


model = MLPNet()
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    with tqdm(train_loader, unit="batch") as tepoch:
        for images, labels in tepoch:
            model.train()
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

            preds = outputs.argmax(dim=1, keepdim=True).squeeze()
            correct = (preds == labels).sum().item()
            accuracy = correct/images.shape[0]

            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
