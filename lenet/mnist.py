import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class LeNet(nn.Module):

    # network structure
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2, bias=False)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.act1  = nn.ReLU()
        self.act2  = nn.ReLU()
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_mnist_data():
    train_dataset = datasets.MNIST(root='./data', train=True).data
    test_dataset = datasets.MNIST(root='./data', train=False).data

    print(f"Training dataset size: {train_dataset.data.shape}")
    print(f"Test dataset size: {test_dataset.data.shape}")

    return train_dataset/255., test_dataset/255.


train_ds, test_ds = load_mnist_data()
train_ds = train_ds.unsqueeze(1).float()
test_ds = test_ds.unsqueeze(1).float()
model = LeNet()
model.eval()
print(train_ds.shape)
with torch.no_grad():
    model(train_ds)