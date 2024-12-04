import torch
import torch.nn as nn
from dataset import MyCifar10
from torch.utils.data import DataLoader

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=3072, out_features=256),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU()
        )
        self.layer6 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x

if __name__ == '__main__':
    epochs = 100
    model = SimpleNeuralNetwork()
    dataset = MyCifar10(root="data/cifar-10-batches-py", is_train=True)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=16,
        num_workers=6,
        shuffle=True,
        drop_last=False
    )
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for images, labels in dataloader:
            output = model(images)
            loss = criterion(output, labels)
            print(epoch, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()