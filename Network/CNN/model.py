import torch.nn as nn
from torch.nn import Module
from torch import flatten
#flattens output
import torch
class CNN(Module):
#This file defines my CNN and its architecture.
    #Uses lenet architecture
    def __init__(self, numChannels, classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= numChannels, out_channels = 12,kernel_size = (3,3), padding = (1,1), stride=(1,1), bias=False)
        self.bc1 = nn.BatchNorm2d(12)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels= 12, out_channels= 30, kernel_size=(3,3), padding = (1,1), stride=(1, 1), bias=False)
        self.bc2 = nn.BatchNorm2d(30)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=30, out_channels=45,kernel_size=(3, 3), padding=(1,1), stride=(1, 1), bias=False)
        self.bc3 = nn.BatchNorm2d(45)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=45, out_channels=60, kernel_size=(3,3), padding=1, stride=(2, 2), bias=False)
        self.bc4 = nn.BatchNorm2d(60)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(60, 75, kernel_size=(3, 3), padding=(1,1), stride=(2, 2), bias=False)
        self.bc5 = nn.BatchNorm2d(75)
        self.relu5 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.fc1 = nn.Linear(in_features=468075, out_features=225)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=225, out_features=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        print(f"conv 1X shape: {x.shape}")
        x = self.bc1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        print(f"conv 2X shape: {x.shape}")
        x = self.bc2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        print(f"conv 3X shape: {x.shape}")
        x = self.bc3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bc4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.bc5(x)
        x = self.relu5(x)
        x = self.maxpool1(x)
        x = flatten(x, 1)
        print(f"X shape: {x.shape}")
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = torch.squeeze(x)
        return x
    @staticmethod
    def weights_init(model) -> None:
        print(f"Heheha")
        for layer in model.children():
            if hasattr(layer, "reset_parameters"):
                # print(f"Before resetting: {layer.bias}")
                layer.reset_parameters()
                # print(f"After resetting: {layer.bias}")
                print(f"{layer} has been initialized")
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                        # print(f"Zeroing bias. Shape: {layer.bias.shape}")
if __name__ == "__main__":
    network = CNN(numChannels=1, classes=1)
    CNN.weights_init(network)

