import torch.nn as nn
from torch.nn import Module
from torch import flatten
import torch
class CNN(Module):
#This file defines my CNN and its architecture.

    def __init__(self, pretrained=False):
        super(CNN, self).__init__()
        self.numChannels = 3
        self.numClasses = 2
        self.pretrained = pretrained

        self.conv1 = nn.Conv2d(in_channels= self.numChannels, out_channels = 12,kernel_size = (3,3), padding = (1,1), stride=(2,2))
        self.bn1 = nn.BatchNorm2d(12)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels= 12, out_channels= 30, kernel_size=(3,3), padding = (1,1), stride=(2, 2))
        self.bn2 = nn.BatchNorm2d(30)
        self.relu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(in_channels=30, out_channels=45,kernel_size=(3, 3), padding=(1,1), stride=(2, 2))
        self.bn3 = nn.BatchNorm2d(45)
        self.relu3 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(in_channels=45, out_channels=60, kernel_size=(3,3), padding=1, stride=(2, 2))
        self.bn4 = nn.BatchNorm2d(60)
        self.relu4 = nn.LeakyReLU()
        self.conv5 = nn.Conv2d(60, 75, kernel_size=(3, 3), padding=(1,1), stride=(2, 2))
        self.bn5 = nn.BatchNorm2d(75)
        self.relu5 = nn.LeakyReLU()
        self.conv6 = nn.Conv2d(in_channels=75, out_channels=90, kernel_size=(2,2), padding=1, stride=(2, 2))
        self.bn6 = nn.BatchNorm2d(90)
        self.relu6 = nn.LeakyReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(in_features=2250, out_features=1024)
        self.bn7 = nn.BatchNorm1d(1024)
        self.relu7 = nn.LeakyReLU()
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.bn8 = nn.BatchNorm1d(512)
        self.relu8 = nn.LeakyReLU()
        self.fc3 = nn.Linear(in_features=512, out_features=256)
        self.bn9 = nn.BatchNorm1d(256)
        self.relu9 = nn.LeakyReLU()
        self.fc4 = nn.Linear(in_features=256, out_features=128)
        self.bn10 = nn.BatchNorm1d(128)
        self.relu10 = nn.LeakyReLU()
        self.fc5 = nn.Linear(in_features=128, out_features=1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        print(f"Forward fufnctin called")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.maxpool1(x)
        x = flatten(x, end_dim=1)
        x = self.fc1(x)
        x = self.relu7(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu8(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu9(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.relu10(x)
        x = self.dropout(x)
        x = self.fc5(x)

        x = torch.squeeze(x)
        x = torch.sigmoid(x)
        print("AHHAA")
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
        return None
if __name__ == "__main__":
    network = CNN(numChannels=1, classes=1)
    CNN.weights_init(network)

