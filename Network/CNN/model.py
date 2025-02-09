import torch.nn as nn
from torch.nn import Module
#Pytorch implements nn using classes
from torch.nn import Conv2d
from torch.nn import Linear
#Fully connected layers
from torch.nn import MaxPool2d
#2d max pooling to decrease spatial dimensions of input
from torch.nn import ReLU
#RELU activation function
#Used in softmax layer to show predicted probabilities of class
from torch import flatten
#flattens output
from torch import sigmoid
import torch
class CNN(Module):
#This file defines my CNN and its architecture.
    #Uses lenet architecture
    def __init__(self, numChannels, classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= numChannels, out_channels = 12,kernel_size = (3,3), padding = (1,1))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size = (3,3), stride = (1,1))
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=30,kernel_size=(3, 3), padding=(1,1))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        #Aggresive spatial reduction.
        self.conv3 = nn.Conv2d(30, 60, kernel_size=(3, 3), padding=(1,1))
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        num_features = 60 * 159 * 159
        self.fc1 =nn.Linear(in_features=num_features, out_features=225)
        #self.fc1 = nn.Linear(30*320*320, out_features=225)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=225, out_features=1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        #Forward pass function
        x = self.conv1(x)
        print(f"{x.shape} Device: {x.device}")
        x = self.relu1(x)
        print(f"{x.shape} Device: {x.device}")
        x = self.maxpool1(x)
        print(f"{x.shape} Device: {x.device}")
        x = self.conv2(x)
        print(f"{x.shape} Device: {x.device}")
        x = self.relu2(x)
        print(f"{x.shape} Device: {x.device}")
        x = self.maxpool2(x)
        print(f"{x.shape} Device: {x.device}")
        x = self.conv3(x)
        print(f"{x.shape} Device: {x.device}")
        x = self.relu3(x)
        print(f"{x.shape} Device: {x.device}")
        x = self.maxpool3(x)
        x = flatten(x, 1)
        print(f"{x.shape}, flattening")
        #Atp, x is multidim tensor so we need to flatten it to a 1d list of values for fc layer.
        x = self.fc1(x)
        print(f"{x.shape}, passing through first fc layer... please?!")
        x = self.relu4(x)
        print(f"{x.shape}, passing through third relu func... please?!")
        x = self.dropout(x)
        x = self.fc2(x)
        print(f"{x.shape}, passing through last fc layer... please?!")
        x = torch.squeeze(x)
        output = self.sigmoid(x)
        print("applying sigmoid...")
        return output

    @staticmethod
    def weights_init(model) -> None: 
        if isinstance(model, nn.Conv2d):
            torch.nn.init.xavier_uniform_(model.weight.data)
        #Initialize weights to reload model weights in the case of an emergency.

