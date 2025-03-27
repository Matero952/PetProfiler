from torch.utils.data import Dataset
from PIL import Image
import json
from torchvision.transforms import ToTensor, InterpolationMode
import os
import torch
from torchvision import datasets, transforms


class PetProfilerDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(
                        (600, 600), interpolation=InterpolationMode.BICUBIC
                    ),
                ]
            )
        self.dataset = datasets.ImageFolder(root, self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label
