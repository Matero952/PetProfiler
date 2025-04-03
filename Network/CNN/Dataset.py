from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision import datasets, transforms
import torch.utils.data.dataloader as dataloader
#This class used to be a bit larger but i cut it down a GOOD AMOUNT
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
        #{'dog': 0, 'none': 1}

test_dataset = PetProfilerDataset('~/datasets/petprofiler-dataset/local/')
test_loader = dataloader.DataLoader(test_dataset, batch_size=1)
