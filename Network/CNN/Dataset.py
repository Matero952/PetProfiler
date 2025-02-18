from torch.utils.data import Dataset
from PIL import Image
import json
from torchvision.transforms import ToTensor
import os
import re
import torchvision.transforms as transforms
import time
import torch

class PetProfiler(Dataset):
    def __init__(self, json_p, transform = None):
        self.json_ = json_p
        self.dataset_dic = {}
        self.transform = transform or transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.05),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=1, translate=(0.1, 0.11)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])])
        # self.transform = ToTensor()
        #Converts PIL image to tensor
        self.images = self.get_img_id()

    def __getitem__(self, idx, bbs = None, category_id = None):
        '''Get item function.'''
        print(f"In get item function now")
        json_mapping = {'../../dataset/Test_Annotation/_annotations.coco.json' : "test", "../../dataset/Valid_Annotation/_annotations.coco.json" : "valid", "../../dataset/Train_Annotation/_annotations.coco.json" : "train"}
        tag = json_mapping[self.json_]
        base_dir = f"../../dataset/{tag}/"
        img = self.images[idx]['path']
        img = os.path.join(base_dir, img)
        img = os.path.abspath(img)
        img = Image.open(img)
        if self.transform:
            img = self.transform(img)
        img = transforms.Grayscale(num_output_channels=1)(img)
        print(f"img: {img}")
        class_mapping = {2 : 0, 1 : 1}
        print(f"Image at index: {self.images[idx]}")
        category_id = self.images[idx]['category_id']
        category_id = torch.tensor(class_mapping[category_id])
        #In my json, null(or no dog recognized) is mapped to 2 and 1 is mapped to recognized dog.
        print(f"Category id: {category_id}")
        return img, category_id

    def get_img_id(self) -> dict:
        '''Creates id directories with all img ids and their attributes as entries'''
        id_dict = {}
        with open(self.json_) as f:
           data = json.load(f)
           for image in data['images']:
                img_id = image['id']
                file = image['file_name']
                id_dict[img_id] = {'id': img_id, 'path' : file}
           for annotation in data['annotations']:
               img_id = annotation['image_id']
               category_id = annotation['category_id']
               id_dict[img_id]['category_id'] = category_id
        return id_dict
    
    def __len__(self) -> int:
        return len(self.get_img_id())
if __name__ == "__main__":
    test = PetProfiler("../../dataset/Test_Annotation/_annotations.coco.json")

