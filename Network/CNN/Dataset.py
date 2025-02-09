from torch.utils.data import Dataset
from PIL import Image
import json
from torchvision.transforms import ToTensor
import os
import re
import torchvision.transforms as transforms
import time

class PetProfiler(Dataset):
    def __init__(self, json_p, transform = None):
        self.json_ = json_p
        self.dataset_dic = {}
        self.transform = ToTensor()
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
        category_id = 1
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
                id_dict[img_id] = {'id' : img_id, 'path' : file}
        # print(f"Id_dict: {id_dict}")
        return id_dict
    
    def __len__(self) -> int:
        return len(self.get_img_id())
if __name__ == "__main__":
    #TODO Implement multi threading eventually (:
    jsonpath = "../../dataset/Test_Annotation/_annotations.coco.json"
    pp = PetProfiler(jsonpath)

