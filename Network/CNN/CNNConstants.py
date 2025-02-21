import torch
from torchvision.transforms import v2
import time

#Model
#Paths
MODEL_SAVE_PATH = f"../../results/parameters_{time.strftime('%Y%m%d')}"
TRAIN_JSON = "../../dataset/Train_Annotation/_annotations.coco.json"
VALID_JSON = "../../dataset/Valid_Annotation/_annotations.coco.json"
TEST_JSON = "../../dataset/Test_Annotation/_annotations.coco.json"
TRAIN_PLT_SAVE_PATH = f"../..results/train_graph{time.strftime('%Y%m%d')}"
VALID_PLT_SAVE_PATH = f"../..results/valid_graph{time.strftime('%Y%m%d')}"
LEARNING_RATE = 0.0003
BATCH_SIZE = 12
EPOCHS = 10

#Safety features
EPOCH_PATIENCE = 5
WORST_TRAIN_LOSS = 0
WORST_VALID_LOSS = 0
PATIENCE_COUNTER = 0
BEST_TRAIN_CORRECT = 0
BEST_VALID_CORRECT = 0
ACCURACY_CUTOFF = 0.

#Device and specifics
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRANSFORMATION = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
