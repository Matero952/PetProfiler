import cv2
import torch
from fastai.vision.all import *
from fastai.vision.data import ImageDataLoaders
from fastai.vision.learner import cnn_learner, vision_learner
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from fastai.optimizer import *
from functools import partial
from torch import optim
from fastai.losses import CrossEntropyLossFlat
import matplotlib.pyplot as plt
from pathlib import Path
from fastai.metrics import accuracy, Precision, Recall
from fastai.vision.learner import *
import seaborn as sns
from sklearn.metrics import confusion_matrix
from Network.CNN.model import CNN
def main(model_path):
    import cv2 as cv2
#     cap = cv2.VideoCapture(4)
    print(os.path.exists(model_path))
    learn = load_learner(model_path, cpu=True)
    while True:
        # try:
        #     ret, frame = cap.read()
        #     if not ret:
        #         print("Failed to capture frame")
        #         break
            img = '/home/mateo/Github/PetProfiler/run_images/1.jpg'
            # img = '/home/mateopi/projects/PetProfiler/run_images/frame.jpg'
            # cv2.imwrite(img, frame)
            img = PILImage.create('/home/mateo/Github/PetProfiler/run_images/frame.jpg')
            img = Image._show(img)
            # img = PILImage.create('/home/mateopi/projects/PetProfiler/run_images/frame.jpg')
            breakpoint()
            pred, pred_idx, probs = learn.predict(img)
            # os.remove('/home/mateo/Github/PetProfiler/run_images/frame.jpg')
            # os.remove('/home/mateopi/projects/PetProfiler/run_images/frame.jpg')
            print(f"pred: {pred}, pred_idx: {pred_idx}, probs: {probs}")
            if pred == 'dog':
                print("Pred is 0 - Dog detected")
                # motor.open()
            else:
                print("Pred is 1 - Nothing")
                pass
            # dog is index 0
            breakpoint()
        # except KeyboardInterrupt or KeyError:
        #     cap.release()
        #     cv2.destroyAllWindows()
        #     break

class Pipeline:
    def __init__(self, motor, model, Capture, model_path):
        self.motor = motor
        self.model = model
        self.model_path = model_path
        self.Capture = Capture
    def pipeline(self) -> None:
        cap = cv2.VideoCapture('/dev/video0')
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = 150
        count = 0
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                count += 1
                print(f"count: {count}")
                if count % interval == 0:
                    img = f"frames{count}.jpg"
                    cv2.imwrite(img, frame)
                    result = self.Capture.analyze(img)
                    print(f"Result: {result}")
                    if result == 1:
                        self.motor.open()
                        time.sleep(10)
                        self.motor.close()
                    else:
                        pass
                else:
                    pass
            except KeyboardInterrupt or KeyError:
                cap.release()
                cv2.destroyAllWindows()
                break
if __name__ == '__main__':
    while True:
        main('/home/mateo/Github/PetProfiler/model_v4.pkl')
