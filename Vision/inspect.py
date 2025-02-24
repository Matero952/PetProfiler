import cv2
import os
import time
import Network.CNN.CNNConstants
from Network.CNN.model import CNN
from Network.CNN.CNNConstants import DEVICE as DEVICE
import torch
from torchvision import transforms
from PIL import Image

class Capture:
    def __init__(self, model, model_path):
        self.model = model
        self.model_path = model_path
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.to(device=torch.device('cpu'))
        self.model.eval()
    def analyze(self, frame) -> int:
        image = Image.open(frame)
        transformed_image = transforms.Grayscale(num_output_channels=1)(image)
        transformed_image = transforms.Resize(size=(640,640))(transformed_image)
        transformed_image = transforms.ToTensor()(transformed_image)
        transformed_image_batch = transformed_image.repeat(Network.CNN.CNNConstants.BATCH_SIZE, 1, 1, 1)
        with torch.no_grad():
            image = transformed_image_batch.to(device=torch.device('cpu'))
            pred = self.model(image)
            print(f"Prediction: {pred[0]}")
            dog = True if torch.round(torch.sigmoid(pred[0])) == 1 else False
            #No clue why this works but it does.
            if dog:
                print(f"Dog recognized (:")
                return 1
            else:
                print(f"Dog not recognized (:")
                return 0
    # cap = cv2.VideoCapture(0)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # interval = 60
    # count = 0
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         print("Error: Failed to capture frame.")
    #         break
    #     # cv2.imshow("Camera Feed", frame)
    #     count += 1
    #     if count % interval == 0:
    #         img = f"frame{count}.jpg"
    #         cv2.imwrite(img, frame)
    #         analyze(img)
    #         time.sleep(5)
    #         os.remove(img)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         os.remove(img)
    #         break
    # cap.release()
    # cv2.destroyAllWindows()
if __name__ == "__main__":
    model = CNN(1, 1)
    model.to(device=DEVICE)
    model.eval()
    capture = Capture(model=model, model_path="../results/megusta/model_7epochs_lr:0.00052.pth")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = 60
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        # cv2.imshow("Camera Feed", frame)
        count += 1
        if count % interval == 0:
            img = f"frame{count}.jpg"
            cv2.imwrite(img, frame)
            capture.analyze(img)
            time.sleep(5)
            os.remove(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            os.remove(img)
            break
    cap.release()
    cv2.destroyAllWindows()