import cv2
import os
import time
from Network.CNN.model import CNN
from Network.CNN.CNNConstants import DEVICE as DEVICE
import torch
from torchvision import transforms
from PIL import Image
model = CNN(1, 1)
try:
    model.load_state_dict(torch.load("../../results/modelstate(good).pth"))
except FileNotFoundError:
    model.load_state_dict(torch.load("../results/modelstate(good).pth"))
model.to(device=DEVICE)
model.eval()
loss_fn = torch.nn.BCEWithLogitsLoss()
def analyze(frame) -> int:
    #Transformations applied to frame.
    image = Image.open(frame)
    transformed_image = transforms.Grayscale(num_output_channels=1)(image)
    transformed_image = transforms.Resize(size=(640,640))(transformed_image)
    transformed_image = transforms.ToTensor()(transformed_image)
    transformed_image_batch = transformed_image.repeat(4, 1, 1, 1)
    with torch.no_grad():
        image = transformed_image_batch.to(device=DEVICE)
        pred = model(image)
        print(f"Prediction: {pred}")
        value = pred[0]
        value = torch.sigmoid(value)
        print(f"Value: {value}")
        assert 0 > 1
        try:
                print(f"Prediction: {pred[0]}")
                dog = True if torch.round(torch.sigmoid(pred[0])) > 0.85 else False
        except TypeError as e:
            for prediction in pred:
                print(f"test predict: {pred}, shape: {pred.shape}")
                dog = True if torch.round(torch.sigmoid(pred)) >= 0.85 else False
        #No clue why this works but it does.
        if dog:
            print(f"Dog recognized (:")
            return 1
        else:
            print(f"Dog not recognized (:")
            return 0
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
interval = 60
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    cv2.imshow("Camera Feed", frame)
    count += 1
    if count % interval == 0:
        img = f"frame{count}.jpg"
        cv2.imwrite(img, frame)
        analyze(img)
        time.sleep(20)
        os.remove(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        os.remove(img)
        break
cap.release()
cv2.destroyAllWindows()
