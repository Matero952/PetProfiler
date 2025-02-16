import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import pandas as pd
from Dataset import PetProfiler
import CNNConstants
from Network.CNN.model import CNN
import time

model = CNN(1, 1)
model.load_state_dict(torch.load("../../results/modelstate(good).pth"))
model.to(device=CNNConstants.DEVICE)
model.eval()
loss_fn = torch.nn.BCEWithLogitsLoss()
test_data = PetProfiler(CNNConstants.TEST_JSON)
test_loader = DataLoader(test_data, batch_size=CNNConstants.BATCH_SIZE, shuffle=False)
def determine():
    correct = 0
    seen = 0
    results = []
    test_loss = []
    test_acc = []
    with torch.no_grad():
        for inputs,labels in test_loader:
            inputs = inputs.to(device=CNNConstants.DEVICE)
            labels = labels.to(device=CNNConstants.DEVICE).float()
            pred = model(inputs)
            loss = loss_fn(pred, labels)
            try:
                for prediction in pred:
                    correct += 1 if torch.round(torch.sigmoid(prediction)) >= 0.85 else 0
                    seen += 1
            except TypeError as e:
                print(f"test predict: {pred}, shape: {pred.shape}")
                correct += 1 if torch.round(torch.sigmoid(pred)) >= 0.85 else 0
                seen += 1
            test_acc.append(correct / seen)
            test_loss.append(loss.item())
    print(f"Correct: {correct}. Seen: {seen}. Accuracy: {correct / seen}. Loss: {loss}")
    plt.style.use("ggplot")
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    x1 = [x for x in range(len(test_loader))]
    y1 = test_loss
    y2 = test_acc
    ax1.plot(x1, y1, 'r--', label="Test Loss")
    ax2.plot(x1, y2, 'bs', label="Test Accuracy")
    ax1.set_xlabel("Batches")
    ax1.set_ylabel("Loss", color="r")
    ax2.set_ylabel("Accuracy", color="b")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title("Test Results of PetProfiler")
    results.append({
        "Test Loss" : test_loss,
        "Test Accuracy" : test_acc,
        "Batches" : len(test_loader)
    })
    df = pd.DataFrame(results)
    df.to_csv("../../results/test_results.csv", index=False)
    print(f"Saved testing results to ../../results/test_results.csv")
    plt.savefig(f'../../results/testresults{time.strftime('%Y%m%d')}')
    plt.show()
def predict():

if __name__ == "__main__":
    determine()
    print(f"Testing process done.")