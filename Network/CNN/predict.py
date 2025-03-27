from statistics import harmonic_mean

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Network.CNN.Dataset import PetProfiler
from Network.CNN.CNNConstants import DEVICE as DEVICE
from Network.CNN.CNNConstants import TEST_JSON as TEST_JSON
from Network.CNN.CNNConstants import BATCH_SIZE as BATCH_SIZE
from Network.CNN.model import CNN
from Network.CNN.train import train_loader

model = CNN(1, 1)
model.load_state_dict(torch.load("../../results/megusta/model_7epochs_lr:0.00052.pth"))
model.to(device=DEVICE)
model.eval()
loss_fn = torch.nn.BCEWithLogitsLoss()
test_data = PetProfiler(TEST_JSON)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
def determine():
    correct = 0
    seen = 0
    results = []
    test_loss = []
    test_acc = []
    correct_positives = 0
    predicted_positives = 0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    with torch.no_grad():
        for inputs,labels in test_loader:
            inputs = inputs.to(device=DEVICE)
            labels = labels.to(device=DEVICE).float()
            pred = model(inputs)
            loss = loss_fn(pred, labels)
            for idx, i in enumerate(pred):
                rounded_pred = torch.round(torch.sigmoid(i))  # Use a new variable
                actual = labels[idx].item()
                correct += 1 if rounded_pred == actual else 0
                seen += 1
                correct_positives += 1 if rounded_pred == 1 and actual == 1 else 0
                predicted_positives += 1 if rounded_pred == 1 else 0
                true_positive += 1 if rounded_pred == 1 and actual == 1 else 0
                false_positive += 1 if rounded_pred == 1 and actual == 0 else 0
                true_negative += 1 if rounded_pred == 0 and actual == 0 else 0
                false_negative += 1 if rounded_pred == 0 and actual == 1 else 0
                test_acc.append(correct / seen)
                test_loss.append(loss.item())
    metrics = ["Precision", "Recall", "F1-Score"]
    results = [(true_positive / (true_positive + false_positive)), (true_positive / (true_positive + false_negative)), harmonic_mean([(true_positive / (true_positive + false_positive)), (true_positive / (true_positive + false_negative))])]
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, results, color=['purple', 'blue', 'orange'])
    plt.xlabel('Metrics', fontsize=10)
    plt.ylabel('Score', fontsize=10)
    plt.title('Test Precision, Recall, and F1 Score of Custom CNN', fontsize=15)
    plt.savefig(f"../../results/figs/test_precision_recall_f1_score.png")
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    x1 = [x for x in range(109)]
    y1 = test_loss
    y2 = test_acc
    ax1.plot(x1, y1, 'r--', label="Test Loss")
    ax2.plot(x1, y2, 'bs', label="Test Accuracy")
    ax1.set_xlabel("Batches")
    ax1.set_ylabel("Loss", color="r")
    ax2.set_ylabel("Accuracy", color="b")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title("Test Accuracy and Loss of PetProfiler")
    plt.savefig(f"../../results/figs/test_loss_accuracy.png")
    plt.close()
if __name__ == "__main__":
    determine()
    print(f"Testing process done.")