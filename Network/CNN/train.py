from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import os
from Dataset import PetProfiler
import CNNConstants
import time
import torch.nn as nn
from Network.CNN.CNNConstants import LEARNING_RATE
from Network.CNN.model import CNN
import numpy as np
from statistics import harmonic_mean

#TODO Create confusion matrix, precision(correct predicted positives to all predicted positive cases),
#TODO recall(correctly predicted positive to actual positives in dataset), F1-score(harmonic mean of precision and recall).
# positive_dataset = 546
# negative_dataset = 648
# weight_class_2 = 0.75 * (546 / (648 + 546))
# class_weights =  (torch.tensor([weight_class_2]))
model89 = CNN(1, 1)
adam89 = torch.optim.Adam(model89.parameters(), lr=CNNConstants.LEARNING_RATE, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss()
model89.to(device=CNNConstants.DEVICE)
print("Loading the PetProfiler dataset...")
train_data = PetProfiler(CNNConstants.TRAIN_JSON)
valid_data = PetProfiler(CNNConstants.VALID_JSON)
train_loader = DataLoader(train_data, batch_size=CNNConstants.BATCH_SIZE, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_data, batch_size=CNNConstants.BATCH_SIZE, shuffle=True, drop_last=True)
#For valid_loader, drop_last is set to true to compensate for the even batch_size and the odd amount of images in valid.
def train() -> None:
	CNN.weights_init(model89)
	for param in model89.parameters():
		param.requires_grad = True
	epochs_loss = []
	epochs_acc = []
	val_loss = []
	val_acc = []
	stop = False
	lr = []
	results = []
	correct_positives = 0
	predicted_positives = 0
	for epoch in range(CNNConstants.EPOCHS):
		epoch_accuracy = []
		epoch_loss = []
		seen = 0
		correct = 0
		for (inputs, labels) in train_loader:
			if stop:
				break
			inputs = inputs.to(CNNConstants.DEVICE)
			labels = labels.to(CNNConstants.DEVICE).float()
			print(f"Labels: {labels} shape: {labels.shape}")
			train_predict = model89(inputs)
			train_predict = train_predict.view(-1, 1)
			print(f"Train Predict after view: {train_predict}")
			labels = labels.view(-1, 1)
			print(f"Labels after view: {labels}")
			loss = loss_fn(train_predict, labels)
			if train_predict.dim() == 0:
				break
			print(f"Loss: {loss}")
			print(f"Train predict: {train_predict}, shape: {train_predict.shape}")
			print(f"Labels: {labels}")
			for idx, i in enumerate(train_predict):
				pred = torch.sigmoid(i) >= 0.7
				actual = labels[idx].item()
				correct += 1 if pred == actual else 0
				seen += 1
				correct_positives += 1 if pred == 1 and actual == 1 else 0
				predicted_positives += 1 if pred == 1 else 0
			epoch_loss.append(loss.item())
			print(f"Epoch_loss: {epoch_loss}")
			epoch_accuracy.append(correct/seen)
			assert seen > 0, "Um what?!"
			print(f"Correct: {correct}. Seen: {seen}. Accuracy: {correct / seen}. Loss: {loss}")
			adam89.zero_grad()
			loss.backward()
			adam89.step()
		epochs_loss.append(epoch_loss)
		epochs_acc.append(epoch_accuracy)
		for param_group in adam89.param_groups:
			current_lr = param_group['lr']
		lr.append(current_lr)
		print(f"Batch size: {CNNConstants.BATCH_SIZE}. Batches processed: {len(train_loader)}.")
		print(f"Epoch {epoch + 1}/{CNNConstants.EPOCHS}, Accuracy: {correct / seen}, LR: {current_lr}")

	plt.style.use('ggplot')
	plt.suptitle(f'Training Loss and Accuracy over {CNNConstants.EPOCHS} epochs. LR:{LEARNING_RATE}', fontsize=2)
	rows = int(np.ceil(np.sqrt(CNNConstants.EPOCHS)))  # Square-like layout
	cols = int(np.ceil(CNNConstants.EPOCHS / rows))
	fig, axes = plt.subplots(nrows=rows, ncols=cols)
	for i in range(CNNConstants.EPOCHS):
		row = i // cols  # Get the row index
		col = i % cols  # Get the column index
		ax = axes[row, col]
		x1 = [x for x in range(len(train_loader))]
		y1 = epochs_loss[i]
		y2 = epochs_acc[i]
		ax.set_title(f"Training Loss and Accuracy over {i}/{CNNConstants.EPOCHS} epoch. LR: {current_lr}", fontsize=4)
		ax.set_xlabel(f'Batches over {i}/{CNNConstants.EPOCHS} epoch', fontsize=2)
		ax.set_ylabel(f'Loss over {i}/{CNNConstants.EPOCHS} epoch', color='r', fontsize=2)
		ax.plot(x1, y1, 'r--', label=f"Loss over {i}th epoch")
		ax1 = ax.twinx()
		ax1.plot(x1, y2, 'b-', label=f"Accuracy over {i}/{CNNConstants.EPOCHS} epoch")
		ax1.set_ylabel(f'Accuracy over {i}/{CNNConstants.EPOCHS} epoch', color='b', fontsize=2)
		ax.legend(loc='upper left', fontsize=1)
		ax1.legend(loc='upper right', fontsize=1)
	for j in range(CNNConstants.EPOCHS, rows * cols):
		row = j // cols
		col = j % cols
		axes[row, col].axis('off')
	if not os.path.exists('../../results'):
		os.mkdir('../../results')
	plt.tight_layout()
	plt.savefig(f'../../results/{CNNConstants.EPOCHS}epochs_lr:{current_lr}.png')
	plt.show()
	torch.save(model89.state_dict(), f"../../results/model_{CNNConstants.EPOCHS}epochs_lr:{current_lr}.pth")
	# Precision, recall, and F1 score
	metrics = ["Precision", "Recall", "F1 Score"]
	results.append(correct_positives / predicted_positives)
	results.append(correct_positives / positive_dataset)
	results.append(harmonic_mean(data=[correct_positives / predicted_positives, correct_positives / positive_dataset]))
	plt.figure(figsize=(10, 6))
	plt.bar(metrics, results, color=['red', 'blue', 'yellow'])
	plt.xlabel('Metrics', fontsize=10)
	plt.ylabel('Score', fontsize=10)
	plt.title('Precision, Recall, and F1 Score of Custom CNN', fontsize=15)
	plt.ylim(0, 1)
	plt.show()
	plt.savefig(f"../../results/{CNNConstants.EPOCHS}epochs_lr:{current_lr}PrecisionRecallF1.png")
def valid() -> None:
	start = time.time()
	correct = 0
	seen = 0
	correct_positives = 0
	predicted_positives = 0
	valid_results = []
	current_lr = CNNConstants.LEARNING_RATE
	model89.load_state_dict(torch.load(f"../../results/model_{CNNConstants.EPOCHS}epochs_lr:{current_lr}.pth"))
	with torch.no_grad():
			valid_loss = []
			for (inputs, labels) in valid_loader:
				model89.eval()
				# Sets model to evaluation model for inference.
				inputs = inputs.to(CNNConstants.DEVICE)
				labels = labels.to(CNNConstants.DEVICE).float()
				valid_predict = model89(inputs)
				loss = loss_fn(valid_predict, labels)
				for idx, i in enumerate(valid_predict):
					pred = torch.sigmoid(i) >= 0.7
					actual = labels[idx].item()
					correct += 1 if pred == actual else 0
					seen += 1
					correct_positives += 1 if pred == 1 and actual == 1 else 0
					predicted_positives += 1 if pred == 1 else 0
				valid_loss.append(loss.item())
				print(f"Correct: {correct}. Seen: {seen}. Accuracy: {correct / seen}. Loss: {loss}")
	print(f"Epoch_loss = {valid_loss} Length = {len(valid_loss)}")

	plt.style.use('ggplot')
	fig, ax1 = plt.subplots()
	x1 = [x for x in range(len(valid_loader))]
	print(f"X1: {x1}")
	y1 = valid_loss
	print(f"y1: {y1}")
	ax1.plot(x1, y1, 'r--', label= "Valid Loss")
	ax1.set_xlabel("Batches")
	ax1.set_ylabel("Loss", color='r')
	ax1.legend(loc= "upper left")
	plt.title("Validation Loss")
	plt.savefig(f'../../results/validresults{CNNConstants.EPOCHS}epochs_lr:{current_lr}.png')
	plt.show()
	valid_results.append(correct_positives / predicted_positives)
	valid_results.append(correct_positives / positive_dataset)
	valid_results.append(harmonic_mean(data=[correct_positives / predicted_positives, correct_positives / positive_dataset]))
	metrics = ["Precision", "Recall", "F1 Score"]
	plt.figure(figsize=(10, 6))
	plt.bar(metrics, valid_results, color=['red', 'blue', 'yellow'])
	plt.xlabel('Metrics', fontsize=10)
	plt.ylabel('Score', fontsize=10)
	plt.title('Precision, Recall, and F1 Score of Custom CNN', fontsize=15)
	plt.ylim(0, 1)
	plt.show()
	plt.savefig(f"../../results/Valid*{CNNConstants.EPOCHS}epochs_lr:{current_lr}PrecisionRecallF1.png")

	if not os.path.exists('../../results'):
		os.mkdir('../../results')
if __name__ == "__main__":
	train()
	print(f"Training process done.")
	valid()
	print(f"Validation process done.")