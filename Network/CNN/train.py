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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from statistics import harmonic_mean

#TODO Create confusion matrix, precision(correct predicted positives to all predicted positive cases),
#TODO recall(correctly predicted positive to actual positives in dataset), F1-score(harmonic mean of precision and recall).
positive_dataset = 546
weight_class_2 = (648 / (648 + 546))
class_weights = torch.tensor([weight_class_2])
model75 = CNN(1, 1)
adam75 = torch.optim.Adam(model75.parameters(), lr=CNNConstants.LEARNING_RATE, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(adam75, mode='min', factor=0.3, patience=2, verbose=True)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(CNNConstants.DEVICE))
model75.to(device=CNNConstants.DEVICE)
print("Loading the PetProfiler dataset...")
train_data = PetProfiler(CNNConstants.TRAIN_JSON)
valid_data = PetProfiler(CNNConstants.VALID_JSON)
train_loader = DataLoader(train_data, batch_size=CNNConstants.BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=CNNConstants.BATCH_SIZE, shuffle=True, drop_last=True)
#For valid_loader, drop_last is set to true to compensate for the even batch_size and the odd amount of images in valid.
def train() -> None:
	CNN.weights_init(model75)
	for param in model75.parameters():
		param.requires_grad = True
	start = time.time()
	epochs_loss = []
	epochs_acc = []
	patience_counter = 0
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
			train_predict = model75(inputs)
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
				pred = torch.round(torch.sigmoid(i)).item()
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
			adam75.zero_grad()
			loss.backward()
			# torch.nn.utils.clip_grad_norm_(model72.parameters(), 1)
			adam75.step()
		scheduler.step(np.mean(epoch_loss))
		epochs_loss.append(epoch_loss)
		epochs_acc.append(epoch_accuracy)
		for param_group in adam75.param_groups:
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
	torch.save(model75.state_dict(), f"../../results/model_{CNNConstants.EPOCHS}epochs_lr:{current_lr}.pth")
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
	valid_results = []
	current_lr = CNNConstants.LEARNING_RATE
	model75.load_state_dict(torch.load(f"../../results/model_{CNNConstants.EPOCHS}epochs_lr:{current_lr}.pth"))
	with torch.no_grad():
			valid_accuracy = []
			valid_loss = []
			for (inputs, labels) in valid_loader:
				model75.eval()
				# Sets model to evaluation model for inference.
				inputs = inputs.to(CNNConstants.DEVICE)
				labels = labels.to(CNNConstants.DEVICE).float()
				valid_predict = model75(inputs)
				loss = loss_fn(valid_predict, labels)
				try:
					for prediction in valid_predict:
						correct += 1 if torch.round(torch.sigmoid(prediction)) >= 0.85 else 0
						seen += 1
				except TypeError as e:
					print(f"train_predict: {valid_predict}, shape: {valid_predict.shape}")
					correct += 1 if torch.round(torch.sigmoid(valid_predict)) >= 0.85 else 0
					seen += 1
				valid_loss.append(loss.item())
				valid_accuracy.append(correct / seen)
				print(f"Correct: {correct}. Seen: {seen}. Accuracy: {correct / seen}. Loss: {loss}")
	print(f"Epoch_loss = {valid_loss} Length = {len(valid_loss)}")
	print(f"Epoch_accuracy: {valid_accuracy} Length = {len(valid_accuracy)}")
	plt.style.use('ggplot')
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	x1 = [x for x in range(len(valid_loader))]
	print(f"X1: {x1}")
	y1 = valid_loss
	print(f"y1: {y1}")
	y2 = valid_accuracy
	print(f"y2: {y2}")
	ax1.plot(x1, y1, 'r--', label= "Valid Loss")
	ax1.set_xlabel("Batches")
	ax1.set_ylabel("Loss", color='r')
	ax2.plot(x1, y2, 'bs', label = "Valid Accuracy")
	ax2.set_ylabel("Accuracy", color='b')
	ax1.legend(loc= "upper left")
	ax2.legend(loc = "upper right")
	# plt.title("Normal Validation Example of Loss and Accuracy over Epoch")
	plt.title("Validation Loss and Accuracy")
	# valid_results.append({
	# 	"Train loss over epoch": valid_loss,
	# 	"Train accuracy over epoch": valid_accuracy,
	# 	"Time taken for valid": time.time() - start
	# })
	if not os.path.exists('../../results'):
		os.mkdir('../../results')
	plt.savefig(f'../../results/validresults{CNNConstants.EPOCHS}epochs_lr:{current_lr}.png')
	plt.show()
if __name__ == "__main__":
	train()
	print(f"Training process done.")
	valid()
	print(f"Validation process done.")