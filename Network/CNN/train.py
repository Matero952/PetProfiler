from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import os
from Dataset import PetProfiler
import CNNConstants
import time
import torch.nn as nn
import pandas as pd
import numpy as np

model = CNNConstants.MODEL2
optimizer = torch.optim.Adam(model.parameters(), lr=CNNConstants.LEARNING_RATE)
loss_fn = nn.BCEWithLogitsLoss()
model.to(device=CNNConstants.DEVICE)
print("Loading the PetProfiler dataset...")
train_data = PetProfiler(CNNConstants.TRAIN_JSON)
train_loader = DataLoader(train_data, batch_size=CNNConstants.BATCH_SIZE, shuffle=True)
def train() -> None:
	for layer in model.children():
		if hasattr(layer, "reset_parameters"):
			layer.reset_parameters()
	optimizer.zero_grad(set_to_none=True)
	train_start = time.time()
	epoch_accuracy = []
	epoch_loss = []
	patience_counter = 0
	stop = False
	lr = []
	train_results = []
	time_over_epoch = []
	for epoch in range(CNNConstants.EPOCHS):
		epoch_start = 0
		epoch_start = time.time()
		train_loss = 0
		seen = 0
		correct = 0
		batch_count = 0
		average_loss = 0
		for (inputs, labels) in train_loader:
			if stop:
				break
			inputs = inputs.to(CNNConstants.DEVICE)
			labels = labels.to(CNNConstants.DEVICE).float()
			print(f"Labels: {labels} shape: {labels.shape}")
			train_predict = model(inputs)
			print(f"Train predict: {train_predict}, shape: {train_predict.shape}")
			try:
				for prediction in train_predict:
					correct += 1 if torch.round(prediction) >= 0.5 else 0
					#Acceptable accuracy threshold. Adjust later.
			except TypeError as e:
				train_predict = train_predict.view(-1)
				for prediction in train_predict:
					correct +=1 if train_predict >= 0.5 else 0
			seen += 2
			batch_count += 1
			print(f"Correct: {correct}")
			print(f"Seen count: {seen}")
			loss = loss_fn(train_predict, labels)
			print(f"Loss: {loss}")
			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			optimizer.step()

			print(f"Penis loss: {loss.item()}")
			if isinstance(train_loss, torch.Tensor):
				train_loss += loss.item()
			else:
				train_loss += loss
		for param_group in optimizer.param_groups:
			current_lr = param_group['lr']
		lr.append(current_lr)
		accuracy = correct / seen if seen > 0 else 0
		epoch_accuracy.append(accuracy)
		print(f"Accuracy: {accuracy}")
		print(f"Train Accuracy: {epoch_accuracy}")
		average_loss = train_loss/batch_count
		epoch_loss.append(average_loss)
		time_over_epoch.append(time.time() - epoch_start)
		if accuracy > 0.5:
			pass
		else:
			patience_counter += 1
		if patience_counter >= 5:
			torch.save(model.state_dict(), "../../results/modelstate.pth")
			print(f"Training process emergency stopped.")
			stop = True
		print(f"Epoch {epoch + 1}/{CNNConstants.EPOCHS}, "
			  f"Loss: {train_loss / len(train_loader)}, "
			  f"Accuracy: {correct / seen}, "
			  f"LR: {current_lr}")
	plt.style.use('ggplot')
	x1 = [x+1 for x in range(CNNConstants.EPOCHS)]
	print(f"x: {x1}")
	print(f"Epoch_loss: {epoch_loss}")
	print(f"Learning rate: {lr}")
	print(f"Epoch_accuracy: {epoch_accuracy}")
	y1 = [loss.cpu().detach().numpy() if isinstance(loss, torch.Tensor) else loss for loss in epoch_loss]
	# y1 = epoch_loss
	# y1 = epoch_loss[0].detach().numpy()
	y2 = epoch_accuracy
	y3 = lr
	fig, ax1 = plt.subplots()
	ax1.plot(x1, y1, 'r--', label= "Loss over Epochs")
	ax1.plot(x1, y2, 'bs', label= "Accuracy over Epochs")
	ax1.plot(x1, y3, 'g^', label= "Learning Rate")
	plt.legend(loc= "upper left")
	ax1.set_xlabel("Epochs")
	ax1.set_ylabel("Train Loss")
	#Weight histogram stuff below.

	train_results.append({
		"Train loss over epochs": epoch_loss,
		"Train accuracy over epochs": epoch_accuracy,
		"Learning Rate over epochs": lr,
		"Time taken to train": time.time() - train_start,
		"Time over Epochs": time_over_epoch})
	if not os.path.exists('../../results'):
		os.mkdir('../../results')
	df = pd.DataFrame(train_results)
	df.to_csv("../../results/train_results.csv", index=False)
	print(f"Saved training results to ../../results/train_results.csv")
	plt.savefig(f'../../results/')
	plt.show()
	torch.save(model.state_dict(), "../../results/modelstate.pth")

if __name__ == "__main__":
	train()
	print(f"Training process done.")