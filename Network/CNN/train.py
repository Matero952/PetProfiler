from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import os
from Dataset import PetProfiler
import CNNConstants
import time
import torch.nn as nn
import pandas as pd
from Network.CNN.model import CNN

model2 = CNN(1, 1)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=CNNConstants.LEARNING_RATE, weight_decay=0.05)
loss_fn = nn.BCEWithLogitsLoss()
model2.to(device=CNNConstants.DEVICE)

print("Loading the PetProfiler dataset...")
train_data = PetProfiler(CNNConstants.TRAIN_JSON)
valid_data = PetProfiler(CNNConstants.VALID_JSON)
train_loader = DataLoader(train_data, batch_size=CNNConstants.BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=CNNConstants.BATCH_SIZE, shuffle=True, drop_last=True)
#For valid_loader, drop_last is set to true to compensate for the even batch_size and the odd amount of images in valid.

def train() -> None:
	CNN.weights_init(model2)
	for param in model2.parameters():
		param.requires_grad = True
	start = time.time()
	epochs_loss = []
	epochs_acc = []
	patience_counter = 0
	stop = False
	lr = []
	train_results = []
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
			train_predict = model2(inputs)
			loss = loss_fn(train_predict, labels)
			print(f"Train predict: {train_predict}, shape: {train_predict.shape}")
			try:
				for prediction in train_predict:
					correct += 1 if torch.round(torch.sigmoid(prediction)) >= 0.85 else 0
					#Acceptable accuracy threshold. Adjust later.
					seen += 1
			except TypeError as e:
				print(f"train_predict: {train_predict}, shape: {train_predict.shape}")
				correct +=1 if torch.round(torch.sigmoid(train_predict)) >= 0.85 else 0
				seen += 1
			epoch_loss.append(loss.item())
			epoch_accuracy.append(correct / seen)
			assert seen > 0, "Um what?!"
			print(f"Correct: {correct}. Seen: {seen}. Accuracy: {correct / seen}. Loss: {loss}")
			optimizer2.zero_grad()
			loss.backward()
			optimizer2.step()

		for param_group in optimizer2.param_groups:
			current_lr = param_group['lr']
		lr.append(current_lr)

		if correct/seen > 0.6:
			pass
		else:
			patience_counter += 1
		if patience_counter >= 5:
			torch.save(model2.state_dict(), f"../../results/modelstate{time.strftime('%Y%m%d')}.pth")
			print(f"Training process emergency stopped.")
			stop = True
		print(f"Batch size: {CNNConstants.BATCH_SIZE}. Batches processed: {len(train_loader)}.")
		print(f"Epoch {epoch + 1}/{CNNConstants.EPOCHS}, Accuracy: {correct / seen}, LR: {current_lr}")

	plt.style.use('ggplot')
	#TODO Make subplots 
	# x1 = [x+1 for x in range(CNNConstants.EPOCHS)]
	x1 = [x for x in range(len(train_loader))]
	print(f"Epoch_loss = {epoch_loss} Length = {len(epoch_loss)}")
	y1 = epoch_loss
	print(f"Epoch_accuracy: {epoch_accuracy} Length = {len(epoch_accuracy)}")
	y2 = epoch_accuracy
	fig, ax1 = plt.subplots()
	ax1.plot(x1, y1, 'r--', label= "Loss over Epoch")
	ax1.set_xlabel("Batches")
	ax1.set_ylabel("Loss", color='r')
	ax2 = ax1.twinx()
	ax2.plot(x1, y2, 'bs', label = "Accuracy over Epoch")
	ax2.set_ylabel("Accuracy", color='b')
	ax1.legend(loc= "upper left")
	ax2.legend(loc= "upper right")
	# plt.title("Normal Training Example of Loss and Accuracy over Epoch")
	plt.title("Normal Training Example of Loss and Accuracy over Epoch")

	train_results.append({
		"Train loss over epoch": epoch_loss,
		"Train accuracy over epoch": epoch_accuracy,
		"Learning Rate over epochs": lr,
		"Time taken to train": time.time() - start
	})
	if not os.path.exists('../../results'):
		os.mkdir('../../results')
	df = pd.DataFrame(train_results)
	df.to_csv("../../results/train_results.csv", index=False)
	print(f"Saved training results to ../../results/train_results.csv")
	plt.savefig(f'../../results/trainresults{time.strftime('%Y%m%d')}')
	plt.show()
	torch.save(model2.state_dict(), "../../results/modelstate(good).pth")

def valid() -> None:
	start = time.time()
	correct = 0
	seen = 0
	valid_results = []
	model2.load_state_dict(torch.load("../../results/modelstate(good).pth"))
	with torch.no_grad():
		for epoch in range(CNNConstants.EPOCHS):
			epoch_accuracy = []
			epoch_loss = []
			for (inputs, labels) in valid_loader:
				model2.eval()
				# Sets model to evaluation model for inference.
				inputs = inputs.to(CNNConstants.DEVICE)
				labels = labels.to(CNNConstants.DEVICE).float()
				valid_predict = model2(inputs)
				loss = loss_fn(valid_predict, labels)
				try:
					for prediction in valid_predict:
						correct += 1 if torch.round(torch.sigmoid(prediction)) >= 0.85 else 0
						seen += 1
				except TypeError as e:
					print(f"train_predict: {valid_predict}, shape: {valid_predict.shape}")
					correct += 1 if torch.round(torch.sigmoid(valid_predict)) >= 0.85 else 0
					seen += 1
				epoch_loss.append(loss.item())
				epoch_accuracy.append(correct / seen)
				print(f"Correct: {correct}. Seen: {seen}. Accuracy: {correct / seen}. Loss: {loss}")
	print(f"Epoch_loss = {epoch_loss} Length = {len(epoch_loss)}")
	print(f"Epoch_accuracy: {epoch_accuracy} Length = {len(epoch_accuracy)}")
	plt.style.use('ggplot')
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	x1 = [x for x in range(len(valid_loader))]
	print(f"X1: {x1}")
	y1 = epoch_loss
	print(f"y1: {y1}")
	y2 = epoch_accuracy
	print(f"y2: {y2}")
	ax1.plot(x1, y1, 'r--', label= "Valid Loss over Epoch")
	ax1.set_xlabel("Batches")
	ax1.set_ylabel("Loss", color='r')
	ax2.plot(x1, y2, 'bs', label = "Valid Accuracy over Epoch")
	ax2.set_ylabel("Accuracy", color='b')
	ax1.legend(loc= "upper left")
	ax2.legend(loc = "upper right")
	# plt.title("Normal Validation Example of Loss and Accuracy over Epoch")
	plt.title("Normal Validation Example of Loss and Accuracy over Epoch")
	valid_results.append({
		"Train loss over epoch": epoch_loss,
		"Train accuracy over epoch": epoch_accuracy,
		"Time taken for valid": time.time() - start
	})
	if not os.path.exists('../../results'):
		os.mkdir('../../results')
	df = pd.DataFrame(valid_results)
	df.to_csv("../../results/valid_results.csv", index=False)
	print(f"Saved training results to ../../results/valid_results.csv")
	plt.savefig(f'../../results/validresults{time.strftime('%Y%m%d')}')
	plt.show()

if __name__ == "__main__":
	train()
	print(f"Training process done.")
	valid()
	print(f"Validation process done.")