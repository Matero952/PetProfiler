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

model1 = CNN(1, 1)
optimizer = torch.optim.Adam(model1.parameters(), lr=CNNConstants.LEARNING_RATE)
loss_fn = nn.BCEWithLogitsLoss()
model1.to(device=CNNConstants.DEVICE)
#TODO Implement learning rate scheduler

print("Loading the PetProfiler dataset...")
train_data = PetProfiler(CNNConstants.TRAIN_JSON)
valid_data = PetProfiler(CNNConstants.VALID_JSON)
train_loader = DataLoader(train_data, batch_size=CNNConstants.BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=CNNConstants.BATCH_SIZE, shuffle=True, drop_last=True)
#For valid_loader, drop_last is set to true to compensate for the even batch_size and the odd amount of images in valid.
def train() -> None:
	for param in model1.parameters():
		param.requires_grad = True
	start = time.time()
	epoch_accuracy = []
	epoch_loss = []
	patience_counter = 0
	stop = False
	lr = []
	train_results = []
	for epoch in range(CNNConstants.EPOCHS):
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

			train_predict = model1(inputs)
			loss = loss_fn(train_predict, labels)
			print(f"Train predict: {train_predict}, shape: {train_predict.shape}")
			try:
				for prediction in train_predict:
					correct += 1 if torch.round(prediction) >= 0.5 else 0
					#Acceptable accuracy threshold. Adjust later.
					seen += 2
			except TypeError as e:
				print(f"train_predict: {train_predict}, shape: {train_predict.shape}")
				correct +=1 if train_predict >= 0.5 else 0
				seen += 2
			batch_count += 1
			print(f"Correct: {correct}")
			print(f"Seen count: {seen}")
			print(f"Loss: {loss}")
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			train_loss = loss
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
		if accuracy > 0.5:
			pass
		else:
			patience_counter += 1
		if patience_counter >= 5:
			torch.save(model1.state_dict(), f"../../results/modelstate{time.strftime('%Y%m%d')}.pth")
			print(f"Training process emergency stopped.")
			stop = True
		print(f"Len of train_loader: {len(train_loader)}")
		print(f"Epoch {epoch + 1}/{CNNConstants.EPOCHS}, "
			  f"Average Loss: {train_loss / len(train_loader)}, "
			  f"Accuracy: {correct / seen}, "
			  f"LR: {current_lr}")
	plt.style.use('ggplot')
	x1 = [x+1 for x in range(CNNConstants.EPOCHS)]
	print(f"x: {x1}")
	if isinstance(epoch_loss, torch.Tensor):
		print("true")
	else:
		print("false")
	print(f"epoch_loss: {epoch_loss}")
	y1 = [epoch_loss.cpu().detach().numpy() if isinstance(epoch_loss, torch.Tensor) else loss for loss in epoch_loss]
	y2 = epoch_accuracy
	y3 = lr
	fig, ax1 = plt.subplots()
	ax1.plot(x1, y1, 'r--', label= "Average Loss over Epochs")
	ax1.plot(x1, y2, 'bs', label= "Accuracy over Epochs")
	ax1.plot(x1, y3, 'g^', label= "Learning Rate")
	plt.legend(loc= "upper left")
	ax1.set_xlabel("Epochs")
	ax1.set_ylabel("Train Loss")

	train_results.append({
		"Train loss over epochs": epoch_loss,
		"Train accuracy over epochs": epoch_accuracy,
		"Learning Rate over epochs": lr,
		"Time taken to train": time.time() - start
	})
	if not os.path.exists('../../results'):
		os.mkdir('../../results')
	df = pd.DataFrame(train_results)
	df.to_csv("../../results/train_results.csv", index=False)
	print(f"Saved training results to ../../results/train_results.csv")
	plt.savefig(f'../../results/modelresults{time.strftime('%Y%m%d')}')
	plt.show()
	torch.save(model1.state_dict(), "../../results/modelstate.pth")

if __name__ == "__main__":
	train()
	print(f"Training process done.")