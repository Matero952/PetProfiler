#TODO Please fix this file.
def valid() -> None:
		start = time.time()
		valid_correct_sum = 0
		valid_accuracy = np.zeros(CNNConstants.EPOCHS)
		valid_loss = np.zeros(CNNConstants.EPOCHS)
		model.load_state_dict(torch.load(CNNConstants.MODEL_SAVE_PATH, weights_only= True))
		input_num = 0
		global valid_loader
		valid_dataset = valid_loader
		with torch.no_grad():
			for epoch in range(CNNConstants.EPOCHS):
				loss = 0
				for (inputs, labels) in valid_dataset:
					input_num += 1
					model.eval()
					# Sets model to evaluation model for inference.
					inputs = inputs.to(CNNConstants.DEVICE)
					labels = labels.to(CNNConstants.DEVICE).float()
					valid_predict = model(inputs)
					epoch_loss = loss_fn(valid_predict, labels)
					loss += epoch_loss.item()
					valid_correct_sum += (valid_predict.round().eq(labels)).sum().item()
					accuracy = valid_correct_sum / input_num
			valid_loss[epoch] = loss
			valid_accuracy[epoch] = accuracy
		plt.style.use('ggplot')
		x1 = [x+1 for x in range(CNNConstants.EPOCHS)]
		print(f"x: {x1}")
		y1 = valid_loss
		#Train accuracy over epochs.
		y2 = valid_accuracy
		#Train correct sum to epochs
		fig, ax1 = plt.subplots()
		ax2 = ax1.twinx()
		ax1.plot(x1, y1, 'r--', label= "Valid Losses")
		ax1.plot(x1, y2, 'bs', label= "Valid Accuracy")
		ax1.legend(loc= "upper left")
		ax1.set_xlabel("Epochs")
		ax1.set_ylabel("Valid Loss")
		ax2.set_ylabel("Valid Accuracy")

		if not os.path.exists('../../results'):
			os.mkdir('../../results')
		plt.savefig(f'../../results/validExperimentResults')
		plt.show()
		end = time.time()
		print(f"Time elapsed: {end - start} seconds.")
