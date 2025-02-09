import torch
import gc
def elapsed_time(start, end) -> float:
    elapsed_time = end - start
    print(f"Elapsed time: {elapsed_time} seconds")
    return elapsed_time
    #Calculates time taken to train

def get_class_indices(label):
    print(f"label: {label}")
    if isinstance(label, tuple):
        label = torch.tensor(label)
    if label.dim() != 1:
        #Checks dimensions of label
        label = label.argmax(dim=1)
        #One-hot encoded -> class indices
    return label

def unsqueeze_predict(predict):
    if predict.dim() == 0 or predict.numel() == 0:
        raise ValueError("predict cannot be empty or doesnt have valid predictions")

    if predict.dim() == 1:
        predict = predict.unsqueeze(1)

    if predict.shape[1] == 1:
        predict = (predict > 0.5).float()

    return predict

def check_scalar(predict):
    if len(predict.shape) == 0:
        predict = predict.unsqueeze(-1)
#TODO figure out return value for this function
def empty_memory():
    gc.collect()
    torch.cuda.empty_cache()

def tensor_to_numpy(x, y):
    if not isinstance(x, (list, torch.Tensor)):
        raise TypeError(f"Expected x input to be a list or tensor. Got {type(x)} instead.")
    if not isinstance(y, (list, torch.Tensor)):
        raise TypeError(f"Expected y input to be a list or tensor. Got {type(y)} instead.")

    # Flatten nested lists or arrays if necessary
    if isinstance(x, list):
        try:
            x = torch.tensor([item for sublist in x for item in sublist] if isinstance(x[0], list) else x)
        except Exception as e:
            raise ValueError(f"Error converting x to tensor: {e}")
    else:
        x = torch.tensor(x)

    if isinstance(y, list):
        try:
            y = torch.tensor([item for sublist in y for item in sublist] if isinstance(y[0], list) else y)
        except Exception as e:
            raise ValueError(f"Error converting y to tensor: {e}")
    else:
        y = torch.tensor(y)
        #Turn tensor to numpy

def save_model(model, path):
    torch.save(model.state_dict(), path)

