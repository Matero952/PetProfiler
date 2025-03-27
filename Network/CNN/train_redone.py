from fastai.vision.all import *
from fastai.vision.data import ImageDataLoaders
from fastai.vision.learner import cnn_learner, vision_learner
from torch.utils.data import DataLoader
from Dataset import *
from model import CNN
from fastai.optimizer import *
from functools import partial
from torch import optim
from fastai.losses import BCEWithLogitsLossFlat
import matplotlib.pyplot as plt


opt_func = partial(OptimWrapper, opt=optim.Adam)


def find_learning_rate(data_set, arch, n_out):
    data = DataLoaders.from_dsets(data_set, bs=16)

    learner = vision_learner(dls=data, arch=arch, n_out=n_out, pretrained=False, loss_func=BCEWithLogitsLossFlat(), metrics=accuracy)
    suggested_lrs = learner.lr_find()
    print(f"Suggested_lrs: {suggested_lrs}")
    print(learner.lr)
    ehe = learner.recorder.plot_lr_find()
    lrs = learner.recorder.lrs
    losses = learner.recorder.losses
    print(lrs)

    print(f"Valley learning rate: {lrs[0]}")
    # Plot the data manually
    plt.plot(lrs, losses)
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.xscale('log')  # Optional: Log scale for better view
    plt.title('Learning Rate Finder')
    plt.show()
    return learner

dataset = PetProfilerDataset('~/datasets/petprofiler-dataset/local/')
find_learning_rate(dataset, arch=CNN, n_out=1)