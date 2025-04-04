print(f"0")
from fastai.vision import *
print(f"1")
from fastai.vision.data import ImageDataLoaders
from fastai.learner import *
print("2")
from fastai.vision.learner import cnn_learner, vision_learner
print("3")
from torch.utils.data import DataLoader
print("4")
from sklearn.metrics import accuracy_score
print("5")
from model import CNN
print("6")
from fastai.optimizer import *
print("7")
from functools import partial
print("8")
from torch import optim
print("9")
from fastai.losses import CrossEntropyLossFlat
print("10")
import matplotlib.pyplot as plt
print("11")
from pathlib import Path
print("12")
from fastai.metrics import accuracy, Precision, Recall
print("13")
from fastai.vision.learner import *
print("14")
import seaborn as sns
print("15")
from sklearn.metrics import confusion_matrix
print("16")
import os
from fastai.vision.core import *
import time as time
# opt_func = partial(OptimWrapper, opt=optim.Adam)
#
# batch_size_range = range(26, 32)
# model_v1 = [26, 0.002511886414140463]
# model_v2 = [27, 0.001737800776027143]
# model_v3 = [29, 0.001454397605732083]
# model_v4 = [29, 0.001454397605732083, '25-30ish']
# models = [model_v1, model_v2, model_v3]
# epochs = range(10, 15)
# def train_model(arch, n_out, path):
#     class_weights = torch.tensor([3, 0.3])  # 3 for dog (0), 1/3 for null (1)
#     loss_func = CrossEntropyLossFlat(weight=class_weights)
#     path = Path(path).expanduser().resolve()
#     data = ImageDataLoaders.from_folder(path, valid_pct=0.2, bs= model_v4[0])
#     learner = vision_learner(dls=data, arch=arch, n_out=n_out, pretrained=False, loss_func=loss_func, metrics=[accuracy, Precision(), Recall()])
#     print(f"Train length: {len(data.train)}")
#     learner.fit_one_cycle(25, model_v4[1])
#     learner.show_results()
#     interp = ClassificationInterpretation.from_learner(learner)
#     interp.plot_confusion_matrix(figsize=(6, 6), dpi=100)
#     cm = interp.confusion_matrix()
#     print(cm)
#     tn, fp, fn, tp = interp.confusion_matrix().ravel()
#     print(f"True Positives: {tp}")
#     print(f"True Negatives: {tn}")
#     print(f"False Positives: {fp}")
#     print(f"False Negatives: {fn}")
#     learner.export('/home/mateo/Github/PetProfiler/model_v4.pkl')
#
# def find_learning_rate(arch, n_out, path):
#     class_weights = torch.tensor([3, 0.3])  # 3 for dog (0), 1/3 for null (1)
#     print(torch.cuda.is_available())  # Should return True
#     print(torch.cuda.device_count())  # Should return >0
#     print(torch.cuda.current_device())  # Should return 0 if GPU is recognized
#     print(torch.cuda.get_device_name(0))
#
#     # Applying the class weights in BCEWithLogitsLoss
#     loss_func = CrossEntropyLossFlat(weight=class_weights)
#     path = Path(path).expanduser().resolve()
#     valid_extensions = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
#     all_images = [f for f in path.rglob("*") if f.suffix.lower() in valid_extensions]
#     avif_images = [f for f in path.rglob("*.avif")]
#     print(f"All images: {len(all_images)}, avif: {len(avif_images)}")
#     print(f"PATH: {path}")
#     for model in models:
#         data = ImageDataLoaders.from_folder(path, valid_pct=0.2, bs= model[0])
#         learner = vision_learner(dls=data, arch=arch, n_out=n_out, pretrained=False, loss_func=loss_func, metrics=[accuracy, Precision(), Recall()])
#         # suggested_lrs = learner.lr_find()
#         # ehe = learner.recorder.plot_lr_find()
#         # lrs = learner.recorder.lrs
#         # losses = learner.recorder.losses
#         for epoch in epochs:
#             for _ in range(3):
#                 hehe = learner.fit_one_cycle(epoch, lr_max=model[1])
#                 learner.show_results()
#                 print(f"Batch Size: {model[0]}\n"
#                       f"Learning Rate: {model[1]}\n"
#                       f"Epochs: {epoch}")
#
#
# # find_learning_rate(CNN, 2, '/home/mateo/datasets/petprofiler-dataset/roboflow/petprofiler-updated.v1-revised-dataset.folder')
# # train_model(CNN, 2, '/home/mateo/datasets/petprofiler-dataset/roboflow/petprofiler-updated.v1-revised-dataset.folder')
#
# def label_func(x):
#     return x.parent.name
# def t(model_path, test_path):
#     torch.cuda.empty_cache()
#     test_images = get_image_files(test_path)
#     print(f"Test images: {test_images}")
#     dls = ImageDataLoaders.from_path_func(test_path, test_images, label_func= label_func, valid_pct=0.95, bs=1, shuffle=True)
#     print(f"dlsss: {dls.vocab}")
#     breakpoint()
#     print(f"Len: {len(dls.items)}")
#     learn = load_learner(fname=model_path, cpu=True)
#
#
#     learn.dls = dls
#
#
#     preds, targets, loss = learn.get_preds(with_loss= True)
#     learn.show_results()
#     pred_args = preds.argmax(dim=1)
#
#     # loss = loss.numpy().tolist()
#     # x = range(len(loss))
#     # plt_figg = plt.figure(figsize=(6, 6))
#     # plt.plot(x, loss)
#     # plt.xlabel('Images', size=13, fontweight='bold')
#     # plt.ylabel('Loss', size=13, fontweight='bold')
#     # plt.title('Dry Environment-Model Testing Loss over Images', size=14, pad=15, fontweight='bold')
#     # plt.show()
#     # breakpoint()
#
#     y_true = targets.numpy()
#     y_pred = pred_args.numpy()
#     cm = confusion_matrix(y_true, y_pred)
#
#     plt_fig = plt.figure(figsize=(7,7))
#
#     sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=True,
#                 xticklabels=['Dog Class', 'Null Class'], yticklabels=['Dog Class', 'Null Class'], annot_kws={"size": 45} )
#     plt.title("Confusion Matrix for Test Split Results", size=15, pad=15, fontweight="bold")
#     plt.xlabel('Predicted Label', size= 13, fontweight='semibold')
#     plt.ylabel('True Label', size= 13, fontweight='semibold')
#     plt.tight_layout()
#     plt.show()
#
#     return preds, targets
def main(model_path):
    import cv2 as cv2
#     cap = cv2.VideoCapture(4)
    learn = load_learner(model_path, cpu=True)
    while True:
        # try:
        #     ret, frame = cap.read()
        #     if not ret:
        #         print("Failed to capture frame")
        #         break
        #     img = '/home/mateo/Github/PetProfiler/run_images
        #CANT USE 8 or 9
            # img = '/home/mateopi/projects/PetProfiler/run_images/frame.jpg'
            # cv2.imwrite(img, frame)c
            img = PILImage.create('/home/mateo/Github/PetProfiler/run_images/6.jpg')
            # img = Image._show(img)
            # img = PILImage.create('/home/mateopi/projects/PetProfiler/run_images/frame.jpg')

            pred, pred_idx, probs = learn.predict(img)
            img = Image._show(img)
            # os.remove('/home/mateo/Github/PetProfiler/run_images/frame.jpg')
            # os.remove('/home/mateopi/projects/PetProfiler/run_images/frame.jpg')
            print(f"pred: {pred}, pred_idx: {pred_idx}, probs: {probs}")
            if pred == 'dog':
                print("Pred is 0 - Dog detected")
                # motor.open()
            else:
                print("Pred is 1 - Nothing")
                pass
            # dog is index 0
            breakpoint()
        # except KeyboardInterrupt or KeyError:
        #     cap.release()
        #     cv2.destroyAllWindows()
        #     break


if __name__ == '__main__':
    # main('/home/mateo/Github/PetProfiler/model_v4.pkl')
    main('/home/mateo/Github/PetProfiler/model_v4.pkl')