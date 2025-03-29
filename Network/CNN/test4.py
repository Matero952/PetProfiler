'''/home/mateo/Github/PetProfiler/.venv/bin/python /home/mateo/Github/PetProfiler/Network/CNN/train_redone.py
epoch     train_loss  valid_loss  accuracy  precision_score  recall_score  time
0         0.954046    0.480188    0.611465  0.918919         0.535152      00:14
1         0.742645    0.525969    0.539809  0.956204         0.412382      00:14
2         0.607637    0.394153    0.628981  0.978389         0.522560      00:14
3         0.516129    0.386334    0.777070  0.948069         0.747114      00:14
4         0.437362    0.369712    0.805732  0.952746         0.782791      00:14
5         0.323390    1.352564    0.804140  0.802913         0.983211      00:14
6         0.345012    0.428466    0.788217  0.920441         0.789087      00:14
7         0.299229    0.281614    0.777866  0.973315         0.727177      00:14
8         0.315066    0.238904    0.748408  0.987749         0.676810      00:14
9         0.258436    0.247685    0.885350  0.971995         0.874082      00:14
10        0.218377    0.202430    0.863057  0.987516         0.830010      00:14
11        0.170459    0.186740    0.875796  0.978391         0.855194      00:14
12        0.164549    0.261425    0.862261  0.969880         0.844701      00:14
13        0.139827    0.139912    0.863854  0.994937         0.824764      00:14
14        0.100468    0.106404    0.912420  0.992982         0.890871      00:14
15        0.096458    0.156804    0.948248  0.983660         0.947534      00:14
16        0.089454    0.095751    0.933917  0.992081         0.920252      00:14
17        0.084428    0.098946    0.938694  0.991031         0.927597      00:14
18        0.068183    0.081095    0.943471  0.993289         0.931794      00:14
19        0.053307    0.074064    0.955414  0.994487         0.946485      00:14
20        0.056405    0.064474    0.965764  0.996725         0.958027      00:14
21        0.048126    0.069948    0.952229  0.996663         0.940189      00:14
22        0.054487    0.065259    0.950637  0.996656         0.938090      00:14
23        0.051920    0.067238    0.970541  0.994600         0.966422      00:14
24        0.039394    0.064574    0.956210  0.996681         0.945435      00:14
[[300   3]
 [ 52 901]]
True Positives: 901
True Negatives: 300
False Positives: 3
False Negatives: 52'''

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_train_loss():
    x = range(0, 25)
    train_loss = [
        0.954046, 0.742645, 0.607637, 0.516129, 0.437362, 0.323390,
        0.345012, 0.299229, 0.315066, 0.258436, 0.218377, 0.170459,
        0.164549, 0.139827, 0.100468, 0.096458, 0.089454, 0.084428,
        0.068183, 0.053307, 0.056405, 0.048126, 0.054487, 0.051920,
        0.039394
    ]
    plt.plot(x, train_loss, color='b', label='Train Loss')
    plt.rcParams.update({'font.size': 20})
    plt.xlabel('Epoch', fontweight='bold', fontsize=15)
    plt.ylabel('Train Loss', fontweight='bold', fontsize=15)
    plt.title('Train Loss Over Epochs', fontsize=20, pad = 10, fontweight='bold')

    # Displaying the plot
    plt.legend()
    plt.grid(True)
    plt.show()

def polt_valid_loss_and_accuracy():
    valid_loss = [
        0.480188, 0.525969, 0.394153, 0.386334, 0.369712, 1.352564,
        0.428466, 0.281614, 0.238904, 0.247685, 0.202430, 0.186740,
        0.261425, 0.139912, 0.106404, 0.156804, 0.095751, 0.098946,
        0.081095, 0.074064, 0.064474, 0.069948, 0.065259, 0.067238,
        0.064574
    ]

    # Accuracy list
    accuracy = [
        0.611465, 0.539809, 0.628981, 0.777070, 0.805732, 0.804140,
        0.788217, 0.777866, 0.748408, 0.885350, 0.863057, 0.875796,
        0.862261, 0.863854, 0.912420, 0.948248, 0.933917, 0.938694,
        0.943471, 0.955414, 0.965764, 0.952229, 0.950637, 0.970541,
        0.956210
    ]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    epochs = range(0, 25)
    # Plot valid loss on the first y-axis
    ax1.plot(epochs, valid_loss, label='Validation Loss Over Epochs', color='red', marker='o')
    ax1.set_xlabel('Epoch', size=17, fontweight='bold')
    ax1.set_ylabel('Validation Loss', size=17, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=15)

    # Create a second y-axis
    ax2 = ax1.twinx()

    # Plot accuracy on the second y-axis
    ax2.plot(epochs, accuracy, label='Validation Accuracy', marker='x')
    ax2.set_ylabel('Accuracy', size=17, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=15)

    # Title and layout adjustments
    plt.title('Valid Loss and Accuracy Over Epochs', size=20, pad=10, fontweight='bold')

    # Add legends for each axis
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Show grid
    ax1.grid(True)

    # Display plot
    plt.show()


def plot_valid_conf_matrix():
    #im too lazy for ts
    conf_matrix = np.array([[300, 3],
                            [52, 901]])
    #True Negative, False Positive, False Negative, True Positive
    labels = ['Negative (Pred)', 'Positive (Pred)']
    categories = ['Negative (Actual)', 'Positive (Actual)']

    # Create the heatmap
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", cbar=True,
                xticklabels=labels, yticklabels=categories, annot_kws={"size": 12} )

    # Add title and labels
    plt.title("Validation Confusion Matrix Heatmap", size=14, pad=15, fontweight="bold")
    plt.xlabel("Predicted Label", size= 13, fontweight='bold')
    plt.ylabel("Actual Label", size= 13, fontweight='bold')

    # Show the heatmap
    plt.show()
polt_valid_loss_and_accuracy()