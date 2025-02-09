import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from sklearn.from typing import Countercluster import KMeans
from collections import Counter
import CNNConstants
#TODO Add functionality so that the graph save path is automatically named to the epoch number/train run #.
#TODO Reorganize this file so I can begin training asap and actually save results.

class MatPlot():

    def graph_train(self, graph):

        graph.set_style()
        graph.set_figure()

        fig, ax = plt.subplots()

        x = Constants.EPOCHS
        y1 = Constants.TRAIN_CORRECT_LIST
        y2 = Constants.TRAIN_INCORRECT_LIST

        plt.plot(x, y1, "r--", label = "Train Correct")
        plt.plot(x, y2, "bs", label = "Train Incorrect")

        plt.legend(loc="upper left")

        font = FontProperties("Times New Roman")

        plt.rcParams["font.family"] = "Times New Roman"


        ax.set_xlabel('Epochs', fontsize=12, fontproperties=font, fontweight='bold', color = '0')
        ax.set_ylabel('Amount', fontsize=12, fontproperties=font, fontweight='bold', color = '0')

        ax.xaxis.set_tick_params(labelsize=8)
        ax.yaxis.set_tick_params(labelsize=8 )
        ax.xaxis.set_tick_params(labelsize=8)
        ax.yaxis.set_tick_params(labelsize=8)

        ax.set_title("Train Correct and Train Incorrect over Epochs", fontproperties = font, fontweight='bold', color = '0', pad= 25, fontsize = 20)

        plt.savefig(Constants.TRAIN_PLT_SAVE_PATH)
        plt.show()


    def set_style(self):
        plt.style.use('ggplot')

    def set_figure(self):
        figure = plt.figure()
        figure.set_figheight(1)
        figure.set_figwidth(1)

    def graph_valid(self, graph):
        graph.set_style()
        graph.set_figure()

        fig, ax = plt.subplots()

        x = Constants.EPOCHS
        y1 = Constants.VALID_CORRECT_LIST
        y2 = Constants.VALID_INCORRECT_LIST

        plt.plot(x, y1, "r--", label="Valid Correct")
        plt.plot(x, y2, "bs", label="Valid Incorrect")

        plt.legend(loc="upper left")

        font = FontProperties("Times New Roman")

        plt.rcParams["font.family"] = "Times New Roman"

        ax.set_xlabel('Epochs', fontsize=12, fontproperties=font, fontweight='bold', color='0')
        ax.set_ylabel('Amount', fontsize=12, fontproperties=font, fontweight='bold', color='0')

        ax.xaxis.set_tick_params(labelsize=8)
        ax.yaxis.set_tick_params(labelsize=8)
        ax.xaxis.set_tick_params(labelsize=8)
        ax.yaxis.set_tick_params(labelsize=8)

        ax.set_title("Valid Correct and Valid Incorrect over Epochs", fontproperties=font, fontweight='bold', color='0',
                     pad=25, fontsize=20)

        plt.savefig(Constants.valid_plt_save_path)
        plt.show()
    def graph_scatter(self):
        x = Constants.K_X
        y = Constants.K_Y

        data = np.column_stack((x, y))

        kmeans = KMeans(n_clusters=2)
        kmeans.fit(data)

        cluster_labels = kmeans.labels_

        cluster_counts = Counter(cluster_labels)
        print(cluster_counts)
        centroids = kmeans.cluster_centers_
        plt.figure(figsize=(8, 6))

        # Plot data points, coloring them by their cluster
        for cluster_id in range(len(centroids)):
            cluster_points = data[cluster_labels == cluster_id]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_id}")
        print("Centroid Coordinates (Width, Height):")
        for i, centroid in enumerate(kmeans.cluster_centers_):
            print(f"Centroid {i + 1}: Width = {centroid[0]:.2f}, Height = {centroid[1]:.2f}")

        # Plot cluster centroids
        plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="X", s=200, label="Centroids")

        plt.title("K-Means Clustering")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend(loc="best")
        plt.grid()
        plt.savefig(Constants.K_MEANS_PLT_SAVE_PATH)
        plt.show()
#Safety checks 12/31/2024: All functions work as expected. Just follow comment instruction.

graph = MatPlot()
graph.graph_scatter()

#TODO: For example, for train loss, it could display a list or dictionary of  epochs to train loss.
#TODO: Example output: {e1: 5.2; e2:2.7; etc}