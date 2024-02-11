# Python program implementing k-means clustering algorithm on the Iris dataset
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class KMeansClustering:
    iris_data = pd.read_csv(r"/Users/neverland/Documents/AIProject/src/irisdata.csv")
    data = iris_data.iloc[:, :-1].values  # Extracting features

    def __init__(self, k, maxIterations=100):
        self.k = k
        self.maxIterations = maxIterations
        self.centroids = None
        self.labels = None
        self.clusteres = None

    def get_centroids(self):
        return self.centroids

    def get_k(self):
        return self.k

    def get_maxIterations(self):
        return self.maxIterations

    # Function to calculate distance between two points
    def distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    # Function to initialize centroids
    def initialize_centroids(self, data):
        self.centroids = data[np.random.choice(data.shape[0], self.k, replace=False)]
        return self.centroids

    # Function to assign each data point to the nearest centroid
    def assign_to_centroids(self, data, centroids):
        self.clusters = []
        for point in data:
            distances = [self.distance(point, centroid) for centroid in centroids]
            cluster = np.argmin(distances)
            self.clusters.append(cluster)
        return np.array(self.clusters)

    # Function to update centroids
    def update_centroids(self, data):
        new_centroids = []
        for i in range(self.k):
            cluster_points = data[self.clusters == i]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                new_centroids.append(centroid)
            else:
                new_centroids.append(self.centroids[i])
        self.centroids = np.array(new_centroids)
        return self.centroids

    # Function to calculate objective function D
    def objective_function(self):
        total = 0  # Holds the sum of the objective function

        # Iterate over all the points
        for point in self.data:
            # Add the distance to the closest centroid
            try:
                total += min(
                    [math.dist(centroid, point) ** 2 for centroid in self.centroids]
                )
            except:
                raise Exception(
                    "\n".join(
                        [
                            f"{len(cluster.getCentroid())}, {len(point)}"
                            for cluster in self.clusters
                        ]
                    )
                )

        return total

    # Function to perform k-means clustering
    def k_means(self, data, max_iters=None):
        if max_iters is None:
            max_iters = self.maxIterations
        else:
            max_iters = int(max_iters)
        centroids = self.initialize_centroids(data)
        for i in range(max_iters):
            self.clusters = self.assign_to_centroids(data, centroids)
            new_centroids = self.update_centroids(data)
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids
        return self.centroids, self.clusters

    def predict(self, data):
        predictions = []

        for point in data:
            distances = np.array(
                [np.linalg.norm(point - centroid) for centroid in self.centroids]
            )
            cluster_index = np.argmin(distances)
            predictions.append(cluster_index)

        return np.array(predictions)

    def plot_learning_curves(self, k_values):
        for k in k_values:
            D_values = []
            for i in range(1, self.maxIterations + 1):
                self.centroids, self.clusters = self.k_means(self.data, max_iters=i)
                D_values.append(self.objective_function())
            plt.plot(range(1, len(D_values) + 1), D_values, label=f"K={k}")
        plt.xlabel("Iterations")
        plt.ylabel("Objective Function (D)")
        plt.title("Learning Curves for K-Means Clustering")
        plt.legend()
        plt.show()

    def plot_cluster_centers(self, k_values):
        for k in k_values:
            self.centroids, self.clusters = self.k_means(self.data[:, 2:], k)
            plt.scatter(
                self.data[:, 2],
                self.data[:, 3],
                c=self.clusters,
                cmap="viridis",
                label="Data",
            )
            plt.scatter(
                self.centroids[:, 0],
                self.centroids[:, 1],
                c="red",
                marker="o",
                s=150,
                label="Centroids",
            )
            plt.xlabel("Petal Length")
            plt.ylabel("Petal Width")
            plt.title(f"Cluster Centers for K={k}")
            plt.legend()
            plt.show()
