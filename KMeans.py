import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, n_clusters, max_iter = 100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter  # consider not converge
        self.cluster_centers_ = None # centroid
        self.labels_ = None # cluster label
        self.inertia_ = None # result of cost function
    
    def fit(self, X):
        self.cluster_centers_ = X[np.random.choice(len(X), self.n_clusters, replace = False)]
        # initial centroid
        print(self.cluster_centers_)
        for _ in range(self.max_iter):
            distances = norm(X[:, np.newaxis] - self.cluster_centers_, axis = 2) # L2norm(data - centoid)
            # X : (n_samples, 2), cluster_centers : (n_clusters,2)
            # (n_samples, 1, 2) - (n_clusters, 2) => (n_samples, n_cluster, 2)
            # (1 -> n_cluster) by broad casting
            self.labels_ = np.argmin(distances, axis=1)
            # minimum of distance 
            
            cluster_centers = []
            for i in range(self.n_clusters):
                new_center = X[self.labels_ == i].mean(axis=0)
                cluster_centers.append(new_center)
            
            cluster_centers = np.array(cluster_centers)
            
            if np.all(self.cluster_centers_ == cluster_centers):
                break # optimal
            
            self.cluster_centers_ = cluster_centers
            
        self.inertia_ = np.sum(distances[np.arange(distances.shape[0]), self.labels_]**2)
    
    def predict(self, X):
        distances = norm((X[:, np.newaxis] - self.cluster_centers_), axis = 2)
        return np.argmin(distances, axis = 1) # nearest centroid

if __name__ == "__main__":
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)
    #(300,2)
    model = KMeans(n_clusters=3)
    model.fit(X)
    
    centroids = model.cluster_centers_

    plt.scatter(X[:, 0], X[:, 1], c = model.labels_)
    plt.scatter(centroids[:,0], centroids[:,1], c = "red")
    plt.show()