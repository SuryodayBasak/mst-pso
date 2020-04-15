import numpy as np
from numpy import linalg as LA

class KNNRegressor:
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.k = k

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            nbrs_dist = []

            for i in range(len(self.X)):
                nbrs_dist.append(LA.norm(x - self.X[i])) #Euclidean dist

            sorted_dist_idx = np.argsort(nbrs_dist)
            k_idx = sorted_dist_idx[:self.k]

            y = 0.0
            for j in k_idx:
                y += self.y[j]

            y = y/(self.k)
            y_pred.append(y)
        return y_pred

    def find_all_neighbors(self, X_test):
        neighbors = []
        for x in X_test:
            nbrs_dist = []

            for i in range(len(self.X)):
                nbrs_dist.append(LA.norm(x - self.X[i])) #Euclidean dist

            sorted_dist_idx = np.argsort(nbrs_dist)
            k_idx = sorted_dist_idx[:self.k]

            neighbors.append(sorted(k_idx))
        return neighbors

    def find_neighborhood_std(self, neighbors):
        variances = []
        for neighbor in neighbors:
            y = [self.y[i] for i in neighbor]
            var = np.var(y, ddof = 1)
            variances.append(var)
        return(np.sqrt(sum(variances)/len(variances)))


class KNNClassifier:
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.k = k

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            nbrs_dist = []

            for i in range(len(self.X)):
                nbrs_dist.append(LA.norm(x - self.X[i])) #Euclidean dist

            sorted_dist_idx = np.argsort(nbrs_dist)
            k_idx = sorted_dist_idx[:self.k]
            y_k = []
            for i in range(0, self.k):
                y_k.append(self.y[k_idx[i]])
            # print(y_k)
            (values, counts) = np.unique(y_k,return_counts=True)
            # print('Values = ', values)
            # print('Counts = ', counts)
            y_idx = np.argmax(counts)
            y = values[y_idx]
            y_pred.append(y)
            # print(y)
            # print()

        return y_pred

    def find_all_neighbors(self, X_test):
        neighbors = []
        for x in X_test:
            nbrs_dist = []

            for i in range(len(self.X)):
                nbrs_dist.append(LA.norm(x - self.X[i])) #Euclidean dist

            sorted_dist_idx = np.argsort(nbrs_dist)
            k_idx = sorted_dist_idx[:self.k]

            neighbors.append(sorted(k_idx))
        return neighbors

    def find_neighborhood_entropy(self, neighbors):
        entropies = []
        for neighbor in neighbors:
            y = [self.y[i] for i in neighbor]

            (values,counts) = np.unique(y,return_counts=True)

            ent = 0
            for i in range(len(counts)):
                pi = counts[i]/self.k
                ent += (-pi * np.log(pi))

            entropies.append(ent)
        return(sum(entropies)/len(entropies))
