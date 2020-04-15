import numpy as np
import random as rand
import matplotlib.pyplot as plt

mean_1 = (5, 2)
cov_1 = [[1, 0.8], [0.8, 1]]

mean_2 = (1, 0)
cov_2 = [[1, 0.1], [0.1, 1]]

N = 400
n_div = 200

# Data
X1 = np.random.multivariate_normal(mean_1, cov_1, n_div)
X2 = np.random.multivariate_normal(mean_2, cov_2, N-n_div)

# Noise
X_noise_1 = np.random.uniform(0, 1, N)
X_noise_2 = np.random.uniform(0, 1, N)

# Labels
X1_l = np.zeros(N-n_div)
X2_l = np.ones(n_div)

labels = np.concatenate((X1_l, X2_l), axis = 0)
X = np.concatenate((X1, X2), axis = 0)
print(np.shape(X))
print(np.shape(labels))
print(np.shape(X_noise_1))

data = np.random.rand(N, 5)
data[:, 0:2] = X
# data[:, 2] = X_noise_1
#data[:, 3] = X_noise_2
data[:, 2:4] = np.flip(X, axis=0)
data[:, 4] = labels

#data = np.concatenate((data, [X_noise], [labels]), axis = 1)
print(data)

# plt.scatter(X1[:,0], X1[:,1])
# plt.scatter(X2[:,0], X2[:,1])
# plt.show()

np.savetxt('data.csv', data, delimiter = ',')
