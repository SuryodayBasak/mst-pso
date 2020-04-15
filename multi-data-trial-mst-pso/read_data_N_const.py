import numpy as np
from knn import KNNRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

mean_1 = (5, 2)
cov_1 = [[1, 0.8], [0.8, 1]]

mean_2 = (0, -1)
cov_2 = [[1, 0.1], [0.1, 1]]

N = 100

metric_array = []
mse_array = []

for noise in range(0, 1001, 10):
    iter_noise = noise/100
    #for case in range(1, 11):
    train_filename = 'samples-'+str(N)+'/'+str(N)+'_'+str(iter_noise).replace('.','-')+'_1-train'#+str(case)+'-train'
    test_filename = 'samples-'+str(N)+'/'+str(N)+'_'+str(iter_noise).replace('.','-')+'_1-test'#+str(case)+'-test'

    train_data = np.genfromtxt(train_filename, delimiter=",")
    test_data = np.genfromtxt(test_filename, delimiter=",")

    x_train = train_data[:,:-1]
    y_train = train_data[:,-1]
    x_test = test_data[:,:-1]
    y_test = test_data[:,-1]
 
    #Generate verification data
    X_1 = np.random.multivariate_normal(mean_1, cov_1, 5000)
    X_2 = np.random.multivariate_normal(mean_2, cov_2, 5000)
    x_verif = np.concatenate((X_1, X_2), axis=0)

    #Method 1
    reg = KNNRegressor(x_train, y_train, 5)
    neighbors = reg.find_all_neighbors(x_verif)
    nbh_std = reg.find_neighborhood_std(neighbors)

    y_pred = reg.predict(x_test)
    print(train_filename)
    print(test_filename)
    print(nbh_std)
    print(mean_squared_error(y_test, y_pred)**0.5)

    metric_array.append(nbh_std)
    mse_array.append(mean_squared_error(y_test, y_pred)**0.5)
    del train_data
    del test_data

plot_x = [i for i in range(0, len(metric_array))]

print(plot_x)
print(metric_array)
print(mse_array)

plt.plot(plot_x, metric_array)
plt.plot(plot_x, mse_array)
plt.show()
