import numpy as np
from knn import KNNRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

mean_1 = (5, 2)
cov_1 = [[1, 0.8], [0.8, 1]]

mean_2 = (0, -1)
cov_2 = [[1, 0.1], [0.1, 1]]

#Generate verification data
X_1 = np.random.multivariate_normal(mean_1, cov_1, 1000)
X_2 = np.random.multivariate_normal(mean_2, cov_2, 1000)
x_verif = np.concatenate((X_1, X_2), axis=0)
#Add a column of random numbers to the data
rand_col_verif = 100*np.random.rand(len(x_verif))
x_verif = np.insert(x_verif, 2, rand_col_verif, axis=1)

N = 100

#Select noise
iter_noise = 2.1
#Select training set
train_filename = 'samples-'+str(N)+'/'+str(N)+'_'+str(iter_noise).replace('.','-')+'_1-train'#+str(case)+'-train'
#Select test set
test_filename = 'samples-'+str(N)+'/'+str(N)+'_'+str(iter_noise).replace('.','-')+'_1-test'#+str(case)+'-test'

metric_array = []
mse_array = []

train_data = np.genfromtxt(train_filename, delimiter=",")
test_data = np.genfromtxt(test_filename, delimiter=",")

x_train = train_data[:,:-1]
y_train = train_data[:,-1]
x_test = test_data[:,:-1]
y_test = test_data[:,-1]

#Add a column of random values - these should be bad for performance
rand_col_train = 100*np.random.rand(len(x_train))
x_train = np.insert(x_train, 2, rand_col_train, axis=1)
rand_col_test = 100*np.random.rand(len(x_test))
x_verif = np.insert(x_test, 2, rand_col_test, axis=1)

#3 nested loops - generate scaled weights
lin_combos = []
for i in range(0, 11):
    for j in range(0, 11):
        for k in range(0, 11):
            combo = [float(i/10), float(j/10), float(k/10)]
            lin_combos.append(combo)
lin_combos = np.array(lin_combos)


for combo in lin_combos:
    #Scale input data
    scaled_x_train = np.multiply(x_train, combo)
    #Scale verificaion data
    scaled_x_verif = np.multiply(x_verif, combo)

    #Method 1
    reg = KNNRegressor(scaled_x_train, y_train, 5)
    neighbors = reg.find_all_neighbors(scaled_x_verif)
    nbh_std = reg.find_neighborhood_std(neighbors)

    print("Combo = ", combo, "\tnbh_std = ", nbh_std)
    #y_pred = reg.predict(x_test)
    #print(train_filename)
    #print(test_filename)
    #print(nbh_std)
    #print(mean_squared_error(y_test, y_pred)**0.5)

    metric_array.append(nbh_std)
    #mse_array.append(mean_squared_error(y_test, y_pred)**0.5)

best_idx = np.argmin(metric_array)
print("Best weights = ", lin_combos[best_idx], "\tBest metric:", metric_array[best_idx])
