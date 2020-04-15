import numpy as np
from knn import KNNClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ga import GeneticAlgorithm
from sklearn.datasets import load_iris, load_digits, load_wine

data = np.genfromtxt('data.csv', delimiter=',')

#Test without scaling
print("Testing without scaling")

X = data[:, :-1]
y = data[:, -1]

# Reload data -- iris
iris = load_wine()
X = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
x_train, x_verif, y_train, y_verif = train_test_split(x_train, y_train, test_size=0.33, random_state=42)

clf = KNNClassifier(x_train, y_train, 5)
y_pred = clf.predict(x_test)
print("Accuracy = ", accuracy_score(y_test, y_pred))

#Run GA to find best weights
N_init_pop = 50
N_crossover = 100
N_selection = 50
improv_thresh = 1e-3

weight_ga = GeneticAlgorithm(13, N_init_pop, mu = 0.1)
weight_pop = weight_ga.get_population()
metric_array = np.empty(N_init_pop)
for i in range(len(weight_pop)):
    #Scale input data
    scaled_x_train = np.multiply(x_train, weight_pop[i])
    #Scale verificaion data
    scaled_x_verif = np.multiply(x_verif, weight_pop[i])

    #Method 1
    reg = KNNClassifier(scaled_x_train, y_train, 5)
    neighbors = reg.find_all_neighbors(scaled_x_verif)
    nbh_std = reg.find_neighborhood_entropy(neighbors)
    metric_array[i] = nbh_std

#Update fitness in GA object
weight_ga.set_fitness(metric_array)
weight_ga.selection(N_selection)
new_best_metric = 2.5

#while (best_metric - new_best_metric) > improv_thresh:
count = 0
while (count < 20):
    count += 1
    best_metric = new_best_metric
    #crossover
    weight_ga.crossover(N_crossover)
    
    #get_population
    weight_pop = weight_ga.get_population()
    metric_array = np.empty(N_crossover)
    
    #evaluate and set fitness
    for i in range(len(weight_pop)):
        #Scale input data
        scaled_x_train = np.multiply(x_train, weight_pop[i])
        #Scale verificaion data
        scaled_x_verif = np.multiply(x_verif, weight_pop[i])
        
        #Method 1
        reg = KNNClassifier(scaled_x_train, y_train, 5)
        neighbors = reg.find_all_neighbors(scaled_x_verif)
        nbh_std = reg.find_neighborhood_entropy(neighbors)
        metric_array[i] = nbh_std

    #selection
    #Update fitness in GA object
    weight_ga.set_fitness(metric_array)
    #get_best_sol
    best_weights, new_best_metric = weight_ga.best_sol()
    weight_ga.selection(N_selection)

    print("Best weights = ", best_weights, "\tBest metric = ", new_best_metric)

#Test with scaling after GA
print("Testing with scaling using weights from GA")
# best_weights = np.array([0.73246665,0.7259511,0.04515603])
print(np.multiply(x_train, best_weights))
reg = KNNClassifier(np.multiply(x_train, best_weights), y_train, 5)
y_pred = reg.predict(np.multiply(x_test, best_weights))

print(y_test)
print(y_pred)

print("Accuracy = ", accuracy_score(y_test, y_pred))

"""
#Test with scaling after exhaustive search
print("Testing with scaling using weights from exhaustive search")
best_weights = np.array([0.1, 0.8, 0.0])
reg = KNNRegressor(np.multiply(x_train, best_weights), y_train, 5)
y_pred = reg.predict(np.multiply(x_test, best_weights))
print("MSE = ", mean_squared_error(y_test, y_pred))

"""
#print("Best weights = ", lin_combos[best_idx], "\tBest metric:", metric_array[best_idx])
