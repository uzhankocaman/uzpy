from knn_classifier import knn_classifier
from knn_density_estimator import knn_density_estimator
import numpy as np
import matplotlib.pyplot as plt 


# create data
N = 1000
first_data = np.random.normal(1, 0.4, N)
second_data = np.random.normal(-1, 0.4, N)

# density estimation with KNN
k = 10
first_likelihood = knn_density_estimator(first_data, k, True, 'first', 'blue')
second_likelihood = knn_density_estimator(second_data, k, True, 'second', 'green')

# classification
dataset = np.linspace(-4, 4, 100)
predicted_class = []

for data in dataset:
    predicted_class.append(knn_classifier([first_likelihood, second_likelihood], [1, -1], [N, N], data))
    
#predicted_class = np.stack([dataset, predicted_class], axis=1)

plt.scatter(dataset, predicted_class)

