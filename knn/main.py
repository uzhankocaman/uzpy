from knn_classifier import knn_classifier
from knn_density_estimator import knn_density_estimator
import numpy as np
import matplotlib.pyplot as plt 

if __name__ == '__main__':
    # create data
    N = 1000
    first_dimension = np.random.normal(1, 0.4, N)
    second_dimension = np.random.normal(-1, 0.4, N)
    data_multidimension = np.stack([first_dimension, second_dimension], axis=1)
    
    # density estimation with KNN
    k = 10
    first_likelihood = knn_density_estimator(data_multidimension, k, True)#, 'first', 'blue')
    #second_likelihood = knn_density_estimator(second_data, k, True)#, 'second', 'green')
    
    # classification
    dataset = np.linspace(-4, 4, 100)
    predicted_class = []
    
    for data in dataset:
        predicted_class.append(knn_classifier([first_likelihood, second_likelihood], ['blue', 'green'], [N, N], data))
        
    #predicted_class = np.stack([dataset, predicted_class], axis=1)
    
    for i in range(len(dataset)):
        plt.scatter(dataset[i], dataset[i], c=predicted_class[i])

