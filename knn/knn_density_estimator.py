import numpy as np
import plot 
#from scipy.interpolate import interp2d


def knn_density_estimator(samples, k, make_plot):
    # compute density estimation from samples with KNN
    # Input
    # samples    : NxD matrix of data points
    # k          : number of neighbors
    # Output
    # f          : estimated density function of the samples
    
    density_position = np.arange(2*samples.min(), 2*samples.max(), 0.1)
    try:
        #multidimensional data
        pos = np.stack([density_position for _ in range(0, samples.shape[1])], axis=0).T
    except IndexError:
        #1-dimensional data
        pos = density_position
        
    N = len(samples)
    #D = 1
    d = []
    for po in pos:
            d.append(np.sort([np.linalg.norm(sample-po) for sample in samples]))
           
    d = np.array(d)
    print(d.shape)
    V = 2*d[:, k-1]
    estDensity = (k / (N*V))
    #f = interp2d(pos, estDensity)
    estDensity = np.stack((pos, estDensity), axis=1)
    
    if make_plot:
        plot.plot(estDensity[:, 0], estDensity[:, 1])#, f'{label}', color)
    return estDensity


mean = [1, 1]
cov = [[1, 0], [0, 1]]
N = 5000
samples = np.random.multivariate_normal(mean, cov, N)
samples = np.random.normal(0, 1, N)
k = 100
estDensity = knn_density_estimator(samples, k, True)
























