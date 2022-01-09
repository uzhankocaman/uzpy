import numpy as np
import plot 
from scipy.interpolate import interp1d

def knn_density_estimator(samples=np.random.normal(0, 1, 100), k=30, make_plot=False, label='Estimated Distribution', color='red'):
    # compute density estimation from samples with KNN
    # Input
    # samples    : DxN matrix of data points
    # k          : number of neighbors
    # Output
    # f : estimated density function of the sampls

    pos = np.arange(min(samples)-2, max(samples)+2, 0.1) 
    N = len(samples)
    #D = 1
    d = []
    
    for po in pos:
            d.append(np.sort([np.abs(sample-po) for sample in samples]))
            
    d = np.array(d)
    V = 2*d[:, k-1]
    estDensity = (k / (N*V))
    f = interp1d(pos, estDensity)
    #estDensity = np.stack((pos, estDensity), axis=1)
    
    if make_plot:
        plot.plot(estDensity[:, 0], estDensity[:, 1], f'{label}', color)
    return estDensity

estDensity = knn_density_estimator()
print(estDensity.shape)