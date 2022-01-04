import numpy as np
import plot 

def knn(samples=np.random.normal(0, 1, 100), k=30, make_plot=False):
    # compute density estimation from samples with KNN
    # Input
    # samples    : DxN matrix of data points
    # k          : number of neighbors
    # Output
    # estDensity : estimated density of the samples

    pos = np.arange(min(samples)-2, max(samples)+2, 0.1) 
    N = len(samples)
    #D = 1
    d = []
    for po in pos:
            d.append(np.sort([np.abs(sample-po) for sample in samples]))
    d = np.array(d)
    V = 2*d[:, k-1]
    estDensity = (k / (N*V))
    estDensity = np.stack((pos, estDensity), axis=1)
    
    if make_plot:
        plot.plot(estDensity[:, 0], estDensity[:, 1], 'Estimated Distribution')
    return estDensity