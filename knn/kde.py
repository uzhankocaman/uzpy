import numpy as np
import plot

def sum_of_density(po, samples, h, norm):
    return np.sum([norm*np.exp(-(po-sample)**2/(2*h**2)) for sample in samples])


def kde(samples=np.random.normal(0, 1, 100), h=0.3, make_plot=False):
    # density estimation from given samples with Kernel Density Estimator
    # Input
    # samples    : DxN matrix of data points
    # h          : (half) window size/radius of kernel
    # plot       : True, if plot of density function is desired
    # Output
    # estDensity : estimated density in the range of the sampls
    
    pos = np.arange(min(samples)-2, max(samples)+2, 0.1) 
    N = len(samples)
    estDensity = []
    
    try:
        D = len(samples[0])
    except TypeError:
        D = 1
        
    norm = (1/N)*(1/(((2*np.pi)**(D/2))*h))
    
    for po in pos:    
        estDensity.append(sum_of_density(po, samples, h, norm))
    estDensity = np.stack((pos, estDensity), axis=1)
    
    if make_plot:
        plot(estDensity[:, 0], estDensity[:, 1], 'Estimated Distribution')

    return estDensity
