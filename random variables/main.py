import numpy as np
import matplotlib.pyplot as plt


def empirical_density_estimation(samples: np.ndarray, nbins: int, plot: bool = True):
    # compute density estimation from samples with an histogram approach
    # Input
    # samples    : data points
    # nbins      : number of bins
    # Output
    # epdf       : estimated density function of the samples
    # bins       : bin centers
    
    epdf, bins = np.histogram(samples, bins=nbins, density=True)
    bin_width = bins[1] - bins[0]
    bins = bins[:-1] + bin_width/2
    
    if plot:
        plt.scatter(bins, epdf)
        plt.plot(bins, epdf)
        plt.grid()
        plt.xlabel("x")
        plt.ylabel("p(x)")
        plt.show()
    
    return epdf, bins
      
def lognormal_density_estimation(samples: np.ndarray)

if __name__ == '__main__':
    mu = 3
    sigma = 1
    size = 1000
    y = np.random.lognormal(mu, sigma, size)
    
    epdf_lognormal, x_lognormal = empirical_density_estimation(y, nbins)
    