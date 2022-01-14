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
        #plt.scatter(bins, epdf, 'b')
        plt.plot(bins, epdf, 'b', label='estimated')
        plt.grid()
        plt.xlabel("x")
        plt.ylabel("p(x)")
    
    return epdf, bins
      
def lognormal_density_estimation(samples: np.ndarray, nbins: int):
    mean = np.mean(samples)
    variance = np.var(samples)
    x = np.linspace(min(samples), max(samples), 100)
    pdf = (np.exp(-(np.log(x) - mean)**2 / (2 * variance))/ (x * np.sqrt(2 * np.pi) * variance))
    plt.plot(x, pdf, 'r', label='empirical')
    epdf, bins = empirical_density_estimation(samples, nbins)
    plt.legend(['empirical', 'estimated'])
    return epdf, bins

if __name__ == '__main__':
    mu = 3
    sigma = 1
    size = 100
    y = np.random.lognormal(mu, sigma, size)
    
    nbins=20
    
    epdf_lognormal, x_lognormal = lognormal_density_estimation(y, nbins)
    