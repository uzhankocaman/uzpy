#reference for the formulas: C. M. Bishop \Pattern Recognition and Machine Learning"
import numpy as np
import scipy.stats


def getLogLikelihood(means, weight, covariances, X):
    #Input
    #means      :mean for each Gaussian []
    #weights    :weight vector for the gaussians []
    #covariance :covariance matrices for each gaussian []
    #X          :input data []
    #Output
    #LogLikelihood
    if len(X.shape) > 1:
        N, D = X.shape
    else:
        N = 1
        D = X.shape[0]
        
    K = len(weights)
    LogLikelihood = 0
    
    for i in range(N):
        p = 0 
        for j in range(K):
            if N == 1:
                meansDiff = X - means[j]
            else:
                meansDiff = X[i,:] - means[j]
            covariance = covariances[:, :, j]
            norm = 1. / float(((2 * np.pi) ** (float(D) / 2.)) * np.sqrt(np.linalg.det(covariance)))
            p += weights[j] * norm * np.exp(-0.5 * ((meansDiff.T).dot(np.linalg.lstsq(covariance.T, meansDiff.T)[0].T)))
        LogLikelihood += np.log(p)
    return LogLikelihood


def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    
    #scipy.stats.norm(means)
    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    n_training_samples, dim = X.shape
    K = len(weights)
    gamma = np.zeros((n_training_samples, K))
    for i in range(n_training_samples):
        for j in range(K):
            means_diff = X[i]-means[j]
            covariance = covariances[:,:,j].copy()
            norm = 1./float(((2*np.pi)**(float(dim)/2))*np.sqrt(np.linalg.det(covariances[:,:,j])))
            gamma[i, j] = weights[j] * norm * np.exp(-0.5 * (means_diff.T.dot(np.linalg.lstsq(covariance.T, means_diff.T)[0].T)))
        gamma[i] /= gamma[i].sum()
        
    return [logLikelihood, gamma]


def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).
    
    n_training_samples, dim = X.shape
    K = gamma.shape[1]

    means = np.zeros((K, dim))
    covariances = np.zeros((dim, dim, K))

    Nk = gamma.sum(axis=0)
    weights = Nk / n_training_samples

    means = np.divide(gamma.T.dot(X), Nk[:, np.newaxis])

    for i in range(K):
        auxSigma = np.zeros((dim, dim))
        for j in range(n_training_samples):
            meansDiff = X[j] - means[i]
            auxSigma = auxSigma + gamma[j, i] * np.outer(meansDiff.T, meansDiff)
        covariances[:, :, i] = auxSigma/Nk[i]

    logLikelihood = getLogLikelihood(means, weights, covariances, X)

    return weights, means, covariances, logLikelihood