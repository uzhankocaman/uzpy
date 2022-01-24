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


if __name__ == '__main__':
    # load datasets
    data = [[], [], []]
    data[0] = np.loadtxt('data1')
    data[1] = np.loadtxt('data2')
    data[2] = np.loadtxt('data3')

    weights = [0.341398243018411, 0.367330235091507, 0.291271521890082]
    
    means = [
            [3.006132088737974,  3.093100568285389],
            [0.196675859954268, -0.034521603109466],
            [-2.957520528756456,  2.991192198151507]
            ]
    
    covariances = np.zeros((2, 2, 3))
    
    covariances[:, :, 0] = [
        [0.949104844872119, -0.170637132238246],
        [-0.170637132238246,  2.011158266600814]
    ]
    
    covariances[:, :, 1] = [
        [0.837094104536474, 0.044657749659523],
        [0.044657749659523, 1.327399518241827]
    ]
    
    covariances[:, :, 2] = [
        [1.160661833073708, 0.058151801834449],
        [0.058151801834449, 0.927437098385088]
    ]

    loglikelihoods = [-1.098653352229586e+03, -1.706951862352565e+03, -1.292882804841197e+03]
    for idx in range(3):
        ll = getLogLikelihood(means, weights, covariances, data[idx])
        diff = loglikelihoods[idx] - ll
        print('LogLikelihood is {0}, should be {1}, difference: {2}\n'.format(ll, loglikelihoods[idx], diff))
    # test EStep
    print('\n')
    print('(b) testing EStep function')
    # load gamma values
    testgamma = [[], [], []]
    testgamma[0] = np.loadtxt('gamma1')
    testgamma[1] = np.loadtxt('gamma2')
    testgamma[2] = np.loadtxt('gamma3')
    for idx in range(3):
        _, gamma = EStep(means, covariances, weights, data[idx])
        absdiff = testgamma[idx] - gamma
        print('Sum of difference of gammas: {0}\n'.format(np.sum(absdiff)))