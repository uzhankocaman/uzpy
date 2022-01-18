import numpy as np
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
