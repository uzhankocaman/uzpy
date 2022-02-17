import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os

data = imread('image.jpg')
X = np.mean(data, -1)

U, S, VT = np.linalg.svd(X, full_matrices=False)
Ss = S
S = np.diag(S)

x = np.arange(len(Ss))
f1 = np.log(np.log(Ss))/max(np.log(np.log(Ss)))
f2 = np.log(Ss)/max(np.log(Ss))
g = np.cumsum(Ss)/np.sum(Ss)
plt.plot(x, f1, label='eigenvalues double log-normalized')
plt.plot(x, f2, label='eigenvalues log-normalized')
plt.plot(x, g, label='cumulative sum of eigenvalues normalized')
plt.grid()
idx1 = np.argwhere(np.diff(np.sign(f1 - g))).flatten()
idx2 = np.argwhere(np.diff(np.sign(f2 - g))).flatten()
plt.plot(x[idx1], f1[idx1], 'ro')
plt.plot(x[idx2], f2[idx2], 'ro')
plt.legend()
plt.show()

#N = 20
#values = [int((n/N)*U.shape[0]) for n in range(1, N+1)]
values = [int(idx1), int(idx2)]
keys = np.arange(len(values))
zip_iterator = zip(keys, values)
dimension = dict(zip_iterator)

for i in dimension.keys():
    r = dimension[i]
    X_approx = U[:, :r] @ S[0:r, :r] @ VT[:r, :]
    plt.figure(i)
    img = plt.imshow(X_approx)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title(f'r = {dimension[i]}')
    plt.show()

