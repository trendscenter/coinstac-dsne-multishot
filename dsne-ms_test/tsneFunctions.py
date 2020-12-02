"""
Created on Mon Jan 152017

@author: Deb
"""

import numpy as np
from numpy import dot
from itertools import chain
from scipy.linalg import eigh

def get_all_keys(current_dict):
    children = []
    for k in current_dict:
        yield k
        if isinstance(current_dict[k], dict):
            children.append(get_all_keys(current_dict[k]))
    for k in chain.from_iterable(children):
        yield k


def listRecursive(d, key):
    for k, v in d.items():
        if isinstance(v, dict):
            for found in listRecursive(v, key):
                yield found
        if k == key:
            yield v


def demeanS(Y, average_Y):
    ''' Subtract Y(low dimensional shared value )by the average_Y and
    return the updated Y) '''

    return Y - np.tile(average_Y, (Y.shape[0], 1))


def demeanL(Y, average_Y):
    ''' It will take Y and average_Y of only local site data and return the
    updated Y by subtracting IY'''
    return Y - np.tile(average_Y, (Y.shape[0], 1))


def Hbeta(D=np.array([]), beta=1.0):
    """Compute the perplexity and the P-row for a specific value of the
    precision of a Gaussian distribution.

    Args:
        D (float): matrix of euclidean distances between every pair of points
        beta (float): Precision of Gaussian distribution
                        Given as -1/(2*(sigma**2))

    Returns:
        H (float): Entropy
        P (float): Similarity matrix (matrix of conditional probabilities)

    """
    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=100.0):
    """Performs a binary search to get P-values in such a way that each
    conditional Gaussian has the same perplexity.

    Returns:
        P: P is computed based on euclidean distance, perplexity and variance
        from high dimensional space. Suppose, if point 5 is near of point 1
        in high dimensional space the P value(P51) would be high.
        if point 5 is far of point 1 in high dimensional space the
        P value(P51) would be low.

    """

    # Initialize some variables
    #    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # =============================================================================
        #         # Print progress
        #         if i % 500 == 0:
        #             print("Computing P-values for point ", i, " of ", n, "...")
        # =============================================================================

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries = tries + 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP

    # Return final P-matrix


#    print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
    return P

def pca(x2d, n_comp=50):
    """ data PCA inspited by https://github.com/alvarouc/ica/blob/03b0335d86126bb431353bafe288211888e7c5bd/ica/ica.py#L52 (mine and Rogers' work)
    *Input
    x2d : 2d data matrix of observations by variables
    n_comp: Number of components to retain
    *Output
    Y : PCA projected X (KL-transformed)
    """
    x2d_demean = x2d - x2d.mean(axis=1).reshape((-1, 1))
    NSUB, NVOX = x2d_demean.shape
    if NSUB > NVOX:
        cov = dot(x2d_demean.T, x2d_demean) / (NSUB - 1)
        w, v = eigh(cov, eigvals=(NVOX - n_comp, NVOX - 1))
        x_pca = dot(x2d_demean, v)
    else:
        cov = dot(x2d_demean, x2d_demean.T) / (NVOX - 1)
        w, u = eigh(cov, eigvals=(NSUB - n_comp, NSUB - 1))
        x_pca = dot(u.T, x2d_demean)
    return x_pca

def tsne(X=np.array([]),
         Y=np.array([]),
         Shared_length=0,
         no_dims=2,
         initial_dims=50,
         perplexity=100.0,
         computation_phase='remote'):
    """Runs t-SNE on the dataset in the NxD array X to reduce its
    dimensionality to no_dims dimensions. The syntaxis of the function is
    Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.

    Args:
       X: High dimensional data
       Y: low dimensiona shared data

    Note:
       When computation phase is remote, it is going to do operation only on
       remote data. No local site data presents there.
       When computation phase is local, it will compute gradient based on
       combined data(remote+local). But after computing gradient,
       it will update only local site data.

    Returns:
       Y: low dimensional computed value of X

    """

    '''def updateS(Y, G):
        return Y

    def updateL(Y, G):
        return Y + G

    def demeanS(Y):
        return Y

    def demeanL(Y):
        return Y - np.tile(np.mean(Y, 0), (Y.shape[0], 1))'''

    # Check inputs
    if round(no_dims) != no_dims:
        #        print("Error: number of dimensions should be an integer.")
        return -1



    if computation_phase=='local':
        Site_length,Site_length_Y  = X.shape;
        #difference = Site_length - Shared_length;
        #print(Shared_length, Site_length, difference)
        # Y = Math.random.randn(Site_length, no_dims);
        Y_1 = Y[0:Shared_length,:]
        np.random.seed()
        rx = (max(Y_1[:, 0]) - min(Y_1[:, 0]))
        Yx = np.random.rand(Site_length) * rx + min(Y_1[:, 0]);
        ry = (max(Y_1[:, 1]) - min(Y_1[:, 1]))
        Yy = np.random.rand(Site_length) * ry + min(Y_1[:, 1]);
        Y = np.c_[Yx, Yy]

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.9
    eta = 500
    min_gain = 0.01
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 8  # early exaggeration
    P = np.maximum(P, 1e-12)
    (n1, d1) = P.shape

# Made changes here. Instead of n1 I am now returning n
    if computation_phase is 'local':
        return (Y, dY, iY, gains, P, n)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y),0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)

        # I think this condition can be removed
        if computation_phase is 'remote':
            Y = Y + iY
            Y = Y - np.tile(np.mean(Y, 0), (n, 1))
        '''else:
            Y[:Shared_length, :] = updateS(Y[:Shared_length, :],
                                           iY[:Shared_length, :])
            Y[Shared_length:, :] = updateL(Y[Shared_length:, :],
                                           iY[Shared_length:, :])
            Y[:Shared_length, :] = demeanS(Y[:Shared_length, :])
            Y[Shared_length:, :] = demeanL(Y[Shared_length:, :])'''

        # Compute current value of cost function
#        if (iter + 1) % 10 == 0:
#            C = np.sum(P * np.log(P / Q))

#            print("Iteration ", (iter + 1), ": error is ", C)

# Stop lying about P-values
        if iter == 100:
            P = P / 4

    # Return solution
    return Y


def master_child(Y, dY, iY, gains, n, Shared_length, P, iter, C):
    # Compute pairwise affinities

    #    max_iter = 1000
    initial_momentum = 0.5
    #    middle_momentum = 0.7
    final_momentum = 0.9
    eta = 500
    min_gain = 0.01
    no_dims = 2

    sum_Y = np.sum(np.square(Y), 1)
    num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y))
    num[range(n), range(n)] = 0
    Q = num / np.sum(num)

    #I have to change in here
    Q = np.maximum(Q, 1e-12)
    #Q = np.maximum(Q, .0001)

    # Compute gradient
    PQ = P - Q
    for i in range(n):
        dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

    # Perform the update
    if iter < 20:
        momentum = initial_momentum
    else:
        momentum = final_momentum

    gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0))
    gains[gains < min_gain] = min_gain
    iY = momentum * iY - eta * (gains * dY)


    # Compute current value of cost function
    if (iter + 1) % 10 == 0:
        C = np.sum(P * np.log(P / Q));


    # Stop lying about P-values
    if iter == 100:
        P = P / 4

    return (Y, dY, iY, gains, n, Shared_length, P, C)


def normalize_columns(X=np.array([])):
    '''Take data X and after performing max min normalization
    it will return the normalized X

       Args:
           X: high dimensional raw data
       Returns:
           X: Normalized X
       '''
    rows, cols = X.shape
    for rows in range(rows):
        p = abs(X[rows, :]).max()
        if (p != 0):
            X[rows, :] = X[rows, :] / abs(X[rows, :]).max()
    return X
