import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
import numpy.matlib as nm
import math
import copy
from scipy import interpolate
import scipy.special as spp
from numpy import linalg as LA
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance


def multi_peak(x, weights=[0.3, 1.2], mu=[8, 14], sigma=[10., 10.]):
    x = x.reshape([1, -1])
    return weights[0] * np.exp(- (x[0] - mu[0]) ** 2 / sigma[0]) \
           + weights[1] * np.exp(- (x[0] - mu[1]) ** 2 / sigma[1]) ** 2 


def multi_peak_gradient(x, weights=[0.3, 1.2], mu=[8, 14], sigma=[10., 10.]):
    x = x.reshape([1, -1])
    return (-2 * (+ weights[0] * np.exp(- (x[0] - mu[0]) ** 2 / sigma[0]) * (x[0] - mu[0]) / sigma[0]
                 + weights[1] * np.exp(- (x[0] - mu[1]) ** 2 / sigma[1]) * (x[0] - mu[1]) / sigma[1]
                 )).reshape([-1, 1])


class QSVGD():
    def __init__(self):
        pass

    def svgd_kernel(self, theta, h=-1):
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist) ** 2
        if h < 0:  # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(theta.shape[0] + 1))
        # compute the rbf kernel
        Kxy = np.nan_to_num(np.exp(-pairwise_dists / h ** 2 / 2))
        dxkxy = -np.matmul(Kxy, theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:, i] + np.multiply(theta[:, i], sumkxy)
        dxkxy = dxkxy / (h ** 2)
        return (Kxy, dxkxy)

    def imq_kernel(self, theta, alpha=1, beta=-0.5):
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist) ** 2
        # compute the rbf kernel
        Kxy = (alpha ** 2 + pairwise_dists) ** beta
        dxkxy = np.matmul((alpha ** 2 + pairwise_dists) ** (beta-1), theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:, i] - np.multiply(theta[:, i], sumkxy)
        dxkxy = dxkxy * 2 * beta
        return (Kxy, dxkxy)

    def update(self, x0, n_iter=1000, stepsize=1e-3, alpha=0.9, debug=False, gx=None, pdf=None,
               Alpha=1.0, Tau=1.0, anneal=True, anneal_stepsize=0.99, upper=None, lower=None, rank_weight=False,
               rank_power=-1000, high_dim=False):
        upper = np.array(upper, dtype='float')
        lower = np.array(lower, dtype='float')
        scale = upper - lower
        theta = np.copy(x0)
        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        for iter in range(n_iter):
            if debug and (iter + 1) % 1000 == 0:
                print 'iter ' + str(iter + 1)
            # reset the out-of-range points
            _theta = np.where(theta > upper.reshape([-1, 1]), upper.reshape([-1, 1]) - scale.reshape([-1, 1]) / 10., theta)
            _theta = np.where(_theta < lower.reshape([-1, 1]), lower.reshape([-1, 1]) + scale.reshape([-1, 1]) / 10., _theta)
            if (theta == _theta).any():
                stepsize /= 2.
            theta = _theta

            ftheta = pdf(theta)
            if rank_weight:
                fsort = ftheta.reshape([-1]).tolist()
                fsort.sort()
                weights = (np.array([fsort.index(item) for item in ftheta.reshape([-1])]) + 1) ** float(rank_power)
                lnpgrad = weights.reshape([-1, 1]) * gx(theta)
                Z = np.mean(weights)
            else:
                # use another utility function
                div = nm.repmat(ftheta, theta.shape[0], 1).T / ftheta
                norms = np.nan_to_num(LA.norm(div, Alpha, axis=0))
                ftheta = np.nan_to_num(1 / norms) ** (Alpha - 1)

                lnpgrad = ftheta * gx(theta)
                Z = 1

            if iter == 0:
                # re-scale the learning rate
                stepsize /= np.max(np.abs(lnpgrad))
            if high_dim:
                kxy, dxkxy = self.imq_kernel(theta)
            else:
                kxy, dxkxy = self.svgd_kernel(theta, h=-1)

            grad_theta = (np.matmul(kxy, lnpgrad) / Z + Tau * dxkxy) / x0.shape[0]

            if iter == 0:
                historical_grad = historical_grad + grad_theta ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
            adj_grad = np.divide(grad_theta, fudge_factor + np.sqrt(historical_grad))
            if anneal:
                stepsize *= anneal_stepsize
            theta = theta + stepsize * adj_grad
        return theta

if __name__ == '__main__':
    x0 = np.random.uniform(8, 20, [10, 1])

    # QSVGD
    theta = QSVGD().update(x0, pdf=multi_peak, gx=multi_peak_gradient,
                           Tau=1e-2, upper=[20], lower=[8], n_iter=500, stepsize=2e-2,
                           rank_weight=True, rank_power=-1, anneal_stepsize=0.995)
    print(np.mean(multi_peak(theta)), np.min(multi_peak(theta)))

    # standard SVGD
    theta = QSVGD().update(x0, pdf=multi_peak, gx=multi_peak_gradient,
                           Tau=1, upper=[20], lower=[8], n_iter=500, stepsize=2e-2,
                           rank_weight=True, rank_power=0, anneal_stepsize=0.995)
    print(np.mean(multi_peak(theta)), np.min(multi_peak(theta)))







