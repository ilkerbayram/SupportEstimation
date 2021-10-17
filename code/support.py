"""
implementation of an algorithm for the anomaly detection
formulation in 'Estimating the Support of a High-Dimensional Distribution'
by Scholkopf et al.

ilker bayram, ibayram@ieee.org, 2021
"""

import numpy as np
from functools import reduce


def project(inp, tau: float):
    """
    implements the projection algorithm in the document
    titled 'Projections Onto the Intersection of the 
    Unit Simplex and an l_inf ball', by Ilker Bayram.
    """
    # form the y vector
    y_vec = np.sort(np.concatenate((inp, inp - tau)))

    ## search for lambda
    # declare function f(z,t) to compute
    # sum_i P_[0,tau] (z_i-t)
    proj = lambda z: np.maximum(0, np.minimum(z, tau))
    sum_proj = lambda z: np.sum(proj(z))
    # compute the g vector
    g = [sum_proj(inp - t) for t in y]
    # find the transition vector
    ind = np.count_nonzero(g < 1)
    u = (y_vec[ind - 1] + y_vec[ind]) / 2
    I_nz = np.count_nonzero((0 < inp) & (inp < u))
    # value of lambda
    lam = (sum_proj(inp - u) - 1) / I_nz
    return proj(inp - lam)


class anomaly_detector:
    def __init__(self, data, nu: float = 1) -> None:
        """
        data : 2D array, where each column
        corresponds to a different data point

        nu : see the definition in the paper by Scholkopf et al.
        """
        assert data.ndim == 2, f"Data should have 2 axes, but has {data.ndim}"
        self.data = data
        self.no_pts = data.shape[1]
        self.tau = nu * self.no_pts
        self.alphas = np.zeros(self.no_pts)

    def iterate(self, no_iterations: int = 1):
        """
        applies the Douglas-Rachford algorithm for 
        'no_iterations' iterations to learn alphas

        when iterations are done, updates the coefficients
        """

        return self

