"""
implementation of an algorithm for the anomaly detection
formulation in 'Estimating the Support of a High-Dimensional Distribution'
by Scholkopf et al.

ilker bayram, ibayram@ieee.org, 2021
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve


def project(inp, tau: float):
    """
    implements the projection algorithm in the document
    titled 'Projections Onto the Intersection of the 
    Unit Simplex and an l_inf ball', by Ilker Bayram.
    """
    # form the y vector
    y_vec = np.sort(np.concatenate((inp, inp - tau)))[::-1]

    ## search for lambda
    proj = lambda z: np.maximum(0, np.minimum(z, tau))
    sum_proj = lambda z: np.sum(proj(z))
    # compute the g vector
    g = np.array([sum_proj(inp - t) for t in y_vec])
    # find the transition vector
    ind = np.count_nonzero(g < 1)
    u = (y_vec[ind - 1] + y_vec[ind]) / 2
    ind_nz = (u < inp) & (inp < u + tau)
    I_tau = np.count_nonzero(u + tau < inp)
    I_nz = np.count_nonzero(ind_nz)
    # value of lambda
    lam = (np.sum(inp[ind_nz]) + I_tau * tau - 1) / I_nz
    return proj(inp - lam)


class anomaly_detector:
    def __init__(
        self,
        data=None,
        nu: float = 1e-2,
        sigma: float = 1e-2,
        gamma: float = 1e-1,
        beta: float = 5e-1,
    ) -> None:
        """
        data : 2D array, where each column
        corresponds to a different data point

        nu : see the definition in the paper by Scholkopf et al.
        """
        assert data.ndim == 2, f"Data should have 2 axes, but has {data.ndim}"
        self.data = data
        self.no_pts = data.shape[1]
        self.tau = 1 / (nu * self.no_pts)
        self.alphas = np.random.normal(0, 1e-5, self.no_pts)
        self.sigma = sigma
        self.update_soln()
        self.gamma = gamma
        self.beta = beta
        self.K = self.compute_K_matrix()
        self.K_fac = lu_factor(1 / self.gamma * np.eye(self.no_pts) + self.K)

    def compute_K_matrix(self):
        inner = self.data.transpose().dot(self.data)
        energy = np.sum(self.data ** 2, axis=0)
        K = np.exp(
            -(energy.reshape(-1, 1) + energy.reshape(1, -1) - 2 * inner) / self.sigma
        )
        return K

    def reflected_resolvent_constraint(self, x):
        return 2 * project(x, self.tau) - x

    def reflected_resolvent_K(self, x):
        return 2 * lu_solve(self.K_fac, x / self.gamma) - x

    def iterate(self, num_iter: int = 1):
        """
        applies the Douglas-Rachford algorithm for 
        'num_iter' iterations to learn alphas

        when iterations are done, updates the coefficients
        """
        for _ in range(num_iter):
            # reflected resolvent from the constraints
            x = self.reflected_resolvent_constraint(self.alphas)
            # reflected resolvent from the data term
            x = self.reflected_resolvent_K(x)
            # averaging step
            self.alphas = self.beta * self.alphas + (1 - self.beta) * x
        return self.update_soln()

    def apply_zero_rho(self, x):
        """
        the anomaly function with rho=0
        """
        return self.soln[self.ind] @ np.exp(
            -np.sum((self.data[:, self.ind] - x.reshape(-1, 1)) ** 2, axis=0)
            / self.sigma
        )

    def update_soln(self):
        self.soln = project(self.alphas, self.tau)
        self.ind = self.soln > 0
        ind = (self.soln > 0) & (self.soln < self.tau)
        rhos = [self.apply_zero_rho(x) for x in self.data[:, ind].transpose()]
        self.rho = np.mean(rhos)
        self.convergence = np.var(rhos)
        return self

    def __call__(self, x):
        return self.apply_zero_rho(x) - self.rho


def data_flower(num_pts: int, num_leaves: int = 3):
    samples = np.random.uniform(low=-1, high=1, size=(num_pts, 2))
    radii = np.sqrt(np.sum(samples ** 2, axis=1))
    angs = np.arccos(samples[:, 0] / (radii + 1e-10))
    pts = [
        samp
        for samp, ang, radius in zip(samples, angs, radii)
        if np.cos(num_leaves * ang) > radius
    ]
    return np.array(pts).transpose()
