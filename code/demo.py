#!/usr/bin/env python
import numpy as np
from numpy.ma.core import anom
from support import anomaly_detector, data_flower
import matplotlib.pyplot as plt


def main():
    dataf = data_flower(num_pts=2500)
    fig, ax = plt.subplots()
    ax.scatter(dataf[0, :], dataf[1, :])
    ax.set_aspect(1)
    plt.savefig("../figures/flower.png")
    det = anomaly_detector(dataf, sigma=5e-2, gamma=0.1)
    det.iterate(num_iter=500)
    print(f"convergence : {det.convergence}")

    array = np.linspace(start=-1, stop=1, num=100)
    func = [[det(np.array([x, y])) for y in array] for x in array]
    fig, ax = plt.subplots()
    ax.scatter(dataf[0, :], dataf[1, :], c=det.soln)
    ax.set_aspect(1)
    plt.savefig("../figures/soln.png")

    ind = det.soln > 0
    fig, ax = plt.subplots()
    ax.scatter(dataf[0, ind], dataf[1, ind], c=det.soln[ind])
    ax.set_aspect(1)
    plt.savefig("../figures/solnonzero.png")

    fig, ax = plt.subplots()
    ax.imshow(func)
    ax.contour(func, 0)
    plt.savefig("../figures/func.png")
    # print(f"applied : {det(np.ones(2)*0.6)}")
    # print(f"applied : {det(np.ones(2)*1.1)}")

    print("Done!")
    return None


if __name__ == "__main__":
    main()
