#!/usr/bin/env python
"""
this script demonstrates the use of the anomaly detector
formulated in the paper 'Estimating the support of a
High-Dimensional Distribution', by Scholkopf et al.
The algorithm is different than the one described in the paper,
and is an implementation of the Douglas-Rachford algorithm,
adapted to the formulation in the paper.

Ilker Bayram, ibayram@ieee.org, 2021.
"""
import argparse
import numpy as np
from support import anomaly_detector, data_flower
import matplotlib.pyplot as plt
from utils import wrap_savefig

savefig = wrap_savefig(plt.savefig)


def get_args():
    parser = argparse.ArgumentParser(
        description="generates 2D normal data in the shape of a flower,"
        " runs an anomaly detector, and saves the figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s", "--samples", help="number of sample points", type=int, default=2500
    )
    parser.add_argument(
        "-v",
        "--variance_kernel",
        help="variance (i.e., sigma squared) used in the definition of the kernel",
        type=float,
        default=5e-2,
    )
    parser.add_argument(
        "-n",
        "--nu_parameter",
        help="nu parameter -- see the definition in the paper by Scholkopf et al.",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "-i", "--num_iterations", help="number of iterations", type=int, default=100
    )
    parser.add_argument(
        "-f", "--figname", help="suffix for the figure filenames", type=str, default=""
    )

    return parser.parse_args()


def plot_figures(detector, decision_function, fig_suffix):
    """
    plot some figures
    """
    # input data
    fig, ax = plt.subplots()
    ax.scatter(detector.data[0, :], detector.data[1, :])
    ax.set_aspect(1)
    ax.set_title(f"Number of data points : {detector.data.shape[1]}")
    savefig("flower" + fig_suffix + ".png")

    # "support" points
    ind = detector.soln > 0
    fig, ax = plt.subplots()
    ax.scatter(
        detector.data[0, ind], detector.data[1, ind], c=detector.soln[ind], cmap="RdGy"
    )
    ax.set_aspect(1)
    ax.set_title(f"Number of support points : {np.count_nonzero(ind)}")
    savefig(fname="nonzero" + fig_suffix + ".png")

    # evaluate the function on a 2D grid
    array = np.linspace(start=-1, stop=1, num=100)
    func = [[detector(np.array([x, y])) for x in array] for y in array[::-1]]

    # function whose zero crossing determines the decision boundary
    extent = -1, 1, -1, 1
    fig, ax = plt.subplots()
    ax.imshow(decision_function, extent=extent, origin="lower", cmap="RdGy")
    ax.contour(decision_function, 0, extent=extent, origin="lower")
    ax.set_aspect(1)
    ax.set_title("Decision function and boundary")
    savefig("function" + fig_suffix + ".png")

    return None


def main():
    """
    """
    args = get_args()
    data = data_flower(num_samples=args.samples)

    print("Initializing the detector\n")
    detector = anomaly_detector(
        data, nu=args.nu_parameter, sigma=args.variance_kernel, gamma=0.1
    )

    print(f"Training the detector for {args.num_iterations} iterations\n")
    detector.iterate(num_iter=args.num_iterations)

    print(
        f"Convergence Check : \nThe following number should be numerically zero:\n{detector.convergence}\n"
    )

    # here's how you can construct the decision function
    # by evaluating the anomaly detector on a 2D grid
    array = np.linspace(start=-1, stop=1, num=100)
    func = [[detector(np.array([x, y])) for x in array] for y in array[::-1]]

    print(f"Saving figures\n")
    plot_figures(detector, func, args.figname)

    print("Done!")
    return None


if __name__ == "__main__":
    main()
