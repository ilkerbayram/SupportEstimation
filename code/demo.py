#!/usr/bin/env python
import numpy as np
from numpy.ma.core import anom
from support import anomaly_detector, data_flower
import matplotlib.pyplot as plt


def main():
    dataf = data_flower(num_pts=500)

    data = np.random.uniform(low=0.0, high=1.0, size=(2, 500))
    det = anomaly_detector(data, sigma=1, gamma=0.1)
    det.iterate(num_iter=500)
    print(f"convergence : {det.convergence}")
    print(f"applied : {det(np.ones(2)*0.6)}")
    print(f"applied : {det(np.ones(2)*1.1)}")

    print("Done!")
    return None


if __name__ == "__main__":
    main()
