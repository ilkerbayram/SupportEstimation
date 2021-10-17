#!/usr/bin/env python
import numpy as np
from support import anomaly_detector


def main():
    data = np.random.normal(0, 1, (5, 6))
    det = anomaly_detector(data)
    print("Done!")
    return None


if __name__ == "__main__":
    main()
