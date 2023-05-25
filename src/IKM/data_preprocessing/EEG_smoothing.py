import numpy as np
import os
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
import pandas as pd
from matplotlib import pyplot as plt

data = []
objects = []
alpha = 0.05
path = "/media/data/lazarenkom98dm/objects-second-visit-full"
path_to_save = "/media/data/lazarenkom98dm/objects-first-visit-smoothed"


def smoothing(path, smoothing_type):
    """
    Applies simple, double and triple exponentional smoothing to the data.
    """
    for filename in os.listdir(path):
        data = np.loadtxt(os.path.join(path, filename), delimiter='\t')

        columns = data.shape[1]
        for i in range(columns):
            if smoothing_type == "simple":
                data_smoothed = SimpleExpSmoothing(data[:, i]).fit(smoothing_level=alpha, optimized=False,
                                                                   use_brute=True).fittedvalues
            elif smoothing_type == "double":
                data_smoothed = ExponentialSmoothing(data[:, i], trend="add").fit(smoothing_level=alpha).fittedvalues
            elif smoothing_type == "triple":
                data_smoothed = ExponentialSmoothing(data[:, i], trend="add", seasonal="add", seasonal_periods=12).fit(
                    smoothing_level=alpha).fittedvalues
            data[:, i] = data_smoothed
        np.savetxt(
            f"{path_to_save}-{smoothing_type}/{filename}",
            data)
        print(f"File {filename} smoothed successfully")


if __name__ == '__main__':
    # smoothing(path, "simple")
    # smoothing(path, "double")
    smoothing(path, "triple")
