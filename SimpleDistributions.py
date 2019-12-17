import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import math


class SimpleDistributions:

    def _convert_to_rank(self, vector):
        from scipy.stats import rankdata
        return rankdata(vector, method="dense")

    def plot_pixel_intensities(self, im1, im2, use_log: bool = False, use_ranked: bool = False):
        plt.ylabel = "im1"
        plt.xlabel = "im2"
        X_to_plot = None
        Y_to_plot = None
        if use_ranked:
            X_to_plot = self._convert_to_rank(im1[:, :, 1].flatten())
            Y_to_plot = self._convert_to_rank(im2[:, :, 2].flatten())
        else:
            X_to_plot = im1[:, :, 1].flatten()
            Y_to_plot = im2[:, :, 2].flatten()
        # with logs
        if use_log:
            if use_ranked:
                plt.plot(np.log(X_to_plot), np.log(Y_to_plot), linestyle=":",color="blue")
            else:
                plt.plot(np.log(X_to_plot), np.log(Y_to_plot),linestyle=":",color="red")
        # without logs
        else:
            if use_ranked:
                plt.plot(X_to_plot, Y_to_plot, linestyle=":", color="blue")
            else:
                plt.plot(X_to_plot, Y_to_plot, linestyle=":", color="red")

        plt.show()

