import sys

import numpy as np


class DimSelector:
    def __init__(self, objects):
        """
        Initializes the variables for using the IKM with dimensions selection enhancement.
        """

        self.data = []

        for obj in objects:
            if len(self.data) == 0:
                self.data = obj.data_mean
            else:
                self.data = np.add(self.data, obj.data_mean)

        d = objects[0].d

        self.dimensions = [i for i in range(d)]
        self.distances = []
        self.ind_min_distances = []
        self.ind_max_distances = []

        mean_across_all_tp = self.data.mean(axis=0)

        for value in self.data:
            self.distances.append(np.linalg.norm(mean_across_all_tp - value))

    def remove_min_dist_dimension(self):
        """
        Removes dimensions where the distance between the mean across all time points and the mean across one
        dimension is the smallest.
        """
        ind_min_dist = np.argmin(self.distances)
        self.ind_min_distances.append(ind_min_dist)
        self.distances[ind_min_dist] = sys.maxsize
        dimensions = np.delete(self.dimensions, self.ind_min_distances)
        return dimensions

    def remove_max_dist_dimension(self):
        """
        Removes dimensions where the distance between the mean across all time points and the mean across one
        dimension is the biggest.
        """
        ind_max_dist = np.argmax(self.distances)
        self.ind_max_distances.append(ind_max_dist)
        self.distances[ind_max_dist] = -sys.maxsize - 1
        dimensions = np.delete(self.dimensions, self.ind_max_distances)
        return dimensions
