import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from numpy import linalg as la

import seaborn as sb
from scipy.stats import pearsonr


class DistanceMeasurer():

    def distance_between_coeffs_combined_objs(self, file_path_1, file_path_2):
        """
        Computes distance between coefficients of the combined data from the first and the second visits.
        """
        final_str = ""

        coeff_1 = np.loadtxt(os.path.join(file_path_1), delimiter=' ')
        coeff_2 = np.loadtxt(os.path.join(file_path_2), delimiter=' ')

        final_str += f"{file_path_1}&{file_path_2}\n Coeffs 1: \n {coeff_1} \n Coeffs 2: \n {coeff_2}\n"

        file_path = r'frobenius_norm.txt'

        file = open(file_path, 'a')

        distance = self.frobenius_norm(coeff_1, coeff_2)

        final_str += f"{distance}"

        # file.write(final_str)
        # file.close()

        return distance

    def distance_between_coeffs_each_obj_average(self, dir_path_1, dir_path_2, norm):
        """
        Computes distance between coefficients of each object from different visits and averages it.
        """
        distances = []
        distance = None
        for file_1 in os.listdir(dir_path_1):
            coeff_1 = np.loadtxt(os.path.join(dir_path_1, file_1), delimiter=' ')
            for file_2 in os.listdir(dir_path_2):
                coeff_2 = np.loadtxt(os.path.join(dir_path_2, file_2), delimiter=' ')

                if norm == 'fro':
                    distance = self.frobenius_norm(coeff_1, coeff_2)
                elif norm == 'total':
                    distance = self.total_norm(coeff_1, coeff_2)
                distances.append(distance)
        return self.average_list(distances)

    def average_list(self, lst):
        """
        Computes average value from values in the provided list.
        """
        return sum(lst) / len(lst)

    def frobenius_norm(self, a, b):
        """
        Calculates the Frobenius norm for two matrices.
        """
        if np.shape(a)[0] > np.shape(b)[0]:
            b = np.lib.pad(b, ((0, np.shape(a)[0] - np.shape(b)[0]), (0, 0)), 'constant', constant_values=(0))
        elif np.shape(a)[0] < np.shape(b)[0]:
            a = np.lib.pad(a, ((0, np.shape(b)[0] - np.shape(a)[0]), (0, 0)), 'constant', constant_values=(0))
        diff = a - b
        fro_norm = la.norm(diff, 'fro')
        return fro_norm

    def total_norm(self, a, b):
        """
        Calculates the total norm for two matrices.
        """
        if np.shape(a)[0] > np.shape(b)[0]:
            b = np.lib.pad(b, ((0, np.shape(a)[0] - np.shape(b)[0]), (0, 0)), 'constant', constant_values=(0))
        elif np.shape(a)[0] < np.shape(b)[0]:
            a = np.lib.pad(a, ((0, np.shape(b)[0] - np.shape(a)[0]), (0, 0)), 'constant', constant_values=(0))
        diff = a - b
        tot_norm = np.sum(np.abs(diff))
        return tot_norm

    def create_matrix_coeffs(self, array_paths, names):

        """
        Creates the matrix with the distances between coefficients.
        """

        data = [[] for i in range(len(names))]

        for i, path_1 in enumerate(array_paths):
            for j, path_2 in enumerate(array_paths):
                print(f"{path_1} and {path_2}:{self.distance_between_coeffs(path_1, path_2)}")
                data[i].append(self.distance_between_coeffs(path_1, path_2))

        df = pd.DataFrame(data, names, names)
        return df

    def correlation_map(self, x, y):

        """
        Computing correlation matrix between coefficients using pearsonr function.
        """
        n_row_x = x.shape[0]
        n_row_y = y.shape[0]
        ccmtx_xy = np.empty((n_row_y, n_row_y))
        for n in range(n_row_x):
            for m in range(n_row_y):
                ccmtx_xy[n, m] = pearsonr(x[n, :], y[m, :])[0]

        print(ccmtx_xy.shape)
        return ccmtx_xy

    def corr2_coeff(self, A, B):

        """
        Computing correlation matrix between coefficients using scratch implementation.
        """
        A_mA = A - A.mean(1)[:, None]
        B_mB = B - B.mean(1)[:, None]

        ssA = (A_mA ** 2).sum(1)
        ssB = (B_mB ** 2).sum(1)

        return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))

    def correlation_between_coeffs(self, paths, delimeter, annot, mode='one',Xlabel = "Summer", Ylabel = "Winter",title = "Title",save=""):

        """
        Creating a graph with the correlations.
        Parameters:
            paths (list of strings): list of paths pointing to the files with the coefficients values.
            delimeter (string): delimeter used for the values from the files loading.
            annot (boolean): annotate the graph or not.
            mode (string): 'one' (default) -- generate one graph, 'multiple' -- generate multiple graphs.
        """
        columns = ['cc','o3','pv','cape','blh','d2m','z','relative_humidity','t2m','t100m','t135m','wdir100m','wspeed135m','wspeed100m']

        if mode == 'multiple':
            # Turn interactive plotting off
            # plt.ioff()
            for i, path_1 in enumerate(paths):
                for j, path_2 in enumerate(paths):
                    if i == j:
                        continue
                    coeff_1 = np.loadtxt(os.path.join(path_1), delimiter=delimeter)
                    coeff_2 = np.loadtxt(os.path.join(path_2), delimiter=delimeter)
                    # rho = self.correlation(coeff_1, coeff_2)
                    rho = self.correlation_map(coeff_1, coeff_2)
                    sb.set(rc={'figure.figsize': (640, 480)})
                    heatmap = sb.heatmap(rho, cmap="Blues", annot=annot)

                    heatmap.set(xlabel=f'{os.path.basename(path_1)}', ylabel=f'{os.path.basename(path_2)}')

                    fig = heatmap.get_figure()
                    fig.savefig(f'correlation-coeffs-{os.path.basename(path_1)}-{os.path.basename(path_2)}.png')
                    plt.close(fig)
        elif mode == 'one':
            path_1 = paths[0]
            path_2 = paths[1]
            coeff_1 = np.loadtxt(os.path.join(path_1), delimiter=delimeter)
            coeff_2 = np.loadtxt(os.path.join(path_2), delimiter=delimeter)
            rho = self.correlation_map(coeff_1, coeff_2)
            plt.figure(figsize=(20, 6))
            heatmap = sb.heatmap(rho, cmap="YlGnBu", annot=annot, xticklabels=columns, yticklabels=columns)
            # heatmap.fig.set_size_inches(15,15)

            heatmap.set(xlabel=f'{Xlabel}', ylabel=f'{Ylabel}',)
            plt.title(title)
            print(os.listdir())
            plt.savefig(save, dpi=300, format='png', bbox_inches='tight')
            plt.show()
            # fig = heatmap.get_figure()
            # fig.savefig(f'correlation-coeffs-{os.path.basename(path_1)}-{os.path.basename(path_2)}.png')