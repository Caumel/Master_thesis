import os
import sys

import random
import time

import numpy as np
from ikm.clust.cluster import Cluster
from ikm.utils.data_loader import DataLoader
from ikm.utils.metrics import Metrics

from ikm.model_builder.time_series_object import TSObject
from ikm.utils.dim_selector import DimSelector


class Algorithm:
    def dim_selection_process(self, report_name=None, configuration=None, path=None, path1=None,
                              loading_setup=None, error='eucl'):

        """
        This function is the modification of the IKM algorithm which preselects dimensions based on the distance between the mean across all time points and the mean across one dimension.
        Parameters:
        report_name (string): The name of the report with the clustering results. Default is 'report'
        configuration (string): Can be 'max_dist' or 'min_dist'
        path (string): Path to the first file
        path1 (string): Path to the second file
        loading_setup (string): The configuration for loading the files. Can be 'one_f' -- load one file, 'two_f' -- load two files, 'alco' -- load EEG of alcoholic patients
        """

        start_time = time.time()
        random.seed(1)

        k = 2

        i = 0

        data_loader = DataLoader(delimiter='\t')

        if loading_setup == "two_f":
            df, objects = data_loader.load_data_two_files(path_first_file=path, path_second_file=path1)
        elif loading_setup == "one_f":
            df, objects = data_loader.load_data_one_file(path=path)
        elif loading_setup == "alco":
            df, objects = data_loader.load_alco(path)

        min_error = sys.float_info.max
        best_cl = "..--~~***~~--.."
        best_clusters = None
        best_dimensions = None
        best_purity = None

        # Initialization of variables for dimensions selector algorithm
        dim_selector = DimSelector(objects)
        clusters_purity_init = 0
        threshold = 0.8
        dimensions = dim_selector.dimensions
        init_dim_length = len(dimensions)
        ind_min_distances = []

        final_str = ""

        while len(dimensions) > (threshold * init_dim_length):

            for tries in range(100):

                clusters = [[] for i in range(k)]
                counter = 0
                random.shuffle(objects)
                for time_series_object in objects:
                    current_object = time_series_object
                    clusters[counter % k].append(current_object)
                    counter += 1

                for cluster in clusters:
                    cluster.sort(key=lambda a: a.name)

                cl = Cluster(clusters)

                for i in range(75):
                    cl.step(error)
                    if not (cl.is_changed()):
                        break
                    # print("Step#:" + str(i + 1))
                    # print(cl)
                cl_error = cl.error(error)
                if min_error > cl_error:
                    min_error = cl_error
                    best_cl = cl.__str__()
                    best_clusters = cl.init_best_clusters()

            df = Cluster.label_clusters(df, k, best_clusters)

            metrics = Metrics()

            label = 'Response'
            clusters_purity = metrics.purity(df, label=label)
            rand_index = metrics.rand_index(df, label=label)
            information_criterion = metrics.information_criterion(df, label=label)

            ### DIMENSIONS SELECTION ALGORITHM IMPLEMENTATION

            if clusters_purity > clusters_purity_init:
                clusters_purity_init = clusters_purity
            best_dimensions = dimensions
            best_purity = clusters_purity_init

            if configuration == 'min_dist':
                dimensions = dim_selector.remove_min_dist_dimension()
                ind_min_distances.append(dim_selector.ind_min_distances)

            elif configuration == 'max_dist':
                dimensions = dim_selector.remove_max_dist_dimension()
                ind_min_distances.append(dim_selector.ind_max_distances)

            objects = []
            for filename in os.listdir(path):
                data = np.loadtxt(os.path.join(path, filename), delimiter='\t')
                data = np.delete(data, ind_min_distances, axis=1)
                objects.append(TSObject(filename, data))

        print(
            f"Best clustering:\n{best_cl}\n"
            f"Clusters purity:\n{clusters_purity}\n"
            f"Rand index:{rand_index}\n"
            f"Information criterion: {information_criterion}"
        )
        # print(dimensions)

        final_str += f"Setting: delta waves\n" \
                     f"Best clustering:\n{best_cl}\n" \
                     f"Clusters purity:\n{clusters_purity}\n" \
                     f"Rand index:{rand_index}\n" \
                     f"Information criterion: {information_criterion}\n"

        finish_time = time.time() - start_time
        final_str += f"Time: {finish_time}\n" \
                     f"Best purity: {best_purity}\n" \
                     f"Best dimensions: {best_dimensions}"

        file_path = fr'{report_name}.txt'
        file = open(file_path, 'a')
        file.write(final_str)
        file.close()
