import sys

import random
import time
import os

import pandas as pd
from tqdm import tqdm
from ikm.clust.cluster import Cluster
from ikm.utils.data_loader import DataLoader
from ikm.utils.metrics import Metrics



class IKM:

    def ikm_process(self,box_cox=None, z_normalization=None,z_score=None, excl_wm=None,path=None,
                    report_name='report', num_clusters=2, error='eucl', specific_windmills=None, windmills=None,
                    how_to_process_data = "all", kind_mean = "all",
                    tries = 100, steps = 75, samples_per_file=10, path_save_file_per_event="./"):
        """
            This function runs IKM algorithm.
            Parameters:
            report_name (string): The name of the report with the clustering results. Default is 'report'
            box_cox (boolean): If True, Box-Cox transformation is applied
            dwt_complex (boolean): If True, the clustering is done on the statistics extracted from DWT
            z_normalization (boolean): If True, z-normalization is applied
            z_score (boolean): If True, the clustering is done on the extracted z-score measure from the data
            z_transform_mode (string): Specifies what type of Z transform to return. Can be 'magnitude' and 'phase'
            excl_wm (boolean): If True only specified parts of the electrodes are selected. left_right and front_back parameters are used to exclude the electrodes related to these parts,
            band (string): The name of the band to extract from the data. Can be 'delta', 'theta', 'beta_1', 'beta_2', 'gamma_1', 'gamma_2'
            hilbert (string): Specifies what type of Hilbert transform to return. Either 'phase' for the phase and 'ampl' for amplitude
            path (string): Path to the first file
            error (string): The type of the error used. Can be 'eucl' for the Euclidian distance calculation, 'total' for the total (absolute) norm calculation, 'max' for max norm distance, 'jaccard' for Jaccard distance and 'hamming' for Hamming distance
            specific_windmills (boolean): If True, the electrodes specidied in elelctrodes will be left in the data
            windmills (array of strings): Electrodes to be left in the data
            """

        start_time = time.time()
        random.seed(1)

        k = num_clusters

        i = 0

        print()
        print("----- [ Start preprocessing ] -----")
        print()

        data_loader = DataLoader(delimiter='\t')


        df, objects = data_loader.load_data_one_file(box_cox, 
                                                        z_normalization, 
                                                        z_score, 
                                                        excl_wm, 
                                                        path,
                                                        specific_windmills, 
                                                        windmills,
                                                        how_to_process_data,
                                                        kind_mean,
                                                        samples_per_file,
                                                        path_save_file_per_event)
        
        # print(df)

        print()
        print("----- [ End preprocessing ] -----")

        min_error = sys.float_info.max
        best_cl = "..--~~***~~--.."
        best_clusters = None

        final_str = ""

        print()
        print("----- [ Start cluster ] -----")

        for tries in tqdm(range(tries)): # 100 

            clusters = [[] for i in range(k)]
            counter = 0
            random.shuffle(objects)

            # We put 1 time series in each cluster
            for time_series_object in objects:
                current_object = time_series_object
                clusters[counter % k].append(current_object)
                counter += 1

            for cluster in clusters:
                cluster.sort(key=lambda a: a.name)

            # Create class Cluster
            cl = Cluster(clusters)

            # Steps for cluster # 75
            for i in tqdm(range(steps),desc="step", leave=False): #75

                # We do 75 steps of the cluster.
                cl.step(error)

                # Print updates
                # print("Step#:" + str(i + 1))
                # print(cl)

                # If the cluster dont change, we stop
                if not (cl.is_changed()):
                    break

            cl_error = cl.error(error)
            if min_error > cl_error:
                min_error = cl_error
                best_cl = cl.__str__(error)
                best_clusters = cl.init_best_clusters()

        

        df = Cluster.label_clusters(df, k, best_clusters)

        print()
        print("----- [ End cluster ] -----")

        print()
        print("----- [ Start metrics ] -----")

        metrics = Metrics()

        label = 'Response'
        clusters_purity = metrics.purity(df, label=label)
        clusters_purity_2 = metrics.purity_2(df, label=label)
        rand_index = metrics.rand_index(df, label=label)
        information_criterion = metrics.information_criterion(df, label=label)

        print(
            f"Clusters purity:\n{clusters_purity}\n"
            f"Clusters purity 2:\n{clusters_purity_2}\n"
            f"Rand index:{rand_index}\n"
            f"Information criterion: {information_criterion}"
        )

        final_str += f"\n" \
                     f"Box-cox:{box_cox}\n" \
                     f"Z-normalization:{z_normalization}\n" \
                     f"Z-score:{z_score}\n" \
                     f"Windmill are excluded:{excl_wm}\n" \
                     f"Best clustering:\n" \
                     f"Clusters purity:\n{clusters_purity}\n" \
                     f"Clusters purity 2:\n{clusters_purity_2}\n" \
                     f"Rand index:{rand_index}\n" \
                     f"Information criterion: {information_criterion}\n" \
                     f"How to process data: {how_to_process_data}\n" \
                     f"Kind of mean: {kind_mean}\n" \
                     f"Tries: {tries}\n" \
                     f"Steps: {steps}\n" \

        finish_time = time.time() - start_time
        final_str += f"Time: {finish_time}\n"

        df = pd.read_csv("report.csv")
        df.loc[len(df)] = [path, box_cox, z_normalization, z_score, excl_wm,
                           clusters_purity, rand_index, information_criterion,
                           error, how_to_process_data, kind_mean, tries, steps]
        
        df.to_csv("report.csv", index=False)

        file_path = fr'{report_name}.txt'

        # Safe file information
        file = open(os.path.join(file_path), 'a')
        file.write(final_str)
        file.close()

        # Safe file split files

        file_path = fr'cluster_split_{error}.txt'

        if os.path.exists(file_path):
            os.remove(file_path)

        file = open(os.path.join(file_path), 'a')
        file.write(best_cl)
        file.close()

        print()
        print("----- [ End metrics ] -----")
        print()
