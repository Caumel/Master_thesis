import sys

import random
import time

import pandas as pd
from ikm.clust.cluster import Cluster
from ikm.utils.data_loader import DataLoader
from ikm.utils.metrics import Metrics


class IKM:

    def ikm_process(self,box_cox=None, dwt_complex=None, z_normalization=None,
                    z_score=None, z_transform_mode=None, excl_wm=None,
                    band=None, hilbert=None, path=None, path1=None,
                    loading_setup=None,
                    report_name='report', num_clusters=2, error='eucl', specific_windmills=None, windmills=None):
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
            path1 (string): Path to the second file
            loading_setup (string): The configuration for loading the files. Can be 'one_f' -- load one file, 'two_f' -- load two files, 'alco' -- load EEG of alcoholic patients
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

        data_loader = DataLoader(delimiter='\t')

        if loading_setup == "two_f":
            df, objects = data_loader.load_data_two_files(box_cox, 
                                                          dwt_complex,
                                                          z_normalization, 
                                                          z_score, 
                                                          z_transform_mode,
                                                          excl_wm, 
                                                          band, 
                                                          hilbert, 
                                                          path, 
                                                          path1,
                                                          specific_windmills, 
                                                          windmills)
        elif loading_setup == "one_f":
            df, objects = data_loader.load_data_one_file(box_cox, 
                                                         dwt_complex,
                                                         z_normalization, 
                                                         z_score, 
                                                         z_transform_mode,
                                                         excl_wm, 
                                                         band, 
                                                         hilbert, 
                                                         path,
                                                         specific_windmills, 
                                                         windmills)
        elif loading_setup == "alco":
            df, objects = data_loader.load_alco(path)

        print()
        print("----- [ End preprocessing ] -----")

        min_error = sys.float_info.max
        best_cl = "..--~~***~~--.."
        best_clusters = None

        final_str = ""

        print()
        print("----- [ Start cluster ] -----")

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

            print("Ole llegue aqui")

            cl = Cluster(clusters)

            break

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

        return

        df = Cluster.label_clusters(df, k, best_clusters)

        metrics = Metrics()

        label = 'Response'
        clusters_purity = metrics.purity(df, label=label)
        rand_index = metrics.rand_index(df, label=label)
        information_criterion = metrics.information_criterion(df, label=label)

        print(
            f"Best clustering:\n{best_cl}\n"
            f"Clusters purity:\n{clusters_purity}\n"
            f"Rand index:{rand_index}\n"
            f"Information criterion: {information_criterion}"
        )

        final_str += f"Loading setup:{loading_setup} \n" \
                     f"Box-cox:{box_cox}\n" \
                     f"DWT Complex:{dwt_complex}\n" \
                     f"Z-normalization:{z_normalization}\n" \
                     f"Z-score:{z_score}\n" \
                     f"Electrodes are excluded:{excl_el}\n" \
                     f"Z transform mode:{z_transform_mode}\n" \
                     f"Hilbert mode: {hilbert}\n" \
                     f"Band: {band}\n" \
                     f"Best clustering:\n{best_cl}\n" \
                     f"Clusters purity:\n{clusters_purity}\n" \
                     f"Rand index:{rand_index}\n" \
                     f"Information criterion: {information_criterion}\n"

        finish_time = time.time() - start_time
        final_str += f"Time: {finish_time}\n"

        df = pd.read_csv("datasets/report.csv")
        df.loc[len(df)] = [loading_setup, path, box_cox, dwt, dwt_complex, z_normalization, z_score, excl_el,
                           z_transform_mode, hilbert, band, clusters_purity, rand_index, information_criterion,
                           error]
        df.to_csv("datasets/report.csv", index=False)

        file_path = fr'{report_name}.txt'
        file = open(file_path, 'a')
        file.write(final_str)
        file.close()
