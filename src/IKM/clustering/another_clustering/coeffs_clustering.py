import os

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans

from ikm.utils.data_preprocessor import DataPreprocessor
from ikm.utils.metrics import Metrics


def cluster_coeffs_chosen_el(electrodes=None, excl_el=None, l_r=None, f_b=None, set_expl=''):
    """
    This function runs clustering algorithms k-means, DBSCAN, agglomerative on different parts of electrodes coefficient.
    """
    path = "/media/data/lazarenkom98dm/coeffs-first-visit"
    path1 = "/media/data/lazarenkom98dm/coeffs-second-visit"

    if electrodes is None:
        electrodes = []

    data_preprocessor = DataPreprocessor()
    data_all = []
    data_concat = []
    df = pd.DataFrame(columns=['ID', 'Data', 'Response'])

    for filename_second_file in os.listdir(path1):
        splitted_filename = filename_second_file.split('_')
        splitted_filename[1] = str(int(splitted_filename[1]) - 1)
        filename_first_file = '_'.join(splitted_filename)

        data_first_visit = np.loadtxt(os.path.join(path, filename_first_file), delimiter=' ')
        data_second_visit = np.loadtxt(os.path.join(path1, filename_second_file), delimiter=' ')

        if electrodes:
            data_first_visit = data_preprocessor.choose_parts_electrodes(data_first_visit.transpose(), electrodes)
            data_second_visit = data_preprocessor.choose_parts_electrodes(data_second_visit.transpose(), electrodes)

        if excl_el:
            data_first_visit = data_preprocessor.leave_parts_electrodes(data_first_visit.transpose(),
                                                                        left_right=l_r,
                                                                        front_back=f_b)
            data_second_visit = data_preprocessor.leave_parts_electrodes(data_second_visit.transpose(),
                                                                         left_right=l_r,
                                                                         front_back=f_b)

        arr_data = [data_first_visit, data_second_visit]

        data = np.append(arr_data[0], arr_data[1], axis=0)

        data_reduced_flatten = [data.flatten()]

        df.loc[len(df.index)] = [filename_first_file, data_reduced_flatten, filename_second_file[-5]]

        data_all.append(data_reduced_flatten)
    for data in data_all:
        if len(data_concat) == 0:
            data_concat = data
        else:
            data_concat = np.concatenate((data_concat, data), axis=0)

    n_clusters = 2

    eps = 0.5
    min_samples = 2
    eps_step = 0.5
    min_samples_step = 1
    final_labels_dbscan = None

    metrics = Metrics()
    label = 'Response'
    best_cp = 0

    for i in range(100):
        for j in range(10):
            clustering_dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', metric_params=None,
                                       algorithm='auto', leaf_size=30, p=None, n_jobs=None).fit(data_concat)

            min_samples += min_samples_step
            labels_dbscan = clustering_dbscan.labels_

            for i, value in enumerate(labels_dbscan):
                df.loc[i, 'Cluster'] = value

            clusters_purity = metrics.purity(df, label=label)

            if (clusters_purity > best_cp) | (best_cp == 0):
                final_labels_dbscan = labels_dbscan
                best_cp = clusters_purity
                best_eps = eps
                best_min_samples = min_samples
        eps += eps_step
        min_samples = 2

    clustering_agglomerative = AgglomerativeClustering().fit(data_concat)
    labels_agglomerative = clustering_agglomerative.labels_

    clustering_k_means = KMeans(n_clusters=n_clusters, n_init=100, random_state=0).fit(data_concat)
    labels_k_means = clustering_k_means.labels_

    clustering_labels = [final_labels_dbscan, labels_agglomerative, labels_k_means]

    clustering_names = ['dbscan', 'agglomerative', 'k-means']

    electrodes_string = ','.join(electrodes)

    df_report = pd.DataFrame(columns=['Setting', 'Algorithm', 'Epsilon', 'Min samples', 'Purity', 'Rand index', 'IC'])
    file_path = fr'report.txt'
    file = open(file_path, 'a')
    final_str = f'Clustering on coefficients of {electrodes_string} from two visits:\n'
    final_str += f'{set_expl}\n'

    for j, labels in enumerate(clustering_labels):
        for i, value in enumerate(labels):
            df.loc[i, 'Cluster'] = value

        clusters_purity = metrics.purity(df, label=label)
        rand_index = metrics.rand_index(df, label=label)
        information_criterion = metrics.information_criterion(df, label=label)

        final_str += f"{clustering_names[j]}:\n"

        if clustering_names[j] == 'dbscan':
            final_str += f"Epsilon: {best_eps}\n" \
                         f"Min samples: {best_min_samples}\n" \
                         f"Labels: {labels}\n"
            df_report.loc[j] = [set_expl, clustering_names[j], best_eps, best_min_samples,
                                clusters_purity, rand_index, information_criterion]

        df_report.loc[j] = [set_expl, clustering_names[j], '', '', clusters_purity, rand_index,
                            information_criterion]
        final_str += f"Clusters purity:\n{clusters_purity}\n" \
                     f"Rand index:{rand_index}\n" \
                     f"Information criterion: {information_criterion}\n"

    file.write(final_str)
    file.close()
    df_report.to_csv('report.csv', index=False)
