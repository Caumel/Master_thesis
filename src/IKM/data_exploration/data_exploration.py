import os
import sys
sys.path.append('../../../')

from tqdm import tqdm

import numpy as np
import pandas as pd
import polars as pl

from src.IKM.clustering.ikm.clust.IC_model_builder import ICmodelBuilder

from src.IKM.clustering.ikm.model_builder.model_bilder import ModelBilder
from src.IKM.clustering.ikm.model_builder.time_series_object import TSObject


def medicaments_per_cluster():
    """
    Counts the medicaments used to treat patients for a specific cluster.
    """

    df = pd.read_csv('../datasets/depression_patients_clusters.csv')

    uniqueValues = (df['treatment_1'].append(df['treatment_2']).append(df['treatment_3'])
                    .append(df['treatment_4'])).unique()
    print(uniqueValues)

    data = {"cluster_1_tot": [], "cluster_2_tot": []}
    index = []

    for value in uniqueValues:

        index.append(value)
        for j in range(1, 3):
            calculation = 0
            for i in range(1, 5):
                calculation += df[(df[f'treatment_{i}'] == value) & (df['cluster'] == j)].count()[f'treatment_{i}']
            data[f"cluster_{j}_tot"].append(calculation)

            print(f"{value} for cluster {j}:{calculation}")

    df = pd.DataFrame(data, index=index)
    df.to_csv("medicaments_per_cluster.csv", index=True)


def coeffs_each_obj(d, m, data, predicted_attr_num):
    """
    Computes the coefficients for each data object.
    """
    relative_input = []
    interaction_cluster_error = sys.float_info.max

    parameters_final = None

    YtYs = np.matmul(data[:, predicted_attr_num].transpose(), data[:, predicted_attr_num])

    while 1:
        interaction_cluster_error_old = interaction_cluster_error
        new_relative_input = -1
        add_remove = True

        for i in range(d):
            if i == predicted_attr_num:
                continue

            tmp_input = relative_input.copy()

            if i in relative_input:
                if len(relative_input) < 3:
                    continue
                tmp_input.remove(i)
            else:
                tmp_input.append(i)

            parameters = ICmodelBuilder.least_squares(tmp_input, predicted_attr_num, data, mode='ordinary')
            bic = ICmodelBuilder.bic(tmp_input, predicted_attr_num, parameters, m, data, YtYs)

            if bic < interaction_cluster_error:
                interaction_cluster_error = bic
                new_relative_input = i
                parameters_final = parameters
                add_remove = i not in relative_input

        if interaction_cluster_error == interaction_cluster_error_old:
            break
        if add_remove:
            relative_input.append(new_relative_input)
        else:
            relative_input.remove(new_relative_input)

    parameters_final = parameters_final[..., np.newaxis]
    parameters_final_zeros = np.zeros((d - 1, 1))

    b1i = 0

    for i in range(d):
        if i == predicted_attr_num:
            continue
        ii = i if i < predicted_attr_num else i - 1

        if i in relative_input:
            parameters_final_zeros[ii][0] = parameters_final[b1i][0]
            b1i += 1
        else:
            parameters_final_zeros[ii][0] = 0.0

    model = parameters_final_zeros

    return model


def ideal_coeffs():
    """
    Computes coefficients for ideally separated clusters.
    """

    k = 2
    objs = [[] for i in range(k)]

    path = "/media/data/lazarenkom98dm/objects-combined/"

    error = 'eucl'

    for filename in os.listdir(path):
        #     ### Windows version
        #     if filename[-5] == "0":
        #         objs[0].append(path + "\\" + filename)
        #     elif filename[-5] == "1":
        #         objs[1].append(path + "\\" + filename)

        ### Ubuntu version
        if filename[-5] == "0":
            objs[0].append(path + filename)
        elif filename[-5] == "1":
            objs[1].append(path + filename)

    clusters = [[] for i in range(len(objs))]

    for i, obj in enumerate(objs):
        for path in obj:
            data = np.loadtxt(os.path.join(path), delimiter='\t')
            clusters[i].append(TSObject(path, data))

    cluster_models = []

    for i in range(k):
        if len(clusters) > 0:
            created_model = ModelBilder.create_model(clusters[i], error)
            cluster_models.append(created_model)
        else:
            cluster_models.append(None)

    result_string = ""
    for i in range(k):
        result_string += f"Cl#:{i}\n"

        for j in range(len(clusters[i])):
            result_string += str(clusters[i][j].name) + "\n"

        models_list = []
        for m in cluster_models[i]:
            models_list.append(m.model.transpose())

        concat_models = np.vstack(models_list)
        filename = f"coeffs_{i + 1}"

        np.savetxt(
            fr"/media/data/lazarenkom98dm/coeffs-combined-ideal/{filename}.txt",
            concat_models, fmt='%.3f')

        for m in range(len(cluster_models[i])):
            result_string += (cluster_models[i][m].__str__(m) + "\n")

    print(result_string)

def ideal_coeffs_2():
    """
    Computes coefficients for ideally separated clusters.
    """

    k = 3
    objs = [[] for i in range(k)]

    path = "../../../data/final_files/small/"

    error = 'eucl'

    clusters = [[] for i in range(len(objs))]

    for file in os.listdir(path):

        data = pl.read_csv(os.path.join(path, file))

        categories = np.array(['_'.join(str(item) for item in sublist) for sublist in data[:,1:4].rows()])

        # Get unique categories
        unique_categories = np.unique(categories)
        np.random.shuffle(unique_categories)

        # Split the array into a list of lists based on categories
        for index,category in enumerate(tqdm(unique_categories, desc='Split events, and create TSObject', leave=False)):
            # Select the one event in a big dataframe
            indices = np.where(categories == category)
            subset = data[indices[0]]

            if file == "high_small.csv" and len(clusters[0]) <40:
                clusters[0].append(TSObject(file_name=category, data=subset,z_normalization=True, z_score=True))
            if file == "moderate_small.csv" and len(clusters[1]) <40:
                clusters[1].append(TSObject(file_name=category, data=subset,z_normalization=True, z_score=True))
            if file == "low_small.csv" and len(clusters[2]) <40:
                clusters[2].append(TSObject(file_name=category, data=subset,z_normalization=True, z_score=True))

    # Experimento, crear un ejemplo con 120 elementos

    import random
    for i,obj in enumerate(clusters):
        selected_elements = random.sample(clusters[i], 40)
        clusters[i] = selected_elements.copy()

    cluster_models = []

    print(len(clusters))

    for i in range(k):
        if len(clusters) > 0:
            created_model = ModelBilder.create_model(clusters[i], error)
            cluster_models.append(created_model)
        else:
            cluster_models.append(None)

    result_string = ""
    for i in range(k):
        result_string += f"Cl#:{i}\n"

        for j in range(len(clusters[i])):
            result_string += str(clusters[i][j].name) + "\n"

        models_list = []
        for m in cluster_models[i]:
            models_list.append(m.model.transpose())

        concat_models = np.vstack(models_list)
        filename = f"coeffs_{i + 1}_{error}_ideal"

        np.savetxt(
            fr"./{filename}.txt",
            concat_models, fmt='%.3f')

        # for m in range(len(cluster_models[i])):
        #     result_string += (cluster_models[i][m].__str__(m) + "\n")

    result_string += ("Coefficients" + "\n") #Para poder usar el cluster

    print(result_string)

    file_path = fr'cluster_split_{error}_ideal.txt'

    # Safe file information
    file = open(os.path.join(file_path), 'a')
    file.write(result_string)
    file.close()

ideal_coeffs_2()
