import sys

import numpy as np
from ikm.clust import IC_model_builder
from ikm.model_builder.model_bilder import ModelBilder


class Cluster:
    clusters = []
    clusters_old = []
    cluster_models = None
    k = 0

    def __init__(self, clusters):
        self.k = len(clusters)
        self.clusters = clusters
        self.best_clusters = None

    def step(self, error):
        """
        Executes one step of IKM.
        """
        self.clusters_old = self.clusters
        clusters_new = [[] for i in range(self.k)]
        self.cluster_models = []
        for i in range(self.k):
            if len(self.clusters) > 0:
                created_model = ModelBilder.create_model(self.clusters[i], error)
                self.cluster_models.append(created_model)
            else:
                self.cluster_models.append(None)

        for i in range(self.k):
            for j in range(len(self.clusters[i])):
                ts_object = self.clusters[i][j]
                current_element = 0
                init_error = sys.float_info.max
                for cluster_number in range(self.k):
                    tmp = sys.float_info.max
                    if not len(self.clusters[cluster_number]) == 0:
                        tmp = IC_model_builder.ICmodelBuilder.obj_error_regarding_cluster(
                            self.cluster_models[cluster_number],
                            ts_object, error)
                    if tmp < init_error:
                        init_error = tmp
                        current_element = cluster_number
                clusters_new[current_element].append(ts_object)
        self.clusters = clusters_new

    def error(self, error):
        """
        Calculates the error of clustering.
        """
        result = 0
        for i in range(self.k):
            for j in range(len(self.clusters[i])):
                ts_object = self.clusters[i][j]
                result += IC_model_builder.ICmodelBuilder.obj_error_regarding_cluster(self.cluster_models[i], ts_object,
                                                                                      error)
        return result

    def is_changed(self):
        """
        Checks if a cluster has been changed.
        """
        for i in range(self.k):
            for j in range(len(self.clusters[i])):
                check_object = self.clusters[i][j]
                if check_object not in self.clusters_old[i]:
                    return True
        return False

    def init_best_clusters(self):
        """
        Returns best clusters.
        """
        return self.clusters

    def __str__(self):
        """
        Outputs one clustering run results.
        """
        result_string = ""
        for i in range(self.k):
            result_string += f"Cl#:{i}\n"
            male = 0
            female = 0
            less_49 = 0
            bigger_49 = 0
            response_positive = 0
            response_negative = 0
            for j in range(len(self.clusters[i])):
                result_string += str(self.clusters[i][j].name) + "\n"
                # result_string += "Response:" + str(self.clusters[i][j].response) + "\n"
            #     if self.clusters[i][j].response == 1.0:
            #         response_positive += 1
            #     else:
            #         response_negative += 1
            #     if self.clusters[i][j].sex == 1.0:
            #         male += 1
            #     else:
            #         female += 1
            #     if self.clusters[i][j].age <= 49:
            #         less_49 += 1
            #     else:
            #         bigger_49 += 1
            #
            # result_string += f"Responses: positive {response_positive} negative {response_negative}\n"
            # result_string += f"Gender: male {male} female {female}\n"
            # result_string += f"Age: <=49 -- {less_49} >49 -- {bigger_49}\n"

            models_list = []
            try:
                for m in self.cluster_models[i]:
                    models_list.append(m.model.transpose())
                    concat_models = np.vstack(models_list)
            except Exception as e:
                print("cluster file",e)
                concat_models = np.zeros(0)

            filename = f"coeffs_{i + 1}"

            np.savetxt(
                fr"{filename}.txt",
                concat_models, fmt='%.3f')


        for m in range(len(0 if self.cluster_models[i] == None else self.cluster_models[i])):
                result_string += (self.cluster_models[i][m].__str__(m) + "\n")
        return result_string

    @staticmethod
    def label_clusters(df, k, best_clusters):
        """
        Labels the objects with their cluster allocation.
        """
        for i in range(k):
            for j in range(len(best_clusters[i])):
                df.loc[df['ID'] == best_clusters[i][j].name, 'Cluster'] = int(i)
        return df
