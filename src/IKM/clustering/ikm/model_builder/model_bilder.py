import numpy as np

from src.IKM.clustering.ikm.clust.IC_model_builder import ICmodelBuilder

class ModelBilder:

    @staticmethod
    def create_model(cluster, error):

        """
        Does initialization and calls the functions for building models for a cluster.
        """
        if len(cluster) == 0:
            return None

        d = cluster[0].d
        aggregation = np.zeros((d, d))
        YtY = np.zeros((d, 1))

        for obj in cluster:
            aggregation = np.add(aggregation, obj.comp_data)
            YtY = np.add(YtY, obj.quadrs)

        IC_model = ICmodelBuilder()
        result = []
        for i in range(d):
            model = IC_model.build_model(cluster, i, aggregation, YtY, error)

            result.append(model)
        return result
