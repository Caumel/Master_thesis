import numpy as np

from src.IKM.clustering.ikm.clust.IC_model_builder import ICmodelBuilder

class ModelBilder:

    @staticmethod
    def create_model(cluster, error):

        """
        Does initialization and calls the functions for building models for a cluster.
        """
        if not cluster:
            return None

        d = cluster[0].d #nº of variables
        
        # aggregation = np.zeros((d, d))
        # YtY = np.zeros((d, 1))

        # for obj in cluster:
        #     aggregation = np.add(aggregation, obj.comp_data)
        #     YtY = np.add(YtY, obj.quadrs)

        # IC_model = ICmodelBuilder()
        # result = []
        # for i in range(d):
        #     model = IC_model.build_model(cluster, i, aggregation, YtY, error)

        #     result.append(model)
        # return result


        comp_data_list = [obj.comp_data for obj in cluster]
        quadrs_list = [obj.quadrs for obj in cluster]

        aggregation = np.sum(comp_data_list, axis=0)
        YtY = np.sum(quadrs_list, axis=0)

        IC_model = ICmodelBuilder()
        
        result = [IC_model.build_model(cluster, i, aggregation, YtY, error) for i in range(d)]
        
        return result
