# import statsmodels.api as sm
import collections.abc
import sys
sys.path.append('../../../')


from math import sqrt, pi

import numpy as np
from numpy import linalg as la

# OBJECT_DIM = 19
from src.IKM.clustering.ikm.model_builder.model import Model


class ICmodelBuilder:
    errors_in_models = []

    @staticmethod
    def build_model(objects, predicted_attr_num, aggregation, YtYs, error):
        """
        Builds the model for a dimension.
        """

        if len(objects) == 0:
            return None

        d = objects[0].d
        # d =OBJECT_DIM

        m = 0

        for ts_object in objects:
            # m += ts_object.data.shape[0]
            m += ts_object.m

        relative_input = []
        interaction_cluster_error = sys.float_info.max

        parameters_final = None

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

                parameters = ICmodelBuilder.least_squares(tmp_input, predicted_attr_num, aggregation)
                bic = ICmodelBuilder.bic(tmp_input, predicted_attr_num, parameters, m, aggregation, YtYs, error)

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

        if error == 'eucl':
            error = sqrt(
                ICmodelBuilder.compute_euclid_error(YtYs, aggregation, parameters_final, predicted_attr_num,
                                                    relative_input)) / m
        elif error == 'total':
            error = ICmodelBuilder.compute_total_norm_error(aggregation, parameters_final, predicted_attr_num,
                                                            relative_input) / m
        elif error == 'max':
            error = ICmodelBuilder.compute_max_error(aggregation, parameters_final, predicted_attr_num,
                                                     relative_input) / m
        elif error == 'jaccard':
            error = ICmodelBuilder.compute_jaccard_error(aggregation, parameters_final, predicted_attr_num,
                                                         relative_input) / m
        elif error == 'hamming':
            error = ICmodelBuilder.compute_hamming_error(aggregation, parameters_final, predicted_attr_num,
                                                         relative_input) / m

        model = Model()
        model.model = parameters_final_zeros
        model.error = error

        return model

    ### modes
    # aggregation -- when data is precomputed using cubes
    # ordinary -- when data is raw
    @staticmethod
    def least_squares(tmp_input, predicted_attr_num, data, mode='aggregation'):
        """
        Executes least squares method.
        """

        a = []
        for i in range(len(tmp_input)):
            a.append(tmp_input[i])
        a.sort()
        if mode == 'aggregation':
            XtX = data[np.ix_(a, a)]
            XtY = data[np.ix_(a, [predicted_attr_num])]
        elif mode == 'ordinary':
            XtX = np.matmul(data[:, a].transpose(), data[:, a])
            XtY = np.matmul(data[:, a].transpose(), data[:, predicted_attr_num])
        # TODO:
        #   # I compute the pseudoinverse but is not the same, lets check
        try:
            XtX_inverse = np.linalg.inv(XtX)
        except:
            XtX_inverse = np.linalg.pinv(XtX)
        parameters = np.matmul(XtX_inverse, XtY)
        return parameters

    @staticmethod
    def bic(tmp_input, predicted_attr_num, parameters, m, aggregation, YtYs, error):
        """
        Finds the Bayesian Information Criterion.
        """
        parameters_number = np.size(parameters, 0)
        sigma = ICmodelBuilder.sigma(predicted_attr_num, tmp_input, parameters, m, aggregation, YtYs, error)
        log_likelihood = ICmodelBuilder.log_likelihood(m, sigma)
        result = -2 * log_likelihood + np.log(m) * (parameters_number + 1.0)
        return result

    @staticmethod
    def sigma(predicted_attr_num, tmp_input, parameters, m, aggregation, YtYs, error='eucl'):
        """
        Finds the sigma for calculating the BIC.
        """
        result = 0
        if error == 'eucl':
            result = ICmodelBuilder.compute_euclid_error(YtYs, aggregation, parameters, predicted_attr_num, tmp_input)
        elif error == 'total':
            result = ICmodelBuilder.compute_total_norm_error(aggregation, parameters, predicted_attr_num, tmp_input)
        elif error == 'max':
            result = ICmodelBuilder.compute_max_error(aggregation, parameters, predicted_attr_num, tmp_input)
        elif error == 'jaccard':
            result = ICmodelBuilder.compute_jaccard_error(aggregation, parameters, predicted_attr_num, tmp_input)
        elif error == 'hamming':
            result = ICmodelBuilder.compute_hamming_error(aggregation, parameters, predicted_attr_num, tmp_input)

        result = result / m
        return result

    @staticmethod
    def log_likelihood(m, sigma):
        """
        Finds log-likelihood.
        """
        result = (m / 2.0) * np.log(2 * pi)
        result += (m / 2.0)
        result += (m / 2.0) * np.log(sigma)
        return -result

    @staticmethod
    def compute_Y_X_Xparameters_diff_errors(data, parameters, predicted_attr_num, tmp_input):
        """
        Computes the Y, X and X multiplied by parameters values for all distance errors except Euclidian distance error.
        """
        a = []
        for i in range(len(tmp_input)):
            a.append(tmp_input[i])
        a.sort()

        Y = data[:, predicted_attr_num]
        Y = Y[:, np.newaxis]
        X = data[:, a]
        Xparameters = np.matmul(X, parameters)

        return Y, X, Xparameters

    @staticmethod
    def compute_euclid_error(YtYs, data, parameters, predicted_attr_num, tmp_input):
        """
        Computes the Euclidian distance error.
        """
        a = []
        for i in range(len(tmp_input)):
            a.append(tmp_input[i])
        a.sort()

        if isinstance(YtYs, (collections.abc.Sequence, np.ndarray)):
            YtY = YtYs[predicted_attr_num][0]
            XtX = data[np.ix_(a, a)]
            XtY = data[np.ix_(a, [predicted_attr_num])]


        else:
            YtY = YtYs
            XtX = np.matmul(data[:, a].transpose(), data[:, a])
            XtY = np.matmul(data[:, a].transpose(), data[:, predicted_attr_num])

        try:
            YtXParameters = np.matmul(XtY.transpose(), parameters)

        except:
            print(f"Parameters {parameters}")

        parameterstXtY = np.matmul(parameters.transpose(), XtY)
        parameterstXtX = np.matmul(parameters.transpose(), XtX)
        parameterstXtXparameters = np.matmul(parameterstXtX, parameters)

        if not isinstance(YtXParameters, (collections.abc.Sequence, np.ndarray)) \
                or not isinstance(parameterstXtY, (collections.abc.Sequence, np.ndarray)) \
                or not isinstance(parameterstXtXparameters, (collections.abc.Sequence, np.ndarray)):
            result = YtY - YtXParameters - parameterstXtY + parameterstXtXparameters
        else:
            result = YtY - YtXParameters[0][0] - parameterstXtY[0][0] + parameterstXtXparameters[0][0]

        return result

    @staticmethod
    def compute_total_norm_error(data, parameters, predicted_attr_num, tmp_input):
        """
        Computes the total norm distance error.
        """

        Y, X, Xparameters = ICmodelBuilder.compute_Y_X_Xparameters_diff_errors(data, parameters, predicted_attr_num,
                                                                               tmp_input)

        diff = Y - Xparameters

        result = np.sum(np.abs(diff))

        return result

    @staticmethod
    def compute_max_error(data, parameters, predicted_attr_num, tmp_input):
        """
        Computes the max distance error.
        """
        Y, X, Xparameters = ICmodelBuilder.compute_Y_X_Xparameters_diff_errors(data, parameters, predicted_attr_num,
                                                                               tmp_input)

        diff = Y - Xparameters

        result = np.max(np.abs(diff))

        return result

    @staticmethod
    def compute_jaccard_error(data, parameters, predicted_attr_num, tmp_input):
        """
        Computes the jaccard distance error.
        """
        Y, X, Xparameters = ICmodelBuilder.compute_Y_X_Xparameters_diff_errors(data, parameters, predicted_attr_num,
                                                                               tmp_input)

        minimum = np.minimum(Y, Xparameters)
        maximum = np.maximum(Y, Xparameters)

        fro_norm_min = la.norm(minimum, 'fro')
        fro_norm_max = la.norm(maximum, 'fro')

        result = fro_norm_min / fro_norm_max

        return result

    @staticmethod
    def compute_hamming_error(data, parameters, predicted_attr_num, tmp_input):

        """
        Computes the hamming distance error.
        """
        Y, X, Xparameters = ICmodelBuilder.compute_Y_X_Xparameters_diff_errors(data, parameters, predicted_attr_num,
                                                                               tmp_input)

        minimum = np.minimum(Y, Xparameters)
        maximum = np.maximum(Y, Xparameters)

        fro_norm_min = la.norm(minimum, 'fro')
        fro_norm_max = la.norm(maximum, 'fro')

        n = np.size(Y, 0)

        result = (fro_norm_max - fro_norm_min) / (n * (n - 1))

        return result

    @staticmethod
    def obj_error_regarding_cluster(cluster_models, ts_object, error):

        """
        Computes the error of an object with regard to a cluster.
        """

        result = 0
        relative_input = []
        for i in range(np.size(ts_object.comp_data, 0)):
            relative_input.append(i)

        ICmodelBuilder.errors_in_models = [[] for i in range(len(cluster_models))]
        for i in range(len(cluster_models)):
            relative_input.remove(i)
            parameters = cluster_models[i].model

            if error == 'eucl':
                model_result = ICmodelBuilder.compute_euclid_error(ts_object.quadrs, ts_object.comp_data, parameters, i,
                                                                   relative_input)
            elif error == 'total':
                model_result = ICmodelBuilder.compute_total_norm_error(ts_object.comp_data, parameters, i,
                                                                       relative_input)
            elif error == 'max':
                model_result = ICmodelBuilder.compute_max_error(ts_object.comp_data, parameters, i,
                                                                relative_input)
            elif error == 'jaccard':
                model_result = ICmodelBuilder.compute_jaccard_error(ts_object.comp_data, parameters, i,
                                                                    relative_input)
            elif error == 'hamming':
                model_result = ICmodelBuilder.compute_hamming_error(ts_object.comp_data, parameters, i,
                                                                    relative_input)

            relative_input.append(i)
            ICmodelBuilder.errors_in_models[i] = model_result

            result += model_result
            # try:
            #     return sqrt(result) / len(cluster_models)
            # except ValueError:
            #     print(f"{result}\n{len(cluster_models)}")
            if result < 0:
                return 0

        return sqrt(result) / len(cluster_models)
