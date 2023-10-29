from ikm.clust import IC_model_builder
from joblib import Parallel, delayed
import copy
import time


# 1 - High
# 2 - Moderate
# 3 - Low

class ModelsErrors:
    errors_in_models = []

    def models_errors(self, models, objects, cluster_number, class_number, error, text, objs_label_real_test, objs_label_train_test):
        """
        Calculates errors in the models using leave-one-out cross-validation.
        """

        # 14 models
        true_train_results = 0
        true_test_results = 0
        train_test_results = 0
        if not self.errors_in_models:
            self.errors_in_models = [0 for i in range(len(models[0]))] # 14, 1 por variable

        for index, object_ in enumerate(objects): # Para cada elemento del test
            errors = [[] for i in range(len(models))]
            
            for i in range(len(models)):
                errors[i] = IC_model_builder.ICmodelBuilder.obj_error_regarding_cluster(models[i], object_, error)

                sign = 1 if i == cluster_number else -1 # si estamos en el cluster index, 1 si no -1
                self.errors_in_models = [x if isinstance(y,list) else x + (sign * (y/1)) for x, y in zip(self.errors_in_models, IC_model_builder.ICmodelBuilder.errors_in_models)] 

            # A que cluster pertenecen.
            real_label = objs_label_real_test[index]
            train_label = objs_label_train_test[index]
            test_label = errors.index(min(errors))

            error_computer = errors[cluster_number]
            error_true = min(errors)
            
            differences_respect_train_true_value = [error_true - x for i, x in enumerate(errors) if i != errors.index(error_true)]
            differences_respect_test_value = [error_computer - x for i, x in enumerate(errors) if i != cluster_number]

            text += f"{object_} ==> Real vs Train: {real_label == train_label}, Real vs Test: {real_label == test_label}, Train vs Test: {train_label == test_label}, Real, Train, Test {real_label,train_label,test_label}, min_error: {error_true}, erros {errors}, diff train {differences_respect_train_true_value}, diff test {differences_respect_test_value}\n"
            
            if real_label != train_label:
                true_train_results += 1
            if real_label != test_label:
                true_test_results += 1
            if train_label != test_label:
                train_test_results += 1

        return true_train_results, true_test_results, train_test_results, text
