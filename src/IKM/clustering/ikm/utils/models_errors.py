from ikm.clust import IC_model_builder


class ModelsErrors:
    errors_in_models = []

    def models_errors(self, models, objects, cluster_number, error, text):
        """
        Calculates errors in the models using leave-one-out cross-validation.
        """

        result = 0
        if not self.errors_in_models:
            self.errors_in_models = [0 for i in range(len(models[0]))]

        for object_ in objects:
            errors = [[] for i in range(len(models))]
            for i in range(len(models)):
                errors[i] = IC_model_builder.ICmodelBuilder.obj_error_regarding_cluster(models[i], object_, error)
                sign = 1 if i == cluster_number else -1
                for j in range(len(self.errors_in_models)):
                    self.errors_in_models[j] += sign * (IC_model_builder.ICmodelBuilder.errors_in_models[j] / 1)

            min_index = 0
            for i in range(len(errors)):
                if errors[i] < errors[min_index]:
                    min_index = i

            err_true = errors[cluster_number]
            if cluster_number == 0:
                err_false_1 = errors[(cluster_number + 1)]
                err_false_2 = errors[(cluster_number + 2)]
            elif cluster_number == 1:
                err_false_1 = errors[(cluster_number - 1)]
                err_false_2 = errors[(cluster_number + 1)]
            else:
                err_false_1 = errors[(cluster_number - 2)]
                err_false_2 = errors[(cluster_number - 1)]
            err_r_f_diff_1 = err_true - err_false_1
            err_r_f_diff_2 = err_true - err_false_2

            # print(f"{object_} ==> {cluster_number == min_index}  {err_true}  {err_false_1} {err_r_f_diff_1} {err_false_2} {err_r_f_diff_1}")
            text += f"{object_} ==> {cluster_number == min_index}  {err_true}  {err_false_1} {err_r_f_diff_1} {err_false_2} {err_r_f_diff_2}\n"
            if cluster_number != min_index:
                result += 1

        return result, text
