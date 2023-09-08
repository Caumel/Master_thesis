import copy
import os

import numpy as np
import ntpath

from ikm.model_builder import model_bilder
from ikm.model_builder.time_series_object import TSObject
from ikm.utils.test import Test


def main():
    """
    Implementation of the interpretation algorithm. Computes the errors of an object using leave-one-out cross-validation method.
    """

    number_objs = 0
    classes_ = []
    test = Test()

    objs = [
        [
            "../data/file_per_event/current_experiment/0_0_2000",
            "../data/file_per_event/current_experiment/0_102_2000",
            "../data/file_per_event/current_experiment/0_103_2000",
            "../data/file_per_event/current_experiment/0_10_2000",
            "../data/file_per_event/current_experiment/0_113_2000",
            "../data/file_per_event/current_experiment/0_11_2000",
            "../data/file_per_event/current_experiment/0_12_2000",
            "../data/file_per_event/current_experiment/0_13_2000",
            "../data/file_per_event/current_experiment/0_14_2000",
            "../data/file_per_event/current_experiment/0_15_2000",
            "../data/file_per_event/current_experiment/0_17_2000",
            "../data/file_per_event/current_experiment/0_19_2000",
            "../data/file_per_event/current_experiment/0_16_2000",
            "../data/file_per_event/current_experiment/0_18_2000"
        ],
        [
            "../data/file_per_event/current_experiment/0_210_2000",
            "../data/file_per_event/current_experiment/0_1043_2000",
            "../data/file_per_event/current_experiment/0_1621_2000",
            "../data/file_per_event/current_experiment/0_174_2000",
            "../data/file_per_event/current_experiment/0_215_2000",
            "../data/file_per_event/current_experiment/0_246_2000",
            "../data/file_per_event/current_experiment/0_211_2000",
            "../data/file_per_event/current_experiment/0_212_2000"
        ],
        [
            "../data/file_per_event/current_experiment/0_1318_2000",
            "../data/file_per_event/current_experiment/0_1475_2000",
            "../data/file_per_event/current_experiment/0_201_2000",
            "../data/file_per_event/current_experiment/0_1237_2000",
            "../data/file_per_event/current_experiment/0_1308_2000",
            "../data/file_per_event/current_experiment/0_1354_2000",
            "../data/file_per_event/current_experiment/0_143_2000",
            "../data/file_per_event/current_experiment/0_1513_2000"
        ]
        ]

    error = 'eucl'
    electrodes = ['Fp1', 'F7', 'F3', 'Cz', 'C3', 'T3', 'P3', 'T5', 'O1']

    class_number = len(objs)
    for i in range(class_number):
        objs_cluster = objs[i]
        number_objs += len(objs_cluster)
        objs_cluster.sort()

        class_ = []
        j = len(objs_cluster) - 1
        while j >= 0:
            data = np.loadtxt(os.path.join(objs_cluster[j]), delimiter=' ')
            filename = ntpath.basename(objs_cluster[j])
            class_.append(TSObject(file_name=filename, data=data, box_cox=True, z_score=True, excl_el=True,
                                   specific_electrodes=electrodes))
            j = j - 1
        classes_.append(class_)

    errors = [0 for c in range(class_number)]
    for cl in range(class_number):
        for obj_num_in_cluster in range(len(classes_[cl])):

            mod_set = copy.deepcopy(classes_)
            test_set = []
            test_set.append(mod_set[cl][obj_num_in_cluster])
            mod_set[cl].pop(obj_num_in_cluster)
            training_mod_set = []
            for i in range(class_number):
                training_mod_set.append(mod_set[i])

            class_models = []
            for i in range(class_number):
                model = model_bilder.ModelBilder.create_model(mod_set[i], error)
                class_models.append(model)

            errors[cl] += test.models_errors(class_models, test_set, cl, error)

    err_number = 0
    print(": err.  = (")
    for error in errors:
        err_number += error
        print(f"{error} ")

    print(f") ({err_number});  {round((1 - (err_number + 0.0) / number_objs) * 1000) / 10.0} %")
    for i in range(len(test.errors_in_models)):
        print(f"{i + 1}:{test.errors_in_models[i]}")


if __name__ == '__main__':
    main()
