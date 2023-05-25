import os

import numpy as np

from clustering.ikm.utils.data_loader import DataLoader

from clustering.ikm.model_builder.time_series_object import TSObject

from clustering.ikm.utils.data_preprocessor import DataPreprocessor
import data_exploration
from distances import DistanceMeasurer


def calc_norm(norm='fro'):
    """
    Calculating the distances between coefficients from the data from different visits.
    Parameters:
        norm (string): can be 'fro' for Frobenius distance and 'total' for total distance
    """

    distance_measurer = DistanceMeasurer()
    dirs = [
        r"/media/data/lazarenkom98dm/coeffs-first-visit-non-responders",
        r"/media/data/lazarenkom98dm/coeffs-first-visit-responders",
        r"/media/data/lazarenkom98dm/coeffs-second-visit-non-responders",
        r"/media/data/lazarenkom98dm/coeffs-second-visit-responders",
    ]

    final_str = ""

    for dir_1 in dirs:
        for dir_2 in dirs:
            if dir_1 == dir_2:
                continue

            file_path = rf'distance.txt'
            distance = distance_measurer.distance_between_coeffs_each_obj_average(dir_1, dir_2, norm)

            final_str += f"{os.path.basename(dir_1)}-{os.path.basename(dir_2)}\n{distance}\n"

            file = open(file_path, 'a')

            file.write(final_str)
            file.close()
            final_str = ''


def create_matrix_corr():
    """
    Creating a correlation matrix between coefficients.
    """

    distance_measurer = DistanceMeasurer()

    names = ['1st-visit-non-responded', '1st-visit-responded', '2nd-visit-non-responded', '2nd-visit-responded']
    file_paths = ['',
                  '',
                  '',
                  '', ]
    df = distance_measurer.create_matrix_coeffs(file_paths, names)
    # print(df)

    distance_measurer.correlation_between_coeffs(paths=file_paths, delimeter=' ', annot=False, mode='one')


def find_coeffs_specific_obj_each_day(selected_el=None):

    """
    Finding coefficients for each object for a specific day using least squares method.
    """

    path = "/media/data/lazarenkom98dm/objects-combined"

    objects = []
    data_loader = DataLoader()
    data_preprocessor = DataPreprocessor()
    if selected_el:
        for filename in os.listdir(path):
            data = np.loadtxt(os.path.join(path, filename), delimiter='\t')

            data_reduced = data_preprocessor.choose_parts_electrodes(data, selected_el)
            objects.append(TSObject(filename, data_reduced))
    else:
        df, objects = data_loader.load_data_one_file(path)

    for obj in objects:
        models = []
        models = np.array(models)
        for dim in range(obj.d):
            coeffs = data_exploration.coeffs_each_obj(obj.d, obj.m, obj.data, dim).transpose()
            if models.size != 0:
                models = np.append(models, coeffs, axis=0)
            else:
                models = coeffs
        np.savetxt(

            fr"/media/data/lazarenkom98dm/coeffs-combined/{obj.name}",
            models, fmt='%.3f')
        print(f"File {obj.name} has been created.")
        # obj_coeffs.append(models)


def main():
    # electrodes = ['Fp1', 'F7', 'F3']
    find_coeffs_specific_obj_each_day()


if __name__ == '__main__':
    main()
