import os

import numpy as np
import pandas as pd
import polars as pl


from ikm.model_builder.time_series_object import TSObject

VERY_SMALL_CLOSEST_ZERO = 0.0000000000000000000001


class DataLoader:
    delimiter = ','

    def __init__(self, delimiter):
        self.delimiter = delimiter

    def load_data_two_files(self, box_cox=None,
                            dwt_complex=None, z_normalization=None,
                            z_score=None, z_transform_mode=None, excl_wm=None,
                            band=None, hilbert=None, path_first_file=None,
                            path_second_file=None, specific_windmills=None, windmills=None):
        pass
        # """
        # Loads the depressed patients data from two visits.
        # """

        # objects = []
        # df = pd.DataFrame(columns=['ID', 'Data', 'Response'])
        # for filename_second_file in os.listdir(path_second_file):
        #     splitted_filename = filename_second_file.split('_')
        #     splitted_filename[1] = str(int(splitted_filename[1]) - 1)
        #     filename_first_file = '_'.join(splitted_filename)
        #     # filename_first_file = f'objects-first-visit-smoothed-triple\{filename_first_file}'

        #     data_first_visit = np.loadtxt(os.path.join(path_first_file, filename_first_file), delimiter=self.delimiter)
        #     data_second_visit = np.loadtxt(os.path.join(path_second_file, filename_second_file),
        #                                    delimiter=self.delimiter)
        #     arr_data = [data_first_visit, data_second_visit]

        #     data = np.append(arr_data[0], arr_data[1], axis=0)
        #     objects.append(
        #         TSObject(
        #             file_name=filename_first_file, data=data,
        #             box_cox=box_cox,
        #             dwt_complex=dwt_complex, z_normalization=z_normalization, z_score=z_score,
        #             z_transform_mode=z_transform_mode,
        #             excl_wm=excl_wm, band=band, hilbert=hilbert,
        #             specific_electrodes=specific_electrodes, electrodes=electrodes
        #         ))
        #     df.loc[len(df.index)] = [filename_first_file, data, filename_second_file[-5]]

        # return df, objects

    def load_data_one_file(self, box_cox=None,
                           dwt_complex=None, z_normalization=None, z_score=None,
                           z_transform_mode=None, excl_wm=None,
                           band=None, hilbert=None, path=None,
                           specific_windmills=None, windmills=None):
        """
        Loads the depressed patients data from one visit.
        """

        objects = []
        df = pd.DataFrame(columns=['ID', 'Data', 'Response'])
        for filename in os.listdir(path):

            # data = np.loadtxt(os.path.join(path, filename), delimiter=self.delimiter)
            data = pl.read_csv(os.path.join(path, filename))
            objects.append(
                TSObject(file_name=filename, # Filename
                         data=data,          # Information 
                         box_cox=box_cox,   # Apply Box_cox transformation
                         dwt_complex=dwt_complex, # Clustering base on the DWT info
                         z_normalization=z_normalization, # Z normalization apply
                         z_score=z_score, # Clustering base on the z_score info
                         z_transform_mode=z_transform_mode, # Transform returned, 'magnitude' or 'phase'
                         excl_wm=excl_wm,  # Exclude some electrodes (Can i do it with wind farms ?)
                         band=band,  #Kind of band, Delta, Theta, Beta1, Beta2, 'gamma_1', 'gamma_2'
                         hilbert=hilbert, # Kind of hilbert transform
                         specific_windmills=specific_windmills, 
                         windmills=windmills))
            
            return
            
            df.loc[len(df.index)] = [filename, data, filename[-5]]
            # Df [ file, data, something ?]
        return df, objects

    # def load_alco(self, path):

    #     """
    #     Loads the data from https://archive-beta.ics.uci.edu/dataset/121/eeg+database.
    #     """
    #     objects = []
    #     df = pd.DataFrame(columns=['ID', 'Data', 'Response'])
    #     for filename in os.listdir(path):
    #         data = np.loadtxt(os.path.join(path, filename), delimiter=' ')
    #         objects.append(TSObject(filename, data))
    #         if filename[3] == "c":
    #             df.loc[len(df.index)] = [filename, data, 0]
    #         elif filename[3] == "a":
    #             df.loc[len(df.index)] = [filename, data, 1]
    #     return df, objects
