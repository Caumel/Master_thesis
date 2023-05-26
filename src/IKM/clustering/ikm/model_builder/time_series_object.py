import numpy as np

from ikm.utils.data_preprocessor import DataPreprocessor


class TSObject:

    def __init__(self, file_name='', data=None,
                 box_cox=None,
                 dwt_complex=None,
                 z_normalization=None, z_score=None, z_transform_mode=None, excl_wm=None,
                 band=None, hilbert=None, specific_windmills=None, windmills=None):

        """
        Initializes an object for IKM. Applies transformation(s) to the data.
        """

        self.name = file_name # Name of the file

        self.data = data  #df 
        self.data_mean = self.data.mean()

        fs = 250.0
        data_preprocessor = DataPreprocessor()


        # Applying the sinus function
        # self.data = data_preprocessor.sin(self.data)

        # Remove some wind mills
        if excl_wm:

            # excl_wm = [1,2,3,4,5,10]
            self.data = data_preprocessor.leave_windmills(data,excl_wm)

        if specific_windmills:

            # windmills = [1,2,3,4,5]
            self.data = data_preprocessor.choose_parts_windmills(data, windmills=windmills)

        # if band is not None:
        #     self.data = data_preprocessor.butter_eeg_bands_extraction(self.data, fs, band)

        # if hilbert == 'phase':
        #     self.data = data_preprocessor.hilbert_phase(self.data)
        # elif hilbert == 'ampl':
        #     self.data = data_preprocessor.hilbert_amplitude(self.data)

        # box-cox transformation
        if box_cox:
            self.data = data_preprocessor.box_cox_transform(self.data)

        # # ????
        # # DWT transformation with statistics extraction
        # if dwt_complex:
        #     self.data = data_preprocessor.get_eeg_features(self.data, 'db4', 5)

        # z-normalization applied
        if z_normalization:
            self.data = data_preprocessor.z_normalize(self.data)

        # z-score transformation
        if z_score:
            self.data = data_preprocessor.z_score(self.data)

        return

        # Z transformation

        z = 1j

        if z_transform_mode == 'magnitude':

            z_transformed, self.data, phase = data_preprocessor.z_transform(self.data, z)

        elif z_transform_mode == 'phase':
            z_transformed, magnitude, self.data = data_preprocessor.z_transform(self.data, z)

        # Remove dimensions
        # array_dimensions_rmv = [0, 1, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15,  17, 18]
        # self.data = data_preprocessor.remove_dimensions(self.data, array_dimensions_rmv)

        self.m = np.size(self.data, 0)
        self.d = np.size(self.data, 1)

        self.comp_data = data_preprocessor.compute_data(self.data, self.d, self.m)

        self.quadrs = data_preprocessor.compute_quadrs(self.data, self.d, self.m)

    def __str__(self):
        return self.name
