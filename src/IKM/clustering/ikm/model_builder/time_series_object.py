import numpy as np

from ikm.utils.data_preprocessor import DataPreprocessor


class TSObject:

    def __init__(self, file_name='', data=None, leave_one_third_part_mode=None, leave_every_nth_el=None,
                 delete_every_nth_el=None,
                 box_cox=None, dwt=None,
                 dwt_complex=None,
                 z_normalization=None, z_score=None, z_transform_mode=None, excl_wm=None,
                 left_right=None, front_back=None, band=None, hilbert=None, specific_electrodes=None, electrodes=None):

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

            # self.data = data_preprocessor.leave_parts_electrodes(data, left_right=left_right,
            #                                                      front_back=front_back)
            print(data.shape)
            print(data)
            self.data = data_preprocessor.leave_windmills(data,excl_wm)
            print(data.shape)        

        return

        if specific_electrodes:
            self.data = data_preprocessor.choose_parts_electrodes(data, electrodes=electrodes)

        if leave_one_third_part_mode is not None:
            self.data = data_preprocessor.leave_one_third_part(self.data, leave_one_third_part_mode)

        if leave_every_nth_el is not None:
            self.data = data_preprocessor.leave_every_nth_el(self.data, leave_every_nth_el)

        if delete_every_nth_el is not None:
            self.data = data_preprocessor.delete_every_nth_el(self.data, delete_every_nth_el)

        # self.data = data_preprocessor.theta_alpha_extraction(self.data)

        # self.data = data_preprocessor.EEG_bandstop(self.data, fs)

        if band is not None:
            self.data = data_preprocessor.butter_eeg_bands_extraction(self.data, fs, band)

        if hilbert == 'phase':
            self.data = data_preprocessor.hilbert_phase(self.data)
        elif hilbert == 'ampl':
            self.data = data_preprocessor.hilbert_amplitude(self.data)

        # print(self.data)

        # box-cox transformation
        if box_cox:
            self.data = data_preprocessor.box_cox_transform(self.data)

        # DWT transformation with statistics extraction
        if dwt_complex:
            self.data = data_preprocessor.get_eeg_features(self.data, 'db4', 5)

        # DWT transformation simple
        if dwt:
            self.data = data_preprocessor.dwt(self.data, 'db4')

        # z-normalization applied
        if z_normalization:
            self.data = data_preprocessor.z_normalize(self.data)

        # z-score transformation
        if z_score:
            self.data = data_preprocessor.z_score(self.data)

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