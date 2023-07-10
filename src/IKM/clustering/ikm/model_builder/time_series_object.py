import numpy as np

from ikm.utils.data_preprocessor import DataPreprocessor
from tqdm import tqdm


class TSObject:

    def __init__(self, file_name='', data=None, box_cox=None,
                 z_normalization=None, z_score=None, excl_wm=None,
                 specific_windmills=None, windmills=None,kind_mean="all"):

        """
        Initializes an object for IKM. Applies transformation(s) to the data.
        """

        self.name = file_name # Name of the file

        self.data = data  #df
        if kind_mean == "all":
            self.data_mean = self.data[:,4:].mean()
        else:
            self.data_mean = np.mean(data[:,4:], axis=0)

        data_preprocessor = DataPreprocessor()

        # Remove some wind mills
        if excl_wm:

            # excl_wm = [1,2,3,4,5,10]
            self.data = data_preprocessor.leave_windmills(data,excl_wm)

        if specific_windmills:

            # windmills = [1,2,3,4,5]
            self.data = data_preprocessor.choose_parts_windmills(data, windmills=windmills)

        # box-cox transformation

        if box_cox:
            self.data = data_preprocessor.box_cox_transform(self.data)

        # z-normalization applied
        if z_normalization:
            self.data = data_preprocessor.z_normalize(self.data)
            # Here self.data is a polars 
        
        # z-score transformation
        if z_score:
            self.data = data_preprocessor.z_score(self.data)
            # Here self.data is a numpy 
        
        ## Compute Data and Quadrs
        self.numbers = self.data[:,4:]
        self.rest = self.data[:,:4]

        self.m = np.size(self.numbers, 0)
        self.d = np.size(self.numbers, 1)

        self.comp_data = data_preprocessor.compute_data(self.numbers, self.d, self.m)

        self.quadrs = data_preprocessor.compute_quadrs(self.numbers, self.d, self.m) 
        
    def __str__(self):
        return self.name
