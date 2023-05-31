import math
from collections import Counter
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
import pywt
from scipy.signal import butter, sosfilt, hilbert
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:

    def leave_windmills(self, data, excl_wm):
        """
        Leaving only specific dimensions (windmills) in the data.
        """
        return self.remove_windmills(data, excl_wm)
    
    def remove_windmills(self, data, excl_wm):
        """
        Removing chosen dimensions from the data.
        """
        df_filtered = data.filter(~data["index"].is_in(excl_wm))
        return df_filtered
    
    def choose_parts_windmills(self, data, windmills):
        """
        This function removes dimensions corresponding to the windmills.
        Parameters:
            data (array-like): EEG data
            windmills (array of strings): Windmills to be left in the data
        """
        df_filtered = data.filter(data["index"].is_in(windmills))
        return df_filtered
    
    # def butter_eeg_bands_extraction(self, data, fs, band='delta'):
    #     """
    #     Extracting bands using Butterworth filter.
    #     """
    #     if band == 'delta':
    #         delta = self.butter_bandpass_filter(data, 0.5, 4, fs)
    #         return delta

    #     elif band == 'theta':

    #         theta = self.butter_bandpass_filter(data, 4, 7, fs)
    #         return theta
    #     elif band == 'alpha':
    #         alpha = self.butter_bandpass_filter(data, 8, 13, fs)

    #         return alpha
    #     elif band == 'beta_1':
    #         beta_1 = self.butter_bandpass_filter(data, 14, 24, fs)

    #         return beta_1
    #     elif band == 'beta_2':
    #         beta_2 = self.butter_bandpass_filter(data, 25, 35, fs)
    #         return beta_2
    #     elif band == 'beta':
    #         beta = self.butter_bandpass_filter(data, 14, 30, fs)
    #         return beta

    #     elif band == 'gamma_1':
    #         gamma_1 = self.butter_bandpass_filter(data, 36, 58, fs)

    #         return gamma_1
    #     elif band == 'gamma_2':
    #         gamma_2 = self.butter_bandpass_filter(data, 62, 100, fs)

    #         return gamma_2
    #     elif band == 'gamma':
    #         gamma = self.butter_bandpass_filter(data, 31, 100, fs)
    #         return gamma

    # def hilbert_amplitude(self, data):

    #     """
    #     Computing Hilbert amplitude.
    #     """
    #     z = hilbert(data)
    #     inst_amplitude = np.abs(z)
    #     return inst_amplitude

    # def hilbert_phase(self, data):
    #     """
    #     Computing Hilbert phase.
    #     """
    #     z = hilbert(data)
    #     inst_phase = np.unwrap(np.angle(z))
    #     return inst_phase
    
    def box_cox_transform(self,data):

        """
        Applies Box-Cox transformation.
        """
        list_columns = data.columns
        data_temporal = data.drop("time").to_numpy()
        dimensions = np.size(data_temporal, 1)
        data_temporal = self.shift_to_positive(data_temporal)
        for col in range(3,dimensions):
            try:
                info, lambda_box_cox = stats.boxcox(data_temporal[:, col])
                data.with_column(pl.Series(list_columns[col], info))
            except:
                continue
        return data

    def shift_to_positive(self,x):
        """
        Shift all numbers in an array to positive numbers.
        """
        min_value = np.min(x)
        if min_value > 0:
            return x
        shift_value = np.abs(min_value) + 1
        return x + shift_value
    
    # def get_eeg_features(self, eeg_data, waveletname, level):

    #     """
    #     Get the statistics from EEG data.
    #     """
    #     list_features = []
    #     for electrode in eeg_data.T:
    #         list_coeff = pywt.wavedec(electrode, waveletname, level=level)
    #         features = []
    #         for coeff in list_coeff:
    #             features += self.get_features(coeff)
    #         list_features.append(features)
    #     list_features = np.stack(list_features, axis=1)
    #     return list_features

    # def get_features(self, list_values):

    #     """
    #     Gets entropy, crossings, and statistics.
    #     """
    #     entropy = self.calculate_entropy(list_values)
    #     crossings = self.calculate_crossings(list_values)
    #     statistics = self.calculate_statistics(list_values)
    #     return [entropy] + crossings + statistics

    # def calculate_entropy(self, list_values):
    #     """
    #     Calculates entropy.
    #     """

    #     counter_values = Counter(list_values).most_common()
    #     probabilities = [elem[1] / len(list_values) for elem in counter_values]
    #     entropy = stats.entropy(probabilities)
    #     return entropy

    # def calculate_statistics(self, list_values):
    #     """
    #     Calculates different statistics such as mean, standard deviation, variance and others.
    #     """
    #     n5 = np.nanpercentile(list_values, 5)
    #     n25 = np.nanpercentile(list_values, 25)
    #     n75 = np.nanpercentile(list_values, 75)
    #     n95 = np.nanpercentile(list_values, 95)
    #     median = np.nanpercentile(list_values, 50)
    #     mean = np.nanmean(list_values)
    #     std = np.nanstd(list_values)
    #     var = np.nanvar(list_values)
    #     rms = np.nanmean(np.sqrt(list_values ** 2))
    #     return [n5, n25, n75, n95, median, mean, std, var, rms]

    # def calculate_crossings(self, list_values):
    #     """
    #     Calculates crossings.
    #     """
    #     zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    #     no_zero_crossings = len(zero_crossing_indices)
    #     mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    #     no_mean_crossings = len(mean_crossing_indices)
    #     return [no_zero_crossings, no_mean_crossings]

    def z_normalize(self,data):
        """
        Applies z-normalization.
        """
        list_columns = data.columns
        data_temporal = data[:,3:]
        scaler = StandardScaler()
        scaler.fit(data_temporal)
        z_normalized_data = scaler.transform(data_temporal)
        for col in range(0,data_temporal.shape[1]):
            data.with_column(pl.Series(list_columns[col+3], z_normalized_data[col]))

        return data
    
    def z_score(self,data):
        """
        Applies z-score transformation.
        """
        numbers = data.to_numpy()[:,3:].astype(float)
        rest = data.to_numpy()[:,:3]
        numbers_update = stats.zscore(numbers, axis=None)
        data = np.hstack((rest, numbers_update))

        return data
    
    def z_transform(self, data, z):
        """
        Applies Z transform.
        """

        numbers = data[:,3:]
        rest = data[:,:3]


        d = np.size(numbers, 1)
        m = np.size(numbers, 0)
        z_transformed = np.zeros((m, d), dtype=complex)

        for dim in range(d):
            for point in range(m):
                value = numbers[point][dim] * z ** (-point)
                z_transformed[point][dim] = value

        # Compute the magnitude of the z-transform
        magnitude = np.abs(z_transformed)

        # Compute the phase of the z-transform
        phase = np.angle(z_transformed)

        magnitude = np.hstack((rest, magnitude))
        phase = np.hstack((rest, phase))

        return z_transformed, magnitude, phase

    def compute_data(self, data, dimensions, numb_data_points):
        """
        Computes V^tA.
        """
        comp_data = np.zeros((dimensions, dimensions))
        for i in range(dimensions):
            for j in range(i, dimensions):
                tmp = 0
                for h in range(numb_data_points):
                    tmp += data[h][i] * data[h][j]
                comp_data[i][j] = tmp
                comp_data[j][i] = tmp
        return comp_data

    def compute_quadrs(self, data, dimensions, numb_data_points):
        """
        Computes V^tV.
        """
        quadrs = np.zeros((dimensions, 1))
        for i in range(dimensions):
            for j in range(numb_data_points):
                tmp = data[j][i]
                tmp *= tmp
                tmp += quadrs[i][0]
                quadrs[i][0] = tmp
        return quadrs


    # def multi_level_dwt(self, data):
    #     """
    #     Applies multi-level Discrete Wavelet Transform.
    #     """
    #     coeffs = 0
    #     for col in range(data.ndim + 1):
    #         coeffs = pywt.wavedec(data[:, col], 'bior6.8', mode='sym', level=2)
    #     return coeffs

    # # EEG bands extraction using FFT method
    # def fft_eeg_bands_extraction(self, eeg_data):

    #     """
    #     Extracts EEG bands using Fast Fourier Transform.
    #     """
    #     fs = 250  # Sampling rate (250 Hz)

    #     # Get real amplitudes of FFT (only in postive frequencies)
    #     fft_vals = np.absolute(np.fft.rfft(eeg_data))

    #     # Get frequencies for amplitudes in Hz
    #     fft_freq = np.fft.rfftfreq(len(eeg_data), 1.0 / fs)

    #     # Define EEG bands
    #     eeg_bands = {'Delta': (0, 4),
    #                  'Theta': (4, 8),
    #                  'Alpha': (8, 12),
    #                  'Beta': (12, 30),
    #                  'Gamma': (30, 45)}

    #     eeg_band_fft = dict()

    #     for band in eeg_bands:
    #         freq_ix = np.where((fft_freq >= eeg_bands[band][0]) &
    #                            (fft_freq <= eeg_bands[band][1]))[0]
    #         eeg_band_fft[band] = fft_vals[freq_ix]

    #     return eeg_band_fft

    # def butter_bandpass(self, lowcut, highcut, fs, order=5):

    #     """
    #     Computes second order sections for Butterworth bandpass filter.
    #     """
    #     nyq = 0.5 * fs
    #     low = lowcut / nyq
    #     high = highcut / nyq
    #     sos = butter(order, [low, high], btype='band', output='sos')
    #     return sos

    # def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
    #     """
    #     Applying Butterworth bandpass filter.
    #     """

    #     sos = self.butter_bandpass(lowcut, highcut, fs, order=order)
    #     y = sosfilt(sos, data)
    #     return y

    # def butter_bandstop(self, lowcut, highcut, fs, order=5):
    #     """
    #     Computes second order sections for Butterworth bandstop filter.
    #     """
    #     nyq = 0.5 * fs
    #     low = lowcut / nyq
    #     high = highcut / nyq
    #     sos = butter(order, [low, high], btype='bandstop', analog=False, output='sos')
    #     return sos

    # def butter_bandstop_filter(self, data, lowcut, highcut, fs, order=5):
    #     """
    #     Computes second order sections for Butterworth bandstop filter.
    #     """
    #     sos = self.butter_bandstop(lowcut, highcut, fs, order=order)
    #     y = sosfilt(sos, data)
    #     return y

    # def sin(self, eeg_data):

    #     """
    #     Applying sinus function to the data.
    #     """
    #     for electrode in eeg_data.T:
    #         sin_data = []
    #         for value in electrode:
    #             sin_data.append(math.sin(value))
    #         eeg_data = np.column_stack((eeg_data, sin_data))
    #     return eeg_data
