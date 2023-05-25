import math
from collections import Counter
import numpy as np
import pandas as pd
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
    
    # def leave_parts_electrodes(self, data, left_right=None, front_back=None):
    #     """
    #     Leaving only specific dimensions (electrodes) in the data.
    #     """
    #     df = self.get_electrodes_data()
    #     indeces_for_deletion = self.find_ind_deletion(df, left_right, front_back)
    #     return self.remove_dimensions(data, indeces_for_deletion)
    
    def remove_dimensions(self, data, array_dimensions):
        """
        Removing chosen dimensions from the data.
        """
        data_without_dims = np.delete(data, array_dimensions, axis=1)
        return data_without_dims
    
    def get_electrodes_data(self):
        """
        Creating the data about the electrodes position.
        """

        electrodes_data = [['Fp1', 'left', 'front', 0], ['Fp2', 'right', 'front', 1], ['F3', 'left', 'front', 2],
                           ['F4', 'right', 'front', 3], ['C3', 'left', 'center', 4], ['C4', 'right', 'center', 5],
                           ['P3', 'left', 'back', 6], ['P4', 'right', 'back', 7], ['O1', 'left', 'back', 8],
                           ['O2', 'right', 'back', 9], ['F7', 'left', 'front', 10], ['F8', 'right', 'front', 11],
                           ['T3', 'left', 'center', 12], ['T4', 'right', 'center', 13], ['T5', 'left', 'back', 14],
                           ['T6', 'right', 'back', 15], ['Fz', 'center', 'front', 16], ['Cz', 'center', 'center', 17],
                           ['Pz', 'center', 'back', 18]]

        df = pd.DataFrame(electrodes_data, columns=['Channel', 'Left-right position', 'Front-back position', 'Index'])
        return df
    
    def find_ind_deletion(self, df, left_right=None, front_back=None):
        """
        Finding the indexes for dimensions delection.
        """
        to_delete_df = pd.DataFrame()
        if left_right is not None:
            to_delete_df = df[(df['Left-right position'] == left_right)]

        if front_back is not None:
            to_delete_df = df[(df['Front-back position'] == front_back)]

        return np.array(to_delete_df['Index'].tolist())

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

    def box_cox_transform(self, data):

        """
        Applies Box-Cox transformation.
        """

        dimensions = np.size(data, 1)
        data = DataPreprocessor.shift_to_positive(self, data)
        for col in range(dimensions):
            data[:, col], lambda_box_cox = stats.boxcox(data[:, col])
        return data

    def shift_to_positive(self, x):
        """
        Shift all numbers in an array to positive numbers.
        """
        min_value = np.min(x)
        if min_value > 0:
            return x
        shift_value = np.abs(min_value) + 1
        return x + shift_value

    def dwt(self, data, waveletname):
        """
        Applies Discrete Wavelet Transform.
        """
        coeffs = []
        for electrode in data.T:
            cA, cD = pywt.dwt(electrode, waveletname, 'per')
            coeffs.append(cD)
        coeffs = np.stack(coeffs, axis=1)
        return coeffs

    def multi_level_dwt(self, data):
        """
        Applies multi-level Discrete Wavelet Transform.
        """
        coeffs = 0
        for col in range(data.ndim + 1):
            coeffs = pywt.wavedec(data[:, col], 'bior6.8', mode='sym', level=2)
        return coeffs

    def z_score(self, data):
        """
        Applies z-score transformation.
        """
        return stats.zscore(data, axis=None)

    def z_normalize(self, data):
        """
        Applies z-normalization.
        """
        scaler = StandardScaler()
        scaler.fit(data)
        z_normalized_data = scaler.transform(data)
        return z_normalized_data

    def z_transform(self, data, z):
        """
        Applies Z transform.
        """

        d = np.size(data, 1)
        m = np.size(data, 0)
        z_transformed = np.zeros((m, d), dtype=complex)

        for dim in range(d):
            for point in range(m):
                z_transformed[point][dim] = data[point][dim] * z ** (-point)

        # Compute the magnitude of the z-transform
        magnitude = np.abs(z_transformed)

        # Compute the phase of the z-transform
        phase = np.angle(z_transformed)

        return z_transformed, magnitude, phase

    def calculate_entropy(self, list_values):
        """
        Calculates entropy.
        """

        counter_values = Counter(list_values).most_common()
        probabilities = [elem[1] / len(list_values) for elem in counter_values]
        entropy = stats.entropy(probabilities)
        return entropy

    def calculate_statistics(self, list_values):
        """
        Calculates different statistics such as mean, standard deviation, variance and others.
        """
        n5 = np.nanpercentile(list_values, 5)
        n25 = np.nanpercentile(list_values, 25)
        n75 = np.nanpercentile(list_values, 75)
        n95 = np.nanpercentile(list_values, 95)
        median = np.nanpercentile(list_values, 50)
        mean = np.nanmean(list_values)
        std = np.nanstd(list_values)
        var = np.nanvar(list_values)
        rms = np.nanmean(np.sqrt(list_values ** 2))
        return [n5, n25, n75, n95, median, mean, std, var, rms]

    def calculate_crossings(self, list_values):
        """
        Calculates crossings.
        """
        zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
        no_zero_crossings = len(zero_crossing_indices)
        mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
        no_mean_crossings = len(mean_crossing_indices)
        return [no_zero_crossings, no_mean_crossings]

    def get_features(self, list_values):

        """
        Gets entropy, crossings, and statistics.
        """
        entropy = self.calculate_entropy(list_values)
        crossings = self.calculate_crossings(list_values)
        statistics = self.calculate_statistics(list_values)
        return [entropy] + crossings + statistics

    def get_eeg_features(self, eeg_data, waveletname, level):

        """
        Get the statistics from EEG data.
        """
        list_features = []
        for electrode in eeg_data.T:
            list_coeff = pywt.wavedec(electrode, waveletname, level=level)
            features = []
            for coeff in list_coeff:
                features += self.get_features(coeff)
            list_features.append(features)
        list_features = np.stack(list_features, axis=1)
        return list_features

    def delete_every_nth_el(self, eeg_data, n):

        """
        Deletes every n element in all dimensions of the data.
        """

        shortened_eeg_data = []
        for i, electrode in enumerate(eeg_data.T):
            shortened_eeg_data.append(np.delete(electrode, np.arange(n - 1, electrode.size, n)))

        shortened_eeg_data = np.stack(shortened_eeg_data, axis=1)
        return shortened_eeg_data

    def leave_every_nth_el(self, eeg_data, n):

        """
        Leaves every n element in all dimensions of the data.
        """

        shortened_eeg_data = []
        for i, electrode in enumerate(eeg_data.T):
            shortened_eeg_data.append(electrode[n - 1::n])

        shortened_eeg_data = np.stack(shortened_eeg_data, axis=1)
        return shortened_eeg_data

    # EEG bands extraction using FFT method
    def fft_eeg_bands_extraction(self, eeg_data):

        """
        Extracts EEG bands using Fast Fourier Transform.
        """
        fs = 250  # Sampling rate (250 Hz)

        # Get real amplitudes of FFT (only in postive frequencies)
        fft_vals = np.absolute(np.fft.rfft(eeg_data))

        # Get frequencies for amplitudes in Hz
        fft_freq = np.fft.rfftfreq(len(eeg_data), 1.0 / fs)

        # Define EEG bands
        eeg_bands = {'Delta': (0, 4),
                     'Theta': (4, 8),
                     'Alpha': (8, 12),
                     'Beta': (12, 30),
                     'Gamma': (30, 45)}

        eeg_band_fft = dict()

        for band in eeg_bands:
            freq_ix = np.where((fft_freq >= eeg_bands[band][0]) &
                               (fft_freq <= eeg_bands[band][1]))[0]
            eeg_band_fft[band] = fft_vals[freq_ix]

        return eeg_band_fft

    # Theta and alpha bands using FFT method
    def theta_alpha_extraction(self, eeg_data):

        """
        Extracts EEG theta and alpha bands using Fast Fourier Transform and concatenates them.
        """

        eeg_band_fft = self.fft_eeg_bands_extraction(eeg_data)

        theta_alpha_data = []

        for band in eeg_band_fft:

            if band == 'Theta' or band == 'Alpha':
                theta_alpha_data.append(eeg_band_fft[band])

        theta_alpha_data = np.concatenate((theta_alpha_data[0], theta_alpha_data[1]), axis=0)
        return theta_alpha_data

    def butter_bandpass(self, lowcut, highcut, fs, order=5):

        """
        Computes second order sections for Butterworth bandpass filter.
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], btype='band', output='sos')
        return sos

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        """
        Applying Butterworth bandpass filter.
        """

        sos = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y

    def butter_bandstop(self, lowcut, highcut, fs, order=5):
        """
        Computes second order sections for Butterworth bandstop filter.
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], btype='bandstop', analog=False, output='sos')
        return sos

    def butter_bandstop_filter(self, data, lowcut, highcut, fs, order=5):
        """
        Computes second order sections for Butterworth bandstop filter.
        """
        sos = self.butter_bandstop(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y

    def EEG_bandstop(self, signal, fs=1024):
        """
        Applying Butterworth bandstop filter. Filters out 54-66 Hz noise
        """
        filtered_EEG = self.butter_bandstop_filter(signal, 54, 66, fs)
        return filtered_EEG

    def butter_eeg_bands_extraction(self, data, fs, band='delta'):
        """
        Extracting bands using Butterworth filter.
        """
        if band == 'delta':
            delta = self.butter_bandpass_filter(data, 0.5, 4, fs)
            return delta

        elif band == 'theta':

            theta = self.butter_bandpass_filter(data, 4, 7, fs)
            return theta
        elif band == 'alpha':
            alpha = self.butter_bandpass_filter(data, 8, 13, fs)

            return alpha
        elif band == 'beta_1':
            beta_1 = self.butter_bandpass_filter(data, 14, 24, fs)

            return beta_1
        elif band == 'beta_2':
            beta_2 = self.butter_bandpass_filter(data, 25, 35, fs)
            return beta_2
        elif band == 'beta':
            beta = self.butter_bandpass_filter(data, 14, 30, fs)
            return beta

        elif band == 'gamma_1':
            gamma_1 = self.butter_bandpass_filter(data, 36, 58, fs)

            return gamma_1
        elif band == 'gamma_2':
            gamma_2 = self.butter_bandpass_filter(data, 62, 100, fs)

            return gamma_2
        elif band == 'gamma':
            gamma = self.butter_bandpass_filter(data, 31, 100, fs)
            return gamma

    def hilbert_amplitude(self, data):

        """
        Computing Hilbert amplitude.
        """
        z = hilbert(data)
        inst_amplitude = np.abs(z)
        return inst_amplitude

    def hilbert_phase(self, data):
        """
        Computing Hilbert phase.
        """
        z = hilbert(data)
        inst_phase = np.unwrap(np.angle(z))
        return inst_phase

    def sin(self, eeg_data):

        """
        Applying sinus function to the data.
        """
        for electrode in eeg_data.T:
            sin_data = []
            for value in electrode:
                sin_data.append(math.sin(value))
            eeg_data = np.column_stack((eeg_data, sin_data))
        return eeg_data

    def leave_one_third_part(self, eeg_data, mode=None):

        """
        Leaving only one-third part of the data.
        Parameters:
            mode (string): can be start, middle, finish
        """
        m = np.size(eeg_data, 0)
        one_third_data = int(m / 3)
        two_third_data = int((2 * m) / 3)
        if mode == 'middle':
            eeg_data = eeg_data[one_third_data:two_third_data, :]
        elif mode == 'start':
            eeg_data = eeg_data[:one_third_data, :]
        elif mode == 'finish':
            eeg_data = eeg_data[two_third_data:, :]
        return eeg_data

    def choose_parts_electrodes(self, data, electrodes):
        """
        This function removes dimensions corresponding to the electrodes.
        Parameters:
            data (array-like): EEG data
            electrodes (array of strings): Electrodes to be left in the data
        """
        df = self.get_electrodes_data()
        df_reduced = df.loc[~df['Channel'].isin(electrodes)]
        to_delete = df_reduced['Index'].tolist()
        data_reduced = self.remove_dimensions(data, to_delete)
        return data_reduced
