import numpy as np
from matplotlib import pyplot as plt

from clustering.ikm.utils.data_preprocessor import DataPreprocessor

"""
Visualizing the bands obtained from the data for a specific subject.
"""

data_preprocessor = DataPreprocessor()
path = r""

data = np.loadtxt(path, delimiter='\t')
time = np.arange(1000)
data = data[:1000, 0]

fig, axs = plt.subplots(6, sharex=True)
fig.suptitle('Subject 2. Second visit')

axs[0].set_title('Original')
axs[0].plot(data)

bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']

fs = 250

for i in range(len(axs)):
    if i == (len(axs) - 1):
        break
    data_bands = data_preprocessor.butter_eeg_bands_extraction(data, fs, bands[i])
    axs[i + 1].set_title(bands[i].capitalize())
    axs[i + 1].plot(data_bands)

# data_theta = data_preprocessor.butter_eeg_bands_extraction(data, fs, 'delta')
# axs[2].set_title('Theta')
# axs[2].plot(data_theta)
#
# data_theta = data_preprocessor.butter_eeg_bands_extraction(data, fs, 'theta')
# axs[2].set_title('Theta')
# axs[2].plot(data_theta)
#
# data_alpha = data_preprocessor.butter_eeg_bands_extraction(data, fs, 'alpha')
# axs[3].set_title('Alpha')
# axs[3].plot(data_alpha)
#
# data_beta = data_preprocessor.butter_eeg_bands_extraction(data, fs, 'beta')
# axs[4].set_title('Beta')
# axs[4].plot(data_beta)
#
# data_gamma = data_preprocessor.butter_eeg_bands_extraction(data, fs, 'gamma')
# axs[5].set_title('Gamma')
# axs[5].plot(data_gamma)

plt.show()
