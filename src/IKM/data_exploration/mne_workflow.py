import os

import matplotlib
import mne
import numpy as np

matplotlib.use('Qt5Agg')

from matplotlib import animation, pyplot as plt


def create_evoked(path, info):
    """
    Creating the evoked array.
    """
    data = np.loadtxt(os.path.join(path), delimiter='\t').transpose()
    data = data * 10 ** (-6)
    data_mean = data.mean(axis=0)
    data_mean.reshape(1, -1)
    # Create the Evoked object
    evoked_array = mne.EvokedArray(data, info, tmin=-0.5,
                                   nave=data.shape[0], comment='simulated')
    return evoked_array


def create_info(sampling_freq, ch_names):
    """
    Creating the info object.
    """
    info = mne.create_info(ch_names, sfreq=sampling_freq, ch_types='eeg')
    info.set_montage('standard_1020')
    return info


def create_animation(path, animation_name, ch_names, sampling_freq=250):
    """
    Creating the topomap animation. samling_freq parameter is in Hertz.
    """
    info = create_info(sampling_freq, ch_names)
    evoked_array = create_evoked(path, info)

    times = np.arange(1, 500, 10)
    topo, anim = evoked_array.animate_topomap(
        times=times, ch_type='eeg', frame_rate=2, time_unit='s', blit=False)

    f = fr"./{animation_name}.mp4"
    writer_gif = animation.FFMpegWriter(fps=2)
    anim.save(f, writer=writer_gif)


def animate_alcohol_data(ch_names):
    """
    Creating the topomap animation for the alcoholics EEG dataset.
    """
    info = mne.create_info(ch_names, sfreq=sampling_freq, ch_types='eeg')
    info.set_montage('standard_1020', on_missing='ignore', match_case=False)

    delete_dimensions = [31, 62, 63]
    path = r""

    data = np.loadtxt(os.path.join(path), delimiter=' ')
    data = np.delete(data, delete_dimensions, axis=1)
    data = data * 10 ** (-6)
    data_mean = data.mean(axis=0)
    data_mean.reshape(1, -1)
    # Create the Evoked object
    evoked_array = mne.EvokedArray(data.transpose(), info, tmin=-0.5,
                                   nave=data.shape[0], comment='simulated')

    animation_name = 'alcohol'

    # times = np.arange(1, 500, 10)
    topo, anim = evoked_array.animate_topomap(ch_type='eeg', frame_rate=1, time_unit='s', blit=False)

    f = fr"./{animation_name}.mp4"
    writer_gif = animation.FFMpegWriter(fps=2)
    anim.save(f, writer=writer_gif)


sampling_freq = 250  # in Hertz
ch_names = ['Fp1',
            'Fp2',
            'F3',
            'F4',
            'C3',
            'C4',
            'P3',
            'P4',
            'O1',
            'O2',
            'F7',
            'F8',
            'T3',
            'T4',
            'T5',
            'T6',
            'Fz',
            'Cz',
            'Pz', ]

path1 = r""
path = r""

create_animation(path, 'second-visit-subj113-responded-25-male', ch_names)
ch_names = ['Fp1',
            'Fp2',
            'F7',
            'F8',
            'AF1',
            'AF2',
            'FZ',
            'F4',
            'F3',
            'FC6',
            'FC5',
            'FC2',
            'FC1',
            'T8',
            'T7',
            'CZ',
            'C3',
            'C4',
            'CP5',
            'CP6',
            'CP1',
            'CP2',
            'P3',
            'P4',
            'PZ',
            'P8',
            'P7',
            'PO2',
            'PO1',
            'O1',
            'O2',
            # 'X',
            'AF7',
            'AF8',
            'F5',
            'F6',
            'FT7',
            'FT8',
            'FPZ',
            'FC4',
            'FC3',
            'C6',
            'C5',
            'F2',
            'F1',
            'TP8',
            'TP7',
            'AFZ',
            'CP3',
            'CP4',
            'P5',
            'P6',
            'C1',
            'C2',
            'PO7',
            'PO8',
            'FCZ',
            'POZ',
            'OZ',
            'P2',
            'P1',
            'CPZ',
            # 'nd',
            # 'Y',
            ]
animate_alcohol_data(ch_names)
