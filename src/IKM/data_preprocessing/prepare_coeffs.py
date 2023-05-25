import collections.abc
import os

import numpy as np


def coeff_file_transform(path, delimiter):
    """
    Transforming the files with coefficients to be able to use them for clustering.
    """
    final_responded = []
    final_responded = np.asarray(final_responded)
    final_non_responded = []
    final_non_responded = np.asarray(final_non_responded)
    for filename in os.listdir(path):
        if filename[-9] == '1':
            if final_responded.size != 0:

                final_responded = np.concatenate(
                    (final_responded, np.loadtxt(os.path.join(path, filename), delimiter=delimiter)), axis=0)

            else:
                final_responded = np.loadtxt(os.path.join(path, filename), delimiter=delimiter)
        elif filename[-9] == '0':
            if final_non_responded.size != 0:

                final_non_responded = np.concatenate(
                    (final_responded, np.loadtxt(os.path.join(path, filename), delimiter=delimiter)), axis=0)

            else:
                final_non_responded = np.loadtxt(os.path.join(path, filename), delimiter=delimiter)

    np.savetxt(
        fr"/media/data/lazarenkom98dm/final_responded",
        final_responded, fmt='%.3f')
    np.savetxt(
        fr"/media/data/lazarenkom98dm/final_non_responded",
        final_non_responded, fmt='%.3f')


path = "/media/data/lazarenkom98dm/objects-second-visit-full"

coeff_file_transform(path, delimiter='\t')
