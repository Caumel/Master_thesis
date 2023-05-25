import os

import numpy as np

"""
Extracts the data from the EEG of alcoholics in the suitable for IKM format.
"""

SKIP_LINES = 5
counter = 0
index = 0
DIMENSIONS = 64

for directory in os.listdir(r""):
    for file in os.listdir(
            fr"\{directory}"):
        f = open(fr"\{directory}\{file}",
                 "r")
        array = [[] for i in range(DIMENSIONS)]
        for line in f:
            if counter < SKIP_LINES:
                counter += 1
                continue
            if line.startswith("#"):
                index += 1
                continue
            array[index].append(float(line.split(" ")[3].strip()))
        converted_array = np.c_[array].transpose()
        np.savetxt(
            fr"\{file}",
            converted_array)
        f.close()
        print(f"File {file} is preprocessed.\n")
        counter = 0
        index = 0
