import os
import shutil

"""
Separating files with coefficients based on the response of a subject.
"""

source_folder = r"/media/data/lazarenkom98dm/coeffs-second-visit/"

destination_folder_1 = r"/media/data/lazarenkom98dm/coeffs-second-visit-responders/"
destination_folder_2 = r"/media/data/lazarenkom98dm/coeffs-second-visit-non-responders/"

for file_name in os.listdir(source_folder):
    source = source_folder + file_name

    if file_name[-5] == "1":
        destination_1 = destination_folder_1 + file_name

        shutil.copy(source, destination_1)
        print('copied', file_name)
    elif file_name[-5] == "0":
        destination_2 = destination_folder_2 + file_name

        shutil.copy(source, destination_2)
        print('copied', file_name)
