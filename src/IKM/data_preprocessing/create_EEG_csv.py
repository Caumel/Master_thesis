import os

import numpy as np
import pandas as pd
import h5py
from pymatreader import read_mat

# OBJECT_LENGTH = 135001
# OBJECT_DIM = 19

path_mat = r''
path_csv = r''
path_dir = r''


def create_eeg_csv_cubes():
    """
    Generating the csv file out of EEG of depressed patients with precomputed cubes.
    """
    data_csv = pd.read_csv(path_csv, encoding="utf-8")
    mat = read_mat(path_mat)

    objects = {"id": [], "response": [], "compData": [], "quadrs": [], "age": [], "sex": [], "srate": [],
               "treatment_1": [], "treatment_2": [], "treatment_3": [], "treatment_4": [], "visit": []}
    counter = 0

    for row in range(len(mat['data']['id'])):
        if mat['data']['visit'][row] == 2:
            EEG_data = mat['data']['EEG_prepro'][row]['data'].transpose()
            objects["id"].append(data_csv.iloc[counter]['id'])
            objects["response"].append(data_csv.iloc[counter]['RESP_MATLAB'])
            objects["age"].append(data_csv.iloc[counter]['age'])
            objects["sex"].append(data_csv.iloc[counter]['sex'])
            objects["srate"].append(data_csv.iloc[counter]['original_srate'])
            objects["visit"].append(2)
            objects["treatment_1"].append(data_csv.iloc[counter]['treatment_1'])
            objects["treatment_2"].append(data_csv.iloc[counter]['treatment_2'])
            objects["treatment_3"].append(data_csv.iloc[counter]['treatment_3'])
            objects["treatment_4"].append(data_csv.iloc[counter]['treatment_4'])

            m = np.size(EEG_data, 0)
            d = np.size(EEG_data, 1)

            comp_data = compute_cubes(m, d, EEG_data)
            quadrs = compute_AtA(m, d, EEG_data)
            objects["compData"].append(comp_data)
            objects["quadrs"].append(quadrs)
            print(f"Object {objects['id'][counter]} has been created\n")
            counter += 1

    print("Objects have been created\n")

    objects = pd.DataFrame(objects)
    objects.to_csv("depression_patients.csv", index=False)

    print("Objects have been saved\n")


def create_eeg_csv():
    """
    Generating the csv file out of EEG of depressed patients.
    """
    data_csv = pd.read_csv(path_csv, encoding="utf-8")
    mat = read_mat(path_mat)

    objects = {"id": [], "response": [], "data": [], "age": [], "sex": [], "original_srate": [],
               "treatment_1": [], "treatment_2": [], "treatment_3": [], "treatment_4": [], "visit": []}
    counter = 0

    for row in range(len(mat['data']['id'])):
        if mat['data']['visit'][row] == 2:
            EEG_data = mat['data']['EEG_prepro'][row]['data'].transpose()
            objects["id"].append(data_csv.iloc[counter]['id'])
            objects["response"].append(data_csv.iloc[counter]['RESP_MATLAB'])
            objects["age"].append(data_csv.iloc[counter]['age'])
            objects["sex"].append(data_csv.iloc[counter]['sex'])
            objects["original_srate"].append(data_csv.iloc[counter]['original_srate'])
            objects["visit"].append(2)
            objects["treatment_1"].append(data_csv.iloc[counter]['treatment_1'])
            objects["treatment_2"].append(data_csv.iloc[counter]['treatment_2'])
            objects["treatment_3"].append(data_csv.iloc[counter]['treatment_3'])
            objects["treatment_4"].append(data_csv.iloc[counter]['treatment_4'])

            objects["data"].append(EEG_data)

            print(f"Object {objects['id'][counter]} has been created\n")
            counter += 1

    print("Objects have been created\n")

    objects = pd.DataFrame(objects)
    objects.to_csv("depression_patients_raw.csv", index=False)

    print("Objects have been saved\n")


def compute_cubes(m, d, data):
    """
    Computes V^tA.
    """
    comp_data = np.zeros((d, d))
    for i in range(d):
        for j in range(i, d):
            tmp = 0
            for h in range(m):
                tmp += data[h][i] * data[h][j]

            comp_data[i][j] = tmp
            comp_data[j][i] = tmp

    return comp_data


def compute_AtA(m, d, data):
    """
    Computes V^tV.
    """
    quadrs = np.zeros((d, 1))
    for i in range(d):
        for j in range(m):
            tmp = data[j][i]
            tmp *= tmp
            tmp += quadrs[i][0]
            quadrs[i][0] = tmp

    return quadrs


create_eeg_csv()
