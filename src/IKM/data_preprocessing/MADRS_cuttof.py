import math

import pandas as pd

df_main = pd.read_csv(r'..\datasets\depression_patients.csv')
df_madrs = pd.read_csv(r'..\datasets\Clinics_MADRS_clean_wo_descr.csv', encoding='utf-8')

df_madrs = df_madrs[['id', 'M_4', 'M_F']]
df_madrs = df_madrs.dropna(how='all', subset=['M_F', 'M_4'])
df_result = pd.merge(df_main, df_madrs, on='id', how='inner')


def madrs_cutoff_column(row):
    """
    Selects either final or fourth measured MADRS score based on their availability in the data.
    """

    if math.isnan(row['M_4']):
        return get_madrs_cutoff(row['M_F'])
    else:
        return get_madrs_cutoff(row['M_4'])


def get_madrs_cutoff(score):
    """
    Assigns MADRS into classes based on its score.
    """
    if score in range(0, 7):
        return 'normal'
    elif score in range(7, 20):
        return 'mild depression'
    elif score in range(20, 35):
        return 'moderate depression'
    elif score > 34:
        return 'severe depression'


df_result['madrs_cuttof'] = df_result.apply(madrs_cutoff_column, axis=1)
df_result = df_result.drop(columns=['M_4', 'M_F'])
df_result.to_csv(r"..\datasets\depression_patients_madrs.csv", index=False)
