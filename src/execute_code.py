import polars as pl
import pandas as pd
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
plt.style.use('ggplot')

import numpy as np
import utils
import random

url_data = "./data"

os.chdir('../')

speed_high = 15
speed_low = 2
speed_moderate_down = 6
speed_moderate_up = 8
utils.set_speed(speed_high,speed_low,speed_moderate_down,speed_moderate_up)

# years = ["2000","2001","2002","2003","2004","2005","2006","2007","2008","2009",\
#          "2010","2011","2012","2013","2014","2015","2016","2017","2018","2019","2020"]

# range_cut = 96
# path_save = os.path.join(url_data,"dataset_split_events")

# for year in years:
#     print(year)
#     df = pl.read_csv(os.path.join(url_data,f"data_{year}.csv"))
#     utils.get_datasets(df, range_cut, path_save, year)
#     # print(utils.count_event_per_file(os.path.join(url_data,"dataset_split_events",f"{year}_{range_cut}_15_high.csv"))['events'].sum())
#     # print(utils.count_event_per_file(os.path.join(url_data,"dataset_split_events",f"{year}_{range_cut}_moderate.csv"))['events'].sum())
#     # print(utils.count_event_per_file(os.path.join(url_data,"dataset_split_events",f"{year}_{range_cut}_low.csv"))['events'].sum())



range_cut = 96

years = ["2000"]#,"2006","2009","2010","2013","2014","2017"]
kinds = ["moderate","low"]
path_read = os.path.join(url_data,"dataset_split_events")

amount = [2385]#,3154,2204,3876,6387,3542,6037]

for index,year in enumerate(years):
    print(year)
    for kind in kinds:
        n_to_take = int(amount[index]/38)
        df = pl.read_csv(os.path.join(path_read,f"{year}_{range_cut}_{kind}.csv"))
        new_df = pl.DataFrame()
        for windfarm in range(0,38):
            print("windfarm nº",windfarm)
            df_new = df.filter(pl.col("index") == windfarm)
            n_events = df_new["n_event"].unique().shape[0]
            random_indices = random.sample(range(1,n_events+1), n_to_take)
            df_selected = df_new.filter(df_new['n_event'].is_in(random_indices))
            new_df = pl.concat([new_df,df_selected], rechunk=True)
        new_df.write_csv(os.path.join(url_data,"cut_dataset",f"{year}_{range_cut}_{kind}_cut.csv"))
