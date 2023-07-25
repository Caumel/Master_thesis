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

years = ["2000","2001","2002","2003","2004","2005","2006","2007","2008","2009",\
         "2010","2011","2012","2013","2014","2015","2016","2017","2018","2019","2020"]

range_cut = 96
path_save = os.path.join(url_data,"dataset_split_events_10")

for year in years:
    print(year)
    df = pl.read_csv(os.path.join(url_data,f"data_{year}.csv"))
    utils.get_datasets(df, range_cut, path_save, year,percentage=0.1, split_sw = True)

for year in years:
    print(year)
    df = pl.read_csv(os.path.join(url_data,f"data_{year}.csv"))
    utils.get_datasets(df, range_cut, path_save, year, percentage = 0.1)
