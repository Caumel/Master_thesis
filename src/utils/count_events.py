import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np
import utils

url_data = "../../data"

print(os.listdir(url_data))

range_cut = 96
path_save = os.path.join(url_data,"dataset_split_events")

# years = ["2000","2001","2002","2003","2004","2005","2006","2007","2008","2009",\
#          "2010","2011","2012","2013","2014","2015","2016","2017","2018","2019","2020"]

# for year in years:
#     try:
#         print(year, "high", utils.count_event_per_file(os.path.join(url_data,"dataset_split_events",f"{year}_{range_cut}_15_high.csv"))['events'].sum())
#     except Exception as e:
#         print(year, "high", e)
#     try:
#         print(year, "low", utils.count_event_per_file(os.path.join(url_data,"dataset_split_events",f"{year}_{range_cut}_low.csv"))['events'].sum())
#     except Exception as e:
#         print(year, "high", e)
#     try:
#         print(year, "moderate", utils.count_event_per_file(os.path.join(url_data,"dataset_split_events",f"{year}_{range_cut}_moderate.csv"))['events'].sum())
#     except Exception as e:
#         print(year, "high", e)

print()
name_folder = "normal_15_30_summer_equal"
for file in os.listdir(os.path.join(url_data,"final_files",name_folder)):
    utils.count_event_per_file(os.path.join(url_data,"final_files",name_folder,file),file)['events'].sum()
