import polars as pl
import os
from datetime import datetime

ini_winder = datetime.strptime("2000-01-01 00:00:00",'%Y-%m-%d %H:%M:%S')
end_winter = datetime.strptime("2000-06-30 23:59:59",'%Y-%m-%d %H:%M:%S')
ini_summer = datetime.strptime("2000-07-01 00:00:00",'%Y-%m-%d %H:%M:%S')
end_summer = datetime.strptime("2000-12-31 23:59:59",'%Y-%m-%d %H:%M:%S')

md_ini_winter = (ini_winder.month, ini_winder.day)
md_end_winter = (end_winter.month, end_winter.day)
md_ini_summer = (ini_summer.month, ini_summer.day)
md_end_summer = (end_summer.month, end_summer.day)

for file in os.listdir("../data/dataset_split_events_15/"):
    df = pl.read_csv("../data/dataset_split_events_15/" + file)

    cut_size = 96

    middle = int(cut_size/2)

    winter_df = pl.DataFrame()
    summer_df = pl.DataFrame()

    for windfarm in range(0,38):
        # print("windfarm nº",windfarm)
        df_new = df.filter(pl.col("index") == windfarm)
        n_events = df_new["n_event"].unique().shape[0]
        # Compute number of rows in summer and in winter for the events in the middle

        count_winter = 0
        count_summer = 0

        for event in range(0,n_events):
            df_new_event = df_new.filter(pl.col("n_event") == event)
            time_split = datetime.strptime(df_new_event[middle,0],'%Y-%m-%d %H:%M:%S')
            date_md = (time_split.month,time_split.day)

            if md_ini_winter <= date_md <= md_end_winter:
                df_new_event = df_new_event.with_column(pl.lit(count_winter).alias("n_event"))
                winter_df = pl.concat([winter_df,df_new_event], rechunk=True)
                count_winter += 1 
            else:
                df_new_event = df_new_event.with_column(pl.lit(count_summer).alias("n_event"))
                summer_df = pl.concat([summer_df,df_new_event], rechunk=True)
                count_summer += 1 

    summer_df.write_csv(os.path.join(f"../data/dataset_summer_split_15/",f"{file[:-4]}summer.csv"))
    winter_df.write_csv(os.path.join(f"../data/dataset_winter_split_15/",f"{file[:-4]}winter.csv"))