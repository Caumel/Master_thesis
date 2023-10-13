import random
import polars as pl
import os
import utils

import shutil

range_cut = 96
url_data = "../data"
os.chdir('../')

years = ["2000","2001","2002","2003","2004","2005","2006","2007","2008","2009",\
         "2010","2011","2012","2013","2014","2015","2016","2017","2018","2019","2020"]
kind = "_summer"
path_read = os.path.join(url_data,"final_files",f"normal{kind}")

ruta_original = os.path.join(url_data,"final_files",f"normal_15_30{kind}_reducido",f"moderate.csv")
ruta_destino = os.path.join(url_data,"final_files",f"normal_15_30{kind}_equal",f"moderate.csv")
shutil.copy2(ruta_original, ruta_destino)

ruta_original = os.path.join(url_data,"final_files",f"normal_15_30{kind}_reducido",f"low.csv")
ruta_destino = os.path.join(url_data,"final_files",f"normal_15_30{kind}_equal",f"low.csv")
shutil.copy2(ruta_original, ruta_destino)

size_wish = 3179

file = "high.csv"
df = pl.read_csv(os.path.join(path_read,file))
years = list(df["year"].unique())
print(len(years))
n_to_take = int(size_wish/len(years)/38) + 2 # El 7 es por que quedaban no suficientes valores
print(n_to_take)
new_df = pl.DataFrame()
for index,year in enumerate(years):
    print(year)
    df_year = df.filter(pl.col("year") == year)
    for windfarm in range(0,38):
        # print("windfarm nº",windfarm)
        df_new = df_year.filter(pl.col("index") == windfarm)
        n_events = df_new["n_event"].unique().shape[0]
        random_indices = random.sample(range(1,n_events+1), min(n_events,n_to_take))
        df_selected = df_new.filter(df_new['n_event'].is_in(random_indices))
        new_df = pl.concat([new_df,df_selected], rechunk=True)
new_df.write_csv(os.path.join(url_data,"final_files",f"normal_15_30{kind}_equal",file))
print(new_df.shape[0]/96)

