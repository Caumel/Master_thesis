import random
import polars as pl
import os

range_cut = 96
url_data = "../data"
os.chdir('../')

years = ["2000","2001","2002","2003","2004","2005","2006","2007","2008","2009",\
         "2010","2011","2012","2013","2014","2015","2016","2017","2018","2019","2020"]
split = ""
path_read = os.path.join(url_data,"dataset_split_events_rare")
n_high = [13512,9662, 10976, 12547, 13456, 7167, 13842, 15192, 22805, 15679, 18378, 12802, 10960, 24988, 14141,9412,13178, 16719, 1116,20293, 20965]

kinds = ["moderate","low"]

for index,year in enumerate(years):
    print(year)
    for kind in kinds:
        n_to_take = int(n_high[index]/38)
        df = pl.read_csv(os.path.join(path_read,f"{year}_{range_cut}_{kind}{split}.csv" ))
        new_df = pl.DataFrame()
        for windfarm in range(0,38):
            # print("windfarm nº",windfarm)
            df_new = df.filter(pl.col("index") == windfarm)
            n_events = df_new["n_event"].unique().shape[0]
            random_indices = random.sample(range(1,n_events+1), min(n_events,n_to_take))
            df_selected = df_new.filter(df_new['n_event'].is_in(random_indices))
            new_df = pl.concat([new_df,df_selected], rechunk=True)
        new_df.write_csv(os.path.join(url_data,"cut_files",f"{year}_{range_cut}_{kind}_cut.csv"))


