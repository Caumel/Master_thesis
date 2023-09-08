import os
import polars as pl

range_cut = 96
url_data = "../data"
os.chdir('../')

path = os.path.join(url_data,"cut_winter_13")

kinds = ["high","moderate","low"]

# years = ["2000","2001","2002","2003","2004","2005","2006","2007","2008","2009",\
#          "2010","2011","2012","2013","2014","2015","2016","2017","2018","2019","2020"]
# split = ""
# years_summer = ["2000","2001","2002","2009","2010","2013","2014","2017","2019","2020"]
# split = "_summer"
years_winter = ["2000","2001","2002","2003","2004","2006","2007","2008","2009",\
          "2010","2011","2012","2013","2014","2015","2016","2017","2019","2020"]
split = "_winter"
range_cut = 96


for kind in kinds:
    print(kind)
    new_df = pl.DataFrame()
    for index,year in enumerate(years_winter):
        print(year)
        if kind == "high":
            df = pl.read_csv(os.path.join(path,f"{year}_{range_cut}_13_{kind}{split}.csv"))
        else:
            df = pl.read_csv(os.path.join(path,f"{year}_{range_cut}_{kind}_cut.csv"))
        df = df.with_column(pl.lit(year).alias("year"))
        # print(df.columns)
        new_df = pl.concat([new_df,df], rechunk=True)
        # print(new_df.columns)

    new_df = new_df.select(['time', 'index', 'n_event', 'year', 'cc', 'o3', 'pv', 'cape', 'blh', 'd2m', 'z', 'relative_humidity', 't2m', 't100m', 't135m', 'wdir100m', 'wspeed135m', 'wspeed100m'])
    new_df.write_csv(os.path.join(url_data,f"final_files","normal_13_winter",f"{kind}.csv"))