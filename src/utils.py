####################################
#                                  #
#                                  #
#                                  #
####################################

# Imports

import polars as pl
import pandas as pd
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
plt.style.use('ggplot')

import numpy as np

speed_high = 0
speed_low = 0
speed_moderate_down = 0
speed_moderate_up = 0

ini_winder = datetime.strptime("2000-01-01 00:00:00",'%Y-%m-%d %H:%M:%S')
end_winter = datetime.strptime("2000-06-30 23:59:59",'%Y-%m-%d %H:%M:%S')
ini_summer = datetime.strptime("2000-07-01 00:00:00",'%Y-%m-%d %H:%M:%S')
end_summer = datetime.strptime("2000-12-31 23:59:59",'%Y-%m-%d %H:%M:%S')

md_ini_winter = (ini_winder.month, ini_winder.day)
md_end_winter = (end_winter.month, end_winter.day)
md_ini_summer = (ini_summer.month, ini_summer.day)
md_end_summer = (end_summer.month, end_summer.day)

def set_speed(speed_high_new,speed_low_new,speed_moderate_down_new,speed_moderate_up_new):
    global speed_high
    global speed_low
    global speed_moderate_down
    global speed_moderate_up

    speed_high = speed_high_new
    speed_low = speed_low_new
    speed_moderate_down = speed_moderate_down_new
    speed_moderate_up = speed_moderate_up_new

def get_speeds():
    print(speed_high)
    print(speed_low)
    print(speed_moderate_up)
    print(speed_moderate_down)

####################################
#                                  #
#            Join data             #
#                                  #
####################################

def read_data_year(type,year,url_data):
    if type == "pressure":
        path = "ERA5_downscaled_data_turbines_pressurebased_" + year + ".txt"
    else:
        path = "ERA5_downscaled_data_turbines_surfacebased_" + year + ".txt"
    df = pl.read_csv(os.path.join(url_data,"original",path), sep='\t')
    df = df.with_column(pl.col('time').str.strptime(pl.Datetime, fmt='%Y-%m-%d %H:%M:%S'))
    return df

def select_columns(df,columns):
    df_new = df.select(pl.col(columns))
    return df_new

def join_cape_info(df_original, df_cape):
    merged_df = df_original.join(df_cape, on='time', how='left')
    return merged_df

def join_pressure_surface(df_pressure,df_surface):
    merged_df = df_pressure.join(df_surface, on=['time','index'], how='left')
    return merged_df

def join_files(year,url_data,pressure_columns,surface_columns):
    df_pressure = read_data_year("pressure",year,url_data)
    df_surface = read_data_year("surface",year,url_data)

    df_pressure = select_columns(df_pressure,pressure_columns)
    df_surface = select_columns(df_surface,surface_columns)

    df_cape = pl.read_csv(os.path.join(url_data,"CAPE.csv"))
    df_cape = df_cape.with_column(pl.col('time').str.strptime(pl.Datetime, fmt='%Y-%m-%d %H:%M:%S'))

    start_date =  datetime(int(year), 1, 1, 0, 0, 0)
    end_date =  datetime(int(year) + 1, 1, 1, 0, 0, 0)
    cape_year = df_cape.filter((df_cape['time'] >= start_date) & (df_cape['time'] < end_date))

    df_pressure = join_cape_info(df_pressure,df_cape)

    df_total = join_pressure_surface(df_pressure,df_surface)

    # df_pressure.filter((pl.col("time") == datetime(2000, 1, 1, 0, 0, 0)) & (pl.col("index") == 0))

    df_total.write_csv(os.path.join(url_data,f"data_{year}.csv"), datetime_format='%Y-%m-%d %H:%M:%S')

# Split data

####################################
#                                  #
#           Split data             #
#                                  #
####################################

def to_datetime(df,column):
    try:
        df = df.with_column(pl.col(column).str.strptime(pl.Datetime, fmt='%Y-%m-%d %H:%M:%S'))
    except Exception as e:
        print(e)
    finally:
        return df
    
def prepare_data(df,presure):
    df_new = df.filter(pl.col("level") == presure)
    df_new = to_datetime(df_new,"time")
    return df_new
    
def prepare_data_per_presure(df):
    # Presure 900
    df_900 = prepare_data(df,900)
    df_925 = prepare_data(df,925)
    df_950 = prepare_data(df,950)
    df_975 = prepare_data(df,975)
    df_1000 = prepare_data(df,1000)

    return df_900, df_925, df_950, df_975, df_1000

def get_data_per_windfarm(df,n_windfarm):
    return df.filter(pl.col("index") == n_windfarm)

def get_list_colors(df):
    colors = []

    for i in range(len(df)):
        if df[i, 'wspeed135m'] >= float(speed_high):
            colors.append('red')
        elif df[i,'wspeed135m'] <= float(speed_low):
            colors.append('blue')
        elif (df[i, 'wspeed135m'] >= float(speed_moderate_down)) & (df[i, 'wspeed135m'] <= float(speed_moderate_up)):
            colors.append('yellow')
        else:
            colors.append('green')
            
    return colors

def get_extreme_events(df):

    df_high_speed = df.filter(df['wspeed135m'] >= float(speed_high))
    df_low_speed = df.filter(df['wspeed135m'] <= float(speed_low))
    df_moderate_speed = df.filter((df['wspeed135m'] >= float(speed_moderate_down)) & (df['wspeed135m'] <= float(speed_moderate_up)))

    return df_high_speed,df_low_speed,df_moderate_speed

def plot_date_range(df,start_date,end_date):
    df_datetime_filter = df.filter((df['time'] >= start_date) & (df['time'] <= end_date))

    # Set the colors for the regions based on the condition
    colors = get_list_colors(df_datetime_filter)

    # Plot the data
    plt.figure()
    fig, ax = plt.subplots(figsize=(18, 4), dpi=100)

    plt.scatter(df_datetime_filter['time'].dt.strftime('%d-%m %H:%M'), df_datetime_filter['wspeed135m'],color=colors)
    plt.plot(df_datetime_filter['time'].dt.strftime('%d-%m %H:%M'), df_datetime_filter['wspeed135m'],color="black")
    plt.xlabel('Time')
    plt.xticks(df_datetime_filter['time'].dt.strftime('%d-%m %H:%M')[::8], rotation=0)
    plt.ylim([0, 20])
    plt.ylabel('Wind Speed (m/s)')
    plt.title(f'Wind Speed from: {start_date} to: {end_date-timedelta(hours=1)}')

    high_patch = mpatches.Patch(color='red', label=f'High speed events >= {str(speed_high)} m / s', alpha=0.6)
    moderate_patch = mpatches.Patch(color='yellow', label=f'Moderate speed events  {str(speed_moderate_down)} m / s <= ws <= {str(speed_moderate_up)} m / s', alpha=0.6)
    low_patch = mpatches.Patch(color='blue', label=f'Low speed events <= {str(speed_low)} m / s', alpha=0.6)
    now_event_patch = mpatches.Patch(color='green', label='No event', alpha=0.6)


    # Customize the background color
    for i, color in enumerate(colors):
        ax.add_patch(Rectangle((0 + i, 0), 1, 20, facecolor=color, alpha=0.25))

    ax.axhline(y=2, color='blue', linestyle='--', alpha=0.6)
    ax.axhline(y=6, color='yellow', linestyle='--', alpha=0.6)
    ax.axhline(y=8, color='yellow', linestyle='--', alpha=0.6)
    ax.axhline(y=15, color='red', linestyle='--', alpha=0.6)

    plt.legend(handles=[high_patch,moderate_patch,low_patch, now_event_patch])


    # Display the plot
    plt.show()
    return df_datetime_filter


# Create dataset with extreme events

##################################################
#                                                #
#        Create dataset with extreme event       #
#                                                #
##################################################

# def filter_date_with_extremes(df,min_date,max_date,start_date,end_date,hours):

#     # Given start_data and end_date, take the slot fo range "date" from the dataframe, taking in care if pass the limits

#     # Add half the difference to the first datetime to find the middle datetime
#     middle_datetime = start_date + timedelta(hours=hours/2)

#     if start_date < min_date:

#         diff_time = int((min_date - start_date).total_seconds() / 3600)
#         start_date = min_date
#         end_date = middle_datetime + timedelta(hours= hours/2 + diff_time-1)
#         # Filter the dates to keep only those within the desired range
#         df_filter = df.filter((df['time'] > start_date - timedelta(hours=1)) & (df['time'] < end_date + timedelta(hours=1)))

#     elif end_date > max_date:
        
#         diff_time = int((end_date - max_date).total_seconds() / 3600)
#         start_date = middle_datetime - timedelta(hours= hours/2 + diff_time-1)
#         end_date = max_date

#         # Filter the dates to keep only those within the desired range
#         df_filter = df.filter((df['time'] > start_date - timedelta(hours=1)) & (df['time'] < end_date + timedelta(hours=1)))

#     else:
#         # Filter the dates to keep only those within the desired range
#         df_filter = df.filter((df['time'] >= start_date) & (df['time'] < end_date))

#     return df_filter

# def join_related_events(df_events):

#     """
#     Metodo para dado la lista de listas de eventos extremos individuales, si son consecutivos, se unen.
#     """

#     # Method given all the events, join the one that are consecutives

#     days = df_events["time"].to_list()
#     sublists = []
#     sublist = [days[0]]
#     for i in range(1, len(days)):
#         diff = days[i] - days[i-1]
#         if diff >= timedelta(0) and diff <= timedelta(hours=1):
#             sublist.append(days[i])
#         else:
#             sublists.append(sublist)
#             sublist = [days[i]]
#     sublists.append(sublist)
#     return sublists

# def split_consecutives_events_in_subevents(list_of_lists, max_length):

#     """
#     Metodo para dado la lista de listas de eventos extremos, dividir con un max_lenght para dividir eventos.
#     """

#     # Given a max_lenght, split in sublists

#     new_list_of_lists = []
#     for lst in list_of_lists:
#         if len(lst) <= max_length:
#             new_list_of_lists.append(lst)
#         else:
#             for i in range(0, len(lst), max_length):
#                 new_list_of_lists.append(lst[i:i+max_length])
#     return new_list_of_lists

# def find_extreme_events_without_split(df,sublists,range_cut):

#     # Second and thred aprox where we take as a big extreme event

#     list_extrem_events = []

#     min_date = df['time'].min()
#     max_date = df['time'].max()

#     for sublist in sublists:
#         len_sublist = len(sublist)

#         #Know the size of the two extremes.
#         if len_sublist % 2 == 0:
#             lenght = int((range_cut / 2) - (len_sublist / 2))
#             start_date = sublist[0] - timedelta(hours=lenght)
#             end_date = sublist[-1] + timedelta(hours=lenght + 1)

#         else:
#             lenght = int((range_cut / 2) - ((len_sublist-1) / 2))
#             start_date = sublist[0] - timedelta(hours=lenght)
#             end_date = sublist[-1] + timedelta(hours=lenght)

#         #Take range
        
#         df_filter = filter_date_with_extremes(df,min_date,max_date,start_date,end_date,range_cut)

#         list_extrem_events.append(df_filter)
    
#     return list_extrem_events

# def find_extreme_events_1(df_events,df,range_cut):

#     # First aprox creating a extreme event dataframe for each row.

#     list_extrem_events = []

#     min_date = df['time'].min()
#     max_date = df['time'].max()

#     for extrem_event in df_events.rows(named=True):

#         start_date = extrem_event[1] - timedelta(hours=range_cut/2)
#         end_date = extrem_event[1] + timedelta(hours=range_cut/2)

#         # If the extremes are smaller than the size that i want to take
#         df_filter = filter_date_with_extremes(df,min_date,max_date,start_date,end_date,range_cut)

#         list_extrem_events.append(df_filter)

#     return list_extrem_events

# def find_extreme_events_2(df_events,df,range_cut,max_length):

#     # Second approch, we split consecutives with max_lenght

#     sublists = join_related_events(df_events)
#     split_sublists = split_consecutives_events_in_subevents(sublists, max_length)
#     list_extrem_events = find_extreme_events_without_split(df,split_sublists,range_cut)
#     return list_extrem_events

# def find_extreme_events_3(df_events,df,range_cut):

#     # Third approch, we take consecutives as 1 event.

#     sublists = join_related_events(df_events)
#     list_extrem_events = find_extreme_events_without_split(df,sublists,range_cut)
#     return list_extrem_events

# def find_extreme_events_high_low_moredate(df,df_high_speed,df_low_speed,df_moderate_speed,range_cut):

#     list_high_extreme_events = find_extreme_events_1(df_high_speed,df,range_cut)
#     list_low_extreme_events = find_extreme_events_1(df_low_speed,df,range_cut)
#     list_moderate_extreme_events = find_extreme_events_1(df_moderate_speed,df,range_cut)

#     return list_high_extreme_events, list_low_extreme_events, list_moderate_extreme_events


# Create dataset final of extreme events

##################################################
#                                                #
#    Create dataset final of extreme events      #
#                                                #
##################################################

def generate_dataset_by_windows(df,range_cut,percentage=0.2):

    high_extreme_events = []
    low_extreme_events = []
    moderate_extreme_events = []

    for position in range(0,df.shape[0]-range_cut):

        # Selecciono el dataframe
        start_date = df[position,1]
        end_date = df[position,1] + timedelta(hours=range_cut)

        cut_df = df.filter((df['time'] >= start_date) & (df['time'] < end_date))

        # Calculo si hay un 20 %

        df_high_speed = cut_df.filter(cut_df['wspeed135m'] >= float(speed_high))
        df_low_speed = cut_df.filter(cut_df['wspeed135m'] <= float(speed_low))
        df_moderate_speed = cut_df.filter((cut_df['wspeed135m'] >= float(speed_moderate_down)) & (cut_df['wspeed135m'] <= float(speed_moderate_up)))

        if df_high_speed.shape[0] >= range_cut * percentage:
            high_extreme_events.append(cut_df)
        if df_low_speed.shape[0] >= range_cut * percentage:
            low_extreme_events.append(cut_df)
        if df_moderate_speed.shape[0] >= range_cut * percentage:
            moderate_extreme_events.append(cut_df)
        
        # print(df_high_speed.shape[0],range_cut * 0.2)
        # print(df_low_speed.shape[0],range_cut * 0.2)
        # print(df_moderate_speed.shape[0],range_cut * 0.2)
    return high_extreme_events, low_extreme_events, moderate_extreme_events

def generate_dataset_winter_summer(df,range_cut,percentage=0.2):

    high_extreme_events_winter = []
    low_extreme_events_winter = []
    moderate_extreme_events_winter = []
    high_extreme_events_summer = []
    low_extreme_events_summer = []
    moderate_extreme_events_summer = []

    middle = int(range_cut/2)

    for position in range(0,df.shape[0]-range_cut):

        # Selecciono el dataframe
        start_date = df[position,1]
        end_date = df[position,1] + timedelta(hours=range_cut)

        cut_df = df.filter((df['time'] >= start_date) & (df['time'] < end_date))

        # Calculo si hay un 20 %

        df_high_speed = cut_df.filter(cut_df['wspeed135m'] >= float(speed_high))
        df_low_speed = cut_df.filter(cut_df['wspeed135m'] <= float(speed_low))
        df_moderate_speed = cut_df.filter((cut_df['wspeed135m'] >= float(speed_moderate_down)) & (cut_df['wspeed135m'] <= float(speed_moderate_up)))


        # time_split = datetime.strptime(,'%Y-%m-%d %H:%M:%S')
        time_split = cut_df[middle,1]
        date_md = (time_split.month,time_split.day)

        if df_high_speed.shape[0] >= range_cut * percentage:
            if md_ini_winter <= date_md <= md_end_winter:
                high_extreme_events_winter.append(cut_df)
            else:
                high_extreme_events_summer.append(cut_df)
                print(time_split,df_high_speed.shape[0], range_cut * percentage)
        if df_low_speed.shape[0] >= range_cut * percentage:
            if md_ini_winter <= date_md <= md_end_winter:
                low_extreme_events_winter.append(cut_df)
            else:
                low_extreme_events_summer.append(cut_df)
        if df_moderate_speed.shape[0] >= range_cut * percentage:
            if md_ini_winter <= date_md <= md_end_winter:
                moderate_extreme_events_winter.append(cut_df)
            else:
                moderate_extreme_events_summer.append(cut_df)
        
        # print(df_high_speed.shape[0],range_cut * 0.2)
        # print(df_low_speed.shape[0],range_cut * 0.2)
        # print(df_moderate_speed.shape[0],range_cut * 0.2)
    return high_extreme_events_winter, low_extreme_events_winter, moderate_extreme_events_winter, \
    high_extreme_events_summer, low_extreme_events_summer, moderate_extreme_events_summer

def generate_datasets(df, range_cut, percentage,split_sw=False):
    if not split_sw:
        return generate_dataset_by_windows(df,range_cut,percentage)
    else:
        return generate_dataset_winter_summer(df,range_cut,percentage)

# Generate datasets per windfarm and pressure, join it and save it.

############################################################################
#                                                                          #
#    Generate datasets per windfarm and pressure, join it and save it.     #                                                    
#                                                                          #
############################################################################

def join_datasets(list_high_extreme_events, list_low_extreme_events, list_moderate_extreme_events, windfarm, pressure, df_high, df_low, df_moderate):

    # Save high events
    if len(list_high_extreme_events) != 0:
        for index,df in enumerate(list_high_extreme_events):
            list_high_extreme_events[index] = df.with_columns(pl.lit(index).alias('n_event'))
        df_high_speed = pl.concat(list_high_extreme_events, rechunk=True)
        df_high = pl.concat([df_high, df_high_speed], rechunk=True)

    # Save low events
    if len(list_low_extreme_events) != 0:
        for index,df in enumerate(list_low_extreme_events):
            list_low_extreme_events[index] = df.with_columns(pl.lit(index).alias('n_event'))
        df_low_speed = pl.concat(list_low_extreme_events, rechunk=True)
        df_low = pl.concat([df_low, df_low_speed], rechunk=True)

    # Save moderate events
    if len(list_moderate_extreme_events) != 0:
        for index,df in enumerate(list_moderate_extreme_events):
            list_moderate_extreme_events[index] = df.with_columns(pl.lit(index).alias('n_event'))
        df_moderate_speed = pl.concat(list_moderate_extreme_events, rechunk=True)
        df_moderate = pl.concat([df_moderate, df_moderate_speed], rechunk=True)

    return df_high, df_low, df_moderate

def get_datasets_per_pressure(df,pressure,range_cut,percentage,split_sw=False):
    if not split_sw:
        df_high = pl.DataFrame()
        df_low = pl.DataFrame()
        df_moderate = pl.DataFrame()
    else:
        df_high_winter = pl.DataFrame()
        df_low_winter = pl.DataFrame()
        df_moderate_winter = pl.DataFrame()
        df_high_summer = pl.DataFrame()
        df_low_summer = pl.DataFrame()
        df_moderate_summer = pl.DataFrame()

    for windfarm in range(0,38):
        print(f"windfarm nº {windfarm}")
        # print(f"Pressure: {pressure}, nº wind farm: {str(windfarm)}")

        # Split dataframe by pressure and windfarm (To use it separately)
        df_windfarm = get_data_per_windfarm(df,windfarm)

        # Get the list of individual extreme events
        # df_high_speed,df_low_speed,df_moderate_speed = get_extreme_events(df_windfarm)

        # Get the range of 4 days, of the list of events.
        # TODO:
        #       I have to change inside of the method if i want to change how i take the events.
        # list_high_extreme_events, list_low_extreme_events, list_moderate_extreme_events = find_extreme_events_high_low_moredate(df_windfarm,df_high_speed,df_low_speed,df_moderate_speed,range_cut)
        
        if not split_sw:
            list_high_extreme_events, list_low_extreme_events, list_moderate_extreme_events = generate_datasets(df_windfarm, range_cut, percentage)
            print("lista de eventos",len(list_high_extreme_events),len(list_low_extreme_events),len(list_moderate_extreme_events))

            df_high, df_low, df_moderate = join_datasets(list_high_extreme_events, list_low_extreme_events, list_moderate_extreme_events, windfarm, pressure, df_high, df_low, df_moderate)

            print("df total",df_high.shape, df_low.shape, df_moderate.shape)
        else:
            list_high_extreme_events_winter, list_low_extreme_events_winter, list_moderate_extreme_events_winter,list_high_extreme_events_summer, list_low_extreme_events_summer, list_moderate_extreme_events_summer = generate_datasets(df_windfarm, range_cut,percentage, split_sw)
            
            print("lista de eventos",len(list_high_extreme_events_winter),len(list_low_extreme_events_winter),len(list_moderate_extreme_events_winter))
            print("lista de eventos",len(list_high_extreme_events_summer),len(list_low_extreme_events_summer),len(list_moderate_extreme_events_summer))


            df_high_winter, df_low_winter, df_moderate_winter = join_datasets(list_high_extreme_events_winter,\
                                                         list_low_extreme_events_winter,\
                                                         list_moderate_extreme_events_winter,\
                                                         windfarm,\
                                                         pressure,\
                                                         df_high_winter, df_low_winter, df_moderate_winter)
            df_high_summer, df_low_summer, df_moderate_summer = join_datasets(list_high_extreme_events_summer,\
                                                         list_low_extreme_events_summer,\
                                                         list_moderate_extreme_events_summer,\
                                                         windfarm,\
                                                         pressure,\
                                                         df_high_summer, df_low_summer, df_moderate_summer)
    if not split_sw:
        return df_high, df_low, df_moderate
    else:
        return df_high_winter, df_low_winter, df_moderate_winter,df_high_summer, df_low_summer, df_moderate_summer

def get_datasets(df,range_cut, path, year, percentage = 0.2, split_sw=False):
    # pressures = ["900","925","950","975","1000"]
    pressures = ["925"]
    df_900, df_925, df_950, df_975, df_1000 = prepare_data_per_presure(df)

    # for index,df_pressure in enumerate([df_900, df_925, df_950, df_975, df_1000]):
    for index,df_pressure in enumerate([df_925]):
        # print(pressures[index])
        if not split_sw:
            df_high, df_low, df_moderate = get_datasets_per_pressure(df_pressure,pressures[index],range_cut,percentage)
            save_datasets(df_high, df_low, df_moderate, path, year, range_cut)
        else:
            df_high_winter, df_low_winter, df_moderate_winter,df_high_summer, df_low_summer, df_moderate_summer = get_datasets_per_pressure(df_pressure,pressures[index],range_cut, percentage, split_sw)
            save_datasets(df_high_winter, df_low_winter, df_moderate_winter, "./data/dataset_winter_split_10/", year, range_cut,type_save="winter")
            save_datasets(df_high_summer, df_low_summer, df_moderate_summer, "./data/dataset_summer_split_10/", year, range_cut,type_save="summer")

def reorder_columns(df_high, df_low, df_moderate):

    # TODO:
    #       Si añadimos otras columnas hay que cambiar esto.

    try:
        df_high = df_high.select(['time', 'index', 'n_event', 'cc', 'o3', 'pv', 'cape', 'blh', 'd2m', 'z', 'relative_humidity', 't2m', 't100m', 't135m', 'wdir100m', 'wspeed135m', 'wspeed100m'])
    except:
        print("df empty")
    try:
        df_low = df_low.select(['time', 'index', 'n_event', 'cc', 'o3', 'pv', 'cape', 'blh', 'd2m', 'z', 'relative_humidity', 't2m', 't100m', 't135m', 'wdir100m', 'wspeed135m', 'wspeed100m'])
    except:
        print("df empty")
    try:
        df_moderate = df_moderate.select(['time', 'index', 'n_event', 'cc', 'o3', 'pv', 'cape', 'blh', 'd2m', 'z', 'relative_humidity', 't2m', 't100m', 't135m', 'wdir100m', 'wspeed135m', 'wspeed100m'])
    except:
        print("df empty")
    return df_high, df_low, df_moderate

def save_datasets(df_high, df_low, df_moderate, path, year, range_cut,type_save=''):

    df_high, df_low, df_moderate = reorder_columns(df_high,df_low,df_moderate)

    df_high.write_csv(os.path.join(path,f"{year}_{range_cut}_{speed_high}_high_{type_save}.csv"), datetime_format='%Y-%m-%d %H:%M:%S')
    # df_low.write_csv(os.path.join(path,f"{year}_{range_cut}_low_{type_save}.csv"), datetime_format='%Y-%m-%d %H:%M:%S')
    # df_moderate.write_csv(os.path.join(path,f"{year}_{range_cut}_moderate_{type_save}.csv"), datetime_format='%Y-%m-%d %H:%M:%S')


############################################
#                                          #
#             Study dataset                #
#                                          #
############################################


def count_event_per_file(path):
    df = pl.read_csv(path)
    dic = []
    for windfarm in range(0,38):
        df_new = df.filter(pl.col("index") == windfarm)
        dic.append(df_new["n_event"].unique().shape[0])
    return pl.DataFrame({"index":range(0,38),"events": dic})