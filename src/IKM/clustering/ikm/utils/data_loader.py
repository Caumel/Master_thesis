import os

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm


from ikm.model_builder.time_series_object import TSObject

VERY_SMALL_CLOSEST_ZERO = 0.0000000000000000000001


class DataLoader:
    delimiter = ','

    def __init__(self, delimiter):
        self.delimiter = delimiter

    def load_data_one_file(self, box_cox=None, z_normalization=None, z_score=None, excl_wm=None,path=None,
                           specific_windmills=None, windmills=None, how_to_process_data="all",kind_mean="all",
                           samples_per_file = 10, path_save_file_per_event="./"):
        """
        Loads the depressed patients data from one visit.
        """

        objects = []
        df = pd.DataFrame(columns=['ID', 'index', 'n_event', 'year', 'Date_init', 'Date_end', 'Data','Response'])


        for filename in os.listdir(path):
            print(f"processing file: {filename}")

            # TODO:
            #   This say us 0 high 1 moderate 2 low
            class_event = self.find_class(filename)

            data = pl.read_csv(os.path.join(path, filename))

            # TODO:
            #   - Cuando realizo transformaciones, deberia de hacerlo sobre todos los windmill a la vez o diferenciar por windmill
            #       - A esto me refiero si las estadisticas de la medio y eso se hacen sobre cada 1 o todas. 
            #   - Tengo que crear un objeto de TSObject por timeseries verdad ?

            if how_to_process_data == "for_windmill":       # Per windmill
                data = self.split_data_per_windmill(data)
            else:                                           # All the windmill
                data = [data]
                
            for data_element in data:
                            
                events, df_events = self.split_events(data_element,
                                                      box_cox,   # Apply Box_cox transformation
                                                      z_normalization, # Z normalization apply
                                                      z_score, # Clustering base on the z_score info
                                                      excl_wm,  # Exclude some electrodes (Can i do it with wind farms ?)
                                                      specific_windmills, 
                                                      windmills,
                                                      kind_mean,
                                                      class_event,
                                                      samples_per_file,
                                                      path_save_file_per_event)

                df = pd.concat([df, df_events])


            # Concanet objects
            objects.extend(events)

        return df, objects
    
    def find_class(self,name):
        if "high" in name:
            return 0
        elif "moderate" in name:
            return 1
        else:
            return 2
    
    def split_data_per_windmill(self,data):

        list_windmill = []

        for n_windmill in range(0,38):
            list_windmill.append(data.filter(pl.col("index") == n_windmill))

        return list_windmill
    
    def split_events(self,data,box_cox, z_normalization,z_score, \
                     excl_wm, specific_windmills, windmills,kind_mean, \
                     class_event,samples_per_file, path_save_file_per_event):
        
        # Columns time,index,n_event,year,cc,o3,pv,cape,blh,d2m,z,relative_humidity,t2m,t100m,t135m,wdir100m,wspeed135m,wspeed100m

        list_events = []

        categories = np.array(['_'.join(str(item) for item in sublist) for sublist in data[:,1:4].rows()])
        updated_list = np.array([test + "_" + str(class_event) for test in categories])

        # Get unique categories
        unique_categories = np.unique(updated_list)
        np.random.shuffle(unique_categories)

        # Split the array into a list of lists based on categories
        df = pd.DataFrame(columns=['ID', 'index', 'n_event', 'year', 'Date_init', 'Date_end', 'Data','Response'])
        for index,category in enumerate(tqdm(unique_categories, desc='Split events, and create TSObject', leave=False)):
            # Select the one event in a big dataframe
            indices = np.where(updated_list == category)
            subset = data[indices[0]]

            # Create TSObject with this event
            df.loc[len(df.index)] = [category,subset[0,1],subset[0,2],subset[0,3],subset[0,0],subset[-1,0],subset,class_event]
            
            time_series = TSObject(file_name=category, # Filename
                                    data=subset,          # Information 
                                    box_cox=box_cox,   # Apply Box_cox transformation
                                    z_normalization=z_normalization, # Z normalization apply
                                    z_score=z_score, # Clustering base on the z_score info
                                    excl_wm=excl_wm,  # Exclude some electrodes (Can i do it with wind farms ?)
                                    specific_windmills=specific_windmills, 
                                    windmills=windmills,
                                    kind_mean=kind_mean)
                
            list_events.append(time_series)

            self.create_files_per_event(subset,category,path_save_file_per_event)

            # break

            # TODO:
            #   - We only take 1 event (for test the results.)
            if samples_per_file != None and index == samples_per_file:
                break

        return list_events, df

    def create_files_per_event(self,event,name,path):

        if not os.path.exists(os.path.join(path,f'{name}.txt')):
            event.write_csv(os.path.join(path,f'{name}.txt'))

        return None