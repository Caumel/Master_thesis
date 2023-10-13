import polars as pl
import numpy as np
import os
from tqdm import tqdm

path = "../../data/final_files/small/"
path_save_file_per_event = "../../data/file_per_event/current_experiment"

def create_files_per_event(event,name,path):

    if not os.path.exists(os.path.join(path,f'{name}.txt')):
        event.write_csv(os.path.join(path,f'{name}.txt'))
        
    return None

print(os.listdir(path))

for file in os.listdir(path):

    data = pl.read_csv(os.path.join(path, file))

    categories = np.array(['_'.join(str(item) for item in sublist) for sublist in data[:,1:4].rows()])

    # Get unique categories
    unique_categories = np.unique(categories)
    np.random.shuffle(unique_categories)

    # Split the array into a list of lists based on categories
    for index,category in enumerate(tqdm(unique_categories, desc='Split events, and create TSObject', leave=False)):
        # Select the one event in a big dataframe
        indices = np.where(categories == category)
        subset = data[indices[0]]

        create_files_per_event(subset,category,path_save_file_per_event)