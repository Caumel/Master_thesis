import copy
import os

import numpy as np
import ntpath
import polars as pl
from tqdm import tqdm

import sys
sys.path.append('../../../')

from src.IKM.clustering.ikm.model_builder import model_bilder
from src.IKM.clustering.ikm.model_builder.time_series_object import TSObject
from src.IKM.clustering.ikm.utils.models_errors import ModelsErrors

def create_list_events():

    path = "./"
    file_clusters = "cluster_split_eucl_ideal.txt"

    with open(os.path.join(path,file_clusters), 'r') as file:
        # Lee el contenido del archivo
        data = file.readlines()

    new_data = []
    for line in data:
        new_data.append(line.strip().split('\n')[0])

    index_1 = new_data.index('Cl#:0')
    index_2 = new_data.index('Cl#:1')
    index_3 = new_data.index('Cl#:2')
    index_4 = new_data.index('Coefficients')

    cluster_1 = new_data[index_1+1 : index_2]
    cluster_2 = new_data[index_2+1 : index_3]
    cluster_3 = new_data[index_3+1 : index_4]
    coeff = new_data[index_4+1 :]

    for index,info in enumerate(cluster_1):
        cluster_1[index] = "../../../data/file_per_event/current_experiment/" + info + ".txt"
        # print("\"" + cluster_1[index] + ".txt\",")

    for index,info in enumerate(cluster_2):
        cluster_2[index] = "../../../data/file_per_event/current_experiment/" + info + ".txt"
        # print("\"" + cluster_2[index] + ".txt\",")

    for index,info in enumerate(cluster_3):
        cluster_3[index] = "../../../data/file_per_event/current_experiment/" + info + ".txt"
        # print("\"" + cluster_3[index] + ".txt\",")

    return [cluster_1,cluster_2,cluster_3]

def main():
    """
    Implementation of the interpretation algorithm. Computes the errors of an object using leave-one-out cross-validation method.
    """

    number_objs = 0
    classes_ = [] 
    model_error = ModelsErrors()

    # print(os.listdir("../../../"))

    objs = create_list_events()
    
    error = 'total'

    # Leer elementos y guardarlos en classes_

    class_number = len(objs)
    for i in tqdm(range(class_number)): # Tomo cada array de cada cluster
        objs_cluster = objs[i] # cluster i
        number_objs += len(objs_cluster) # numero de elementos en total de todos los clusters
        objs_cluster.sort() # Ordeno

        class_ = []
        j = len(objs_cluster) - 1
        while j >= 0:
            data = pl.read_csv(os.path.join(objs_cluster[j]))
            # data = np.loadtxt(os.path.join(objs_cluster[j]), delimiter=' ', skiprows=1)
            filename = ntpath.basename(objs_cluster[j])
            class_.append(TSObject(file_name=filename, data=data,z_normalization=True, z_score=True))

            j = j - 1
        classes_.append(class_)

    text = ""

    errors = [0 for c in range(class_number)] # defino error a 0
    for cl in range(class_number): # Para cada cluster.

        print()
        print(f"Cluster: {cl}")
        text += "\n"
        text += f"Cluster: {cl}\n"

        for obj_num_in_cluster in tqdm(range(len(classes_[cl]))): #Para cada elemento en el cluster cl de la lista classes_  [[],[],[]]

            mod_set = copy.deepcopy(classes_)
            test_set = []
            test_set.append(mod_set[cl][obj_num_in_cluster]) # Selecciono el evento y lo añado a test_set
            mod_set[cl].pop(obj_num_in_cluster) # Y lo quito de la lista original.

            training_mod_set = []
            for i in range(class_number): # Añado todos los elementos parecido al original pero sin el que estoy analizando.
                training_mod_set.append(mod_set[i])

            class_models = []
            for i in range(class_number): #Error para cada dimension
                model = model_bilder.ModelBilder.create_model(mod_set[i], error)
                class_models.append(model)
            
            #Calcula el error del elemento con el cluster, el primer true or false es si parece que pertenece al cluster asociado.

            result, text = model_error.models_errors(class_models, test_set, cl, error, text)
            errors[cl] += result

    print(model_error)

    # Numero de errores por cluster.

    err_number = 0
    print()
    print(": err.  = (")
    text += "\n"
    text += ": err.  = (\n"
    for error in errors:
        err_number += error
        print(f"{error} ")
        text += f"{error} \n"

    print()
    print(f") ({err_number});  {round((1 - (err_number + 0.0) / number_objs) * 1000) / 10.0} %")
    text += "\n"
    text += f") ({err_number});  {round((1 - (err_number + 0.0) / number_objs) * 1000) / 10.0} % \n"

    print(model_error.errors_in_models)

    for i in range(len(model_error.errors_in_models)):
        print(f"{i + 1}:{model_error.errors_in_models[i]}")
        text += f"{i + 1}:{model_error.errors_in_models[i]}\n"



    file_path = fr'test_results.txt'

    if os.path.exists(file_path):
        os.remove(file_path)

    file = open(os.path.join(file_path), 'a')
    file.write(text)
    file.close()



if __name__ == '__main__':
    main()
