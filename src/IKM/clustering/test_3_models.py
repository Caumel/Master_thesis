import copy
import os
from joblib import Parallel, delayed
import time
import random
from sklearn.model_selection import KFold,StratifiedKFold



import numpy as np
import ntpath
import polars as pl
from tqdm import tqdm

import sys
sys.path.append('../../../')

from src.IKM.clustering.ikm.model_builder import model_bilder
from src.IKM.clustering.ikm.model_builder.time_series_object import TSObject
from src.IKM.clustering.ikm.utils.models_errors import ModelsErrors

path = "../../../current_experiment/"

def find_unique_max_indices(lst):
    # Create a list to store the result
    result = []
    
    # Create a list to keep track of the indices that have been used
    used_indices = [False, False, False]
    
    # Step 1: Assign indices to the largest element in each sublist
    for sublist in lst:
        # Find the index of the largest element
        max_idx = sublist.index(max(sublist))
        result.append(max_idx)
        # Mark this index as used
        used_indices[max_idx] = True
    
    if result[0] == result[1]:
        result = update_list(result,lst,0,1)
    elif result[0] == result[2]:
        result = update_list(result,lst,0,2)
    elif result[1] == result[2]:
        result = update_list(result,lst,1,2)
    return result
    
def update_list(result,lst,first,second):

    sublist_1 = lst[first]
    sublist_2 = lst[second]

    max_sublist_1 = max(sublist_1)
    max_sublist_2 = max(sublist_2)

    number_not_put = list(set(list(range(3))) - set(result))[0]

    if (sum(sublist_2) - max_sublist_2) > max_sublist_1:
        result[second] = number_not_put
    elif (sum(sublist_1) - max_sublist_1) > max_sublist_2:
        result[first] = number_not_put
    elif max_sublist_1 > max_sublist_2:
        result[second] = number_not_put
    elif max_sublist_2 > max_sublist_1:
        result[first] = number_not_put

    return result

def create_list_events(file_clusters):

    global path

    count_classes = [[0,0,0],[0,0,0],[0,0,0]]
    count_classes_real = [[0,0,0],[0,0,0],[0,0,0]]

    with open(os.path.join(path,"input",file_clusters), 'r') as file:
        data = file.readlines()

    new_data = []
    for line in data:
        new_data.append(line.strip().split('\n')[0])

    index_1 = new_data.index('Cl#:0')
    index_2 = new_data.index('Cl#:1')
    index_3 = new_data.index('Cl#:2')
    index_4 = new_data.index('Coefficients')

    objs = [new_data[index_1+1 : index_2],new_data[index_2+1 : index_3],new_data[index_3+1 : index_4]]
    objs_train_label_real = [new_data[index_1+1 : index_2],new_data[index_2+1 : index_3], new_data[index_3+1 : index_4]]
    objs_train_label_train = [new_data[index_1+1 : index_2],new_data[index_2+1 : index_3], new_data[index_3+1 : index_4]]

    coeff = new_data[index_4+1 :]

    for index,info in enumerate(objs[0]):
        objs[0][index] = path + "files/" + info + ".txt"
        objs_train_label_real[0][index] = int(info[-1])
        objs_train_label_train[0][index] = 0
        count_classes = know_cluster_class(info,0,count_classes)
        # print("\"" + cluster_1[index] + ".txt\",")

    for index,info in enumerate(objs[1]):
        objs[1][index] = path + "files/" + info + ".txt"
        objs_train_label_real[1][index] = int(info[-1])
        objs_train_label_train[1][index] = 1
        count_classes = know_cluster_class(info,1,count_classes)
        # print("\"" + cluster_2[index] + ".txt\",")

    for index,info in enumerate(objs[2]):
        objs[2][index] = path + "files/" + info + ".txt"
        objs_train_label_real[2][index] = int(info[-1])
        objs_train_label_train[2][index] = 2
        count_classes = know_cluster_class(info,2,count_classes)
        # print("\"" + cluster_3[index] + ".txt\",")
    
    n_cluster = find_unique_max_indices(count_classes)
    # n_cluster = [count_classes[0].index(max(count_classes[0])),count_classes[1].index(max(count_classes[1])),count_classes[2].index(max(count_classes[2]))]

    # Este codigo esta aqui para calcular los valores actualizadoas en el caso de que el cluster 1 tenga la clase 2.
    mapping = {0:n_cluster[0],1:n_cluster[1],2:n_cluster[2]}
    objs_train_label_real = [[mapping.get(item, item) for item in sublist] for sublist in objs_train_label_real]

    objs_real = [[],[],[]]
    # if n_cluster is [1,0,2]
    # in cluster_real_label go [..._1,..._0,..._2]
    objs_label_real = [[],[],[]]
    objs_label_train = [[],[],[]]

    new_data = objs[0] + objs[1] + objs[2]

    #### MI LABEL ES EL CLUSTER AL QUE PERTENECE ####

    # I create the perfect clasification.
    count_classes_real = [[0,0,0],[0,0,0],[0,0,0]]
    for index,info in enumerate(new_data):
        if info[-5] == "0":
            where_go = n_cluster.index(int(info[-5]))
            objs_real[where_go].append(info)
            objs_label_real[where_go].append(where_go)
            if info in objs[0]:
                objs_label_train[where_go].append(0)
            elif info in objs[1]:
                objs_label_train[where_go].append(1)
            elif info in objs[2]:
                objs_label_train[where_go].append(2)
        elif info[-5] == "1":
            where_go = n_cluster.index(int(info[-5]))
            objs_real[where_go].append(info)
            objs_label_real[where_go].append(where_go)
            if info in objs[0]:
                objs_label_train[where_go].append(0)
            elif info in objs[1]:
                objs_label_train[where_go].append(1)
            elif info in objs[2]:
                objs_label_train[where_go].append(2)
        elif info[-5] == "2":
            where_go = n_cluster.index(int(info[-5]))
            objs_real[where_go].append(info)
            objs_label_real[where_go].append(where_go)
            if info in objs[0]:
                objs_label_train[where_go].append(0)
            elif info in objs[1]:
                objs_label_train[where_go].append(1)
            elif info in objs[2]:
                objs_label_train[where_go].append(2)

    #TODO:
    # Es necesario hacer un asterico aqui ya que si en el label tenemos que hay mas clases del 2 en el primer cluster el id seria [2,1,0] y 
    # creamos el perfecto de esta forma

    count_classes_real[0][n_cluster[0]] = len(objs_real[0])
    count_classes_real[1][n_cluster[1]] = len(objs_real[1])
    count_classes_real[2][n_cluster[2]] = len(objs_real[2])



    return objs, objs_train_label_real, objs_train_label_train, objs_real, objs_label_real, objs_label_train, count_classes, count_classes_real

def know_cluster_class(info,cluster,count_classes):
    if info.endswith("_0"):
        count_classes[cluster][0] += 1
    elif info.endswith("_1"):
        count_classes[cluster][1] += 1
    elif info.endswith("_2"):
        count_classes[cluster][2] += 1
    return count_classes

# def leave_one(classes_,cl,obj_num_in_cluster,error,class_number,model_error,errors):
#     mod_set = copy.deepcopy(classes_)
#     test_set = []
#     test_set.append(mod_set[cl][obj_num_in_cluster]) # Selecciono el evento y lo añado a test_set
#     mod_set[cl].pop(obj_num_in_cluster) # Y lo quito de la lista original.

#     # start = time.time()

#     class_models = Parallel(n_jobs=-1)(delayed(model_bilder.ModelBilder.create_model)(mod_set[i], error) for i in range(class_number))
    
#     # end = time.time()
#     # print('{:.4f} s'.format(end-start))

#     # class_models = []
#     # for i in range(class_number): #Error para cada dimension
#     #     model = model_bilder.ModelBilder.create_model(mod_set[i], error)
#     #     class_models.append(model)            # class_models = Parallel(n_jobs=3)(delayed(model_bilder.ModelBilder.create_model)(mod_set[i], error) for i in range(class_number))

    
#     #Calcula el error del elemento con el cluster, el primer true or false es si parece que pertenece al cluster asociado.
#     result, text = model_error.models_errors(class_models, test_set, cl, error, text)
#     errors[cl] += result
#     return result, text, errors

def main(file_clusters,error,dataset,size_cut=1000):
    """
    Implementation of the interpretation algorithm. Computes the errors of an object using k-fold cross-validation method.
    """

    number_objs = 0
    classes_ = [] 
    model_error = ModelsErrors()

    objs, objs_train_label_real, objs_train_label_train, objs_real, objs_label_real, objs_label_train, count_classes, count_classes_real = create_list_events(file_clusters)

    print("Perfect",count_classes_real)
    print("Train",count_classes)
    print()

    # print(objs_real[2], objs_label_real[2], objs_label_train[2])
    
    if dataset == "real":
        objs = objs_real
        objs_train_label_real = objs_label_real
        objs_train_label_train = objs_label_train
        count_classes = count_classes_real
    
        # if file_clusters == "winter_total.txt":
        #     objs[0] = objs[0][:8000]
        #     objs_train_label_real[0] = objs_train_label_real[0][:8000]
        #     objs_train_label_train[0] = objs_train_label_train[0][:8000]
        #     count_classes[0][1] = 8000
        # elif file_clusters == "winter_eucl.txt":
        #     objs[1] = objs[1][:8000]
        #     objs_train_label_real[1] = objs_train_label_real[1][:8000]
        #     objs_train_label_train[1] = objs_train_label_train[1][:8000]
        #     count_classes[1][1] = 8000


    n_cluster = find_unique_max_indices(count_classes)
    print(f"Cluster 0: {n_cluster[0]}, 1: {n_cluster[1]}, 2: {n_cluster[2]}")
    # n_cluster = [count_classes[0].index(max(count_classes[0])),count_classes[1].index(max(count_classes[1])),count_classes[2].index(max(count_classes[2]))]

    # Leer elementos y guardarlos en classes_

    #### IMPORTANT ####

    # I take the labels from the train labels, but as i know the true labels, is better if i take the reals one.

    for i in tqdm(range(len(n_cluster))): # Tomo cada array de cada cluster
        objs_cluster = objs[i] # cluster i
        number_objs += len(objs_cluster) # numero de elementos en total de todos los clusters
        # objs_cluster.sort() # Ordeno

        class_ = []
        for j in range(len(objs[i])):
            data = pl.read_csv(os.path.join(objs_cluster[j]))
            # data = np.loadtxt(os.path.join(objs_cluster[j]), delimiter=' ', skiprows=1)
            filename = ntpath.basename(objs_cluster[j])
            class_.append(TSObject(file_name=filename, data=data,z_normalization=True, z_score=True))
        classes_.append(class_)
        
    text = ""

    print()
    print("----- [ Error for element ] -----")
    print()

    # errors = [0 for c in range(len(n_cluster))] # defino error a 0
    true_train_results_list = [0 for c in range(len(n_cluster))]
    true_test_results_list = [0 for c in range(len(n_cluster))]
    train_test_results_list = [0 for c in range(len(n_cluster))]
    for index,cl in enumerate(n_cluster): # Para cada cluster. #Index: cluster, cl: class 

        print()
        print(f"Cluster: {index}, class: {cl}, size: {len(classes_[cl])}")
        print()
        print(f"Element for cluster: {count_classes[index]}")
        print()
        text += "\n"
        text += f"Cluster: {index}\n"

        ### Leave one cross validations

        # for obj_num_in_cluster in tqdm(range(len(classes_[cl]))): #Para cada elemento en el cluster cl de la lista classes_  [[],[],[]]
            
        #     result, text, errors_new = leave_one(classes_,cl,obj_num_in_cluster,error,n_cluster,model_error,errors)
        #     error = copy.deepcopy(errors_new)

        ### Kfold cross validations

        # if len(classes_[index])>10000:
        #     size_cut = 3000
        # else:
        #     size_cut = 1000

        time_run = 1 if len(classes_[index])<=size_cut else ((len(classes_[index])//size_cut) + 1)
        # print(time_run)

        # If we use 
        kf = StratifiedKFold(n_splits=time_run)
        set = copy.deepcopy(classes_)

        # Tomo como referencia los labels de real si es el real y train si es el train.
        if dataset == "real":
            labels = objs_train_label_real
        else:
            labels = objs_train_label_train
        
        for train_index, test_index in tqdm(kf.split(np.array(set[index]),np.array(labels[index])), total=kf.get_n_splits(), desc="k-fold"):
            
            mod_set = copy.deepcopy(classes_)

            mod_set[index], test_set = [set[index][i] for i in train_index], [set[index][i] for i in test_index]

            objs_label_real_test = [objs_train_label_real[index][i] for i in test_index]
            objs_label_train_test = [objs_train_label_train[index][i] for i in test_index]

            # print(test_set,objs_label_real_test,objs_label_train_test)

            # Create a model for each of the dataset [[Cros-validation],[Todo],[Todo]]
            class_models = []
            for i in range(len(n_cluster)):
                model = model_bilder.ModelBilder.create_model(mod_set[i], error)
                class_models.append(model)
            # class_models = Parallel(n_jobs=-1)(delayed(model_bilder.ModelBilder.create_model)(mod_set[i], error) for i in range(len(n_cluster)))

            # Compute the error of our test set in each of the model
            true_train_results, true_test_results, train_test_results, text = model_error.models_errors(class_models, test_set, index, cl, error, text, objs_label_real_test, objs_label_train_test)
            
            true_train_results_list[index] += true_train_results
            true_test_results_list[index] += true_test_results
            train_test_results_list[index] += train_test_results


    print()
    text += "\n"
    print("----- [ Numero de errores por cluster ] -----")
    print()


    err_number = 0
    print()
    print(": err true-train.  = (")
    text += ": err true-train.  = ( "

    for error in true_train_results_list:
        err_number += error
        print(f"{error} ")
        text += f"{error} "

    print(f") ({err_number});  {round((1 - (err_number + 0.0) / number_objs) * 1000) / 10.0} % \n")
    text += f") ({err_number});  {round((1 - (err_number + 0.0) / number_objs) * 1000) / 10.0} % \n"

    err_number = 0
    print()
    print(": err true-test.  = (")
    text += ": err true-test.  = ( "
    
    for error in true_test_results_list:
        err_number += error
        print(f"{error} ")
        text += f"{error} "

    print(f") ({err_number});  {round((1 - (err_number + 0.0) / number_objs) * 1000) / 10.0} % \n")
    text += f") ({err_number});  {round((1 - (err_number + 0.0) / number_objs) * 1000) / 10.0} % \n"

    err_number = 0
    print()
    print(": err train-test.  = (")
    text += ": err train-test.  = ( "

    for error in train_test_results_list:
        err_number += error
        print(f"{error} ")
        text += f"{error} "

    print(f") ({err_number});  {round((1 - (err_number + 0.0) / number_objs) * 1000) / 10.0} % \n")
    text += f") ({err_number});  {round((1 - (err_number + 0.0) / number_objs) * 1000) / 10.0} % \n"

    print()
    text += "\n"
    print("----- [ Var Error ] -----")
    print()


    # print(model_error.errors_in_models)

    # Order error list to put lower the first one.
    arr2 = np.array(model_error.errors_in_models)
    sorted_indices = np.argsort(arr2)
    sorted_list1 = np.array(range(1,len(model_error.errors_in_models)+1))[sorted_indices].tolist()
    sorted_list2 = arr2[sorted_indices].tolist()

    for index,element in enumerate(sorted_list2):
        print(f"{sorted_list1[index]}:{element}")
        text += f"{sorted_list1[index]}:{element}\n"



    file_path = fr'test_results_{size_cut}_{dataset}_{file_clusters[:-4]}.txt'

    if os.path.exists(file_path):
        os.remove(file_path)

    file = open(os.path.join("../../../aa_winter_try/results",file_path), 'a')
    file.write(text)
    file.close()



if __name__ == '__main__':

    # path = "../../../current_experiment/"
    # file_clusters = "winter_total.txt"
    # dataset = "real"
    # error = "total"
    # size_cut = 3000
    # main(file_clusters,error,dataset,size_cut)

    path = "../../../aa_winter_try/"
    dataset = ["real","train"]
    size_cut = 1000

    for file in os.listdir(os.path.join(path,"input")):
        print()
        print(file)
        error = file.split("_")[-1].split(".")[-2]
        for type_dataset in dataset:
            print()
            print(type_dataset)
            print()
            main(file,error,type_dataset,size_cut)

    # Me acabo de dar cuenta de una cosa, al printear el tamaño del cluster
    # Si es 0,2,1 en el 0 printeo el 0, pero en el 1 printeo el 2 pero uso la posicion 1, esta bien entrenado pero los datos esos cambiados
    # Para tenerlo en cuenta.
