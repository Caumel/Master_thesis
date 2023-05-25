import ast
import sys

import random
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cluster import Cluster
from time_series_object import TSObject

if __name__ == '__main__':

    """
    Running the IKM algorithm for different k.
    """

    start_time = time.process_time()
    data = pd.read_csv("../datasets/depression_patients.csv")
    objects = []
    for ind in data.index:
        data["compData"][ind] = ' '.join(data["compData"][ind].split())
        data["compData"][ind] = data["compData"][ind].replace('[ ', '[').replace(' ', ',')
        data["quadrs"][ind] = ' '.join(data["quadrs"][ind].split())
        data["quadrs"][ind] = data["quadrs"][ind].replace('[ ', '[').replace(' ', ',')

    for ind in data.index:
        objects.append(TSObject(data["id"][ind], np.array(ast.literal_eval(data["compData"][ind])),
                                np.array(ast.literal_eval(data["quadrs"][ind])), data["response"][ind],
                                data["age"][ind], data["sex"][ind], data["srate"][ind], data["treatment_1"][ind],
                                data["treatment_2"][ind], data["treatment_3"][ind], data["treatment_4"][ind]))

    min_error = sys.float_info.max
    bestCl = "..--~~***~~--.."
    max_k = 8
    k_errors = [[] for i in range(max_k-1)]
    K = range(1, max_k)

    for k in K:
        print("k#:" + str(k))
        for tries in range(1):
            print("Try#:" + str(tries + 1))

            clusters = [[] for i in range(k)]

            counter = 0
            random.shuffle(objects)
            for time_series_object in objects:
                current_object = time_series_object
                clusters[counter % k].append(current_object)
                counter += 1

            for cluster in clusters:
                cluster.sort(key=lambda a: a.name)

            cl = Cluster(clusters)

            for i in range(75):
                cl.step()
                if not (cl.is_changed()):
                    break
                # print("Step#:" + str(i + 1))
                # print(cl)
            error = cl.error()
            if min_error > error:
                min_error = error
                bestCl = cl.__str__()
                k_errors[k-1] = error

    K = [i for i in range(max_k-1)]

    plt.figure(figsize=(16, 8))
    plt.plot(K, k_errors, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Error')
    plt.title('The Elbow Method showing the optimal k')
    plt.savefig('elbow_plot.png')
    plt.show()


    finish_time = time.process_time() - start_time
    print(f"Time: {finish_time}\nBest clustering:\n{bestCl}\n")
    for i in range(len(k_errors)):
        print(f"k:{i+1} Error: {k_errors[i]}\n")
    file = open(r'\reports\optimal_k_results.txt', 'w')
    file.write(f"Time: {finish_time}\nBest clustering:\n{bestCl}\n")
    for i in range(len(k_errors)):
        file.write(f"k:{i+1} Error: {k_errors[i]}\n")
    file.close()
