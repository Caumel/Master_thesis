import os

path = "./IKM/clustering/"
file_clusters = "cluster_split.txt"

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

print()

for index,info in enumerate(cluster_2):
    cluster_2[index] = "../../../data/file_per_event/current_experiment/" + info + ".txt"
    # print("\"" + cluster_2[index] + ".txt\",")

print()

for index,info in enumerate(cluster_3):
    cluster_3[index] = "../../../data/file_per_event/current_experiment/" + info + ".txt"
    # print("\"" + cluster_3[index] + ".txt\",")

print(cluster_1)

print()
print(coeff)