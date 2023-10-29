import sys
import os
sys.path.append('../../')
os.chdir('../../')

from src.IKM.data_exploration.distances import DistanceMeasurer
distance_measurer = DistanceMeasurer()

dirs = [
    r"./resultados/ideal_clusters/normal_15_summer/coeffs_1_eucl_ideal.txt",
    r"./resultados/ideal_clusters/normal_15_winter/coeffs_1_eucl_ideal.txt"
]

distance = distance_measurer.correlation_between_coeffs(dirs," ",True)