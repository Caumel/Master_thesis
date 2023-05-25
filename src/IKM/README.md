# Clustering Brain Regions by Similar Interaction Patterns Based on Multivariate Neural Signals for Identifying the Response to Antidepressants

# Overview

This code is part of the Master Thesis with the title "Clustering Brain Regions by Similar Interaction Patterns Based on
Multivariate Neural Signals for Identifying the Response to Antidepressants".

It provides the ability to run the Interaction k-means (IKM) algorithm (Plant et al. 2013) on the data of the depressed
patients. Moreover, it provides the scripts for analyzing the data and transforming it using different methods such
as Box-Cox, z-score, Hilbert transform and others. This repository provides the code for clustering on the data using
IKM. It also provides the implementation for clustering on the data coefficients (obtained from the least squares
method) using such algorithms as k-means, DBSCAN and
hierarchical clustering.

# Folder structure

* `datasets` contains various datasets for the computation
  
* `ikm` contains the code for running the IKM and enhanced (with the preselection of dimensions implementation) IKM
  algorithms, and for calculating
  metrics (`metrics.py`). Also, it contains interpretation algorithm implementation, code for the data
  transformation (`data_preprocessor.py`) and for loading the
  data (`data_loader.py`)

    * Almost all code in the sub folders `clust`, `interptet`, and `model_builder` was recoded from the original
      implementation of Plant et al. 2013. The exceptions are some changes in the files `dim_selection.py`
      and `time_series_object.py`.
      
    * In the folder `utils`, code in the `models_errors.py` was recoded from the original implementation of Plant
      et al. 2013 as well as functions `compute_quadrs` and `compute_data` in the file `data_preprocessor.py`. The
      functions `calculate_entropy`, `calculate_statistics`, `calculate_crossings` and `get_features` in the
      file `data_preprocessor.py` are based
      on (A guide for using the wavelet transform in machine learning, 2021). The functions `box_cox_transform`
      and `shift_to_positive` are based on (Bialowas, 2022). The function `fft_eeg_bands_extraction` is based on
      the (How to correctly compute the EEG frequency bands with python?). The functions `butter_bandstop_filter`
      , `butter_bandstop`, `butter_bandpass_filter`, `butter_bandpassAll` are based on (Brain2Speech). The functions
      `hilbert_phase` and `hilbert_amplitude` are based on (Viswanathan, 2020). All other code
      in the file was developed by me, Mykola Lazarenko.

* `data_exploration` contains the scripts for calculating the Frobenius distances between the coefficients extracted
  from the depression patients data, the scripts for using the elbow method for finding the optimal k number of clusters
  and other scripts for preliminary data analysis.
    * Code in the functions `ideal_coeffs` and `coeffs_each_obj` in the file `data_exploration.py` was based on the
      original implementation of Plant et al. 2013. The same applies for some parts of the code in the `optimal_k.py`
      file.
    * The code in files `mne_data_exploration.ipynb` and `mne_workflow.py` are based on the examples from MNE Python
      package.
    * The function `corr2_coeff` in the file `data_exploration.py` is based on (Computing the correlation coefficient
      between two multi-dimensional arrays).
    * All other code in other files of this folder was developed by me, Mykola Lazarenko.

* `data_preprocessing` contains the scripts for transforming the structure of the data in the correct one for running
  the IKM algorithm on it. Also, there are the scripts for other data preprocessing steps such as files renaming,
  separating data, exponentially smoothing the data, etc.
    * Only functions `compute_AtA` and `compute_cubes` in the file `create_EEG_csv.py` were based on the
      original implementation of Plant et al. 2013. All other code was developed by me, Mykola Lazarenko

* `data_visualization` contains the scripts for visualizing interaction between electrodes, clustering results and bands
  visualizations
    * All code was developed by me, Mykola Lazarenko.








# How to run it?

## Main

To run the IKM algorithm, execute the command `python __main__.py` in the command-line interpreter while being in the
project directory.

The function that runs IKM algorithm is called `ikm_process`. The parameters it can take are described in the `ikm.py`
file.

## Other algorithms

To run the interpretation algorithm, execute the command `python class_leave_one_out.py` in the `ikm/interpret` folder.

To run k-means, DBSCAN and hierarchical clustering on the coefficients, the `cluster_coeffs_chosen_el` function from the
`coeffs_clustering.py` file can be called.

To run the dimension selection IKM algorithm, the function `dim_selection_process` from the
`dim_selection.py` can be called.

## Data transformations

Data transformations are regulated by the parameters of the `ikm_process` function. The calls of the specific
transformation
function is done in `time_series_object.py` file which is responsible for the creation of the objects
that are a final input to the IKM algorithm. To choose the specific data transformation function, call the `ikm_process`
function in the `__main.py__` file with the corresponding parameters.

# Versions

Python 3.10 was used for the implementation of this project.

# References

* C. Plant, A. Zherdin, C. Sorg, A. Meyer-Baese, and A. M. Wohlschläger, in the scope of ‘Mining Interaction Patterns
  among Brain Regions by Clustering’, 2013 paper.
* A guide for using the wavelet transform in machine learning. ML
  Fundamentals. https://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/. Accessed
  5
  March 2023
* Bialowas, R. (2022, August 26). Box-Cox Transformation, explained. Medium.
  Medium. https://medium.com/@radoslaw.bialowas/box-cox-transformation-explained-da8450295668. Accessed 11 March 2023
* How to correctly compute the EEG frequency bands with python? Signal Processing Stack
  Exchange. https://dsp.stackexchange.com/questions/45345/how-to-correctly-compute-the-eeg-frequency-bands-with-python.
  Accessed 11 March 2023
* Brain2Speech. Algorithms/KNN_NN_LSTM at master · Brain2Speech/Algorithms.
  GitHub. https://github.com/Brain2Speech/Algorithms/tree/master/knn_nn_lstm. Accessed 11 March 2023
* Viswanathan, M. (2020, November 19). Extract envelope, phase using Hilbert transform: Demo.
  GaussianWaves. https://www.gaussianwaves.com/2017/04/extract-envelope-instantaneous-phase-frequency-hilbert-transform/
  . Accessed 11 March 2023
* Open-source Python package for exploring, visualizing, and analyzing human neurophysiological data: MEG, EEG, sEEG,
  ECoG, NIRS, and more. Documentation overview - MNE 1.3.1 documentation. https://mne.tools/stable/overview/index.html.
  Accessed 11 March 2023
* Computing the correlation coefficient between two multi-dimensional arrays. Stack
  Overflow. https://stackoverflow.com/questions/30143417/computing-the-correlation-coefficient-between-two-multi-dimensional-arrays
  . Accessed 12 March 2023

# Author

Mykola Lazarenko


 