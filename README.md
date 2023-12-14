# Clustering of wind related time series in a wind turbine farm
Repository for master thesis with notebooks and code.

# Overview

This code has been developed for the thesis of Luis Caumel Morales, for the thesis Clustering of wind related time series in a wind turbine farm.
The Interaction k-means (IKM) algorithm (Plant et al. 2013) has been implemented for wind turbine data. It contains both
scripts for the preprocessing of the data as well as the training and analysis of the results. This repository provides the code for clustering on the data using IKM.

# Folder structure

* `data`: Datos utilizados para la thesis, esta informacion no se encuentra en el repositorio.
* `docs`: Documentos utilizados para el desarrollo de la thesis
* `images`: Imagenes utilizadas para la memoria de la thesis
* `resultados`: Resultados de la thesis, para mostrar en el documento de entrega, esta informacion no se encuentra en el repositorio
* `src`
    * `IKM`
        * `clustering` contains the code for running the IKM and enhanced IKM algorithms, and for calculating
        metrics (`test_3_models.py`). Also, it contains interpretation algorithm implementation, code for the data
        transformation (`data_preprocessor.py`) and for loading the
        data (`data_loader.py`)
            * Almost all code in the sub folders `clust`, `interptet`, and `model_builder` was recoded from the
            implementation of Mykola Lazarenko. The exceptions are some changes in the file `data_preprocessor.py`
            and `time_series_object.py`, to adapt to my data.
            * In the folder `utils`, code in the `models_errors.py` was recoded from the implementation of Mi
            et al. 2013 as well as functions `compute_quadrs` and `compute_data` in the file `data_preprocessor.py`. The
            functions `calculate_entropy`, `calculate_statistics`, `calculate_crossings` and `get_features` in the
            file `data_preprocessor.py` are based on (A guide for using the wavelet transform in machine learning, 2021).
            The functions `box_cox_transform` and `shift_to_positive` are based on (Bialowas, 2022). The function 
            `fft_eeg_bands_extraction` is based on the (How to correctly compute the EEG frequency bands with python?).
            The functions `butter_bandstop_filter`, `butter_bandstop`, `butter_bandpass_filter`, `butter_bandpassAll` 
            are based on (Brain2Speech). The functions `hilbert_phase` and `hilbert_amplitude` are based on (Viswanathan, 2020).
            The updates of the code was developed by me, Luis Caumel Morales.
            * `__main__.py`: Codigo para la ejecucion del algoritmo IKM, preporcesado y print de los resultados
            * `test_3_models.py`: Codigo para el testeo de los resultados del modelo, ademas de preparar los datos.
        * `data_exploration` contains the scripts for calculating the diferent distances between the coefficients extracted
        from the data.
            * Code in the functions `ideal_coeffs` and `coeffs_each_obj` in the file `data_exploration.py` was based on the
            original implementation of Plant et al. 2013. The same applies for some parts of the code in the `optimal_k.py`
            file.
            * The code in files `mne_data_exploration.ipynb` and `mne_workflow.py` are based on the examples from MNE Python
            package.
            * The function `corr2_coeff` in the file `data_exploration.py` is based on (Computing the correlation coefficient
            between two multi-dimensional arrays).
            * All code in this folder was developed by Mykola Lazarenko, as I did not modify it.
    * `notebooks`: Notebooks en los cuales se muestran diferentes facelas de la implementacion
        * `create_dataset.ipynb`: Notebook para la creacion del dataset
        * `draft.ipynb`: Notebook para  el testeo de metodos
        * `resultados.ipynb`: Notebook para el printeo de los resultados
    * `utils`
        * `computer_correlation.py`: Fichero para el calculo de correlacion dados dos matrices
        * `count_events.py`: Fichero para contar el numero de eventos de cada tipo
        * `create_list_files_results.py`: Fichero para crear una lista de eventos.
        * `createDataset.py`: Fichero para crear el dataset de la thesis
        * `file_per_event.py`: Fichero para crear un fichero por cada evento
        * `join_files.py`: Fichero para unir ficheros
        * `reduce_dataset_after_join.py`: Fichero para reducir el numero de eventos antes de unirlos todos
        * `reduce_dataset_before_join.py`: Fichero para reducir el numero de eventos despues de unir todos los eventos
        * `split_summer_winter.py`: Fichero para dividir el dataset en invierno y verano
        * `utils.py`: Fichero con metodos utilizados por otros ficheros

# How to run it?

## Main

To run the IKM algorithm, execute the command `python __main__.py` in the command-line interpreter in the path `./master_thesis`

The function that runs IKM algorithm is called `ikm_process`. The parameters it can take are described in the `__main__.py` file.


## Data transformations

Data transformations are regulated by the parameters of the `ikm_process` function. The calls of the specific
transformation function is done in `time_series_object.py` file which is responsible for the creation of the objects
that are a final input to the IKM algorithm. To choose the specific data transformation function, call the `ikm_process`
function in the `__main.py__` file with the corresponding parameters.

# Versions

Python 3.10.2 was used for the implementation of this project.

# Author

Luis Caumel Morales


 