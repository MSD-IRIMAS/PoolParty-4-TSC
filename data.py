"""
The following code is inspired by
https://github.com/MSD-IRIMAS/LITE/blob/e2c2680fda0e0f4f9d1384ef959e33580cc7f473/utils/utils.py
"""

import os
import numpy as np
import sklearn.preprocessing
import aeon.datasets

import constants


data_folder = 'datasets/'


def minmax_normalization(x):
    xmin = np.min(x, axis=1, keepdims=True)
    xmax = np.max(x, axis=1, keepdims=True)
    max_min = xmax - xmin
    # avoid 0 division
    max_min[max_min == 0.0] = 1.0
    
    return (x - xmin) / max_min


def z_normalization(x):
    stds = np.std(x, axis=1, keepdims=True)
    means = np.mean(x, axis=1, keepdims=True)
    # avoid 0 division
    stds[stds == 0.0] = 1.0
    
    return (x - means) / stds


def load_tsv_data(file_name, folder_path=f'{data_folder}UCRArchive_2018/'):
    # missing value
    if file_name in constants.UNIVARIATE_DATASET_NAMES_2018_MISSING_VALUES:
        folder_path += 'Missing_value_and_variable_length_datasets_adjusted/'
    train_path = folder_path + f'{file_name}/{file_name}' + '_TRAIN.tsv'
    test_path = folder_path + f'{file_name}/{file_name}' + '_TEST.tsv'

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError('')

    train = np.loadtxt(train_path, dtype=np.float64)
    test = np.loadtxt(test_path, dtype=np.float64)

    x_train = train[:, 1:]
    x_test = test[:, 1:]
    
    y_train = train[:, 0]
    y_test = test[:, 0]

    return x_train, y_train, x_test, y_test


def load_ts_data(file_name, folder_path=f'{data_folder}Multivariate2018_ts/'):
    train_path = folder_path + f'{file_name}/{file_name}' + '_TRAIN.ts'
    test_path = folder_path + f'{file_name}/{file_name}' + '_TEST.ts'

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print(train_path)
        raise FileNotFoundError('')
    
    x_train, y_train = aeon.datasets.load_from_ts_file(train_path, return_meta_data=False, return_type=np.float64)
    x_test,  y_test  = aeon.datasets.load_from_ts_file(test_path, return_meta_data=False, return_type=np.float64)

    if file_name in constants.MULTIVARIATE_DATASET_NAMES_2018_ALL:
        # multivariate (sample, channel, length) -> (sample, length, channel)
        x_train = np.swapaxes(x_train, 1, 2)
        x_test  = np.swapaxes(x_test, 1, 2)
    
    return x_train, y_train, x_test, y_test


def load_dataset(file_name, normalization=z_normalization, one_hot=True):
    # load raw data
    if file_name in constants.UNIVARIATE_DATASET_NAMES_2018:
        x_train, y_train, x_test, y_test = load_tsv_data(file_name)
    elif file_name in constants.MULTIVARIATE_DATASET_NAMES_2018_ALL:
        x_train, y_train, x_test, y_test = load_ts_data(file_name)
    else:
        raise NotImplementedError
    
    # x preprocess
    if file_name in constants.UNIVARIATE_DATASET_NAMES_2018:
        # univariate
        x_train = np.expand_dims(x_train, axis=-1)
        x_test  = np.expand_dims(x_test, axis=-1)

    x_train = normalization(x_train)
    x_test  = normalization(x_test)

    # y preprocess
    le = sklearn.preprocessing.LabelEncoder()

    y_train = le.fit_transform(y_train)
    y_test  = le.fit_transform(y_test)

    if one_hot:
        y_train = np.expand_dims(y_train, axis=-1)
        y_test  = np.expand_dims(y_test, axis=-1)
        
        ohe = sklearn.preprocessing.OneHotEncoder(sparse_output=False)
        
        y_train = ohe.fit_transform(y_train)
        y_test  = ohe.fit_transform(y_test)
    
    return x_train, y_train, x_test, y_test
