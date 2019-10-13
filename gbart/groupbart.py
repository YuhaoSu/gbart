#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 13:41:42 2019

@author: suyuhao
"""

import numpy as np
import gbart.utilities as ut
import copy
from gbart.modified_bartpy.sklearnmodel import SklearnModel
from sklearn.ensemble import RandomForestRegressor


def helper_model_acc(dataset, pair_list):
    np.random.shuffle(dataset)
    b = int(0.8 * np.shape(dataset)[0])
    Data_train = dataset[:b, :]
    Data_predict = dataset[b:, :]
    x_data = Data_train[:, :-1]
    y_data = Data_train[:, -1]
    # use original bart
    # build model
    model = SklearnModel(sublist=pair_list,
                         n_trees=50,
                         n_chains=4,
                         n_samples=50,
                         n_burn=300,
                         thin=0.1,
                         n_jobs=4)
    # fit and prediction
    _model_samples = model.fit(x_data, y_data)
    y_pred = model.predict(Data_predict[:, :-1])
    y_true = Data_predict[:, -1]
    acc = ut.get_error_reg(y_pred, y_true)
    return acc


def build_group_wise_model(dataset, pair_list):
    np.random.shuffle(dataset)
    b = int(0.8 * np.shape(dataset)[0])
    Data_train = dataset[:b, :]
    Data_predict = dataset[b:, :]
    x_data = Data_train[:, :-1]
    y_data = Data_train[:, -1]
    # use original bart
    # build model
    model = SklearnModel(sublist=pair_list,
                         n_trees=200,
                         n_chains=4,
                         n_samples=200,
                         n_burn=300,
                         thin=0.1,
                         n_jobs=4)
    # fit and prediction
    model.fit(x_data, y_data)
    y_pred = model.predict(Data_predict[:, :-1])
    y_true = Data_predict[:, -1]
    acc_g = ut.get_error_reg(y_pred, y_true)
    print("group wise accuracy:", acc_g)
    return acc_g


def build_true_group_model(dataset, pair_list):
    np.random.shuffle(dataset)
    b = int(0.8 * np.shape(dataset)[0])
    Data_train = dataset[:b, :]
    Data_predict = dataset[b:, :]
    x_data = Data_train[:, :-1]
    y_data = Data_train[:, -1]
    # use original bart
    # build model
    model = SklearnModel(sublist=pair_list,
                         n_trees=200,
                         n_chains=4,
                         n_samples=200,
                         n_burn=300,
                         thin=0.1,
                         n_jobs=4)
    # fit and prediction
    model.fit(x_data, y_data)
    y_pred = model.predict(Data_predict[:, :-1])
    y_true = Data_predict[:, -1]
    acc_g = ut.get_error_reg(y_pred, y_true)
    print("true group wise accuracy:", acc_g)
    return acc_g


def build_original_model(dataset):
    np.random.shuffle(dataset)
    b = int(0.8 * np.shape(dataset)[0])
    Data_train = dataset[:b, :]
    Data_predict = dataset[b:, :]
    x_data = Data_train[:, :-1]
    y_data = Data_train[:, -1]
    # use original bart
    # build model
    model = SklearnModel(sublist=None,
                         n_trees=200,
                         n_chains=4,
                         n_samples=200,
                         n_burn=300,
                         thin=0.1,
                         n_jobs=4)
    # fit and prediction
    model.fit(x_data, y_data)
    y_pred = model.predict(Data_predict[:, :-1])
    y_true = Data_predict[:, -1]
    acc_o = ut.get_error_reg(y_pred, y_true)
    print("original accuracy:", acc_o)
    return acc_o


def helper_rf_acc(dataset):
    np.random.shuffle(dataset)
    b = int(0.8 * np.shape(dataset)[0])
    Data_train = dataset[:b, :]
    Data_predict = dataset[b:, :]
    x_data = Data_train[:, :-1]
    y_data = Data_train[:, -1]
    # use original bart
    # build model
    model = RandomForestRegressor(
        n_estimators=100, max_depth=20, max_features=3, bootstrap=True)
    model.fit(x_data, y_data)
    y_pred = model.predict(Data_predict[:, :-1])
    y_true = Data_predict[:, -1]
    acc_rf = ut.get_error_reg(y_pred, y_true)
    print("random forest accuracy:", acc_rf)
    return acc_rf


def helper_flatten(list_A):
    flattened = []
    for sublist in list_A:
        for val in sublist:
            flattened.append(val)
    return flattened


def get_pair(dataset):
    # pre processing to delete variables
    num_var = np.shape(dataset)[1] - 1
    pair_list = None
    # get its original accuracy
    acc_best = helper_model_acc(dataset, pair_list)
    print("the original accuracy=", acc_best)
    # variable selection, delete unused variable
    temp_index_var = np.arange(num_var).tolist()
    temp_delete = []
    
    print("start variable selection!")
    for i in range(num_var):
        part1 = [temp_index_var[i]]
        temp_pair_list = list(set(temp_index_var) - set(part1))
        acc_i = helper_model_acc(dataset, [temp_pair_list])
        # recording the index of variable to be deleted
        if acc_i <= acc_best *0.85 :
            temp_delete.append(i)
    
    # delete variables
    index_var_remain = list(set(temp_index_var) - set(temp_delete))
    print("remaining variable:", index_var_remain)
    var_num_ = len(index_var_remain)

    # to get the index list
    acc_array = np.zeros(var_num_)
    for i in range(var_num_):
        part1 = [index_var_remain[i]]
        part2 = list(set(index_var_remain) - set(part1))
        temp_pair_list = [part1, part2]
        acc_array[i] = helper_model_acc(dataset, temp_pair_list)

    acc_array = acc_array.tolist()
    index_var_temp = sorted(acc_array, reverse=True)
    index_var = [index_var_remain[acc_array.index(
        ele)] for ele in index_var_temp]

    print("ranking finished! ranked variable:", index_var)

    output_pair = []
    while len(index_var) > 1:
        # try to find the first pair
        # fix the first variable, try to find its pair
        pair = [index_var[0]]
        index_var.pop(0)
        for i in range(len(index_var)):
            if len(output_pair) != 0:
                temp = copy.deepcopy(output_pair)
            else:
                temp = []
            # make the first pair
            pair.append(index_var[i])
            temp.append(pair)
            # make the first partition
            comp_pair = list(set(index_var) - set(helper_flatten(temp)))
            if len(comp_pair) != 0:
                temp.append(comp_pair)
            # check temp
            # get the associated acc
            acc_temp = helper_model_acc(dataset, temp)
            if acc_temp < acc_best:
                # print("temp",temp)
                pair_list = copy.deepcopy(temp)
                acc_best = acc_temp
            pair.pop()

        if pair_list is None:
            print("Partition finished in advanced")
            if len(output_pair) != 0:
                comp_pair = list(set(index_var_remain) -
                                 set(helper_flatten(output_pair)))
                output_pair.append(comp_pair)
                print("output the best current partition", output_pair)
                return output_pair
            else:
                print("No existing partition! Exit and we will use original BART!")
                return None

        else:
            index_var = copy.deepcopy(pair_list[-1])
            pair_list.pop()
            # print("pair_list",pair_list)
            output_pair = copy.deepcopy(pair_list)
            # print("output_pair",output_pair)
            pair_list = None
    comp_pair = list(set(index_var_remain) - set(helper_flatten(output_pair)))
    output_pair.append(comp_pair)
    print("partition process finished!")
    print("the best partition", output_pair)
    return output_pair

