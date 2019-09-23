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
import gbart.create_dataset as cd
# load dataset

def helper_model_acc(dataset, pair_list):
    b = int(0.8 * np.shape(dataset)[0])  
    Data_train = dataset[:b,:]
    Data_predict = dataset[b:,:]
    x_data = Data_train[:,:-1]
    y_data = Data_train[:,-1]
    # use original bart
    # build model
    model = SklearnModel(sublist=pair_list,
                     n_trees=50,
                     n_chains=4,
                     n_samples=50,
                     n_burn=200,
                     thin=0.1,
                     n_jobs=4)
    # fit and prediction 
    _model_samples = model.fit(x_data, y_data)
    y_pred = model.predict(Data_predict[:,:-1])
    y_true = Data_predict[:,-1]
    acc = ut.get_error_reg(y_pred, y_true)
    #var = ut.get_error_var(y_pred, y_true, acc)
    #print("pair_list:", pair_list)
    #print("accuracy:", acc)
    #print("variance:", var)
    return acc#,model,x_data,y_data

def build_group_wise_model(dataset,pair_list):
    b = int(0.8 * np.shape(dataset)[0])  
    Data_train = dataset[:b,:]
    Data_predict = dataset[b:,:]
    x_data = Data_train[:,:-1]
    y_data = Data_train[:,-1]
    # use original bart
    # build model
    model = SklearnModel(sublist=pair_list,
                     n_trees=200,
                     n_chains=4,
                     n_samples=200,
                     n_burn=200,
                     thin=0.1,
                     n_jobs=4)
    # fit and prediction 
    model.fit(x_data, y_data)
    y_pred = model.predict(Data_predict[:,:-1])
    y_true = Data_predict[:,-1]
    acc_g = ut.get_error_reg(y_pred, y_true)
    print("group wise accuracy:",acc_g)
    return acc_g

def build_original_model(dataset):
    b = int(0.8 * np.shape(dataset)[0])  
    Data_train = dataset[:b,:]
    Data_predict = dataset[b:,:]
    x_data = Data_train[:,:-1]
    y_data = Data_train[:,-1]
    # use original bart
    # build model
    model = SklearnModel(sublist=None,
                     n_trees=200,
                     n_chains=4,
                     n_samples=200,
                     n_burn=200,
                     thin=0.1,
                     n_jobs=4)
    # fit and prediction 
    model.fit(x_data, y_data)
    y_pred = model.predict(Data_predict[:,:-1])
    y_true = Data_predict[:,-1]
    acc_o = ut.get_error_reg(y_pred, y_true)
    print("original accuracy:",acc_o)
    return acc_o
    

def helper_rf_acc(dataset):
    b = int(0.8 * np.shape(dataset)[0])  
    Data_train = dataset[:b,:]
    Data_predict = dataset[b:,:]
    x_data = Data_train[:,:-1]
    y_data = Data_train[:,-1]
    # use original bart
    # build model
    model = RandomForestRegressor(n_estimators =200, max_depth =20, max_features =3,bootstrap =True )
    model.fit(x_data,y_data)
    y_pred = model.predict(Data_predict[:,:-1])
    y_true = Data_predict[:,-1]
    acc_rf = ut.get_error_reg(y_pred, y_true)
    #var = ut.get_error_var(y_pred, y_true, acc)
    print("random forest accuracy:", acc_rf)
    #print("variance:", var)
    return acc_rf#,model,x_data,y_data
#acc = helper_model_acc(multiply_1, [[0, 1], [2, 4], [3, 5]])



def helper_flatten(list_A):
    flattened = []
    for sublist in list_A :
        for val in sublist:
            flattened.append(val)
    return flattened

def get_pair(dataset):
    # pre processing to delete variables
    num_var = np.shape(dataset)[1] - 1 
    pair_list = None
    # get its original accuracy
    acc_best = helper_model_acc(dataset,pair_list)
    print("the original accuracy=",acc_best)
    # variable selection, delete unused variable
    temp_index_var = np.arange(num_var).tolist()
    temp_delete = []
    print("start variable selection!")
    for i in range(num_var):
        part1 = [temp_index_var[i]]
        temp_pair_list = list(set(temp_index_var) - set(part1))
        acc_i = helper_model_acc(dataset,[temp_pair_list])
        # recording the index of variable to be deleted
        if acc_i <= acc_best*0.95:
            temp_delete.append(i)
    
    # delete variables
    index_var_remain = list(set(temp_index_var) - set(temp_delete))
    print("remaining variable:",index_var_remain)
    var_num_ = len(index_var_remain)

    # to get the index list
    acc_array = np.zeros(var_num_)
    for i in range(var_num_):
        part1 = [index_var_remain[i]]
        part2 = list(set(index_var_remain) - set(part1))
        temp_pair_list = [part1, part2]
        acc_array[i] = helper_model_acc(dataset,temp_pair_list)
    
    acc_array = acc_array.tolist()
    index_var_temp = sorted(acc_array,reverse=True)
    index_var = [index_var_remain[acc_array.index(ele)] for ele in index_var_temp ]
    
    print("ranking finished! ranked variable:",index_var)
    
    output_pair = []
    while len(index_var) > 1:
        # try to find the first pair
        # fix the first variable, try to find its pair
        pair = [index_var[0]]
        index_var.pop(0)
        for i in range(len(index_var)):
            #print("output_pair",output_pair)
            if len(output_pair) != 0: 
                temp = copy.deepcopy(output_pair)
            else:
                temp = []
            # make the first pair
            pair.append(index_var[i])
            #print("temp",temp)
            temp.append(pair)
            # make the first partition
            comp_pair = list(set(index_var) - set(helper_flatten(temp)))
            if len(comp_pair) != 0: temp.append(comp_pair)
            # check temp
            # get the associated acc
            acc_temp = helper_model_acc(dataset,temp)
            #print("this is ",i,"th partition:",temp)
            #print("corresponding accuracy:",acc_temp)
            if acc_temp < acc_best:
                #print("temp",temp)
                pair_list = copy.deepcopy(temp)
                acc_best = acc_temp
            pair.pop()
        #print("one loop finished!")
        
        if pair_list is None:
            print("Partition finished in advanced")
            if len(output_pair) != 0:
                comp_pair = list(set(index_var_remain) - set(helper_flatten(output_pair)))
                output_pair.append(comp_pair)
                print("output the best current partition",output_pair)
                return output_pair
            else:
                print("No existing partition! Exit and we will use original BART!")
                return None

        else:
            index_var = copy.deepcopy(pair_list[-1])
            pair_list.pop()
            #print("pair_list",pair_list)
            output_pair = copy.deepcopy(pair_list)
            #print("output_pair",output_pair)
            pair_list = None
    comp_pair = list(set(index_var_remain) - set(helper_flatten(output_pair)))
    output_pair.append(comp_pair)
    print("partition process finished!")
    print("the best partition", output_pair)
    return output_pair

#output_pair = get_pair(square_1)
#output_pair = get_pair(square_2)
#acc,model,x_data,y_data = helper_model_acc(square_2,None)
#original_model, null_distribution = feature_importance(model,x_data,y_data,variable=0)

#acc = helper_model_acc(square_3rd_3,[[3,4],[0,1,2,5]])

#print("multiply_1")
#output_pair = get_pair(multiply_1)

def simulation(dataset):
    output_pair = get_pair(dataset)
    if output_pair is None:   
        acc_g = build_group_wise_model(dataset,output_pair)
        acc_o = acc_g
        acc_rf = helper_rf_acc(dataset)
    else:
        acc_g = build_group_wise_model(dataset,output_pair)
        acc_o = build_original_model(dataset)
        acc_rf = helper_rf_acc(dataset)
    return acc_g, acc_o, acc_rf

#print("multiply_2")
#acc_g, acc_o = simulation(multiply_2)


def test_normal_six(k,size,t,u,v,key,time):
    acc_g = np.zeros(time)
    acc_o = np.zeros(time)
    acc_rf = np.zeros(time)
    for i in range(time):
        dataset = cd.create_dataset_six(k,size,t,u,v,key)
        acc_g[i], acc_o[i],acc_rf[i] = simulation(dataset)
    gw_acc = np.mean(acc_g)
    gw_std = np.std(acc_g)/np.sqrt(time)
    or_acc = np.mean(acc_o)
    or_std = np.std(acc_o)/np.sqrt(time)
    rf_acc = np.mean(acc_rf)
    fr_std = np.std(acc_rf)/np.sqrt(time)
    print("final group wise accuracy:",gw_acc, "std:",gw_std)
    print("final original accuracy:",or_acc,"std:",or_std)
    print("final original accuracy:",rf_acc,"std:",fr_std)
    return gw_acc,gw_std,or_acc,or_std,rf_acc,fr_std


def test_friedman(time):
    acc_g = np.zeros(time)
    acc_o = np.zeros(time)
    acc_rf = np.zeros(time)
    for i in range(time):
        dataset = cd.create_friedman()
        acc_g[i], acc_o[i],acc_rf[i] = simulation(dataset)
    gw_acc = np.mean(acc_g)
    gw_std = np.std(acc_g)/np.sqrt(time)
    or_acc = np.mean(acc_o)
    or_std = np.std(acc_o)/np.sqrt(time)
    rf_acc = np.mean(acc_rf)
    fr_std = np.std(acc_rf)/np.sqrt(time)
    print("final group wise accuracy:",gw_acc, "std:",gw_std)
    print("final original accuracy:",or_acc,"std:",or_std)
    print("final original accuracy:",rf_acc,"std:",fr_std)
    return gw_acc,gw_std,or_acc,or_std,rf_acc,fr_std
    
"""
print("multiply_1")
gw_acc,gw_var,or_acc,or_var = test(k=2,size=500,t=5,u=1,v=0.2,key="multiply",time=10)

print("square_1")
gw_acc,gw_var,or_acc,or_var = test(k=2,size=500,t=5,u=1,v=0.2,key="square",time=10)
"""
#print("multiply_2")
#gw_acc,gw_var,or_acc,or_var = test(k=2,size=500,t=1,u=1,v=1,key="multiply",time=10)


#print("square_2")
#gw_acc,gw_var,or_acc,or_var = test(k=2,size=500,t=1,u=1,v=1,key="square",time=10)
#print("multiply_2_ss")
#gw_acc,gw_var,or_acc,or_var = test(k=2,size=500,t=1,u=1,v=1,key="multiply_ss",time=3)


#print("square_1")
#acc_g, acc_o = simulation(square_1)

#print("multiply_2_ss")
#index_var_4 = get_pair(multiply_2_ss)
"""
dataset = cd.create_friedman()
b = 400  
Data_train = dataset[:b,:]
Data_predict = dataset[b:,:]
x_data = Data_train[:,:-1]
y_data = Data_train[:,-1]
model = SklearnModel(sublist=[[0,1,2,3,4]],
                     n_trees=50,
                     n_chains=4,
                     n_samples=50,
                     n_burn=200,
                     thin=0.1,
                     n_jobs=4)
    # fit and prediction 
_model_samples = model.fit(x_data, y_data)
y_pred = model.predict(Data_predict[:,:-1])
y_true = Data_predict[:,-1]
acc = ut.get_error_reg(y_pred, y_true)
"""
#print(_model_samples)
#print(len(_model_samples))
#print(_model_samples[0].trees)
#print(len(_model_samples[0].trees))