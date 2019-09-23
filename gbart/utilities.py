#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 01:19:44 2018

@author: suyuhao
"""
import numpy as np

"""=======BEFORE BUIDING RANDOM FOREST======="""
"""load dataset"""
def loadDataSet(r):
    dataSet = []
    f = open(r)
    #print(f)
    #("sss")
    fr = f.readlines()
    for line in fr:
        line = line.strip().split(',')
        linef = [float(li) for li in line]
        dataSet.append(linef)
    dataSet = np.array(dataSet)
    #print(dataSet)
    #dataSetMat = mat(dataSet)
    #print(dataSetMat) 
    return dataSet


"""=======AFTER BUIDING RANDOM FOREST======="""
#计算回归时误差，输出相对误差的均值
def get_error_reg(y_pred, y_true):
    m = np.shape(y_pred)[0]
    e = np.zeros((m))
    for i in range(m):
        e[i] = np.square(y_pred[i]-y_true[i])
    e = np.mean(e)
    return e


def get_error_var(y_pred, y_true, exp):
    #exp = get_error_reg(y_pred,y_true)
    m = np.shape(y_pred)[0]
    e = np.zeros((m))
    for i in range(m):
        e[i] = (y_pred[i]-y_true[i])**4 
    var = np.mean(e) - exp**2
    return var

#计算分类时误差，输出正确率(错误率)
def get_error_cla(y_pred,y_true):
    m = np.shape(y_pred)[0]
    j =0
    wronglist = []
    for i in range(m):
        if y_pred[i] == y_true[i]:
            j = j+1
        else:
            wronglist.append(i)
    #print()
    #print("this is wrongindex for classification",wronglist)
    return 1-j/m
