#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:28:54 2018
# see the multinomial logistic dist. https://en.wikipedia.org/wiki/
@author: suyuhao
"""

#
import numpy as np


def create_dataset_six(size, gcov, t, u, v, key):
    """
    size := the number of sample points
    gcov: set up the cov inside of group, range [0,1]
    t,u,v: the coefficinets for these three variable communities
    """
    a = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            a[i, j] = gcov
    for i in range(2):
        a[i, i] = 1
    cov = np.block([
        [a, np.zeros((2, 2)), np.zeros((2, 2))],
        [np.zeros((2, 2)), a, np.zeros((2, 2))],
        [np.zeros((2, 2)), np.zeros((2, 2)), a]
    ])
    #print("check cov:")
    #print(cov)
    
    #cov = np.identity(6)
    mean = np.ones(6)
    
    y1 = np.zeros(size)
    y2 = np.zeros(size)
    y3 = np.zeros(size)
    y = np.zeros(size)
    data = np.random.multivariate_normal(mean, cov, size)
    bias = np.random.normal(0, 0.5, size)

    if key == "square":
        for i in range(size):
            y1[i] = t * (data[i, 0] + data[i, 1])**2
            y2[i] = u * (data[i, 2] + data[i, 3])**2
            y3[i] = v * (data[i, 4] + data[i, 5])**2
            y[i] = y1[i] + y2[i] + y3[i] + bias[i]

    elif key == "multiply":
        for i in range(size):
            y1[i] = t * (data[i, 0] * data[i, 1])
            y2[i] = u * (data[i, 2] * data[i, 3])
            y3[i] = v * (data[i, 4] * data[i, 5])
            y[i] = y1[i] + y2[i] + y3[i] + bias[i]
            #y[i] = y1[i] +y2[i] +y3[i] + data[i,6]**2+ bias[i]

    elif key == "multiply_2s":
        #data = np.delete(data,-1,axis=1)
        data[:, -1] = np.random.normal(1, 1, size)
        data[:, -2] = np.random.normal(1, 1, size)
        for i in range(size):
            y1[i] = t * (data[i, 0] * data[i, 1])
            y2[i] = u * (data[i, 2] * data[i, 3])
            y3[i] = v * data[i, 4] + v * data[i, 5]
            y[i] = y1[i] + y2[i] + y3[i] + bias[i]

    elif key == "square_2s":
        data[:, -1] = np.random.normal(1, 1, size)
        data[:, -2] = np.random.normal(1, 1, size)
        for i in range(size):
            y1[i] = t * (data[i, 0] + data[i, 1])**2
            y2[i] = u * (data[i, 2] + data[i, 3])**2
            y3[i] = v * data[i, 4]**2 + v * data[i, 5]**2
            y[i] = y1[i] + y2[i] + y3[i] + bias[i]

    elif key == "multiply_3sr":
        data[:, -1] = np.random.normal(1, 1, size)
        data[:, -2] = np.random.normal(1, 1, size)
        data[:, -3] = np.random.normal(1, 1, size)
        data[:, -4] = np.random.normal(1, 1, size)
        for i in range(size):
            y1[i] = t * (data[i, 0] * data[i, 1])
            y2[i] = u * data[i, 2] + u * data[i, 3]
            y3[i] = v * data[i, 4]
            y[i] = y1[i] + y2[i] + y3[i] + bias[i]

    elif key == "square_3sr":
        data[:, -1] = np.random.normal(1, 1, size)
        data[:, -2] = np.random.normal(1, 1, size)
        data[:, -3] = np.random.normal(1, 1, size)
        data[:, -4] = np.random.normal(1, 1, size)
        for i in range(size):
            y1[i] = t * (data[i, 0] + data[i, 1])**2
            y2[i] = u * data[i, 2]**2 + u * data[i, 3]**2
            y3[i] = v * data[i, 4]**2
            y[i] = y1[i] + y2[i] + y3[i] + bias[i]

    elif key == "sin_square":
        for i in range(size):
            y1[i] = t * np.sin(data[i, 0]) * np.sin(data[i, 1])
            y2[i] = u * (data[i, 2] + data[i, 3])**2
            y3[i] = v * (data[i, 4] + data[i, 5])**2
            y[i] = y1[i] + y2[i] + y3[i] + bias[i]

    elif key == "sin_multiply":
        for i in range(size):
            y1[i] = t * np.sin(data[i, 0]) * np.sin(data[i, 1])
            y2[i] = u * (data[i, 2] * data[i, 3])
            y3[i] = v * (data[i, 4] * data[i, 5])
            y[i] = y1[i] + y2[i] + y3[i] + bias[i]

    data_reg = np.concatenate((data, y[:, None]), axis=1)
    #np.savetxt('temp.csv', data_reg, delimiter=',')
    return data_reg


def create_dataset_twenty(size, gcov, t, u, v, z, key):
    """
    size := the number of sample points
    t,u,v: the coefficinets for these three variable communities
    """
    
    a = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            a[i, j] = gcov
    for i in range(2):
        a[i, i] = 1
    cov = np.block([
        [a, np.zeros((2, 2)), np.zeros((2, 2))],
        [np.zeros((2, 2)), a, np.zeros((2, 2))],
        [np.zeros((2, 2)), np.zeros((2, 2)), a]
    ])
    #print("check cov:")
    #print(cov)
    mean = np.ones(6)
    y1 = np.zeros(size)
    y2 = np.zeros(size)
    y3 = np.zeros(size)
    y4 = np.zeros(size)
    y = np.zeros(size)
    data_1 = np.random.multivariate_normal(mean, cov, size)
    data_2 = np.random.uniform(0, 1, [size, 14])
    data = np.concatenate((data_1, data_2), axis=1)
    bias = np.random.normal(0, 0.5, size)

    if key == "square":
        for i in range(size):
            y1[i] = t * (data_1[i, 0] + data_1[i, 1])**2
            y2[i] = u * (data_1[i, 2] + data_1[i, 3])**2
            y3[i] = v * (data_1[i, 4] + data_1[i, 5])**2
            y4[i] = z * np.sum(np.square(data_2[i, :]))
            y[i] = y1[i] + y2[i] + y3[i] + y4[i] + bias[i]

    elif key == "multiply":
        for i in range(size):
            y1[i] = t * (data[i, 0] * data[i, 1])
            y2[i] = u * (data[i, 2] * data[i, 3])
            y3[i] = v * (data[i, 4] * data[i, 5])
            y4[i] = z * np.sum(data_2[i, :])
            y[i] = y1[i] + y2[i] + y3[i] + y4[i] + bias[i]
            #y[i] = y1[i] +y2[i] +y3[i] + data[i,6]**2+ bias[i]

    elif key == "square_ar":
        for i in range(size):
            y1[i] = t * (data_1[i, 0] + data_1[i, 1])**2
            y2[i] = u * (data_1[i, 2] + data_1[i, 3])**2
            y3[i] = v * (data_1[i, 4] + data_1[i, 5])**2
            #y4[i] = z* np.sum(data_2[i,:])
            y[i] = y1[i] + y2[i] + y3[i] + bias[i]
            #y[i] = y1[i] +y2[i] +y3[i] + data[i,6]**2+ bias[i]

    elif key == "multiply_ar":
        for i in range(size):
            y1[i] = t * (data[i, 0] * data[i, 1])
            y2[i] = u * (data[i, 2] * data[i, 3])
            y3[i] = v * (data[i, 4] * data[i, 5])
            #y4[i] = z* np.sum(data_2[i,:])
            y[i] = y1[i] + y2[i] + y3[i] + bias[i]
            #y[i] = y1[i] +y2[i] +y3[i] + data[i,6]**2+ bias[i]

    elif key == "sin_square":
        for i in range(size):
            y1[i] = t * np.sin(data[i, 0]) * np.sin(data[i, 1])
            y2[i] = u * (data[i, 2] + data[i, 3])**2
            y3[i] = v * (data[i, 4] + data[i, 5])**2
            y4[i] = z * np.sum(data_2[i, :])
            y[i] = y1[i] + y2[i] + y3[i] + y4[i] + bias[i]

    elif key == "sin_multiply":
        for i in range(size):
            y1[i] = t * np.sin(data[i, 0]) * np.sin(data[i, 1])
            y2[i] = u * (data[i, 2] * data[i, 3])
            y3[i] = v * (data[i, 4] * data[i, 5])
            y4[i] = z * np.sum(data_2[i, :])
            y[i] = y1[i] + y2[i] + y3[i] + y4[i] + bias[i]

    data_reg = np.concatenate((data, y[:, None]), axis=1)
    #np.savetxt('temp.csv', data_reg, delimiter=',')
    return data_reg


def create_friedman():
    """
    x1, x2, …xp ~ i.i.d. uniform(0,1), noise ~ N(0,1) p=10 n=100
    Y = 10* sin(pi*x1x2) +20*(x3-0.5)^2 +10*x4+5*x5+noise
    """
    data = np.random.uniform(0, 1, [500, 7])
    y1 = np.zeros(500)
    y2 = np.zeros(500)
    y3 = np.zeros(500)
    y = np.zeros(500)
    bias = np.random.normal(0, 1, 500)
    for i in range(500):
        y1[i] = 10 * np.sin(np.pi * data[i, 0] * data[i, 1])
        y2[i] = 20 * (data[i, 2] - 0.5) ** 2
        y3[i] = 10 * data[i, 3] + 5 * data[i, 4]
        y[i] = y1[i] + y2[i] + y3[i] + bias[i]
    data_reg = np.concatenate((data, y[:, None]), axis=1)
    #np.savetxt('friedman.csv', data_reg, delimiter=',')
    return data_reg

#data_six = create_dataset_six(size=500, gcov=0.1, t=5, u=1, v=0.2, key="square")
#data_twenty = create_dataset_twenty(size=500, gcov=0.1, t=5, u=1, v=0.2, z=0.04, key="square")