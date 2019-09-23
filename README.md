# GBART


## Introduction

GBART is a pure python package to implement our proposed algorithm Gbart, it is based on Bayesian additive regression tree(BART) framework of Chipman et al. 

Through Gbart, We try to find potential grouping of variables in the sense that there is no interaction term between variables of different groups. For details please visit our recent paper.

This python package is build based on BartPy package, the pure python version of BART, for details please refer to the offical website of BartPy.


## Installation
### Using pip
```
pip install gbart
Note: a typo is fixing, somehow using pip will result in a missed subfolder(modified_bart), we recommend install it manually.
```
### Mannually installation
```
1. Visit our website https://github.com/augusHsu/gbart

2. Directly download folder "gbart"

3. put folders into place where you usually import packages, generally, it should be in 
 ~/bin/python3.7/site-packages/ 
 or put it with your script in same folder, then it will be fine.
```


## Usage and examples
### The easiest way to run and obtain result

* Preparation

```
import copy
import numpy as np
import gbart.utilities as ut  # provide helper functions
import gbart.create_dataset as cd  # data generator
from gbart.groupbart import * 
```
* generate dataset

```
dataset = cd.create_friedman()
# the last column of dataset is dependent variable, the output Y.
```
* Build model and get accuracy by using BART.

```
acc_o = build_original_model(dataset)
# This function splits the whole dataset into training and testing (80% for training)
# This function returns accuracy in testing data.
```

* Get the grouping information

```
output_pair = get_pair(dataset)
# return the the grouping information. The first phase of Gbart algorithm.

```

* Build the gbart model 

```
acc_g = build_group_wise_model(dataset, output_pair)
# take "output_pair" as pair_list 
# This function returns accuracy in testing data.

```
### To design a customization version and/or tune model parameters. Please consider the following. 

* Write your own helper function instead of calling functions in *groupbart.py*, the only thing you may need in *groupbart.py* is  *get_pair(dataset)*, which will help you find the proper grouping information.


* Tune the parameters of both BART and GBART model. Here is an easy example.

```
import numpy as np
import gbart.utilities as ut
from gbart.modified_bartpy.sklearnmodel import SklearnModel

# Data preparation 
b = int(0.8 * np.shape(dataset)[0])  
Data_train = dataset[:b,:]
Data_predict = dataset[b:,:]
x_data = Data_train[:,:-1]
y_data = Data_train[:,-1]


# Building the model
model = SklearnModel(sublist=None,
                     n_trees=50,
                     n_chains=4,
                     n_samples=50,
                     n_burn=200,
                     thin=0.1,
                     n_jobs=1)
# This GBART model inherited BART model, a new feature named sublist is added. 
# sublist can either take "None" or list of groups of variables as input.
# when sublist is None, it will be the exact BART model.
# when sublist is a list, it will build GBART model.
# To tune other parameters, please refer to BartPy usage guide.

# fit and prediction 
model.fit(x_data, y_data)
y_pred = model.predict(Data_predict[:,:-1])
y_true = Data_predict[:,-1]
acc = ut.get_error_reg(y_pred, y_true)

```



## Acknowledge  

We truly thank Mr. Jake Coltman for his contribution to BartPy package.


