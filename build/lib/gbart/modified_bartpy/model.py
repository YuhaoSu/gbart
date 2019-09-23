from copy import deepcopy
from typing import List, Generator

import numpy as np
import pandas as pd

from gbart.modified_bartpy.data import Data
from gbart.modified_bartpy.sigma import Sigma
from gbart.modified_bartpy.tree import Tree, LeafNode, deep_copy_tree
from gbart.modified_bartpy.split import Split


class Model:

    def __init__(self, sublist: list, data: Data, sigma: Sigma, trees=None, n_trees: int = 50, alpha: float=0.95, beta: int=2., k: int=2.):
        self.sublist = sublist
        self.data = data
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.k = k
        self._sigma = sigma
        #print("showing:", self.sublist, self.data, self.sigma, n_trees, self.alpha, self.beta)


        if trees is None:
            #print("showing:", self.sublist, self.data, self.sigma, n_trees, self.alpha, self.beta)
            self.n_trees = n_trees
            self._trees = self.initialize_trees()
            """
            index = []
            
            if self.sublist is not None:
                
                for ele in range(len(self.sublist)):
                    index.append([])
                    for i in range(len(self.trees)):
                        if self.sublist[ele] == self.trees[i]._nodes[0].s_list:
                            index[ele].append(i)
                print(index)
            """
        else:
            self.n_trees = len(trees)
            self._trees = trees

        self._prediction = None
        

    def initialize_trees(self) -> List[Tree]:
        
        tree_data = deepcopy(self.data)
        tree_data._y = tree_data.y / self.n_trees
        
        def sub_list(d):  
            if d is None:
                return None
            else:
                len_d = len(d)
                var_list = d[np.random.choice(len_d)]
                return var_list
        
        trees = [Tree([LeafNode(Split(self.data))]) for _ in range(self.n_trees)]
        for _ in range(self.n_trees):
            
            trees[_]._nodes[0].s_list = sub_list(self.sublist)
            #print("i=",_," ", trees[_]._nodes[0].s_list)
            
            #print()
            #print(type(trees[_]._nodes[0]))
        print("initialize_trees done!")
        
        
        """
        for _ in range(self.n_trees):
            trees[_].is_root = True
        """
        return trees
    
    
    def residuals(self) -> np.ndarray:
        return self.data.y - self.predict()

    def unnormalized_residuals(self) -> np.ndarray:
        return self.data.unnormalized_y - self.data.unnormalize_y(self.predict())

    def predict(self, X: np.ndarray=None) -> np.ndarray:
        if X is not None:
            return self._out_of_sample_predict(X)
        #if X is not None and index is not None:
        #    return self.post_predict(index,X)
        
        return np.sum([tree.predict() for tree in self.trees], axis=0)

    def _out_of_sample_predict(self, X: np.ndarray):
        if type(X) == pd.DataFrame:
            X = X.values

        return np.sum([tree.predict(X) for tree in self.trees], axis=0)
    
    def par_predict(self, X: np.ndarray, group:list):

        if type(X) == pd.DataFrame:
            X = X.values
        
        pred = [tree.predict(X) for tree in self.trees]
        partial = []
        for _ in group:
            partial.append(pred[_])
        
        total = np.sum(pred, axis=0)
        part = np.sum(partial, axis=0)
        return part,total ,partial,pred
    
        #self.trees[0,5,9,11,15,16,17,19]
        

    @property
    def trees(self) -> List[Tree]:
        return self._trees

    def refreshed_trees(self) -> Generator[Tree, None, None]:
        if self._prediction is None:
            self._prediction = self.predict()
        for tree in self.trees:
            self._prediction -= tree.predict()
            tree.update_y(self.data.y - self._prediction)
            yield tree
            self._prediction += tree.predict()

    @property
    def sigma_m(self):
        return 0.5 / (self.k * np.power(self.n_trees, 0.5))

    @property
    def sigma(self):
        return self._sigma


def deep_copy_model(model: Model) -> Model:
    copied_model = Model(None, deepcopy(model.sublist), deepcopy(model.sigma), [deep_copy_tree(tree) for tree in model.trees])
    return copied_model
