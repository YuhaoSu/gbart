from typing import Callable, List, Mapping, Optional, Tuple
from operator import le, gt

import numpy as np

from gbart.modified_bartpy.errors import NoSplittableVariableException, NoPrunableNodeException
from gbart.modified_bartpy.mutation import TreeMutation, GrowMutation, PruneMutation
from gbart.modified_bartpy.node import LeafNode, DecisionNode, split_node
from gbart.modified_bartpy.samplers.treemutation.proposer import TreeMutationProposer
from gbart.modified_bartpy.split import SplitCondition
from gbart.modified_bartpy.tree import Tree


def uniformly_sample_grow_mutation(tree: Tree) -> TreeMutation:
    node = random_splittable_leaf_node(tree)
    updated_node = sample_split_node(node)
    updated_node.s_list = node.s_list
    return GrowMutation(node, updated_node)
    """
    if updated_node is None:
        _node = random_prunable_decision_node(tree)
        _updated_node = LeafNode(_node.split, depth=_node.depth)
        _updated_node.s_list = _node.s_list
        return PruneMutation(_node, _updated_node)
    else:
        updated_node.s_list = node.s_list
        return GrowMutation(node, updated_node)
    """


def uniformly_sample_prune_mutation(tree: Tree) -> TreeMutation:
    node = random_prunable_decision_node(tree)
    updated_node = LeafNode(node.split, depth=node.depth)
    updated_node.s_list = node.s_list
    return PruneMutation(node, updated_node)


class UniformMutationProposer(TreeMutationProposer):

    def __init__(self, prob_method: List[float]=None, prob_method_lookup: Mapping[Callable[[Tree], TreeMutation], float]=None):
        if prob_method_lookup is not None:
            self.prob_method_lookup = prob_method_lookup
        else:
            if prob_method is None:
                prob_method = [0.5, 0.5]
            self.prob_method_lookup = {x[0]: x[1] for x in zip([uniformly_sample_grow_mutation, uniformly_sample_prune_mutation], prob_method)}
        self.methods = list(self.prob_method_lookup.keys())
        self.proposals = None
        self.refresh_proposal_cache()

    def refresh_proposal_cache(self):
        self.proposals = list(np.random.choice(list(self.prob_method_lookup.keys()), p=list(self.prob_method_lookup.values()), size=250))
       
    def sample_mutation_method(self) -> Callable[[Tree], TreeMutation]:        
        prop = self.proposals.pop()    
        if len(self.proposals) == 0:
            self.refresh_proposal_cache()
      
        return prop

    def propose(self, tree: Tree) -> TreeMutation:
        method = self.sample_mutation_method()
        
        try:
            return method(tree)
        except NoSplittableVariableException:
            return self.propose(tree)
        except NoPrunableNodeException:
            return self.propose(tree)


def random_splittable_leaf_node(tree: Tree) -> LeafNode:
    """
    Returns a random leaf node that can be split in a non-degenerate way
    i.e. a random draw from the set of leaf nodes that have at least two distinct values in their covariate matrix
    """
    splittable_nodes = tree.splittable_leaf_nodes
    if len(splittable_nodes) > 0:
        return np.random.choice(splittable_nodes)
    else:
        raise NoSplittableVariableException()


def random_prunable_decision_node(tree: Tree) -> DecisionNode:
    """
    Returns a random decision node that can be pruned
    i.e. a random draw from the set of decision nodes that have two leaf node children
    """
    leaf_parents = tree.prunable_decision_nodes
    if len(leaf_parents) == 0:
        raise NoPrunableNodeException()
    return np.random.choice(leaf_parents)

"""
import utilities as ut

def sampling_var_list(node: LeafNode) -> list:
    
    d = [0,1,2]
    b = [3,4]
    c = [d,b]
    
    if node.is_root is True:
        #print(np.random.choice(2))
        var_list = c[np.random.choice(2)]
        np.savetxt("/Users/suyuhao/Documents/RandomForest/var_list.csv", var_list, delimiter=',')

    if node.is_root is False:
        var_list = np.array(ut.loadDataSet("/Users/suyuhao/Documents/RandomForest/var_list.csv")).flatten().astype(int)
        var_list = var_list.tolist()

    #print(var_list)
    return var_list

"""

def sample_split_condition(node: LeafNode) -> Optional[Tuple[SplitCondition, SplitCondition]]:
    """
    Randomly sample a splitting rule for a particular leaf node
    Works based on two random draws

      - draw a node to split on based on multinomial distribution
      - draw an observation within that variable to split on

    Returns None if there isn't a possible non-degenerate split
    
    """
    """
    a = [0,1,2]
    b = [3,4]
    c = [a,b]
    i = np.random.choice(2)
    #print(list(node.split.data.splittable_variables()))
    split_variable = np.random.choice(c[i])
    #split_variable = np.random.choice([0,1,2,3,4])
    #print(c[i])
    """
    #if node.s_list is not None : print(node.s_list)
    if node.s_list is None:
        available_list = list(node.split.data.splittable_variables())
        #print(len(available_list))
        split_variable = np.random.choice(available_list)
    else:           
    #print(list(node.split.data.splittable_variables()))
        splittable_variables_list = set(node.split.data.splittable_variables())
        s_variables_list = set(node.s_list)
        available_list = list(s_variables_list & splittable_variables_list)
        if len(available_list) == 0:
            raise NoSplittableVariableException()
        else:
            split_variable = np.random.choice(available_list)
        
    #split_variable = np.random.choice(list(node.split.data.splittable_variables()))
    split_value = node.data.random_splittable_value(split_variable)
    if split_value is None:
        return None
    return SplitCondition(split_variable, split_value, le), SplitCondition(split_variable, split_value, gt)


def sample_split_node(node: LeafNode) -> DecisionNode:
    """
    Split a leaf node into a decision node with two leaf children
    The variable and value to split on is determined by sampling from their respective distributions
    """
    conditions = sample_split_condition(node)
    if conditions is None:
        return None
    else:
        return split_node(node, conditions)