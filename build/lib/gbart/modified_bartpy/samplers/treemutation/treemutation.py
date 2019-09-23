from typing import Optional

import numpy as np

from gbart.modified_bartpy.model import Model
from gbart.modified_bartpy.mutation import TreeMutation
from gbart.modified_bartpy.samplers.treemutation.proposer import TreeMutationProposer
from gbart.modified_bartpy.samplers.treemutation.likihoodratio import TreeMutationLikihoodRatio
from gbart.modified_bartpy.tree import Tree, mutate
from gbart.modified_bartpy.samplers.sampler import Sampler


class TreeMutationSampler(Sampler):
    """
    A sampler for tree mutation space.
    Responsible for producing samples of ways to mutate a tree within a model

    Works by combining a proposer and likihood evaluator into:
     - propose a mutation
     - assess likihood
     - accept if likihood higher than a uniform(0, 1) draw

    Parameters
    ----------
    proposer: TreeMutationProposer
    likihood_ratio: TreeMutationLikihoodRatio
    """

    def __init__(self, proposer: TreeMutationProposer, likihood_ratio: TreeMutationLikihoodRatio):
        self.proposer = proposer
        self.likihood_ratio = likihood_ratio

    def sample(self, model: Model, tree: Tree) -> Optional[TreeMutation]:
        proposal = self.proposer.propose(tree)
        ratio = self.likihood_ratio.log_probability_ratio(model, tree, proposal)
        if np.log(np.random.uniform(0, 1)) < ratio:
            return proposal
        else:
            return None

    def step(self, model: Model, tree: Tree) -> None:
        mutation = self.sample(model, tree)
        if mutation is not None:
            mutate(tree, mutation)
