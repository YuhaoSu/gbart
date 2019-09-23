from abc import abstractmethod

from gbart.modified_bartpy.model import Model
from gbart.modified_bartpy.tree import Tree


class Sampler:

    @abstractmethod
    def step(self, model: Model, tree: Tree):
        raise NotImplementedError()