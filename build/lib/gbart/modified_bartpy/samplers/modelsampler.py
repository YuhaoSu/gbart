from copy import deepcopy

import numpy as np
#from tqdm import tqdm

from gbart.modified_bartpy.model import Model, deep_copy_model
from gbart.modified_bartpy.samplers.schedule import SampleSchedule
from gbart.modified_bartpy.samplers.sampler import Sampler


from collections import defaultdict
from typing import List, Mapping, Union, Any, Type, Tuple


from gbart.modified_bartpy.trace import TraceLogger


Chain = Mapping[str, Union[List[Any], np.ndarray]]


class ModelSampler(Sampler):

    def __init__(self, schedule: SampleSchedule):
        self.schedule = schedule

    def step(self, model: Model):
        for step in self.schedule.steps(model):
            step()

    def samples(self, model: Model, n_samples: int, n_burn: int, thin: float=0.1, store_in_sample_predictions: bool=True) -> Tuple[List[Model], np.ndarray]:
        #print("Starting burn")
        #for _ in tqdm(range(n_burn)):
        for _ in range(n_burn):
            self.step(model)
        trace = []
        model_trace = []
        #print("Starting sampling")

        thin_inverse = 1. / thin

        #for ss in tqdm(range(n_samples)):
        for ss in range(n_samples):
            self.step(model)
            if ss % thin_inverse == 0:
                if store_in_sample_predictions:
                    trace.append(model.predict())
                model_trace.append(deep_copy_model(model))
        return model_trace, np.array(trace)
    
""" 
class ModelSampler(Sampler):

    def __init__(self,
                 schedule: SampleSchedule,
                 trace_logger_class: Type[TraceLogger]=TraceLogger):
        self.schedule = schedule
        self.trace_logger_class = trace_logger_class

    def step(self, model: Model, trace_logger: TraceLogger):
        step_result = defaultdict(list)
        for step_kind, step in self.schedule.steps(model):
            result = step()
            log_message = trace_logger[step_kind](result)
            if log_message is not None:
                step_result[step_kind].append(log_message)
        return {x: np.mean([1 if y else 0 for y in step_result[x]]) for x in step_result}

    def samples(self, model: Model,
                n_samples: int,
                n_burn: int,
                thin: float=0.1,
                store_in_sample_predictions: bool=True,
                store_acceptance: bool=True) -> Chain:
        #print("Starting burn")

        trace_logger = self.trace_logger_class()

        #for _ in tqdm(range(n_burn)):
        for _ in range(n_burn):
            self.step(model, trace_logger)
        trace = []
        model_trace = []
        acceptance_trace = []
        #print("Starting sampling")

        thin_inverse = 1. / thin

        #for ss in tqdm(range(n_samples)):
        for ss in range(n_samples):
            step_trace_dict = self.step(model, trace_logger)
            if ss % thin_inverse == 0:
                if store_in_sample_predictions:
                    in_sample_log = trace_logger["In Sample Prediction"](model.predict())
                    if in_sample_log is not None:
                        trace.append(in_sample_log)
                if store_acceptance:
                    acceptance_trace.append(step_trace_dict)
                model_log = trace_logger["Model"](model)
                if model_log is not None:
                    model_trace.append(model_log)
        return {
            "model": model_trace,
            "acceptance": acceptance_trace,
            "in_sample_predictions": trace
        }
"""