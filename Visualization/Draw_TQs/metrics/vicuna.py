import numpy
import pickle
import pathlib

from typing import Any, List, Literal

from . import Task

class Vicuna(Task):
    @classmethod
    def pre_process(cls, multihops_path: pathlib.Path, specifics_path: pathlib.Path) -> tuple[Any, Any]:
        with open(multihops_path, 'rb') as f:
            multihops = pickle.load(f)
        with open(specifics_path, 'rb') as f:
            specifics = pickle.load(f)
        return multihops, specifics

    @classmethod
    def get_qualities_thresholds(cls, origin_data: List, metric: Literal['top1-Acc'] = 'top1-Acc'):
        thresholds = numpy.array([thrsh for thrsh, _ in origin_data])
        qualities = numpy.array([all_round_qualities for thrsh, all_round_qualities in origin_data])
        if metric == 'top1-Acc':
            selected_qualities = qualities

        return selected_qualities, thresholds
    
    
