import numpy
import pickle
import pathlib

from typing import Any, List, Literal

from . import Task

class DETR(Task):
    @classmethod
    def pre_process(cls, multihops_path: pathlib.Path, specifics_path: pathlib.Path) -> tuple[Any, Any]:
        with open(multihops_path, 'rb') as f:
            multihops = pickle.load(f)
        with open(specifics_path, 'rb') as f:
            specifics = pickle.load(f)
        return multihops, specifics

    @classmethod
    def get_qualities_thresholds(cls, origin_data: List, metric: Literal['mAP'] = 'mAP'):
        thresholds = numpy.array([thrsh for thrsh, _ in origin_data])
        qualities = [all_round_qualities for thrsh, all_round_qualities in origin_data]
        # start: remove nan
        max_len = max(len(item) for item in qualities[0] if isinstance(item, list))
        zero_list = [0.0] * max_len
        qualities = numpy.array([
            [numpy.where(numpy.isnan(tup), zero_list, tup) for tup in row]
            for row in qualities
        ])
        # end: remove nan
        if metric == 'mAP':
            selected_qualities = numpy.array([[tup[1] for tup in row] for row in qualities])

        return selected_qualities, thresholds
    
    
