import numpy

from typing import Any

class Task(object):
    @classmethod
    def pre_process(cls, multihops_path, specifics_path) -> tuple[Any, Any]:
        # This classmethod should return multihops, specifics
        raise NotImplementedError

    @classmethod
    def get_qualities_thresholds(cls, origin_data, metric):
        raise NotImplementedError

    @classmethod
    def get_multihops_qualities_thresholds(cls, multihops, metric):
        multihops_qualities, multihops_thresholds = cls.get_qualities_thresholds(multihops)
        return multihops_qualities, multihops_thresholds
    
    @classmethod
    def get_specifics_qualities_thresholds(cls, specifics, metric):
        specifics_qualities, specifics_thresholds = cls.get_qualities_thresholds(specifics)
        return specifics_qualities, specifics_thresholds
    
    @classmethod
    def combine_two_multihops_with_specifics(cls, multihops_thresholds_with_outlier, multihops_thresholds_without_outlier, 
                                            specifics_thresholds, multihops_qualities_with_outlier, 
                                            multihops_qualities_without_outlier, specifics_qualities):
        multihops_thresholds = numpy.concatenate((multihops_thresholds_with_outlier, multihops_thresholds_without_outlier, specifics_thresholds), axis=0)
        multihops_qualities = numpy.concatenate((multihops_qualities_with_outlier, multihops_qualities_without_outlier, specifics_qualities), axis=0)
        sorted_indices = numpy.argsort(multihops_thresholds)
        multihops_thresholds = multihops_thresholds[sorted_indices]
        multihops_qualities = multihops_qualities[sorted_indices]
        return multihops_qualities, multihops_thresholds 
    
    @classmethod
    def get_combined_multihops_and_specifics(cls, multihop_with_outlier_path, multihop_without_outlier_path, specific_path, metric):
        multihops_with_outlier, specifics = cls.pre_process(multihop_with_outlier_path, specific_path)
        multihops_without_outlier, _ = cls.pre_process(multihop_without_outlier_path, specific_path)

        multihops_qualities_with_outlier, multihops_thresholds_with_outlier = cls.get_qualities_thresholds(multihops_with_outlier, metric)
        multihops_qualities_without_outlier, multihops_thresholds_without_outlier = cls.get_qualities_thresholds(multihops_without_outlier, metric)
        specifics_qualities, specifics_thresholds = cls.get_qualities_thresholds(specifics, metric)
        
        multihops_qualities, multihops_thresholds = cls.combine_two_multihops_with_specifics(
                                                        multihops_thresholds_with_outlier, multihops_thresholds_without_outlier, 
                                                        specifics_thresholds, multihops_qualities_with_outlier, 
                                                        multihops_qualities_without_outlier, specifics_qualities) 

        return multihops_qualities, multihops_thresholds, specifics_qualities, specifics_thresholds 
    
