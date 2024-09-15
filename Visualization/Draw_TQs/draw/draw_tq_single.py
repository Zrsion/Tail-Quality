import numpy
import pathlib
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt

from metrics              import Task
from metrics.detr         import DETR
from metrics.vicuna       import Vicuna
from metrics.light_gcn    import LightGCN
from metrics.hybrid_nets  import HybridNets
from metrics.mobile_net   import MobileNet
from metrics.emotion_flow import EmotionFlow

from draw.draw_detr import draw_detr_A100_total_combined
from draw.draw_vicuna import draw_vicuna_A100_total_combined
from draw.draw_lightgcn import draw_lightgcn_A100_total_combined
from draw.draw_mobilenet import draw_mobilenet_A100_total_combined
from draw.draw_emotionflow import draw_emotionflow_A100_total_combined
from draw.draw_hybridnets import draw_hybridnets_A100_total_combined

from draw.draw_detr import draw_detr_A100_inference_combined
from draw.draw_vicuna import draw_vicuna_A100_inference_combined
from draw.draw_lightgcn import draw_lightgcn_A100_inference_combined
from draw.draw_mobilenet import draw_mobilenet_A100_inference_combined
from draw.draw_emotionflow import draw_emotionflow_A100_inference_combined
from draw.draw_hybridnets import draw_hybridnets_A100_inference_combined

tasks: dict[str, Task] = dict(
    detr         = DETR,
    vicuna       = Vicuna,
    light_gcn    = LightGCN,
    hybrid_nets  = HybridNets,
    mobile_net   = MobileNet,
    emotion_flow = EmotionFlow,
)

tls = ['90%', '95%', '99%', '99.9%']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="visualize Tail Quality results")
    parser.add_argument('--multihop-with-outlier-filepath', type=str, required=True)
    parser.add_argument('--multihop-without-outlier-filepath', type=str, required=True)
    parser.add_argument('--specific-filepath', type=str, required=True)
    parser.add_argument('--save-dir', type=str, required=True)
    parser.add_argument('--metric', type=str, required=True,
                        choices=['mAP', 'top1-Acc', 'top5-Acc', 'precision', 
                                 'recall', 'ndcg', 'miou', 'iou', 'weighted-f1'])

    parser.add_argument('--task-name', type=str, choices=tasks.keys(), required=True)
    parser.add_argument('--time-type', type=str, required=True)
    parser.add_argument('--save-name', type=str, required=True)
    parser.add_argument('--remove-outlier', action='store_true')
    parser.add_argument('--is-subplot', action='store_true', default=False)

    args = parser.parse_args()

    task_name = args.task_name 
    task = tasks[task_name]
    time_type = args.time_type
    metric = args.metric
    save_name = args.save_name
    is_subplot = args.is_subplot

    multihop_with_outlier_path = pathlib.Path(args.multihop_with_outlier_filepath)
    multihop_without_outlier_path = pathlib.Path(args.multihop_without_outlier_filepath)
    specific_path = pathlib.Path(args.specific_filepath)
    save_dir = pathlib.Path(args.save_dir)

    multihops_qualities, multihops_thresholds, specifics_qualities, specifics_thresholds  = task.get_combined_multihops_and_specifics(multihop_with_outlier_path, multihop_without_outlier_path, specific_path, metric)
    print(save_name)
    if save_name == 'DETR_PyTorch_A100_total_mAP' or save_name == 'DETR_PyTorch_A100_inference_mAP':
        if time_type == 'total':
            draw_detr_A100_total_combined(multihops_qualities, multihops_thresholds, specifics_qualities, specifics_thresholds, metric, time_type, save_name, save_dir, is_subplot)
        else: 
            draw_detr_A100_inference_combined(multihops_qualities, multihops_thresholds, specifics_qualities, specifics_thresholds, metric, time_type, save_name, save_dir, is_subplot)
    if save_name == 'Vicuna_PyTorch_A100_total_top1-Acc' or save_name == 'Vicuna_PyTorch_A100_inference_top1-Acc':
        if time_type == 'total':
            draw_vicuna_A100_total_combined(multihops_qualities, multihops_thresholds, specifics_qualities, specifics_thresholds, metric, time_type, save_name, save_dir, is_subplot)
        else:
            draw_vicuna_A100_inference_combined(multihops_qualities, multihops_thresholds, specifics_qualities, specifics_thresholds, metric, time_type, save_name, save_dir, is_subplot)
    if save_name == 'LightGCN_PyTorch_A100_total_ndcg' or save_name == 'LightGCN_PyTorch_A100_inference_ndcg':
        if time_type == 'total':
            draw_lightgcn_A100_total_combined(multihops_qualities, multihops_thresholds, specifics_qualities, specifics_thresholds, metric, time_type, save_name, save_dir, is_subplot)
        else:
            draw_lightgcn_A100_inference_combined(multihops_qualities, multihops_thresholds, specifics_qualities, specifics_thresholds, metric, time_type, save_name, save_dir, is_subplot)
    if save_name == 'MobileNet_PyTorch_A100_total_top1-Acc' or save_name == 'MobileNet_PyTorch_A100_inference_top1-Acc':
        if time_type == 'total':
            draw_mobilenet_A100_total_combined(multihops_qualities, multihops_thresholds, specifics_qualities, specifics_thresholds, metric, time_type, save_name, save_dir, is_subplot)
        else:
            draw_mobilenet_A100_inference_combined(multihops_qualities, multihops_thresholds, specifics_qualities, specifics_thresholds, metric, time_type, save_name, save_dir, is_subplot)
    if save_name == 'EmotionFlow_PyTorch_A100_total_weighted-f1' or save_name == 'EmotionFlow_PyTorch_A100_inference_weighted-f1':
        if time_type == 'total':
            draw_emotionflow_A100_total_combined(multihops_qualities, multihops_thresholds, specifics_qualities, specifics_thresholds, metric, time_type, save_name, save_dir, is_subplot)
        else:
            draw_emotionflow_A100_inference_combined(multihops_qualities, multihops_thresholds, specifics_qualities, specifics_thresholds, metric, time_type, save_name, save_dir, is_subplot)
    if save_name == 'HybridNets_PyTorch_A100_total_mAP' or save_name == 'HybridNets_PyTorch_A100_inference_mAP':
        if time_type == 'total':
            draw_hybridnets_A100_total_combined(multihops_qualities, multihops_thresholds, specifics_qualities, specifics_thresholds, metric, time_type, save_name, save_dir, is_subplot)
        else:
            draw_hybridnets_A100_inference_combined(multihops_qualities, multihops_thresholds, specifics_qualities, specifics_thresholds, metric, time_type, save_name, save_dir, is_subplot)
