import numpy
import pathlib
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt

from itertools import chain
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec

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

tasks: dict[str, Task] = dict(
    detr         = DETR,
    vicuna       = Vicuna,
    light_gcn    = LightGCN,
    hybrid_nets  = HybridNets,
    mobile_net   = MobileNet,
    emotion_flow = EmotionFlow,
)

tls = ['90%', '95%', '99%', '99.9%']
metrics = ['mAP', 'top1-Acc', 'top5-Acc', 'precision', 'recall', 'ndcg', 'miou', 'iou', 'weighted-f1']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="visualize Tail Quality results")
    parser.add_argument('--detr-multihop-with-outlier-filepath', type=str, required=True)
    parser.add_argument('--detr-multihop-without-outlier-filepath', type=str, required=True)
    parser.add_argument('--detr-specific-filepath', type=str, required=True)
    parser.add_argument('--detr-metric', type=str, required=True, choices=metrics)

    parser.add_argument('--vicuna-multihop-with-outlier-filepath', type=str, required=True)
    parser.add_argument('--vicuna-multihop-without-outlier-filepath', type=str, required=True)
    parser.add_argument('--vicuna-specific-filepath', type=str, required=True)
    parser.add_argument('--vicuna-metric', type=str, required=True, choices=metrics)

    parser.add_argument('--lightgcn-multihop-with-outlier-filepath', type=str, required=True)
    parser.add_argument('--lightgcn-multihop-without-outlier-filepath', type=str, required=True)
    parser.add_argument('--lightgcn-specific-filepath', type=str, required=True)
    parser.add_argument('--lightgcn-metric', type=str, required=True, choices=metrics)

    parser.add_argument('--hybridnets-multihop-with-outlier-filepath', type=str, required=True)
    parser.add_argument('--hybridnets-multihop-without-outlier-filepath', type=str, required=True)
    parser.add_argument('--hybridnets-specific-filepath', type=str, required=True)
    parser.add_argument('--hybridnets-metric', type=str, required=True, choices=metrics)

    parser.add_argument('--mobilenet-multihop-with-outlier-filepath', type=str, required=True)
    parser.add_argument('--mobilenet-multihop-without-outlier-filepath', type=str, required=True)
    parser.add_argument('--mobilenet-specific-filepath', type=str, required=True)
    parser.add_argument('--mobilenet-metric', type=str, required=True, choices=metrics)

    parser.add_argument('--emotionflow-multihop-with-outlier-filepath', type=str, required=True)
    parser.add_argument('--emotionflow-multihop-without-outlier-filepath', type=str, required=True)
    parser.add_argument('--emotionflow-specific-filepath', type=str, required=True)
    parser.add_argument('--emotionflow-metric', type=str, required=True, choices=metrics)

    parser.add_argument('--save-dir', type=str, required=True)
    parser.add_argument('--time-type', type=str, required=True)
    parser.add_argument('--is-subplot', action='store_true', required=True, default=False)

    args = parser.parse_args()

    time_type = args.time_type
    is_subplot = args.is_subplot
    save_name = f'PyTorch_A100_{time_type}'
    save_dir = pathlib.Path(args.save_dir)

    detr_multihop_with_outlier_path = pathlib.Path(args.detr_multihop_with_outlier_filepath)
    detr_multihop_without_outlier_path = pathlib.Path(args.detr_multihop_without_outlier_filepath)
    detr_specific_path = pathlib.Path(args.detr_specific_filepath)
    detr_metric = args.detr_metric

    vicuna_multihop_with_outlier_path = pathlib.Path(args.vicuna_multihop_with_outlier_filepath)
    vicuna_multihop_without_outlier_path = pathlib.Path(args.vicuna_multihop_without_outlier_filepath)
    vicuna_specific_path = pathlib.Path(args.vicuna_specific_filepath)
    vicuna_metric = args.vicuna_metric

    lightgcn_multihop_with_outlier_path = pathlib.Path(args.lightgcn_multihop_with_outlier_filepath)
    lightgcn_multihop_without_outlier_path = pathlib.Path(args.lightgcn_multihop_without_outlier_filepath)
    lightgcn_specific_path = pathlib.Path(args.lightgcn_specific_filepath)
    lightgcn_metric = args.lightgcn_metric

    hybridnets_multihop_with_outlier_path = pathlib.Path(args.hybridnets_multihop_with_outlier_filepath)
    hybridnets_multihop_without_outlier_path = pathlib.Path(args.hybridnets_multihop_without_outlier_filepath)
    hybridnets_specific_path = pathlib.Path(args.hybridnets_specific_filepath)
    hybridnets_metric = args.hybridnets_metric

    mobilenet_multihop_with_outlier_path = pathlib.Path(args.mobilenet_multihop_with_outlier_filepath)
    mobilenet_multihop_without_outlier_path = pathlib.Path(args.mobilenet_multihop_without_outlier_filepath)
    mobilenet_specific_path = pathlib.Path(args.mobilenet_specific_filepath)
    mobilenet_metric = args.mobilenet_metric

    emotionflow_multihop_with_outlier_path = pathlib.Path(args.emotionflow_multihop_with_outlier_filepath)
    emotionflow_multihop_without_outlier_path = pathlib.Path(args.emotionflow_multihop_without_outlier_filepath)
    emotionflow_specific_path = pathlib.Path(args.emotionflow_specific_filepath)
    emotionflow_metric = args.emotionflow_metric

    detr_multihops_qualities, detr_multihops_thresholds, detr_specifics_qualities, detr_specifics_thresholds  = DETR.get_combined_multihops_and_specifics(detr_multihop_with_outlier_path, detr_multihop_without_outlier_path, detr_specific_path, detr_metric)
    vicuna_multihops_qualities, vicuna_multihops_thresholds, vicuna_specifics_qualities, vicuna_specifics_thresholds  = Vicuna.get_combined_multihops_and_specifics(vicuna_multihop_with_outlier_path, vicuna_multihop_without_outlier_path, vicuna_specific_path, vicuna_metric)
    lightgcn_multihops_qualities, lightgcn_multihops_thresholds, lightgcn_specifics_qualities, lightgcn_specifics_thresholds  = LightGCN.get_combined_multihops_and_specifics(lightgcn_multihop_with_outlier_path, lightgcn_multihop_without_outlier_path, lightgcn_specific_path, lightgcn_metric)
    hybridnets_multihops_qualities, hybridnets_multihops_thresholds, hybridnets_specifics_qualities, hybridnets_specifics_thresholds  = HybridNets.get_combined_multihops_and_specifics(hybridnets_multihop_with_outlier_path, hybridnets_multihop_without_outlier_path, hybridnets_specific_path, hybridnets_metric)
    mobilenet_multihops_qualities, mobilenet_multihops_thresholds, mobilenet_specifics_qualities, mobilenet_specifics_thresholds  = MobileNet.get_combined_multihops_and_specifics(mobilenet_multihop_with_outlier_path, mobilenet_multihop_without_outlier_path, mobilenet_specific_path, mobilenet_metric)
    emotionflow_multihops_qualities, emotionflow_multihops_thresholds, emotionflow_specifics_qualities, emotionflow_specifics_thresholds  = EmotionFlow.get_combined_multihops_and_specifics(emotionflow_multihop_with_outlier_path, emotionflow_multihop_without_outlier_path, emotionflow_specific_path, emotionflow_metric)

    fig = plt.figure(figsize=(24,11))
    # fig.subplots_adjust(bottom=0.2) 
    specs = GridSpec(ncols=3, nrows=2, figure=fig)
    specs.update(wspace=0.4, hspace=0.3) 
    
    ax1 = draw_detr_A100_total_combined(detr_multihops_qualities, detr_multihops_thresholds, detr_specifics_qualities, detr_specifics_thresholds, detr_metric, time_type, save_name, save_dir, is_subplot=True, spec=specs[0,0])
    ax2 = draw_vicuna_A100_total_combined(vicuna_multihops_qualities, vicuna_multihops_thresholds, vicuna_specifics_qualities, vicuna_specifics_thresholds, vicuna_metric, time_type, save_name, save_dir, is_subplot=True, spec=specs[0,1])
    ax3 = draw_lightgcn_A100_total_combined(lightgcn_multihops_qualities, lightgcn_multihops_thresholds, lightgcn_specifics_qualities, lightgcn_specifics_thresholds, lightgcn_metric, time_type, save_name, save_dir, is_subplot=True, spec=specs[0,2])
    ax4 = draw_hybridnets_A100_total_combined(hybridnets_multihops_qualities, hybridnets_multihops_thresholds, hybridnets_specifics_qualities, hybridnets_specifics_thresholds, hybridnets_metric, time_type, save_name, save_dir, is_subplot=True, spec=specs[1,0])
    ax5 = draw_mobilenet_A100_total_combined(mobilenet_multihops_qualities, mobilenet_multihops_thresholds, mobilenet_specifics_qualities, mobilenet_specifics_thresholds, mobilenet_metric, time_type, save_name, save_dir, is_subplot=True, spec=specs[1,1])
    ax6 = draw_emotionflow_A100_total_combined(emotionflow_multihops_qualities, emotionflow_multihops_thresholds, emotionflow_specifics_qualities, emotionflow_specifics_thresholds, emotionflow_metric, time_type, save_name, save_dir, is_subplot=True, spec=specs[1,2])
    
    lines_labels = [ax1.get_legend_handles_labels()[0]]
    lines, labels = [list(chain(*lol)) for lol in zip(*lines_labels)]
    # fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 2), ncol=4)    
    fig.legend(lines, labels, loc='lower center', ncol=4) 
    plt.savefig(save_dir.joinpath(f'{save_name}.jpg'), format='jpg', dpi=300)
    # plt.savefig(save_dir.joinpath(f'{save_name}.pdf'), format='pdf', dpi=300)
    print(f' - Fig saved') 


