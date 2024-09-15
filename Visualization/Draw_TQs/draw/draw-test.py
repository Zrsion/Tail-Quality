import numpy
import pathlib
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt

from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec

from metrics              import Task
from metrics.detr         import DETR
from metrics.vicuna       import Vicuna
from metrics.light_gcn    import LightGCN
from metrics.hybrid_nets  import HybridNets
from metrics.mobile_net   import MobileNet
from metrics.emotion_flow import EmotionFlow

tasks: dict[str, Task] = dict(
    detr         = DETR,
    vicuna       = Vicuna,
    light_gcn    = LightGCN,
    hybrid_nets  = HybridNets,
    mobile_net   = MobileNet,
    emotion_flow = EmotionFlow,
)

tls = ['90%', '95%', '99%', '99.9%']


def draw_without_outlier(multihops_qualities, multihops_thresholds, specifics_qualities, specifics_thresholds, metric, time_type, save_name, save_dir):
    mul_avgs = numpy.average(multihops_qualities, axis=-1)
    mul_mins = numpy.min(multihops_qualities, axis=-1)
    mul_maxs = numpy.max(multihops_qualities, axis=-1)
    mul_stds = numpy.std(multihops_qualities, axis=-1)

    specifics_qualities = specifics_qualities[:-1]
    specifics_thresholds = specifics_thresholds[:-1] 
    spe_mins = numpy.min(specifics_qualities, axis=-1)
    spe_maxs = numpy.max(specifics_qualities, axis=-1)

    origin_quality = multihops_qualities[-1][-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    cmap = plt.get_cmap('tab20')
    ogn_color = cmap(6)
    max_color = cmap(10)
    maxs_color = cmap(12)
    min_color = cmap(14)
    mins_color = cmap(16)
    avg_color = cmap(0)
    vln_color = cmap(15) 

    mpl.rcParams['font.size'] = 14
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['legend.fontsize'] = 14

    ax.plot(multihops_thresholds*1000, mul_maxs, label=f'Maximum', color=max_color, linewidth=2.0)
    ax.plot(multihops_thresholds*1000, mul_mins, label=f'Minimum', color=min_color, linewidth=2.0)
    ax.plot(multihops_thresholds*1000, mul_avgs, label=f'Average', color=avg_color, linewidth=2.0)
    ax.fill_between(multihops_thresholds*1000, numpy.minimum(mul_avgs + mul_stds, mul_maxs), numpy.maximum(mul_avgs - mul_stds, mul_mins), color=avg_color, alpha=0.2)
    
    ax.vlines(specifics_thresholds*1000, spe_mins, spe_maxs, ls=':', color=vln_color, zorder=4)#, label=r'All $\alpha$s at $\theta$')
   
    # ax.scatter(multihops_thresholds[-1]*1000, origin_quality, label='No Time Limit', color=ogn_color, marker='*', s=15, zorder=3)
    ax.scatter(specifics_thresholds*1000, spe_maxs, marker='v', edgecolors=maxs_color, facecolors='1', s=15, zorder=4)
    ax.scatter(specifics_thresholds*1000, spe_mins, marker='^', edgecolors=mins_color, facecolors='1', s=15, zorder=4)
    
    print('spe_maxs:', spe_maxs)
    print('spe_mins:', spe_mins)

    text = 'Tail Quality:\n'

    for index ,(thr, min, max) in enumerate(zip(specifics_thresholds, spe_mins, spe_maxs), start=0):
        print(max)
        text += f'{min*100:.2f}~{max*100:.2f} ({tls[index]}={int(thr*1000)}ms)'
        if index != len(specifics_thresholds)-1:
            text += '\n' 
    
    ax.annotate(
        text,
        xy=(0.70, 0.05),  
        xycoords='axes fraction',
        ha='left',
        va='bottom',
        fontsize=12,
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='white',
            edgecolor='gray'
        )
    )
    if time_type == 'inference':
        ax.set_xlabel('Inference Time (w/o pre/postprocess) Thresholds (Milliseconds)',fontsize=12)
    else:
        ax.set_xlabel('Inference Time (w/ pre/postprocess) Thresholds (Milliseconds)',fontsize=12) 
    ax.set_ylabel(f'Inference Quality ({metric})',fontsize=12)

    lines_labels = [ax.get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='lower center', ncol=4)
    plt.tight_layout(rect=[0, 0.1, 1, 1]) 

    plt.savefig(save_dir.joinpath(f'{save_name}-removed-outliers.jpg'), format='jpg', dpi=300)
    # plt.savefig(save_dir.joinpath(f'{save_name}-removed-outliers.pdf'), format='pdf', dpi=300)
    print(f' - Fig saved') 


def draw_with_outlier(multihops_qualities, multihops_thresholds, specifics_qualities, specifics_thresholds, metric, time_type, save_name, save_dir):
    mul_avgs = numpy.average(multihops_qualities, axis=-1)
    mul_mins = numpy.min(multihops_qualities, axis=-1)
    mul_maxs = numpy.max(multihops_qualities, axis=-1)
    mul_stds = numpy.std(multihops_qualities, axis=-1)

    specifics_qualities = specifics_qualities[:-1]
    specifics_thresholds = specifics_thresholds[:-1] 
    spe_mins = numpy.min(specifics_qualities, axis=-1)
    spe_maxs = numpy.max(specifics_qualities, axis=-1)

    origin_quality = multihops_qualities[-1][-1]

    # fig = plt.figure(figsize=(10,8))
    # spec = GridSpec(ncols=1, nrows=1, figure=fig)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8)) 

    cmap = plt.get_cmap('tab20')
    ogn_color = cmap(6)
    max_color = cmap(10)
    maxs_color = cmap(12)
    min_color = cmap(14)
    mins_color = cmap(16)
    avg_color = cmap(0)
    vln_color = cmap(15) 

    mpl.rcParams['font.size'] = 14
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['legend.fontsize'] = 14

    # default_xticks = ax.get_xticks()
    # ax = brokenaxes(xlims=((100, 200), (12100, 12190)), hspace=.05, subplot_spec=spec[0,0])

    ax.plot(multihops_thresholds*1000, mul_maxs, label=f'Maximum', color=max_color, linewidth=2.0)
    ax.plot(multihops_thresholds*1000, mul_mins, label=f'Minimum', color=min_color, linewidth=2.0)
    ax.plot(multihops_thresholds*1000, mul_avgs, label=f'Average', color=avg_color, linewidth=2.0)
    ax.fill_between(multihops_thresholds*1000, numpy.minimum(mul_avgs + mul_stds, mul_maxs), numpy.maximum(mul_avgs - mul_stds, mul_mins), color=avg_color, alpha=0.2)
    
    ax.vlines(specifics_thresholds*1000, spe_mins, spe_maxs, ls=':', color=vln_color, zorder=4)#, label=r'All $\alpha$s at $\theta$')

    ax.scatter(multihops_thresholds[-1]*1000, origin_quality, label='No Time Limit', color=ogn_color, marker='*', s=15, zorder=3)
    ax.scatter(specifics_thresholds*1000, spe_maxs, marker='v', edgecolors=maxs_color, facecolors='1', s=15, zorder=4)
    ax.scatter(specifics_thresholds*1000, spe_mins, marker='^', edgecolors=mins_color, facecolors='1', s=15, zorder=4)
    
    print('spe_maxs:', spe_maxs)
    print('spe_mins:', spe_mins)
    text = 'Tail Quality:\n'

    for index ,(thr, min, max) in enumerate(zip(specifics_thresholds, spe_mins, spe_maxs), start=0):
        print(max)
        text += f'{min*100:.2f}~{max*100:.2f} ({tls[index]}={int(thr*1000)}ms)'
        if index != len(specifics_thresholds)-1:
            text += '\n' 
    
    ax.annotate(
        text,
        xy=(0.70, 0.05),  
        xycoords='axes fraction',
        ha='left',
        va='bottom',
        fontsize=12,
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='white',
            edgecolor='gray'
        )
    )
    if time_type == 'inference':
        ax.set_xlabel('Inference Time (w/o pre/postprocess) Thresholds (Milliseconds)',fontsize=12)
    else:
        ax.set_xlabel('Inference Time (w/ pre/postprocess) Thresholds (Milliseconds)',fontsize=12) 
    ax.set_ylabel(f'Inference Quality ({metric})',fontsize=12)

    lines_labels = [ax.get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='lower center', ncol=4)
    plt.tight_layout(rect=[0, 0.1, 1, 1]) 

    plt.savefig(save_dir.joinpath(f'{save_name}.jpg'), format='jpg', dpi=300)
    # plt.savefig(save_dir.joinpath(f'{save_name}.pdf'), format='pdf', dpi=300)
    print(f' - Fig saved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="visualize Tail Quality results")
    parser.add_argument('--multihop-filepath', type=str, required=True)
    parser.add_argument('--specific-filepath', type=str, required=True)
    parser.add_argument('--save-dir', type=str, required=True)
    parser.add_argument('--metric', type=str, required=True,
                        choices=['mAP', 'top1-Acc', 'top5-Acc', 'precision', 
                                 'recall', 'ndcg', 'miou', 'iou', 'weighted-f1'])

    parser.add_argument('--task-name', type=str, choices=tasks.keys(), required=True)
    parser.add_argument('--time-type', type=str, required=True)
    parser.add_argument('--save-name', type=str, required=True)
    parser.add_argument('--remove-outlier', action='store_true')

    args = parser.parse_args()

    task_name = args.task_name 
    task = tasks[task_name]
    time_type = args.time_type
    metric = args.metric
    save_name = args.save_name

    multihop_path = pathlib.Path(args.multihop_filepath)
    specific_path = pathlib.Path(args.specific_filepath)
    save_dir = pathlib.Path(args.save_dir)

    multihops, specifics = task.pre_process(multihop_path, specific_path)
    multihops_qualities, multihops_thresholds = task.get_qualities_thresholds(multihops, metric)
    specifics_qualities, specifics_thresholds = task.get_qualities_thresholds(specifics, metric)

    if args.remove_outlier:
        draw_without_outlier(multihops_qualities, multihops_thresholds, specifics_qualities, specifics_thresholds, metric, time_type, save_name, save_dir)
    else:
        draw_with_outlier(multihops_qualities, multihops_thresholds, specifics_qualities, specifics_thresholds, metric, time_type, save_name, save_dir)
    
