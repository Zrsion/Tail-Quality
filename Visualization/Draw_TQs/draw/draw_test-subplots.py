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
# from metrics.vicuna       import Vicuna
# from metrics.light_gcn    import LightGCN
# from metrics.hybrid_nets  import HybridNets
# from metrics.mobile_net   import MobileNet
# from metrics.emotion_flow import EmotionFlow

tasks: dict[str, Task] = dict(
    detr         = DETR,
    # vicuna       = Vicuna,
    # light_gcn    = LightGCN,
    # hybrid_nets  = HybridNets,
    # mobile_net   = MobileNet,
    # emotion_flow = EmotionFlow,
)

tls = ['90%', '95%', '99%', '99.9%']

def draw_two_subplots(multihops_qualities, multihops_thresholds, specifics_qualities, specifics_thresholds, metric, time_type, save_name, save_dir):
    fig = plt.figure(figsize=(8,6))
    specs = GridSpec(ncols=2, nrows=1, figure=fig)
    ax1 = draw_without_outlier(specs[0,0], multihops_qualities, multihops_thresholds, specifics_qualities, specifics_thresholds, metric, time_type, save_name, save_dir)
    ax2 = draw_without_outlier(specs[0,1], multihops_qualities, multihops_thresholds, specifics_qualities, specifics_thresholds, metric, time_type, save_name, save_dir)
    lines_labels = [ax1.get_legend_handles_labels()[0]]
    lines, labels = [list(chain(*lol)) for lol in zip(*lines_labels)]

    plt.savefig(save_dir.joinpath(f'test_two_subplots.jpg'), format='jpg', dpi=300)
    # ax.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.15), ncol=2)     
    # plt.savefig(save_dir.joinpath(f'{save_name}.jpg'), format='jpg', dpi=300)
    # plt.savefig(save_dir.joinpath(f'{save_name}.pdf'), format='pdf', dpi=300)
    print(f' - Fig saved') 


def draw_without_outlier(spec, multihops_qualities, multihops_thresholds, specifics_qualities, specifics_thresholds, metric, time_type, save_name, save_dir):
    mul_avgs = numpy.average(multihops_qualities, axis=-1)
    mul_mins = numpy.min(multihops_qualities, axis=-1)
    mul_maxs = numpy.max(multihops_qualities, axis=-1)
    mul_stds = numpy.std(multihops_qualities, axis=-1)

    specifics_qualities = specifics_qualities[:-1]
    specifics_thresholds = specifics_thresholds[:-1] 
    spe_mins = numpy.min(specifics_qualities, axis=-1)
    spe_maxs = numpy.max(specifics_qualities, axis=-1)

    origin_quality = multihops_qualities[-1][-1]
    
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

    ax = brokenaxes(xlims=((0, 10),(100, 200), (12160, 12190)), despine=False, tilt=45, d=0.01, hspace=.2, subplot_spec=spec)
    
    ax.plot(multihops_thresholds[:-1]*1000, mul_maxs[:-1], label=f'Maximum', color=max_color, linewidth=2.0)
    ax.plot(multihops_thresholds[:-1]*1000, mul_mins[:-1], label=f'Minimum', color=min_color, linewidth=2.0)
    ax.plot(multihops_thresholds[:-1]*1000, mul_avgs[:-1], label=f'Average', color=avg_color, linewidth=2.0)
    ax.fill_between(multihops_thresholds[:-1]*1000, numpy.minimum(mul_avgs[:-1] + mul_stds[:-1], mul_maxs[:-1]), numpy.maximum(mul_avgs[:-1] - mul_stds[:-1], mul_mins[:-1]), color=avg_color, alpha=0.2)
    
    ax.vlines(specifics_thresholds*1000, spe_mins, spe_maxs, ls=':', color=vln_color, zorder=4)#, label=r'All $\alpha$s at $\theta$')

    ax.scatter(multihops_thresholds[-1]*1000, origin_quality, label='No Time Limit', color=ogn_color, marker='*', s=20, zorder=3)
    ax.scatter(specifics_thresholds*1000, spe_maxs, marker='v', edgecolors=maxs_color, facecolors='1', s=15, zorder=4)
    ax.scatter(specifics_thresholds*1000, spe_mins, marker='^', edgecolors=mins_color, facecolors='1', s=15, zorder=4)
    
    text = 'Tail Quality:\n'
    print(spe_maxs)

    for index ,(thr, min, max) in enumerate(zip(specifics_thresholds, spe_mins, spe_maxs), start=0):
        print(max)
        text += f'{min*100:.2f}~{max*100:.2f} ({tls[index]}={int(thr*1000)}ms)'
        if index != len(specifics_thresholds)-1:
            text += '\n'
        if index == len(specifics_thresholds)-1:
            # text += f'{origin_quality*100:.2f} (Origin Quality)'
            # text += '\n'
            # text += f'Origin Quality: {origin_quality*100:.2f}'
            pass

    ax.annotate(f'{origin_quality*100:.2f}', xy=(multihops_thresholds[-1]*1000, origin_quality),
            # xytext=(-1, numpy.sign(origin_quality)*3), textcoords="offset points",
            xytext=(12, -18), textcoords="offset points",
            horizontalalignment="right",
            verticalalignment="bottom" if origin_quality > 0 else "top",
            fontsize=10
            )
    
    ax.annotate(
        text,
        xy=(0.62, 0.15),  
        xycoords='figure fraction',
        ha='left',
        va='bottom',
        fontsize=10,
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='white',
            edgecolor='gray'
        )
    )    
    ax.set_title('DETR', fontsize=16)

    if time_type == 'inference':
        ax.set_xlabel('Inference Time (w/o pre/postprocess) Thresholds (Milliseconds)',fontsize=14, labelpad=24)
    else:
        ax.set_xlabel('Inference Time (w/ pre/postprocess) Thresholds (Milliseconds)',fontsize=14, labelpad=24) 
    ax.set_ylabel(f'Inference Quality ({metric})',fontsize=14, labelpad=28)
    ax.tick_params(axis='both', which='major', labelsize=10) 
    return ax
    


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

    args = parser.parse_args()

    task_name = args.task_name 
    task = tasks[task_name]
    time_type = args.time_type
    metric = args.metric
    save_name = args.save_name

    multihop_with_outlier_path = pathlib.Path(args.multihop_with_outlier_filepath)
    multihop_without_outlier_path = pathlib.Path(args.multihop_without_outlier_filepath)
    specific_path = pathlib.Path(args.specific_filepath)
    save_dir = pathlib.Path(args.save_dir)

    multihops_with_outlier, specifics = task.pre_process(multihop_with_outlier_path, specific_path)
    multihops_without_outlier, _ = task.pre_process(multihop_without_outlier_path, specific_path)

    multihops_qualities_with_outlier, multihops_thresholds_with_outlier = task.get_qualities_thresholds(multihops_with_outlier, metric)
    multihops_qualities_without_outlier, multihops_thresholds_without_outlier = task.get_qualities_thresholds(multihops_without_outlier, metric)
    specifics_qualities, specifics_thresholds = task.get_qualities_thresholds(specifics, metric)

    multihops_qualities, multihops_thresholds = task.combine_two_multihops_with_specifics(multihops_thresholds_with_outlier, multihops_thresholds_without_outlier, 
                                            specifics_thresholds, multihops_qualities_with_outlier, 
                                            multihops_qualities_without_outlier, specifics_qualities) 
    sorted_indices = numpy.argsort(multihops_thresholds)

    multihops_thresholds = multihops_thresholds[sorted_indices]
    multihops_qualities = multihops_qualities[sorted_indices]

    draw_two_subplots(multihops_qualities, multihops_thresholds, specifics_qualities, specifics_thresholds, metric, time_type, save_name, save_dir)
