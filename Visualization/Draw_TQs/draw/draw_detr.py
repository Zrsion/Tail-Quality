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

tasks: dict[str, Task] = dict(
    detr         = DETR
)

tls = ['90%', '95%', '99%', '99.9%']


def draw_detr_A100_total_combined(multihops_qualities, multihops_thresholds, specifics_qualities, specifics_thresholds, metric, time_type, save_name, save_dir, is_subplot=False, **kwargs):
    from constants.plot_A100 import anno_opts, bax_opts

    mul_avgs = numpy.average(multihops_qualities, axis=-1)
    mul_mins = numpy.min(multihops_qualities, axis=-1)
    mul_maxs = numpy.max(multihops_qualities, axis=-1)
    mul_stds = numpy.std(multihops_qualities, axis=-1)

    specifics_qualities = specifics_qualities[:-1]
    specifics_thresholds = specifics_thresholds[:-1] 
    spe_mins = numpy.min(specifics_qualities, axis=-1)
    spe_maxs = numpy.max(specifics_qualities, axis=-1)

    origin_quality = multihops_qualities[-1][-1]

    if is_subplot:
        spec = kwargs.get('spec', None)
        assert spec is not None, "'spec' must be provided in kwargs if is_subplot."
    else:
        fig = plt.figure(figsize=(8,6))
        spec = GridSpec(ncols=1, nrows=1, figure=fig)[0,0] 

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

    ax = brokenaxes(xlims=((100, 200), (12160, 12190)), despine=False, subplot_spec=spec, **bax_opts)
    
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
            xytext=(12, -18), textcoords="offset points",
            horizontalalignment="right",
            verticalalignment="bottom" if origin_quality > 0 else "top",
            fontsize=10
            )
    
    ax.big_ax.annotate(
        text,
        **anno_opts 
    )    

    ax.set_title('DETR', fontsize=16)

    if time_type == 'inference':
        ax.set_xlabel('Inference Time (w/o pre/postprocess) Thresholds (Milliseconds)',fontsize=14, labelpad=24)
    else:
        ax.set_xlabel('Inference Time (w/ pre/postprocess) Thresholds (Milliseconds)',fontsize=14, labelpad=24) 
    ax.set_ylabel(f'Inference Quality ({metric})',fontsize=14, labelpad=28)
    ax.tick_params(axis='both', which='major', labelsize=10) 

    if is_subplot:
        return ax

    lines_labels = [ax.get_legend_handles_labels()[0]]
    lines, labels = [list(chain(*lol)) for lol in zip(*lines_labels)]

    ax.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.15), ncol=2) 

    plt.savefig(save_dir.joinpath(f'{save_name}.jpg'), format='jpg', dpi=300)
    # plt.savefig(save_dir.joinpath(f'{save_name}.pdf'), format='pdf', dpi=300)
    print(f' - Fig saved')

    
def draw_detr_A100_inference_combined(multihops_qualities, multihops_thresholds, specifics_qualities, specifics_thresholds, metric, time_type, save_name, save_dir, is_subplot=False, **kwargs):
    from constants.plot_A100 import anno_opts, bax_opts

    mul_avgs = numpy.average(multihops_qualities, axis=-1)
    mul_mins = numpy.min(multihops_qualities, axis=-1)
    mul_maxs = numpy.max(multihops_qualities, axis=-1)
    mul_stds = numpy.std(multihops_qualities, axis=-1)

    specifics_qualities = specifics_qualities[:-1]
    specifics_thresholds = specifics_thresholds[:-1] 
    spe_mins = numpy.min(specifics_qualities, axis=-1)
    spe_maxs = numpy.max(specifics_qualities, axis=-1)
    
    origin_quality = multihops_qualities[-1][-1]

    if is_subplot:
        spec = kwargs.get('spec', None)
        assert spec is not None, "'spec' must be provided in kwargs if is_subplot."
    else:
        fig = plt.figure(figsize=(8,6))
        spec = GridSpec(ncols=1, nrows=1, figure=fig)[0,0] 

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

    ax = brokenaxes(xlims=((90, 130), (10155, 10160)), despine=False, subplot_spec=spec, **bax_opts)
    
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
            xytext=(12, -18), textcoords="offset points",
            horizontalalignment="right",
            verticalalignment="bottom" if origin_quality > 0 else "top",
            fontsize=10
            )
    
    ax.big_ax.annotate(
        text,
        **anno_opts 
    )    

    ax.set_title('DETR', fontsize=16)

    if time_type == 'inference':
        ax.set_xlabel('Inference Time (w/o pre/postprocess) Thresholds (Milliseconds)',fontsize=14, labelpad=24)
    else:
        ax.set_xlabel('Inference Time (w/ pre/postprocess) Thresholds (Milliseconds)',fontsize=14, labelpad=24) 
    ax.set_ylabel(f'Inference Quality ({metric})',fontsize=14, labelpad=28)
    ax.tick_params(axis='both', which='major', labelsize=10) 

    if is_subplot:
        return ax

    lines_labels = [ax.get_legend_handles_labels()[0]]
    lines, labels = [list(chain(*lol)) for lol in zip(*lines_labels)]

    ax.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.15), ncol=2) 

    plt.savefig(save_dir.joinpath(f'{save_name}.jpg'), format='jpg', dpi=300)
    # plt.savefig(save_dir.joinpath(f'{save_name}.pdf'), format='pdf', dpi=300)
    print(f' - Fig saved')