#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-09-11 15:49
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy
import pathlib
import argparse

from matplotlib import pyplot

from younger.commons.io import load_json
from younger.commons.logging import logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--l-filepath', type=str, required=True)
    parser.add_argument('--save-filepath', type=str, required=True)
    parser.add_argument('--nrows', type=int, required=True)
    parser.add_argument('--ncols', type=int, required=True)
    parser.add_argument('--fig-w', type=float, required=True)
    parser.add_argument('--fig-h', type=float, required=True)
    args = parser.parse_args()

    l_filepath = pathlib.Path(args.l_filepath)
    save_filepath = pathlib.Path(args.save_filepath)
    nrows, ncols = args.nrows, args.ncols

    all_latencies: list[tuple[int, list[tuple[float, list[float]]]]] = load_json(l_filepath)

    fig, axs = pyplot.subplots(nrows=nrows, ncols=ncols, figsize=(args.fig_w, args.fig_h))
    fig_number = nrows * ncols

    xtick_labels = list()
    specific_latencies = dict()
    for run_index, percentile_latencies in all_latencies:
        xtick_labels.append(run_index)
        for percentile, latencies in percentile_latencies:
            specific_latencies[percentile] = specific_latencies.get(percentile, list()) + [numpy.array(latencies)]

    assert fig_number == len(specific_latencies)
    logger.info(f'Total {len(xtick_labels)} Violins.')

    for index, (percentile, latencies) in enumerate(specific_latencies.items()):
        if nrows == 1 and ncols == 1:
            ax = axs
        if nrows != 1 and ncols == 1:
            ax = axs[index]
        if nrows != 1 and ncols != 1:
            row_id, col_id = divmod(index, nrows)
            ax = axs[row_id, col_id]
        # ax.violinplot(latencies[:4], showmeans=True, showmedians=True)
        ax.boxplot(latencies[:4]+latencies[-4:])
        ax.set_title(f'The Change of Tail Latencies (>={percentile*100}%)')
        ax.set_xticks([tick for tick in range(1, len(xtick_labels[:4] + xtick_labels[-4:])+1)], labels=xtick_labels[:4] + xtick_labels[-4:])
        ax.set_xlabel('Inference Round')
        ax.set_ylabel('Inference Time (seconds)')
    pyplot.savefig(save_filepath)

    for index, (percentile, latencies) in enumerate(specific_latencies.items()):
        if nrows == 1 and ncols == 1:
            ax = axs
        if nrows != 1 and ncols == 1:
            ax = axs[index]
        if nrows != 1 and ncols != 1:
            row_id, col_id = divmod(index, nrows)
            ax = axs[row_id, col_id]
        # ax.violinplot(latencies[:4], showmeans=True, showmedians=True)
        ax.boxplot(latencies[:4]+latencies[-4:])
        ax.set_title(f'The Change of Tail Latencies (>={percentile*100}%)')
        ax.set_xticks([tick for tick in range(1, len(xtick_labels[:4] + xtick_labels[-4:])+1)], labels=xtick_labels[:4] + xtick_labels[-4:])
        ax.set_xlabel('Inference Round')
        ax.set_ylabel('Inference Time (seconds)')
    pyplot.savefig(save_filepath)