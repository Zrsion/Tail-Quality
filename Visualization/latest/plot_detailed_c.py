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
    parser.add_argument('--c-filepath', type=str, required=True)
    parser.add_argument('--save-filepath', type=str, required=True)
    parser.add_argument('--nrows', type=int, required=True)
    parser.add_argument('--ncols', type=int, required=True)
    parser.add_argument('--fig-w', type=float, required=True)
    parser.add_argument('--fig-h', type=float, required=True)
    args = parser.parse_args()

    c_filepath = pathlib.Path(args.c_filepath)
    save_filepath = pathlib.Path(args.save_filepath)
    nrows, ncols = args.nrows, args.ncols

    all_confidence_intervals: list[tuple[int, list[tuple[float, tuple[float], float, float, float, float]]]] = load_json(c_filepath)

    fig, axs = pyplot.subplots(nrows=nrows, ncols=ncols, figsize=(args.fig_w, args.fig_h))
    fig_number = nrows * ncols

    xtick_labels = list()
    specifics = dict()
    for run_index, confidence_itervals  in all_confidence_intervals:
        xtick_labels.append(run_index)
        for confidence, (interval_l, interval_r), mu, sigma, min_time, max_time in confidence_itervals:
            specific_interval_ls, specific_interval_rs, specific_mus, specific_sigmas, specific_mins, specific_maxs = specifics.get(confidence, (list(), list(), list(), list(), list(), list()))
            specific_interval_ls.append(interval_l)
            specific_interval_rs.append(interval_r)
            specific_mus.append(mu)
            specific_sigmas.append(sigma)
            specific_mins.append(min_time)
            specific_maxs.append(max_time)
            specifics[confidence] = (specific_interval_ls, specific_interval_rs, specific_mus, specific_sigmas, specific_mins, specific_maxs)

    assert fig_number == len(specifics)
    xticks = [tick for tick in range(1, len(xtick_labels) + 1)]

    logger.info(f'Total {len(xtick_labels)} Round.')

    for index, (confidence, (ls, rs, mus, sigmas, mins, maxs)) in enumerate(specifics.items()):
        # print(confidence)
        if nrows == 1 and ncols == 1:
            ax = axs
        if nrows != 1 and ncols == 1:
            ax = axs[index]
        if nrows != 1 and ncols != 1:
            row_id, col_id = divmod(index, nrows)
            ax = axs[row_id, col_id]
        # ax.violinplot(latencies[:4], showmeans=True, showmedians=True)
        ls = numpy.array(ls)
        rs = numpy.array(rs)
        mus = numpy.array(mus)
        sigmas = numpy.array(sigmas)
        mins = numpy.array(mins)
        maxs = numpy.array(maxs)
        # ax.plot(xticks, mus, label='Mean')
        # ax.plot(xticks, rs-ls, label='Delta Iterval')
        # ax.fill_between(xticks, mus-sigmas, mus+sigmas, color='red', alpha=0.1, label='Confidence Interval')
        # ax.fill_between(xticks, ls, rs, color='blue', alpha=0.2, label='Confidence Interval')
        ax.fill_between(xticks, mins, maxs, color='cyan', alpha=0.1, label='Min & Max')
        ax.set_title(f'The Change of Statistics (confidence={confidence*100}%)')
        ax.set_xlabel('Inference Round')
        ax.set_ylabel('Statistics')
    # print(rs[0]-ls[0])
    # print(rs[-1]-ls[-1])
    pyplot.savefig(save_filepath)