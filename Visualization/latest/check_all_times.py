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
import scipy
import pickle
import pathlib
import argparse

from younger.commons.logging import logger


def get_confidence_interval(samples: list[float], confidences: list[float]):
    samples = numpy.array(samples)
    mu, sigma = samples.mean(), samples.std(ddof=1)
    N = len(samples)
    confidence_intervals = list()
    for confidence in confidences:
        confidence_intervals.append(scipy.stats.norm.interval(confidence, loc=mu, scale=sigma/numpy.sqrt(N)))
    return confidence_intervals, mu, sigma


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-times-dirpath', type=str, required=True)
    parser.add_argument('--confidences', type=float, nargs='+')
    args = parser.parse_args()

    assert all(0 <= confidence and confidence <= 1 for confidence in args.confidences)

    all_times_dirpath = pathlib.Path(args.all_times_dirpath)
    for times_dirpath in all_times_dirpath.iterdir():
        logger.info(f'Now processing {times_dirpath.name} ...')
        with open(times_dirpath.joinpath('All_Times.pickle'), 'rb') as f:
            times = pickle.load(f)
        inf_times = [batch_inf_time for inf_time in times['inference'] for batch_id, batch_inf_time in inf_time.items()]
        tot_times = [batch_tot_time for tot_time in times['total'] for batch_id, batch_tot_time in tot_time.items()]

        logger.info(f' v Inference:')
        min_inf_t = min(inf_times)
        max_inf_t = max(inf_times)
        confidence_intervals, mu, sigma = get_confidence_interval(inf_times, args.confidences)
        confidence_intervals_string = f'   -> Confidence (Mean={mu:.4g}, N={len(inf_times)}):\n'
        for confidence, confidence_interval in zip(args.confidences, confidence_intervals):
            confidence_intervals_string += f'      {confidence*100}% - ({confidence_interval[0]:.4g}, {confidence_interval[1]:.4g})\n'
        logger.info(
            f'\n'
            f'   -> (Mean, STD) - ({mu:.4g}, {sigma:.4g}) | [Min, Max] - ({min_inf_t:.4g}, {max_inf_t:.4g})\n'
            f'{confidence_intervals_string}'
        )

        logger.info(f' v Total:')
        min_tot_t = min(tot_times)
        max_tot_t = max(tot_times)
        confidence_intervals, mu, sigma = get_confidence_interval(tot_times, args.confidences)
        confidence_intervals_string = f'   -> Confidence (Mean={mu:.4g}, N={len(tot_times)}):\n'
        for confidence, confidence_interval in zip(args.confidences, confidence_intervals):
            confidence_intervals_string += f'      {confidence*100}% - ({confidence_interval[0]:.4g}, {confidence_interval[1]:.4g})\n'
        logger.info(
            f'\n'
            f'   -> (Mean, STD) - ({mu:.4g}, {sigma:.4g}) | [Min, Max] - ({min_tot_t:.4g}, {max_tot_t:.4g})\n'
            f'{confidence_intervals_string}'
        )
