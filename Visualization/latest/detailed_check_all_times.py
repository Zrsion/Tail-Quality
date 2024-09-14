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


import tqdm
import numpy
import scipy
import pickle
import pathlib
import argparse

from typing import Literal

from younger.commons.io import save_json
from younger.commons.logging import logger


def get_latencies(times: list[float], percentiles: list[float]) -> list[tuple[float, list[float]]]:
    times = numpy.array(times)
    latencies = list()
    for percentile in percentiles:
        tail_latency = numpy.percentile(times, percentile*100)
        latencies.append((percentile, times[times >= tail_latency].tolist()))
    return latencies


def get_confidence_intervals(times: list[float], confidences: list[float]) -> list[tuple[float, tuple[float], float, float, float, float]]:
    times = numpy.array(times)
    mu, sigma = times.mean(), times.std(ddof=1)
    N = len(times)
    confidence_intervals = list()
    for confidence in confidences:
        confidence_intervals.append((confidence, scipy.stats.norm.interval(confidence, loc=mu, scale=sigma/numpy.sqrt(N)), mu, sigma, min(times), max(times)))
    return confidence_intervals


def analyze_all_times(all_times: list[dict[int, float]], confidences: list[float], percentiles: list[float], warm_run: int, window_size: int, fit_run_number: int, stage: Literal['fit', 'jsd']) -> tuple[list[tuple[int, list[tuple[float, list[float]]]]], list[tuple[int, list[tuple[float, tuple[float], float, float, float, float]]]]]:
    fit_distribution_number = 0

    all_latencies = list()
    all_confidence_intervals = list()

    current_times = list()
    for already_run, times in tqdm.tqdm(enumerate(all_times, start=1), total=len(all_times)):
        for batch_id, batch_time in times.items():
            current_times.append(batch_time)

        if already_run > warm_run and (already_run - warm_run) % fit_run_number == 0:
            if stage == 'fit':
                all_latencies.append((already_run, (get_latencies(current_times, percentiles))))
                all_confidence_intervals.append((already_run, (get_confidence_intervals(current_times, confidences))))

            if fit_distribution_number % window_size == 0 and fit_distribution_number != 0:
                if stage == 'jsd':
                    all_latencies.append((already_run, (get_latencies(current_times, percentiles))))
                    all_confidence_intervals.append((already_run, (get_confidence_intervals(current_times, confidences))))
            fit_distribution_number += 1
    return all_latencies, all_confidence_intervals


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-times-dirpath', type=str, required=True)
    parser.add_argument('--confidences', type=float, nargs='+', required=True)
    parser.add_argument('--percentiles', type=float, nargs='+', required=True)
    parser.add_argument('--fit-run-number', type=int, required=True)
    parser.add_argument('--window-size', type=int, required=True)
    parser.add_argument('--warm-run', type=int, required=True)
    parser.add_argument('--stage', type=str, choices=['jsd', 'fit'], required=True)
    parser.add_argument('--save-dirpath', type=str, required=True)
    args = parser.parse_args()

    fit_run_number = args.fit_run_number
    window_size = args.window_size
    warm_run = args.warm_run
    stage = args.stage
    save_dirpath = pathlib.Path(args.save_dirpath)

    assert all(0 <= confidence and confidence <= 1 for confidence in args.confidences)
    assert all(0 <= percentile and percentile <= 1 for percentile in args.percentiles)

    confidences = args.confidences
    percentiles = args.percentiles

    logger.info(f' = Stage: {stage}')

    fit_distribution_number = 0
    all_times_dirpath = pathlib.Path(args.all_times_dirpath)
    for times_dirpath in all_times_dirpath.iterdir():
        logger.info(f'Now processing {times_dirpath.name} ...')
        with open(times_dirpath.joinpath('All_Times.pickle'), 'rb') as f:
            times = pickle.load(f)

        inf_all_latencies, inf_all_confidence_intervals = analyze_all_times(times['inference'], confidences, percentiles, warm_run, window_size, fit_run_number, stage)
        tot_all_latencies, tot_all_confidence_intervals = analyze_all_times(times['total'], confidences, percentiles, warm_run, window_size, fit_run_number, stage)

        this_inf_l_stage_filepath = save_dirpath.joinpath(f'{times_dirpath.name}_I_L_{stage.upper()}.json')
        this_inf_c_stage_filepath = save_dirpath.joinpath(f'{times_dirpath.name}_I_C_{stage.upper()}.json')

        this_tot_l_stage_filepath = save_dirpath.joinpath(f'{times_dirpath.name}_T_L_{stage.upper()}.json')
        this_tot_c_stage_filepath = save_dirpath.joinpath(f'{times_dirpath.name}_T_C_{stage.upper()}.json')

        logger.info(f' -> Inference Latencies saved into: {this_inf_l_stage_filepath}')
        save_json(inf_all_latencies, this_inf_l_stage_filepath, indent=2)
        logger.info(f'    Length: {len(inf_all_latencies)}')

        logger.info(f' -> Total Latencies saved into: {this_tot_l_stage_filepath}')
        save_json(tot_all_latencies, this_tot_l_stage_filepath, indent=2)
        logger.info(f'    Length: {len(tot_all_latencies)}')

        logger.info(f' -> Inference Confidences saved into: {this_inf_c_stage_filepath}')
        save_json(inf_all_confidence_intervals, this_inf_c_stage_filepath, indent=2)
        logger.info(f'    Length: {len(inf_all_confidence_intervals)}')

        logger.info(f' -> Total Confidences saved into: {this_tot_c_stage_filepath}')
        save_json(tot_all_confidence_intervals, this_tot_c_stage_filepath, indent=2)
        logger.info(f'    Length: {len(tot_all_confidence_intervals)}')
