#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-09-11 16:12
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pickle
import pathlib
import argparse

from younger.commons.logging import logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rjsd-threshold', type=float, required=True)
    parser.add_argument('--all-rjsds-dirpath', type=str, required=True)
    args = parser.parse_args()

    inf_invalid = list()
    tot_invalid = list()
    all_rjsds_dirpath = pathlib.Path(args.all_rjsds_dirpath)
    for rjsds_dirpath in all_rjsds_dirpath.iterdir():
        logger.info(f' v Now processing {rjsds_dirpath.name} ...')
        with open(rjsds_dirpath.joinpath('All_rJSDs.pickle'), 'rb') as f:
            rjsds = pickle.load(f)
        inf_rjsds = rjsds['inference']
        tot_rjsds = rjsds['total']

        min_inf_rjsd = min(inf_rjsds)
        min_tot_rjsd = min(tot_rjsds)
        logger.info(f' - Inference Min rJSD: {min_inf_rjsd}')
        logger.info(f' - Total Min rJSD: {min_tot_rjsd}')

        if min_inf_rjsd > args.rjsd_threshold:
            inf_invalid.append((rjsds_dirpath.name, min_inf_rjsd))
        if min_tot_rjsd > args.rjsd_threshold:
            tot_invalid.append((rjsds_dirpath.name, min_tot_rjsd))
    logger.info(f' ^ Done')

    logger.info(f' v Inference Invalid')
    for name, rjsd in inf_invalid:
        logger.info(f'  {name} - {rjsd}')

    logger.info(f' v Total Invalid')
    for name, rjsd in tot_invalid:
        logger.info(f'  {name} - {rjsd}')
