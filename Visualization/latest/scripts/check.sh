#!/bin/bash
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-09-14 17:06
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

PROJ_HOME=/younger/0.Sources/Tail-Quality/Visualization/latest

python ${PROJ_HOME}/detailed_check_all_times.py --all-times-dirpath ./TQResults/ --confidences 0.99 0.95 0.90 --percentiles 0.99 0.95 0.90 --fit-run-number 1 --window-size 5 --warm-run 1 --stage jsd --save-dirpath ./analysis
python ${PROJ_HOME}/detailed_check_all_times.py --all-times-dirpath ./TQResults/ --confidences 0.99 0.95 0.90 --percentiles 0.99 0.95 0.90 --fit-run-number 1 --window-size 5 --warm-run 1 --stage fit --save-dirpath ./analysis
