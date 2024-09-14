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
TASK_NAME=DETR_TensorFlow


python ${PROJ_HOME}/plot_detailed_l.py --l-filepath ./analysis/${TASK_NAME}_I_L_FIT.json --save-filepath ./figures/${TASK_NAME}_I_L_FIT.png --nrows 1 --ncols 3 --fig-w 18 --fig-h 5
python ${PROJ_HOME}/plot_detailed_l.py --l-filepath ./analysis/${TASK_NAME}_I_L_JSD.json --save-filepath ./figures/${TASK_NAME}_I_L_JSD.png --nrows 1 --ncols 3 --fig-w 18 --fig-h 5
python ${PROJ_HOME}/plot_detailed_c.py --c-filepath ./analysis/${TASK_NAME}_I_C_FIT.json --save-filepath ./figures/${TASK_NAME}_I_C_FIT.png --nrows 1 --ncols 3 --fig-w 18 --fig-h 5
python ${PROJ_HOME}/plot_detailed_c.py --c-filepath ./analysis/${TASK_NAME}_I_C_JSD.json --save-filepath ./figures/${TASK_NAME}_I_C_JSD.png --nrows 1 --ncols 3 --fig-w 18 --fig-h 5

python ${PROJ_HOME}/plot_detailed_l.py --l-filepath ./analysis/${TASK_NAME}_T_L_FIT.json --save-filepath ./figures/${TASK_NAME}_T_L_FIT.png --nrows 1 --ncols 3 --fig-w 18 --fig-h 5
python ${PROJ_HOME}/plot_detailed_l.py --l-filepath ./analysis/${TASK_NAME}_T_L_JSD.json --save-filepath ./figures/${TASK_NAME}_T_L_JSD.png --nrows 1 --ncols 3 --fig-w 18 --fig-h 5
python ${PROJ_HOME}/plot_detailed_c.py --c-filepath ./analysis/${TASK_NAME}_T_C_FIT.json --save-filepath ./figures/${TASK_NAME}_T_C_FIT.png --nrows 1 --ncols 3 --fig-w 18 --fig-h 5
python ${PROJ_HOME}/plot_detailed_c.py --c-filepath ./analysis/${TASK_NAME}_T_C_JSD.json --save-filepath ./figures/${TASK_NAME}_T_C_JSD.png --nrows 1 --ncols 3 --fig-w 18 --fig-h 5
