#!/bin/bash

get_absolute_path() {
  local relative_path="$1"
  echo "$(cd "$(dirname "$relative_path")"; pwd)/$(basename "$relative_path")"
}

SCRIPTS_ROOT=$(get_absolute_path ".")

MODEL=EmotionFlow
FRAMEWORK=PyTorch
DEVICE=A100
HYPER_PARAMETER=''
TIME_TYPE=inference
METRIC_TYPE=weighted-f1

if [ -z "$HYPER_PARAMETER" ]; then
  ALL_TIMES_NAME=${MODEL}_${FRAMEWORK}_${DEVICE}
else
  ALL_TIMES_NAME=${MODEL}_${FRAMEWORK}_${DEVICE}_${HYPER_PARAMETER}
fi

cd ../..
RESULTS_DIR=../../../TQResults
VISUALIZATION_DIR=../../../TQVisualizations
SUFFIX_WITH_OUTLIER=old
SUFFIX_WITHOUT_OUTLIER=new

if [ -f "${RESULTS_DIR}-${SUFFIX_WITHOUT_OUTLIER}/ALL_Multihops/${ALL_TIMES_NAME}_${TIME_TYPE}/multihop-100.pickle" ]; then
    echo "${RESULTS_DIR}-${SUFFIX_WITHOUT_OUTLIER}/ALL_Multihops/${ALL_TIMES_NAME}_${TIME_TYPE}/multihop-100.pickle exists" 
    mkdir -p ${VISUALIZATION_DIR}/ALL_graphs
else
    echo "${RESULTS_DIR}-${SUFFIX_WITHOUT_OUTLIER}/ALL_Multihops/${ALL_TIMES_NAME}_${TIME_TYPE}/multihop-100.pickle not found"
fi

python -m draw.draw_tq_single \
  --multihop-with-outlier-filepath ${RESULTS_DIR}-${SUFFIX_WITH_OUTLIER}/ALL_Multihops/${ALL_TIMES_NAME}_${TIME_TYPE}/multihop-100.pickle \
  --multihop-without-outlier-filepath ${RESULTS_DIR}-${SUFFIX_WITHOUT_OUTLIER}/ALL_Multihops/${ALL_TIMES_NAME}_${TIME_TYPE}/multihop-100.pickle \
  --specific-filepath ${RESULTS_DIR}-${SUFFIX_WITH_OUTLIER}/ALL_Specifics/${ALL_TIMES_NAME}_${TIME_TYPE}/specific-100.pickle \
  --save-dir ${VISUALIZATION_DIR}/ALL_graphs \
  --time-type ${TIME_TYPE} \
  --metric ${METRIC_TYPE} \
  --save-name ${ALL_TIMES_NAME}_${TIME_TYPE}_${METRIC_TYPE} \
  --task-name emotion_flow 
 
cd ${SCRIPTS_ROOT}