#!/bin/bash

get_absolute_path() {
  local relative_path="$1"
  echo "$(cd "$(dirname "$relative_path")"; pwd)/$(basename "$relative_path")"
}

SCRIPTS_ROOT=$(get_absolute_path ".")

FRAMEWORK=PyTorch
DEVICE=A100
TIME_TYPE=inference

DETR=DETR_${FRAMEWORK}_${DEVICE}_${TIME_TYPE}
DETR_METRIC_TYPE=mAP

VICUNA=Vicuna_${FRAMEWORK}_${DEVICE}_${TIME_TYPE}
VICUNA_METRIC_TYPE=top1-Acc

LIGHTGCN=LightGCN_${FRAMEWORK}_${DEVICE}_${TIME_TYPE}
LIGHTGCN_METRIC_TYPE=ndcg

HYBRIDNETS=HybridNets_${FRAMEWORK}_${DEVICE}_${TIME_TYPE}
HYBRIDNETS_METRIC_TYPE=mAP

MOBILENET=MobileNet_${FRAMEWORK}_${DEVICE}_${TIME_TYPE}
MOBILENET_METRIC_TYPE=top1-Acc

EMOTIONFLOW=EmotionFlow_${FRAMEWORK}_${DEVICE}_${TIME_TYPE}
EMOTIONFLOW_METRIC_TYPE=weighted-f1

cd ..
RESULTS_DIR=../../../TQResults
VISUALIZATION_DIR=../../../TQVisualizations
SUFFIX_WITH_OUTLIER=old
SUFFIX_WITHOUT_OUTLIER=new

if [ -f "${RESULTS_DIR}-${SUFFIX_WITHOUT_OUTLIER}/ALL_Multihops/${DETR}/multihop-100.pickle" ]; then
    echo "${RESULTS_DIR}-${SUFFIX_WITHOUT_OUTLIER}/ALL_Multihops/${DETR}/multihop-100.pickle exists" 
    mkdir -p ${VISUALIZATION_DIR}/ALL_graphs
else
    echo "${RESULTS_DIR}-${SUFFIX_WITHOUT_OUTLIER}/ALL_Multihops/${DETR}/multihop-100.pickle not found"
fi

python -m draw.draw_PyTorch_A100_inference \
  --detr-multihop-with-outlier-filepath ${RESULTS_DIR}-${SUFFIX_WITH_OUTLIER}/ALL_Multihops/${DETR}/multihop-100.pickle \
  --detr-multihop-without-outlier-filepath ${RESULTS_DIR}-${SUFFIX_WITHOUT_OUTLIER}/ALL_Multihops/${DETR}/multihop-100.pickle \
  --detr-specific-filepath ${RESULTS_DIR}-${SUFFIX_WITH_OUTLIER}/ALL_Specifics/${DETR}/specific-100.pickle \
  --detr-metric ${DETR_METRIC_TYPE} \
  --vicuna-multihop-with-outlier-filepath ${RESULTS_DIR}-${SUFFIX_WITH_OUTLIER}/ALL_Multihops/${VICUNA}/multihop-100.pickle \
  --vicuna-multihop-without-outlier-filepath ${RESULTS_DIR}-${SUFFIX_WITHOUT_OUTLIER}/ALL_Multihops/${VICUNA}/multihop-100.pickle \
  --vicuna-specific-filepath ${RESULTS_DIR}-${SUFFIX_WITH_OUTLIER}/ALL_Specifics/${VICUNA}/specific-100.pickle \
  --vicuna-metric ${VICUNA_METRIC_TYPE} \
  --lightgcn-multihop-with-outlier-filepath ${RESULTS_DIR}-${SUFFIX_WITH_OUTLIER}/ALL_Multihops/${LIGHTGCN}/multihop-100.pickle \
  --lightgcn-multihop-without-outlier-filepath ${RESULTS_DIR}-${SUFFIX_WITHOUT_OUTLIER}/ALL_Multihops/${LIGHTGCN}/multihop-100.pickle \
  --lightgcn-specific-filepath ${RESULTS_DIR}-${SUFFIX_WITH_OUTLIER}/ALL_Specifics/${LIGHTGCN}/specific-100.pickle \
  --lightgcn-metric ${LIGHTGCN_METRIC_TYPE} \
  --hybridnets-multihop-with-outlier-filepath ${RESULTS_DIR}-${SUFFIX_WITH_OUTLIER}/ALL_Multihops/${HYBRIDNETS}/multihop-100.pickle \
  --hybridnets-multihop-without-outlier-filepath ${RESULTS_DIR}-${SUFFIX_WITHOUT_OUTLIER}/ALL_Multihops/${HYBRIDNETS}/multihop-100.pickle \
  --hybridnets-specific-filepath ${RESULTS_DIR}-${SUFFIX_WITH_OUTLIER}/ALL_Specifics/${HYBRIDNETS}/specific-100.pickle \
  --hybridnets-metric ${HYBRIDNETS_METRIC_TYPE} \
  --mobilenet-multihop-with-outlier-filepath ${RESULTS_DIR}-${SUFFIX_WITH_OUTLIER}/ALL_Multihops/${MOBILENET}/multihop-100.pickle \
  --mobilenet-multihop-without-outlier-filepath ${RESULTS_DIR}-${SUFFIX_WITHOUT_OUTLIER}/ALL_Multihops/${MOBILENET}/multihop-100.pickle \
  --mobilenet-specific-filepath ${RESULTS_DIR}-${SUFFIX_WITH_OUTLIER}/ALL_Specifics/${MOBILENET}/specific-100.pickle \
  --mobilenet-metric ${MOBILENET_METRIC_TYPE} \
  --emotionflow-multihop-with-outlier-filepath ${RESULTS_DIR}-${SUFFIX_WITH_OUTLIER}/ALL_Multihops/${EMOTIONFLOW}/multihop-100.pickle \
  --emotionflow-multihop-without-outlier-filepath ${RESULTS_DIR}-${SUFFIX_WITHOUT_OUTLIER}/ALL_Multihops/${EMOTIONFLOW}/multihop-100.pickle \
  --emotionflow-specific-filepath ${RESULTS_DIR}-${SUFFIX_WITH_OUTLIER}/ALL_Specifics/${EMOTIONFLOW}/specific-100.pickle \
  --emotionflow-metric ${EMOTIONFLOW_METRIC_TYPE} \
  --save-dir ${VISUALIZATION_DIR}/ALL_graphs \
  --time-type ${TIME_TYPE} \
  --is-subplot
 
cd ${SCRIPTS_ROOT}