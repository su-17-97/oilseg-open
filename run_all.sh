#!/bin/bash
# 按顺序推理所有模型，结果写入 results.txt
# 用法：bash run_all.sh          # CPU 推理（默认）
#       bash run_all.sh --gpu    # GPU 推理
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$SCRIPT_DIR/scripts/infer_onnx.py"
MODELS="$SCRIPT_DIR/onnx_models"
DATA="$SCRIPT_DIR/oil_datasets/test"
RESULTS="$SCRIPT_DIR/results.txt"

# GPU 开关：传 --gpu 则在每个推理命令后追加 --gpu
GPU_FLAG=""
if [ "$1" = "--gpu" ]; then
    GPU_FLAG="--gpu"
fi

echo "============================================================" | tee "$RESULTS"
echo "  OilSeg-Open: ONNX Semantic Segmentation Benchmark" | tee -a "$RESULTS"
echo "  $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$RESULTS"
echo "============================================================" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

echo ">>>> Group 1: Proposed Model" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"
python "$SCRIPT" --model "$MODELS/oilsegsarformer.onnx" --data-root "$DATA" $GPU_FLAG | tee -a "$RESULTS"

echo ">>>> Group 2: Comparison Models (pretrained)" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"
python "$SCRIPT" --model "$MODELS/oneformer.onnx" --data-root "$DATA" $GPU_FLAG | tee -a "$RESULTS"
python "$SCRIPT" --model "$MODELS/swintransformer.onnx" --data-root "$DATA" $GPU_FLAG | tee -a "$RESULTS"
python "$SCRIPT" --model "$MODELS/deeplabv3plus.onnx" --data-root "$DATA" $GPU_FLAG | tee -a "$RESULTS"
python "$SCRIPT" --model "$MODELS/offseg.onnx" --data-root "$DATA" $GPU_FLAG | tee -a "$RESULTS"

echo "" | tee -a "$RESULTS"
echo "============================================================" | tee -a "$RESULTS"
