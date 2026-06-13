# OilSeg-Open

ONNX Runtime-based inference & evaluation toolkit for oil spill segmentation models.

> [**中文文档**](README_zh.md)

## Model List

| Model | Description | Paper |
|-------|-------------|-------|
| OffSeg | Comparison | ICCV 2025 [[link]](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Revisiting_Efficient_Semantic_Segmentation_Learning_Offsets_for_Better_Spatial_and_ICCV_2025_paper.html) |
| SwinTransformer | Comparison | ICCV 2021 [[link]](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html) |
| DeepLabV3+ | Comparison | ECCV 2018 [[link]](https://openaccess.thecvf.com/content_ECCV_2018/html/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.html) |
| OneFormer | Comparison | CVPR 2023 [[link]](https://openaccess.thecvf.com/content/CVPR2023/html/Jain_OneFormer_One_Transformer_To_Rule_Universal_Image_Segmentation_CVPR_2023_paper.html) |
| **OilSegSARFormer** | **Ours** | — |

**Not Included (weights not uploaded due to storage limits):**
| Model | Paper |
|-------|-------|
| SegNeXt | NeurIPS 2022 |
| SegFormer | NeurIPS 2021 |
| Mask2Former | CVPR 2022 |
| RapidNet | WACV 2025 |
| InternImage | CVPR 2023 |

### Why Only These Models?

Due to Git LFS storage constraints (1GB limit), we currently only provide the top-performing models for inference demo. After the paper is published, we will release **all models, full source code, and trained weights** via cloud storage to ensure **experimental authenticity and reproducibility**.

### OilSegSARFormer Results

```
=======================================================
  OilSegSARFormer  |  546 imgs  |  mIoU: 0.7343
=======================================================
  Background         0.9464
  LW                 0.6761
  OS                 0.6615
  RC                 0.6037
  SI                 0.7836
  ----------------------------
  mIoU               0.7343
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

CPU-only onnxruntime is installed by default. For NVIDIA GPU acceleration (requires CUDA + cuDNN):

```bash
pip install onnxruntime-gpu>=1.16.0
(ort-gpu must match your CUDA version, install based on your device)
```

### Run Inference

```bash
# Run all models
bash run_all.sh

# GPU inference
bash run_all.sh --gpu

# Single model
python scripts/infer_onnx.py \
    --model onnx_models/oilsegsarformer.onnx \
    --data-root oil_datasets/test
```

### Directory Structure

```
oilseg-open/
├── onnx_models/          # ONNX model files
├── oil_datasets/test/    # Test dataset
│   ├── image/{lw,os,rc,si}/
│   └── mask/{lw,os,rc,si}/
├── scripts/
│   └── infer_onnx.py     # Inference script
├── run_all.sh            # Batch inference script
├── results.txt           # Inference results
├── requirements.txt
├── README.md             # English
└── README_zh.md          # Chinese
```

## Test Data

The test set covers 4 oil-related classes:
- **LW** (Low Wind)
- **OS** (Oil Spill)
- **RC** (Rain Cell)
- **SI** (Sea Ice)

~136 images per class, 546 total. Images are resized to 384×384 at inference time.

---

## Notes

### ONNX vs PyTorch Precision

ONNX models are converted from PyTorch training weights. Due to framework and operator implementation differences, there are minor floating-point discrepancies between ONNX Runtime and PyTorch inference. **mIoU differences are typically below 10⁻³** and do not affect model ranking or conclusions.

Main causes:
1. **Operator fusion**: ONNX Runtime fuses operators (e.g., Conv+BN+ReLU) during graph optimization, introducing tiny numerical differences from PyTorch's layer-by-layer computation.
2. **SwinTransformer**: Window attention is partially folded into constants during ONNX trace; only batch=1 is supported.
