# OilSeg-Open

基于 ONNX Runtime 的油污分割模型推理仓库，支持多种语义分割模型的快速部署与评测。

[English Version](README.md)

## 模型列表

| 模型 | 说明 | 论文 |
|------|------|------|
| OffSeg | 对比模型 | ICCV 2025 [[link]](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Revisiting_Efficient_Semantic_Segmentation_Learning_Offsets_for_Better_Spatial_and_ICCV_2025_paper.html) |
| SwinTransformer | 对比模型 | ICCV 2021 [[link]](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html) |
| DeepLabV3+ | 对比模型 | ECCV 2018 [[link]](https://openaccess.thecvf.com/content_ECCV_2018/html/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.html) |
| OneFormer | 对比模型 | CVPR 2023 [[link]](https://openaccess.thecvf.com/content/CVPR2023/html/Jain_OneFormer_One_Transformer_To_Rule_Universal_Image_Segmentation_CVPR_2023_paper.html) |
| **OilSegSARFormer** | **本文方法** | — |

**未包含模型（因存储限制未上传权重）：**
| 模型 | 论文 |
|------|------|
| SegNeXt | NeurIPS 2022 |
| SegFormer | NeurIPS 2021 |
| Mask2Former | CVPR 2022 |
| RapidNet | WACV 2025 |
| InternImage | CVPR 2023 |

### 为什么只有这几个模型？

由于 Git LFS 存储限制（1GB 上限），当前仓库仅保留表现最优的模型供推理演示使用。论文正式发表后，我们将通过**网盘**开放**全部模型、完整代码及训练权重**，以保证实验的**真实性和可复现性**。

### OilSegSARFormer 详细结果

```
=======================================================
  OilSegSARFormer  |  546 张  |  mIoU: 0.7343
=======================================================
  Background         0.9464
  LW                 0.6761
  OS                 0.6615
  RC                 0.6037
  SI                 0.7836
  ----------------------------
  mIoU               0.7343
```

## 快速开始

### 环境安装

```bash
pip install -r requirements.txt
```

默认安装 CPU 版 onnxruntime。如有 NVIDIA GPU 且已安装 CUDA + cuDNN，可使用 GPU 加速：

```bash
pip install onnxruntime-gpu>=1.16.0
(ort-gpu 必须和CUDA严格兼容，建议根据自己设备安装特定版本)
```

### 运行推理

```bash
# 运行全部模型
bash run_all.sh

# GPU 推理
bash run_all.sh --gpu

# 单模型推理
python scripts/infer_onnx.py \
    --model onnx_models/oilsegsarformer.onnx \
    --data-root oil_datasets/test 
```

### 目录结构

```
oilseg-open/
├── onnx_models/          # ONNX 模型文件
├── oil_datasets/test/    # 测试集
│   ├── image/{lw,os,rc,si}/
│   └── mask/{lw,os,rc,si}/
├── scripts/
│   └── infer_onnx.py     # 推理脚本
├── run_all.sh            # 批量推理
├── results.txt           # 推理结果
├── requirements.txt
├── README.md             # English
└── README_zh.md          # 中文
```

## 测试数据

测试集包含 4 类油污相关目标：
- **LW**（低风区）
- **OS**（油污）
- **RC**（降雨单元）
- **SI**（海冰）

每类约 136 张图片，共 546 张，图片尺寸不一，推理时统一 resize 至 384×384。

---

## 附注

### ONNX 与 PyTorch 推理精度差异

ONNX 模型由 PyTorch 训练权重转换而来。由于推理框架和算子实现的差异，ONNX Runtime 与原始 PyTorch 推理结果存在微小浮点数误差，**mIoU 差异通常在 10⁻³ 级别以下**，不影响模型性能排序与结论。

主要原因：
1. **算子融合优化**：ONNX Runtime 在图优化阶段会进行算子融合（如 Conv+BN+ReLU），与 PyTorch 逐层计算存在微小数值差异。
2. **SwinTransformer 特殊情况**：窗口注意力机制在 ONNX trace 时部分逻辑被固化为常量，仅支持 batch=1 推理。

