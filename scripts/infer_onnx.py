#!/usr/bin/env python3
"""ONNX semantic segmentation inference: load ONNX models and evaluate on test set.

Outputs per-class IoU and mIoU for each model.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm

NUM_CLASSES = 5
CLASS_NAMES = ["Background", "LW", "OS", "RC", "SI"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = (384, 384)

# Some models have batch dimension fixed during ONNX trace
SINGLE_BATCH_MODELS = {"swintransformer", "swintransformer_scratch"}


def preprocess(img: Image.Image) -> np.ndarray:
    """PIL Image -> (3, H, W) float32, same as OilSegDataset training."""
    img = img.resize(IMG_SIZE, resample=Image.BILINEAR)
    img_np = np.array(img, dtype=np.uint8).astype(np.float32) / 255.0
    img_np = (img_np - MEAN) / STD
    return np.ascontiguousarray(img_np.transpose(2, 0, 1))


def load_session(model_path: str, num_threads: int, use_gpu: bool = False):
    """Load ONNX model, return (session, input_name, has_gpu, used_threads).

    Thread optimizations:
      - auto (num_threads<=0) uses half of physical cores
      - ORT_PARALLEL execution mode
      - CPU memory arena / pattern optimization
    """
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if num_threads <= 0:
        try:
            num_threads = max(1, len(os.sched_getaffinity(0)) // 2)
        except AttributeError:
            num_threads = max(1, (os.cpu_count() or 4) // 2)

    so.intra_op_num_threads = num_threads
    so.inter_op_num_threads = max(2, num_threads // 2)
    so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    so.enable_cpu_mem_arena = True
    so.enable_mem_pattern = True

    has_gpu = False
    providers = ["CPUExecutionProvider"]
    provider_opts = [{}]

    if use_gpu:
        try:
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
                provider_opts.insert(0, {"arena_extend_strategy": "kNextPowerOfTwo"})
                has_gpu = True
        except Exception:
            pass

    sess = ort.InferenceSession(model_path, sess_options=so,
                                providers=providers, provider_options=provider_opts)
    return sess, sess.get_inputs()[0].name, has_gpu, num_threads


def collect_pairs(data_root: str, image_dir: str, mask_dir: str):
    """Collect (image_path, mask_path) pairs, supports subdirectory structure."""
    img_base = Path(data_root) / image_dir
    msk_base = Path(data_root) / mask_dir
    sub_dirs = sorted(d for d in img_base.iterdir() if d.is_dir()) if img_base.is_dir() else []
    if not sub_dirs:
        sub_dirs = [img_base]

    pairs = []
    for sub in sub_dirs:
        sub_name = sub.name if sub != img_base else ""
        msk_sub = msk_base / sub_name if sub_name else msk_base
        if not sub.is_dir():
            continue
        for img_p in sorted(sub.iterdir()):
            if img_p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
                msk_p = msk_sub / img_p.name
                if msk_p.exists():
                    pairs.append((str(img_p), str(msk_p)))
    return pairs


def run_inference(args):
    model_name = Path(args.model).stem
    NC = args.num_classes

    sess, input_name, has_gpu, n_threads = load_session(args.model, args.num_threads, args.gpu)
    batch_size = args.batch_size if args.batch_size > 0 else 2
    if model_name in SINGLE_BATCH_MODELS and batch_size > 1:
        print(f"[{model_name}] single-batch model forced to batch=1")
        batch_size = 1

    pairs = collect_pairs(args.data_root, args.image_dir, args.mask_dir)
    if not pairs:
        print(f"[{model_name}] no test images found")
        return None

    total = len(pairs)
    device = "GPU" if has_gpu else f"CPU({n_threads}t)"
    print(f"[{model_name}] {device}, batch={batch_size}, total={total}")

    all_cm = np.zeros((NC, NC), dtype=np.int64)
    t0 = time.time()
    batch_inp, batch_gt = [], []

    for img_path, msk_path in tqdm(pairs, desc=model_name, unit="img", ncols=100):
        img = Image.open(img_path).convert("RGB")
        msk = Image.open(msk_path)
        if msk.mode != "L":
            msk = msk.convert("L")
        batch_inp.append(preprocess(img))
        batch_gt.append(np.array(msk, dtype=np.int64))

        if len(batch_inp) == batch_size:
            inp = np.stack(batch_inp, axis=0)
            preds = sess.run(None, {input_name: inp})[0].argmax(axis=1).astype(np.int64)
            for j in range(preds.shape[0]):
                pred, gt = preds[j], batch_gt[j]
                if gt.shape != pred.shape:
                    gt = np.array(Image.fromarray(gt.astype(np.uint8)).resize(
                        (pred.shape[1], pred.shape[0]), resample=Image.NEAREST), dtype=np.int64)
                valid = gt >= 0
                all_cm += np.bincount(NC * gt[valid] + pred[valid], minlength=NC * NC).reshape(NC, NC)
            batch_inp, batch_gt = [], []

    # Remainder batch
    if batch_inp:
        inp = np.stack(batch_inp, axis=0)
        preds = sess.run(None, {input_name: inp})[0].argmax(axis=1).astype(np.int64)
        for j in range(preds.shape[0]):
            pred, gt = preds[j], batch_gt[j]
            if gt.shape != pred.shape:
                gt = np.array(Image.fromarray(gt.astype(np.uint8)).resize(
                    (pred.shape[1], pred.shape[0]), resample=Image.NEAREST), dtype=np.int64)
            valid = gt >= 0
            all_cm += np.bincount(NC * gt[valid] + pred[valid], minlength=NC * NC).reshape(NC, NC)

    elapsed = time.time() - t0

    # Compute metrics
    diag = np.diag(all_cm)
    union = all_cm.sum(axis=1) + all_cm.sum(axis=0) - diag
    iou = np.divide(diag, union, out=np.zeros(NC, dtype=np.float64), where=union > 0)
    miou = float(iou.mean())

    print(f"\n{'=' * 55}")
    print(f"  {model_name}  |  {total} imgs  |  mIoU: {miou:.4f}  |  {elapsed:.1f}s")
    print(f"{'=' * 55}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<16s} {iou[i]:>8.4f}")
    print(f"  {'-' * 28}")
    print(f"  {'mIoU':<16s} {miou:>8.4f}\n")

    return {"model": model_name, "mIoU": round(miou, 4),
            "per_class_IoU": {CLASS_NAMES[i]: round(float(iou[i]), 4) for i in range(NC)},
            "images": total, "time_s": round(elapsed, 1)}


def main():
    parser = argparse.ArgumentParser(description="ONNX semantic segmentation inference")
    parser.add_argument("--model", required=True, help="path to ONNX model")
    parser.add_argument("--data-root", default="oil_datasets/test")
    parser.add_argument("--image-dir", default="image")
    parser.add_argument("--mask-dir", default="mask")
    parser.add_argument("--batch-size", type=int, default=0, help="0=default 2, lower if OOM")
    parser.add_argument("--num-threads", type=int, default=20, help="CPU threads, 0=auto half cores")
    parser.add_argument("--gpu", action="store_true", help="enable GPU inference (default CPU)")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
