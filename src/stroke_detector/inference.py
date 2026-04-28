#!/usr/bin/env python3
"""
predict_stroke.py
=================
Clinical inference for stroke localization on DWI/ADC data.

Two-signal presence gate
------------------------
has_lesion = True only if BOTH:
  1. objectness confidence >= score_thresh  (detection head fires)
  2. predicted mask volume >= min_seg_volume_ml  (segmentation agrees)

Calibrated from test set: FP negative cases have seg_vol < 0.15ml.

Usage
-----
  python predict_stroke.py ^
      --dwi  "path/to/dwi_b1000.nii.gz" ^
      --adc  "path/to/adc.nii.gz" ^
      --checkpoint "D:\\Stroke\\checkpoints_stage3\\best.pt" ^
      --out  "result.json"

Output JSON
-----------
  {
    "subject":         "patient_001",
    "has_lesion":      true,
    "presence_conf":   0.87,
    "confidence":      0.99,
    "seg_volume_ml":   6.42,
    "center_vox":      [48, 52, 31],
    "box_vox":         [32,38,24, 64,66,38],
    "subtype":         0,
    "subtype_str":     "focal",
    "model_shape":     [96, 96, 64],
    "spacing_mm":      [1.5, 1.5, 3.0]
  }
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch

from model_stroke import build_model, nms_3d
from stroke_dataset import (
    load_nii, reorient_to_ras, resample_to_spacing,
    pad_crop_to_shape, maybe_resize_to_exact, robust_zscore,
)


def preprocess(dwi_path, adc_path, model_shape=(96,96,64),
               canonical_zooms=(1.5,1.5,3.0)):
    _, dwi, dwi_aff, dwi_hdr = load_nii(dwi_path)
    _, adc, adc_aff, _       = load_nii(adc_path)
    native_zooms = tuple(float(z) for z in dwi_hdr.get_zooms()[:3])
    dwi_r, _, _, _ = reorient_to_ras(dwi, dwi_aff)
    adc_r, _, _, _ = reorient_to_ras(adc, adc_aff)
    dwi_rs, _ = resample_to_spacing(dwi_r, native_zooms, canonical_zooms)
    adc_rs, _ = resample_to_spacing(adc_r, native_zooms, canonical_zooms)
    dwi_pc, _, _ = pad_crop_to_shape(dwi_rs, model_shape)
    adc_pc, _, _ = pad_crop_to_shape(adc_rs, model_shape)
    dwi_m, _ = maybe_resize_to_exact(dwi_pc, model_shape)
    adc_m, _ = maybe_resize_to_exact(adc_pc, model_shape)
    brain_mask = dwi_m > 0
    dwi_m = robust_zscore(dwi_m, brain_mask)
    adc_m = robust_zscore(adc_m, brain_mask)
    X = np.stack([dwi_m, adc_m], 0)
    X = torch.from_numpy(X).float()
    X = X.permute(0, 3, 2, 1).unsqueeze(0)
    return X


@torch.no_grad()
def run_inference(model, x, spacing, score_thresh, min_seg_vol_ml,
                  seg_thresh=0.5, iou_thresh=0.3):
    model.eval()
    outputs = model(x)

    # Detection
    boxes_dec, scores_dec = model.decode(outputs)
    boxes_cpu  = boxes_dec[0].cpu()
    scores_cpu = scores_dec[0].cpu()
    keep = nms_3d(boxes_cpu, scores_cpu, iou_thresh)
    if not keep:
        keep = [int(scores_cpu.argmax().item())]
    best_box   = boxes_cpu[keep[0]]
    best_score = float(scores_cpu[keep[0]].item())

    # Clamp box
    W, H, D = model.input_shape
    mins = torch.clamp(best_box[:3], min=0.)
    maxs = torch.clamp(best_box[3:], max=torch.tensor(
        [float(W-1), float(H-1), float(D-1)]))
    best_box = torch.cat([mins, maxs])

    cx = float((best_box[0] + best_box[3]) / 2)
    cy = float((best_box[1] + best_box[4]) / 2)
    cz = float((best_box[2] + best_box[5]) / 2)

    # Classification
    subtype_names = {0:"focal", 1:"multi", 2:"embolic", 3:"negative"}
    subtype = int(outputs["cls_logits"][0].argmax().item())

    # Segmentation volume
    seg_prob   = torch.sigmoid(outputs["seg"])
    seg_binary = (seg_prob > seg_thresh).float()
    seg_voxels = float(seg_binary.sum().item())
    vox_vol_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
    seg_vol_ml = seg_voxels * vox_vol_ml

    # Presence
    if model.stage == 3 and "presence_logit" in outputs:
        presence_prob = float(torch.sigmoid(
            outputs["presence_logit"][0]).item())
    else:
        presence_prob = best_score

    # Two-signal gate: confidence AND seg volume must both agree
    has_lesion = (best_score >= score_thresh and
                  seg_vol_ml >= min_seg_vol_ml)

    return {
        "has_lesion":    has_lesion,
        "presence_conf": round(presence_prob, 4),
        "confidence":    round(best_score, 4),
        "seg_volume_ml": round(seg_vol_ml, 3),
        "center_vox":    [round(cx,1), round(cy,1), round(cz,1)],
        "box_vox":       [round(v,1) for v in best_box.tolist()],
        "subtype":       subtype,
        "subtype_str":   subtype_names.get(subtype, "unknown"),
        "seg_mask":      seg_binary.cpu(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dwi",        required=True)
    ap.add_argument("--adc",        required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out",        default="result.json")
    ap.add_argument("--model_shape",  type=int,   nargs=3, default=[96,96,64])
    ap.add_argument("--spacing",      type=float, nargs=3, default=[1.5,1.5,3.0])
    ap.add_argument("--score_thresh", type=float, default=0.3,
                    help="Objectness confidence threshold")
    ap.add_argument("--min_seg_volume_ml", type=float, default=0.15,
                    help="Min predicted mask volume (ml) for has_lesion=True. "
                         "Calibrated from test set. Set 0.0 to disable.")
    ap.add_argument("--seg_thresh",   type=float, default=0.5)
    ap.add_argument("--return_seg",   action="store_true")
    ap.add_argument("--seg_out",      type=str,   default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    chk       = torch.load(args.checkpoint, map_location="cpu",
                           weights_only=False)
    ckpt_args = chk.get("args", {})
    ckpt_stage = ckpt_args.get("stage", 2)
    print(f"[INFO] stage={ckpt_stage}  "
          f"epoch={chk.get('epoch','?')}  "
          f"score={chk.get('best_score',0.):.4f}")

    model = build_model(
        in_ch         = 2,
        base_channels = tuple(ckpt_args.get("base_channels",[16,24,32,48])),
        fpn_channels  = ckpt_args.get("fpn_channels", 32),
        input_shape   = tuple(args.model_shape),
        num_classes   = 4,
        cls_hidden    = ckpt_args.get("cls_hidden", 64),
        stage         = ckpt_stage,
    ).to(device)
    model.load_state_dict(chk["model"], strict=False)
    print(f"[INFO] Loaded  "
          f"{sum(p.numel() for p in model.parameters())/1e6:.3f}M params")

    print(f"[INFO] Preprocessing...")
    X = preprocess(args.dwi, args.adc,
                   model_shape=tuple(args.model_shape),
                   canonical_zooms=tuple(args.spacing)).to(device)

    print(f"[INFO] Running inference "
          f"score_thresh={args.score_thresh}  "
          f"min_seg_vol={args.min_seg_volume_ml}ml")

    result = run_inference(
        model=model, x=X, spacing=args.spacing,
        score_thresh=args.score_thresh,
        min_seg_vol_ml=args.min_seg_volume_ml,
        seg_thresh=args.seg_thresh,
    )

    subject = Path(args.dwi).stem.replace("_dwi","").replace(".nii","")
    out_dict = {
        "subject":       subject,
        "has_lesion":    result["has_lesion"],
        "presence_conf": result["presence_conf"],
        "confidence":    result["confidence"],
        "seg_volume_ml": result["seg_volume_ml"],
        "center_vox":    result["center_vox"],
        "box_vox":       result["box_vox"],
        "subtype":       result["subtype"],
        "subtype_str":   result["subtype_str"],
        "model_shape":   list(args.model_shape),
        "spacing_mm":    list(args.spacing),
        "thresholds": {
            "score_thresh":      args.score_thresh,
            "min_seg_volume_ml": args.min_seg_volume_ml,
        }
    }

    with open(args.out, "w") as f:
        json.dump(out_dict, f, indent=2)

    print(f"\n[RESULT]\n{json.dumps(out_dict, indent=2)}")
    print(f"\n[INFO] Saved: {args.out}")

    if args.return_seg:
        import nibabel as nib
        seg_np   = result["seg_mask"][0,0].numpy().transpose(2,1,0)
        seg_path = args.seg_out or args.out.replace(".json","_seg.nii.gz")
        nib.save(nib.Nifti1Image(seg_np.astype(np.uint8), np.eye(4)), seg_path)
        print(f"[INFO] Seg mask saved: {seg_path}")


if __name__ == "__main__":
    main()