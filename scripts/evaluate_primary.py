#!/usr/bin/env python3
"""
evaluate_test.py
================
Unbiased test set evaluation for the trained stroke detection model.

This script evaluates the Stage 2 model on the held-out test split
that was never used during training or validation.

Usage
-----
    python evaluate_test.py ^
        --checkpoint "D:\\Stroke\\checkpoints_stage2\\best.pt" ^
        --root       "D:\\Stroke\\ISLES-2022" ^
        --split_csv  "D:\\Stroke\\ISLES-2022\\isles_final_split.csv" ^
        --out        "D:\\Stroke\\test_results"

Outputs
-------
    test_results/
        summary.json          overall metrics
        per_case.csv          per-case predictions vs GT
        confusion_matrix.txt  classification confusion matrix
        test_report.txt       human-readable report

Metrics reported
----------------
    Detection:
        mAP@0.2, 0.3, 0.4, 0.5
        mAP@0.2-0.5 (primary)
        meanIoU
        Sensitivity  (TP rate on positive cases)
        Specificity  (TN rate on negative cases)
        TP / FP / TN / FN counts

    Segmentation (auxiliary):
        Mean Dice score across all cases
        Mean Dice on positive cases only

    Classification (subtype):
        Overall accuracy
        Per-class accuracy (focal / multi / embolic / negative)
        Confusion matrix

    Per-role breakdown:
        Metrics split by positive_single / positive_multi /
        embolic / negative
"""

import os
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

from stroke_detector.model import (
    build_model, StrokeDetector,
    box_iou_3d, box_cxcycz_whd_to_xyzxyz,
    ROLE_TO_CLS, dice_loss,
)
from stroke_detector.data import StrokeDataset, collate_fn
from torch.utils.data import DataLoader


# =============================================================================
# Helpers
# =============================================================================

def gt_norm_to_corners(gt_norm, w_dim, h_dim, d_dim):
    scale = gt_norm.new_tensor([w_dim, h_dim, d_dim, w_dim, h_dim, d_dim])
    return box_cxcycz_whd_to_xyzxyz(
        (gt_norm * scale).unsqueeze(0)).squeeze(0)


def compute_dice(pred_mask, gt_mask, thresh=0.5):
    pred_b = (pred_mask > thresh).float()
    inter  = (pred_b * gt_mask).sum()
    denom  = pred_b.sum() + gt_mask.sum()
    if denom < 1:
        return 1.0 if inter < 1 else 0.0
    return float((2 * inter + 1) / (denom + 1))


def nms_simple(boxes, scores, iou_thresh=0.3):
    if boxes.numel() == 0:
        return []
    order = scores.argsort(descending=True)
    keep  = []
    while order.numel() > 0:
        i = int(order[0].item()); keep.append(i)
        if order.numel() == 1: break
        rest = order[1:]
        ious = box_iou_3d(boxes[i:i+1], boxes[rest]).squeeze(0)
        order = rest[ious <= iou_thresh]
    return keep


# =============================================================================
# Main evaluation
# =============================================================================

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    # -- Load checkpoint ------------------------------------------------------
    print(f"[INFO] Loading: {args.checkpoint}")
    chk      = torch.load(args.checkpoint, map_location="cpu",
                          weights_only=False)
    ckpt_args = chk.get("args", {})

    input_shape = tuple(ckpt_args.get("input_shape", [96, 96, 64]))
    spacing     = tuple(ckpt_args.get("spacing",     [1.5, 1.5, 3.0]))

    ckpt_stage = ckpt_args.get("stage", 2)
    print(f"[INFO] Checkpoint stage: {ckpt_stage}")
    model = build_model(
        in_ch         = 2,
        base_channels = tuple(ckpt_args.get("base_channels", [16,24,32,48])),
        fpn_channels  = ckpt_args.get("fpn_channels", 32),
        input_shape   = input_shape,
        num_classes   = 4,
        cls_hidden    = ckpt_args.get("cls_hidden", 64),
        stage         = ckpt_stage,
    ).to(device)

    model.load_state_dict(chk["model"], strict=False)
    model.eval()

    val_score = chk.get("best_score", "?")
    val_epoch = chk.get("epoch", "?")
    print(f"[INFO] Checkpoint: epoch={val_epoch}  val_mAP={val_score:.4f}")

    # -- Dataset --------------------------------------------------------------
    test_ds = StrokeDataset(
        root            = args.root,
        split_csv       = args.split_csv,
        split           = "test",
        model_shape     = input_shape,
        canonical_zooms = spacing,
        augment         = False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=collate_fn)

    print(f"[INFO] Test cases: {len(test_ds)}")

    # -- Output dir -----------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Per-case results storage ---------------------------------------------
    cls_names   = {0:"focal", 1:"multi", 2:"embolic", 3:"negative"}
    per_case    = []

    det_iou_thresholds = [0.2, 0.3, 0.4, 0.5]

    # -- Inference loop -------------------------------------------------------
    print(f"\nRunning inference on {len(test_ds)} test cases...\n")

    with torch.no_grad():
        for batch in test_loader:
            imgs       = batch["image"].to(device, non_blocking=True)
            gt_mask    = batch["mask"][0]          # (1,D,H,W) cpu
            box_single = batch["box_single"][0]    # (6,) cpu
            train_role = batch["train_role"][0]
            has_box    = bool(batch["has_box"][0].item())
            cid        = batch["case_id"][0]

            _, _C, d_dim, h_dim, w_dim = imgs.shape

            outputs    = model(imgs)
            pred_seg   = torch.sigmoid(outputs["seg"][0].cpu())  # (1,D,H,W)
            boxes_dec, scores_dec = model.decode(outputs)
            boxes_cpu  = boxes_dec[0].cpu()   # (N, 6)
            scores_cpu = scores_dec[0].cpu()  # (N,)
            cls_pred   = int(outputs["cls_logits"][0].argmax().item())
            true_cls   = ROLE_TO_CLS.get(train_role, 3)

            # Global presence prediction (stage 3)
            if "presence_logit" in outputs:
                presence_prob = float(torch.sigmoid(
                    outputs["presence_logit"][0]).item())
                pred_has_lesion = presence_prob > 0.5
            else:
                presence_prob   = float(scores_cpu.max().item())
                pred_has_lesion = presence_prob > args.score_thresh

            # Best box by IoU (if positive) or by score
            if has_box:
                gt_c     = gt_norm_to_corners(
                    box_single, w_dim, h_dim, d_dim)
                ious_all = box_iou_3d(
                    boxes_cpu, gt_c.view(1,6)).squeeze(1)
                bi_iou   = int(torch.argmax(ious_all))
                best_iou = float(ious_all[bi_iou].item())
                best_box = boxes_cpu[bi_iou]
                best_sc  = float(scores_cpu[bi_iou].item())
                gt_box   = gt_c
            else:
                bi_sc    = int(torch.argmax(scores_cpu))
                best_iou = 0.0
                best_box = boxes_cpu[bi_sc]
                best_sc  = float(scores_cpu[bi_sc].item())
                gt_box   = None

            # Dice
            dice = compute_dice(pred_seg, gt_mask)

            # AP at each threshold
            aps = {}
            for thr in det_iou_thresholds:
                if has_box:
                    aps[f"AP@{thr}"] = 1.0 if best_iou >= thr else 0.0
                else:
                    aps[f"AP@{thr}"] = float("nan")

            # Center of best box
            cx = float((best_box[0] + best_box[3]) / 2)
            cy = float((best_box[1] + best_box[4]) / 2)
            cz = float((best_box[2] + best_box[5]) / 2)

            case_result = {
                "case_id":       cid,
                "train_role":    train_role,
                "true_cls":      true_cls,
                "pred_cls":      cls_pred,
                "cls_correct":   int(cls_pred == true_cls),
                "has_box":       has_box,
                "pred_has_lesion": int(pred_has_lesion),
                "presence_prob": round(presence_prob, 4),
                "true_has_lesion": int(has_box or train_role=="embolic"),
                "presence_correct": int(pred_has_lesion ==
                    (has_box or train_role=="embolic")),
                "best_iou":   round(best_iou, 4),
                "confidence": round(best_sc, 4),
                "dice":       round(dice, 4),
                "center_x":   round(cx, 1),
                "center_y":   round(cy, 1),
                "center_z":   round(cz, 1),
                **{k: round(v, 4) if not np.isnan(v) else None
                   for k, v in aps.items()},
            }
            per_case.append(case_result)

            # PASS = correctly detected (positive) or correctly rejected (negative)
            if has_box:
                status = "PASS" if (pred_has_lesion and best_iou >= 0.3) else "FAIL"
            else:
                status = "PASS" if not pred_has_lesion else "FAIL"
            print(f"  {status} {cid:<30} role={train_role:<16} "
                  f"iou={best_iou:.3f}  conf={best_sc:.3f}  "
                  f"dice={dice:.3f}  "
                  f"cls={cls_names.get(cls_pred,'?')}"
                  f"(true={cls_names.get(true_cls,'?')})")

    # -- Aggregate metrics ----------------------------------------------------
    print("\n" + "="*70)
    print("TEST SET RESULTS")
    print("="*70)

    # Detection metrics
    positive_cases = [c for c in per_case if c["has_box"]]
    negative_cases = [c for c in per_case if not c["has_box"]]

    score_thr = args.score_thresh
    iou_thr   = 0.3

    # Use presence prediction if available (stage 3), else fall back to confidence
    use_presence = any("pred_has_lesion" in c for c in per_case)
    if use_presence:
        # Stage 3: presence head directly predicts has_lesion
        tp = sum(1 for c in positive_cases
                 if c.get("pred_has_lesion",0) and c["best_iou"] >= iou_thr)
        fn = len(positive_cases) - tp
        fp = sum(1 for c in negative_cases
                 if c.get("pred_has_lesion",0))
        tn = len(negative_cases) - fp
    else:
        # Stage 2: use confidence threshold
        tp = sum(1 for c in positive_cases
                 if c["confidence"] >= score_thr and c["best_iou"] >= iou_thr)
        fn = len(positive_cases) - tp
        fp = sum(1 for c in negative_cases
                 if c["confidence"] >= score_thr)
        tn = len(negative_cases) - fp

    sensitivity  = tp / max(1, tp + fn)
    specificity  = tn / max(1, tn + fp)
    ppv          = tp / max(1, tp + fp)   # precision
    f1           = 2*tp / max(1, 2*tp + fp + fn)

    # mAP
    ap_by_thr = {}
    for thr in det_iou_thresholds:
        key    = f"AP@{thr}"
        vals   = [c[key] for c in positive_cases
                  if c[key] is not None]
        ap_by_thr[key] = float(np.mean(vals)) if vals else 0.0

    mean_ap = float(np.mean(list(ap_by_thr.values())))

    # meanIoU on positive cases
    mean_iou_pos = float(np.mean([c["best_iou"]
                                  for c in positive_cases])) \
                   if positive_cases else 0.0

    # Dice
    mean_dice     = float(np.mean([c["dice"] for c in per_case]))
    mean_dice_pos = float(np.mean([c["dice"]
                                   for c in positive_cases])) \
                    if positive_cases else 0.0

    # Classification
    cls_correct = sum(c["cls_correct"] for c in per_case)
    cls_acc     = cls_correct / max(1, len(per_case))

    cls_by_role = defaultdict(lambda: {"correct":0,"total":0})
    for c in per_case:
        role = c["train_role"]
        cls_by_role[role]["total"]   += 1
        cls_by_role[role]["correct"] += c["cls_correct"]

    # Per-role IoU
    iou_by_role = defaultdict(list)
    for c in positive_cases:
        iou_by_role[c["train_role"]].append(c["best_iou"])

    # -- Print report ---------------------------------------------------------
    report_lines = []
    def p(line=""):
        print(line)
        report_lines.append(line)

    p()
    p("="*70)
    p("STROKE DETECTION MODEL — TEST SET EVALUATION")
    p("="*70)
    p(f"Checkpoint : {args.checkpoint}")
    p(f"Val mAP    : {val_score:.4f}  (epoch {val_epoch})")
    p(f"Test cases : {len(per_case)}  "
      f"(pos={len(positive_cases)}  neg={len(negative_cases)})")
    p()

    p("── DETECTION ──────────────────────────────────────────────────────")
    for k, v in ap_by_thr.items():
        p(f"  {k:<12}: {v:.4f}")
    p(f"  mAP@0.2-0.5 : {mean_ap:.4f}   ← primary metric")
    p(f"  meanIoU     : {mean_iou_pos:.4f}  (positive cases only)")
    p()
    p(f"  score_thresh={score_thr}  iou_thresh={iou_thr}")
    p(f"  TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    p(f"  Sensitivity : {sensitivity:.4f}  "
      f"({tp}/{tp+fn} positive cases detected)")
    p(f"  Specificity : {specificity:.4f}  "
      f"({tn}/{tn+fp} negative cases correctly suppressed)")
    p(f"  Precision   : {ppv:.4f}")
    p(f"  F1 score    : {f1:.4f}")
    p()

    p("── SEGMENTATION (auxiliary) ────────────────────────────────────────")
    p(f"  Mean Dice (all cases)     : {mean_dice:.4f}")
    p(f"  Mean Dice (positive only) : {mean_dice_pos:.4f}")
    p()

    p("── CLASSIFICATION ──────────────────────────────────────────────────")
    p(f"  Overall accuracy: {cls_acc:.4f}  ({cls_correct}/{len(per_case)})")
    for role, counts in sorted(cls_by_role.items()):
        acc = counts["correct"]/max(1,counts["total"])
        p(f"  {role:<20}: {acc:.3f}  "
          f"({counts['correct']}/{counts['total']})")
    p()

    p("── PER-ROLE DETECTION IoU ──────────────────────────────────────────")
    for role, ious in sorted(iou_by_role.items()):
        p(f"  {role:<20}: mean={np.mean(ious):.3f}  "
          f"min={np.min(ious):.3f}  max={np.max(ious):.3f}  "
          f"n={len(ious)}")
    p()

    p("── CONFUSION MATRIX (classification) ──────────────────────────────")
    roles_ordered = ["positive_single","positive_multi","embolic","negative"]
    cls_ordered   = [0, 1, 2, 3]
    p("  True \\ Pred    focal  multi  embolic  negative")
    for true_r in roles_ordered:
        true_c = ROLE_TO_CLS.get(true_r, 3)
        row    = [sum(1 for c in per_case
                      if c["true_cls"]==true_c and c["pred_cls"]==pc)
                  for pc in cls_ordered]
        p(f"  {true_r:<20} {row[0]:>5}  {row[1]:>5}  "
          f"{row[2]:>7}  {row[3]:>8}")
    p()
    p("="*70)

    # -- Save outputs ---------------------------------------------------------
    # Summary JSON
    summary = {
        "checkpoint":    str(args.checkpoint),
        "val_mAP":       float(val_score) if isinstance(val_score, float)
                         else val_score,
        "val_epoch":     val_epoch,
        "n_test":        len(per_case),
        "n_positive":    len(positive_cases),
        "n_negative":    len(negative_cases),
        "detection": {
            **{k: round(v, 4) for k, v in ap_by_thr.items()},
            "mAP@0.2-0.5": round(mean_ap, 4),
            "meanIoU":      round(mean_iou_pos, 4),
            "Sensitivity":  round(sensitivity, 4),
            "Specificity":  round(specificity, 4),
            "Precision":    round(ppv, 4),
            "F1":           round(f1, 4),
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "score_thresh": score_thr,
            "iou_thresh":   iou_thr,
        },
        "segmentation": {
            "mean_dice_all":      round(mean_dice, 4),
            "mean_dice_positive": round(mean_dice_pos, 4),
        },
        "classification": {
            "overall_accuracy": round(cls_acc, 4),
            **{role: round(c["correct"]/max(1,c["total"]), 4)
               for role, c in cls_by_role.items()},
        },
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INFO] Summary saved: {summary_path}")

    # Per-case CSV
    csv_path = out_dir / "per_case.csv"
    if per_case:
        fieldnames = list(per_case[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(per_case)
    print(f"[INFO] Per-case CSV: {csv_path}")

    # Text report
    report_path = out_dir / "test_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"[INFO] Report saved: {report_path}")

    return summary


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Test set evaluation for stroke detection model")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to Stage 2 best.pt checkpoint")
    ap.add_argument("--root",       required=True,
                    help="ISLES-2022 root directory")
    ap.add_argument("--split_csv",  required=True,
                    help="Path to isles_final_split.csv")
    ap.add_argument("--out_dir",        default="test_results",
                    help="Output directory for results")
    ap.add_argument("--score_thresh", type=float, default=0.3,
                    help="Confidence threshold for detection (default 0.3)")
    args = ap.parse_args()

    evaluate(args)