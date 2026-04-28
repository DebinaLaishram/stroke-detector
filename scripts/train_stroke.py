#!/usr/bin/env python3
"""
train_stroke.py
===============
Segmentation-Supervised Stroke Detection — two-stage training with MLflow.

Stage 1 (segmentation pretraining)
-----------------------------------
  python train_stroke.py --config configs/stage1.yaml

  Backbone + FPN + SegDecoder trained end-to-end on binary lesion masks.
  Loss: Dice + BCE.
  Monitor: val Dice score.
  Expected: Dice 0.55-0.70 on ISLES-2022 val set (175 train cases).
  Duration: ~120 epochs, ~2-3 hours on T400.

Stage 2 (detection fine-tuning)
---------------------------------
  python train_stroke.py --config configs/stage2.yaml

  Loads Stage 1 backbone+FPN weights.
  Adds detection heads + classification head (masked pooling).
  Loss: GIoU + L1 + objectness + λ_seg×Dice + λ_cls×CE.
  Monitor: val mAP@0.2-0.5.
  Expected: mAP > 0.45 (vs 0.42 previous best with YOLO-only).
  Duration: ~200 epochs, ~4-5 hours on T400.

MLflow
------
  All runs tracked automatically. View UI with:
    mlflow ui --port 5000
  Navigate to http://localhost:5000

  Tracked per epoch:
    - All loss components (seg, giou, l1, obj, cls)
    - Validation metrics (mAP, Dice, Sensitivity, Specificity)
    - Learning rates per param group
    - GPU memory usage

Config
------
  All hyperparameters in YAML config files (configs/stage1.yaml, stage2.yaml).
  Override any param with CLI: --lr 1e-4 --patience 60 etc.
"""

import os
import csv
import math
import argparse
import random
import inspect
import yaml
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

# MLflow
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("[WARN] MLflow not installed. Run: pip install mlflow")
    print("       Continuing without experiment tracking.")

from stroke_detector.model import (
    StrokeDetector, StrokeLoss, build_model,
    box_iou_3d, box_cxcycz_whd_to_xyzxyz,
    predict, ROLE_TO_CLS, dice_loss,
)
from stroke_detector.data import StrokeDataset, collate_fn


# =============================================================================
# Helpers
# =============================================================================

def set_seed(s: int = 2026):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _safe_float(v) -> float:
    if v is None: return 0.
    if isinstance(v, torch.Tensor): return float(v.item())
    return float(v)


def gt_norm_to_corners(gt_norm, w_dim, h_dim, d_dim):
    scale = gt_norm.new_tensor([w_dim, h_dim, d_dim, w_dim, h_dim, d_dim])
    return box_cxcycz_whd_to_xyzxyz(
        (gt_norm * scale).unsqueeze(0)).squeeze(0)


# =============================================================================
# Config loading
# =============================================================================

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def merge_args_config(args: argparse.Namespace, config: dict) -> argparse.Namespace:
    """
    Config file provides defaults. CLI arguments override config values.
    Priority: CLI > config > argparse defaults.
    """
    # Find which args were explicitly set via CLI (not from defaults)
    parser_defaults = {a.dest: a.default
                       for a in _get_parser()._actions
                       if hasattr(a, 'default')}

    for key, val in config.items():
        dest = key.replace("-", "_")
        # Only apply config value if CLI did not explicitly override
        cli_val = getattr(args, dest, None)
        if cli_val == parser_defaults.get(dest, None):
            setattr(args, dest, val)
    return args


# =============================================================================
# Metrics
# =============================================================================

def compute_val_metrics(pred_boxes, gt_boxes,
                        iou_thresholds=(0.2, 0.3, 0.4, 0.5)) -> Dict:
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return {"mAP@0.2-0.5": 0., "meanIoU": 0.,
                **{f"AP@{t:.1f}": 0. for t in iou_thresholds}}
    ious = box_iou_3d(pred_boxes, gt_boxes).diag()
    res = {}; aps = []
    for thr in iou_thresholds:
        ap = (ious >= thr).float().mean().item()
        res[f"AP@{thr:.1f}"] = ap; aps.append(ap)
    res["mAP@0.2-0.5"] = sum(aps) / max(1, len(aps))
    res["meanIoU"]      = ious.mean().item()
    return res


def compute_dice_val(pred_masks, gt_masks) -> float:
    """Compute mean Dice over val set."""
    scores = []
    for pred, gt in zip(pred_masks, gt_masks):
        pred_b = (pred > 0.5).float()
        inter  = (pred_b * gt).sum()
        denom  = pred_b.sum() + gt.sum()
        if denom < 1:
            scores.append(1.0 if inter < 1 else 0.0)
        else:
            scores.append(float((2*inter+1)/(denom+1)))
    return float(np.mean(scores)) if scores else 0.


def compute_cls_accuracy(pred_cls, true_cls) -> Dict:
    from collections import defaultdict
    correct = defaultdict(int); total = defaultdict(int)
    for p, t in zip(pred_cls, true_cls):
        total[t] += 1; correct[t] += int(p == t)
    cls_names = {0:"focal",1:"multi",2:"embolic",3:"negative"}
    out = {}
    for c, name in cls_names.items():
        if total[c] > 0:
            out[f"acc_{name}"] = correct[c]/total[c]
    out["cls_acc"] = sum(correct.values())/max(1,sum(total.values()))
    return out


def compute_sens_spec(pred_boxes, pred_scores, is_positive,
                      gt_boxes_list, iou_thr=0.3, score_thr=0.3) -> Dict:
    """
    Sensitivity = TP / (TP + FN)
      TP: positive case where confidence >= score_thr AND IoU >= iou_thr
      FN: positive case missed (low confidence or poor localisation)

    Specificity = TN / (TN + FP)
      TN: negative case where confidence < score_thr  (correctly suppressed)
      FP: negative case where confidence >= score_thr (false alarm)

    gt_boxes_list: list of (6,) tensors, one per positive case,
                   in same order as pred_boxes[is_positive]
    """
    pos = is_positive.bool()
    neg = ~pos
    tp = fn = fp = tn = 0

    # Sensitivity — evaluate each positive case individually
    pos_indices = pos.nonzero(as_tuple=True)[0]
    for k, idx in enumerate(pos_indices):
        score = float(pred_scores[idx].item())
        iou   = 0.0
        if k < len(gt_boxes_list):
            gt  = gt_boxes_list[k].view(1, 6)
            pb  = pred_boxes[idx].view(1, 6)
            iou = float(box_iou_3d(pb, gt).item())
        if score >= score_thr and iou >= iou_thr:
            tp += 1
        else:
            fn += 1

    # Specificity — evaluate each negative case individually
    neg_indices = neg.nonzero(as_tuple=True)[0]
    for idx in neg_indices:
        score = float(pred_scores[idx].item())
        if score >= score_thr:
            fp += 1   # model fired on healthy brain = false alarm
        else:
            tn += 1   # model correctly quiet on healthy brain

    return {
        "Sensitivity": tp / max(1, tp + fn),
        "Specificity": tn / max(1, tn + fp),
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
    }


# =============================================================================
# LR schedule + AMP
# =============================================================================

def cosine_with_warmup(opt, warmup, total, min_lr=0.0):
    def f(s):
        if warmup > 0 and s < warmup:
            return s / max(1, warmup)
        p = (s - warmup) / max(1, total - warmup)
        return max(min_lr, 0.5*(1+math.cos(math.pi*p)))
    return optim.lr_scheduler.LambdaLR(opt, f)


class AmpContext:
    def __init__(self, enabled, prefer="cuda"):
        self.enabled = bool(enabled and torch.cuda.is_available())
        try:
            from torch import amp as _a
            sig = inspect.signature(_a.autocast)
            if "device_type" in sig.parameters:
                self._ac = lambda: _a.autocast(
                    device_type=prefer, enabled=self.enabled)
            else:
                self._ac = lambda: _a.autocast(enabled=self.enabled)
            self.scaler = _a.GradScaler(enabled=self.enabled)
        except Exception:
            from torch.cuda.amp import GradScaler, autocast
            self.scaler = GradScaler(enabled=self.enabled, init_scale=256.)
            self._ac    = lambda: autocast(enabled=self.enabled)

    def autocast(self): return self._ac()


# =============================================================================
# Param groups
# =============================================================================

def build_param_groups(model: StrokeDetector,
                       base_lr: float,
                       backbone_lr_scale: float,
                       cls_lr_scale: float,
                       weight_decay: float) -> List[Dict]:
    backbone_params  = list(model.backbone.parameters())
    fpn_params       = list(model.fpn.parameters())
    seg_params       = list(model.seg_decoder.parameters())

    backbone_lr = base_lr * backbone_lr_scale
    print(f"[INFO] Backbone LR: {backbone_lr:.2e}  "
          f"(scale={backbone_lr_scale})")
    print(f"[INFO] FPN+Seg LR:  {base_lr:.2e}")

    groups = [
        {"params": fpn_params + seg_params,
         "lr": base_lr, "weight_decay": weight_decay, "name": "fpn_seg"},
        {"params": backbone_params,
         "lr": backbone_lr, "weight_decay": weight_decay, "name": "backbone"},
    ]

    if model.stage in (2, 3):
        det_params = list(model.det_heads.parameters())
        cls_params = list(model.cls_head.parameters())
        cls_lr     = base_lr * cls_lr_scale

        # Stage 3: detection heads train slowly (already converged in stage 2)
        det_lr = base_lr * 0.1 if model.stage == 3 else base_lr

        print(f"[INFO] Det heads LR: {det_lr:.2e}"
              + (" (frozen 0.1x — stage 3)" if model.stage==3 else ""))
        print(f"[INFO] Cls head LR:  {cls_lr:.2e}  (scale={cls_lr_scale})")
        groups.append(
            {"params": det_params,
             "lr": det_lr, "weight_decay": weight_decay, "name": "det_heads"})
        groups.append(
            {"params": cls_params,
             "lr": cls_lr, "weight_decay": weight_decay, "name": "cls_head"})

    if model.stage == 3:
        # Presence head trains at full lr — it is randomly initialised
        pres_params = list(model.presence_head.parameters())
        print(f"[INFO] Presence head LR: {base_lr:.2e}  (new head, full lr)")
        groups.append(
            {"params": pres_params,
             "lr": base_lr, "weight_decay": weight_decay,
             "name": "presence_head"})

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Trainable: {trainable/1e6:.3f}M / {total/1e6:.3f}M")
    return groups


# =============================================================================
# Weighted sampler
# =============================================================================

def build_sampler(dataset, role_weights: Dict[str, float]):
    weights = [role_weights.get(dataset[i]["train_role"], 1.0)
               for i in range(len(dataset))]
    weights_t = torch.tensor(weights, dtype=torch.float)
    sampler   = WeightedRandomSampler(weights_t, len(weights_t),
                                      replacement=True)
    from collections import Counter
    role_counts = Counter(dataset[i]["train_role"]
                          for i in range(len(dataset)))
    print("[INFO] WeightedRandomSampler:")
    for role, count in sorted(role_counts.items()):
        w = role_weights.get(role, 1.0)
        print(f"       {role:<20}: n={count}  weight={w:.1f}x  "
              f"~{count*w:.0f}/epoch")
    return sampler


# =============================================================================
# CSV helpers
# =============================================================================

def init_csv(path, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


def append_csv(path, row):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


# =============================================================================
# Training loop
# =============================================================================

def train(args):
    set_seed(args.seed)
    cuda   = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    stage  = args.stage
    print(f"[INFO] device={device}  stage={stage}  workers={args.workers}")

    if cuda:
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass

    # -- MLflow setup -------------------------------------------------------
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(args.experiment_name)
        run = mlflow.start_run(run_name=args.run_name)
        print(f"[MLflow] Experiment: {args.experiment_name}")
        print(f"[MLflow] Run: {args.run_name}  ID: {run.info.run_id}")
        # Log all hyperparameters
        mlflow.log_params({
            "stage":              stage,
            "lr":                 args.lr,
            "max_epochs":         args.max_epochs,
            "lambda_seg":         args.lambda_seg,
            "lambda_giou":        args.lambda_giou,
            "lambda_l1":          args.lambda_l1,
            "lambda_cls":         args.lambda_cls,
            "pos_obj_weight":     args.pos_obj_weight,
            "giou_warmup_epochs": args.giou_warmup_epochs,
            "backbone_lr_scale":  args.backbone_lr_scale,
            "cls_lr_scale":       args.cls_lr_scale,
            "patience":           args.patience,
            "use_weighted_sampler": args.use_weighted_sampler,
            "base_channels":      str(args.base_channels),
            "fpn_channels":       args.fpn_channels,
            "input_shape":        str(args.input_shape),
            "resume":             str(args.resume) if args.resume else "none",
        })

    # -- Datasets -----------------------------------------------------------
    train_ds = StrokeDataset(
        args.root, args.split_csv, "train",
        model_shape=args.input_shape,
        canonical_zooms=args.spacing,
        augment=True)
    val_ds = StrokeDataset(
        args.root, args.split_csv, "val",
        model_shape=args.input_shape,
        canonical_zooms=args.spacing,
        augment=False)

    # -- Sampler ------------------------------------------------------------
    if args.use_weighted_sampler:
        role_weights = {
            "positive_single": args.weight_positive_single,
            "positive_multi":  args.weight_positive_multi,
            "embolic":         args.weight_embolic,
            "negative":        args.weight_negative,
        }
        sampler      = build_sampler(train_ds, role_weights)
        train_loader = DataLoader(
            train_ds, batch_size=1, sampler=sampler,
            num_workers=args.workers, pin_memory=cuda,
            collate_fn=collate_fn)
    else:
        train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=True,
            num_workers=args.workers, pin_memory=cuda,
            collate_fn=collate_fn)

    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=cuda,
        collate_fn=collate_fn)

    print(f"[INFO] train={len(train_ds)}  val={len(val_ds)}")

    # -- Model --------------------------------------------------------------
    W, H, D = args.input_shape
    model = build_model(
        in_ch         = 2,
        base_channels = tuple(args.base_channels),
        fpn_channels  = args.fpn_channels,
        input_shape   = (W, H, D),
        num_classes   = 4,
        cls_hidden    = args.cls_hidden,
        stage         = stage,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model params: {total_params/1e6:.3f}M")
    if MLFLOW_AVAILABLE:
        mlflow.log_param("model_params_M", round(total_params/1e6, 3))

    # -- Resume -------------------------------------------------------------
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            chk = torch.load(resume_path, map_location="cpu",
                             weights_only=False)
            # Load with strict=False: stage 2 has new det/cls heads
            missing, unexpected = model.load_state_dict(
                chk["model"], strict=False)
            new_heads = [k for k in missing
                         if any(h in k for h in
                                ["det_heads", "cls_head"])]
            other_missing = [k for k in missing if k not in new_heads]
            if new_heads:
                print(f"[INFO] New heads initialised: {len(new_heads)} keys")
            if other_missing:
                print(f"[WARN] Unexpected missing: {other_missing[:5]}")
            resumed_score = chk.get("best_score", "?")
            resumed_epoch = chk.get("epoch", "?")
            print(f"[INFO] Resumed from {resume_path}")
            print(f"       epoch={resumed_epoch}  score={resumed_score}")
            del chk
        else:
            print(f"[WARN] --resume not found: {resume_path}  "
                  f"Starting from scratch.")

    # -- Loss ---------------------------------------------------------------
    criterion = StrokeLoss(
        stage               = stage,
        lambda_giou         = args.lambda_giou,
        lambda_l1           = args.lambda_l1,
        lambda_obj_bg       = args.lambda_obj_bg,
        lambda_cls          = args.lambda_cls,
        lambda_seg          = args.lambda_seg,
        lambda_presence     = getattr(args, "lambda_presence", 1.0),
        pos_obj_weight      = args.pos_obj_weight,
        giou_warmup_epochs  = args.giou_warmup_epochs,
        return_components   = True,
    ).to(device)

    # -- Optimiser ----------------------------------------------------------
    wd          = 1e-2
    param_groups = build_param_groups(
        model,
        base_lr           = args.lr,
        backbone_lr_scale = args.backbone_lr_scale,
        cls_lr_scale      = args.cls_lr_scale,
        weight_decay      = wd,
    )
    optimizer = optim.AdamW(param_groups)

    total_steps  = max(1, args.max_epochs * max(1, len(train_loader))
                       // max(1, args.accum_steps))
    warmup_steps = max(10, int(0.05 * total_steps))
    print(f"[INFO] warmup_steps={warmup_steps}  total_steps={total_steps}")

    scheduler = cosine_with_warmup(optimizer, warmup_steps, total_steps)
    amp       = AmpContext(enabled=bool(args.amp and cuda))

    # -- Output dirs --------------------------------------------------------
    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_header = ["epoch", "train_loss", "seg_loss", "det_loss", "cls_loss",
                  "val_dice", "mAP@0.2-0.5", "meanIoU",
                  "cls_acc", "Sensitivity", "Specificity"]
    log_csv = ckpt_dir / "train_log.csv"
    init_csv(log_csv, log_header)

    # -- Training -----------------------------------------------------------
    best_score  = -1.0
    no_improve  = 0
    opt_steps   = 0
    optimizer.zero_grad(set_to_none=True)

    monitor_key = "val_dice" if stage == 1 else "mAP@0.2-0.5"

    for epoch in range(1, args.max_epochs + 1):
        criterion.set_epoch(epoch)

        # =================== TRAIN =========================================
        model.train()
        run_loss = run_seg = run_det = run_cls = run_pres = 0.
        nb = 0

        for it, batch in enumerate(train_loader, 1):
            imgs       = batch["image"].to(device, non_blocking=True)
            gt_mask    = batch["mask"].to(device, non_blocking=True)
            box_single = batch["box_single"][0].to(device)
            train_role = batch["train_role"][0]

            with amp.autocast():
                outputs     = model(imgs)
                loss, comps = criterion(outputs, gt_mask, box_single,
                                        train_role)
                loss = loss / args.accum_steps

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            amp.scaler.scale(loss).backward()

            if it % args.accum_steps == 0:
                amp.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                amp.scaler.step(optimizer)
                amp.scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_steps += 1

            run_loss += loss.item() * args.accum_steps
            run_seg  += _safe_float(comps.get("seg"))
            run_det  += _safe_float(comps.get("det_bce_pos",
                                    comps.get("suppress")))
            run_cls  += _safe_float(comps.get("cls"))
            run_pres += _safe_float(comps.get("presence"))
            nb       += 1

        # =================== VALIDATE ======================================
        model.eval()
        vp_boxes    = []; vp_scores  = []; vp_flags = []
        vp_pres_scores = []   # stage 3: presence probabilities
        vgt_boxes   = []
        pred_masks  = []; gt_masks_v = []
        pred_cls_l  = []; true_cls_l = []

        with torch.no_grad():
            for batch in val_loader:
                imgs       = batch["image"].to(device, non_blocking=True)
                gt_mask    = batch["mask"][0]        # (1,D,H,W) on cpu
                box_single = batch["box_single"][0]
                train_role = batch["train_role"][0]
                has_box    = bool(batch["has_box"][0].item())
                cid        = batch["case_id"][0]

                outputs = model(imgs)
                pred_seg = torch.sigmoid(outputs["seg"][0].cpu())   # (1,D,H,W)

                pred_masks.append(pred_seg)
                gt_masks_v.append(gt_mask)

                true_cls = ROLE_TO_CLS.get(train_role, 3)
                true_cls_l.append(true_cls)

                if stage in (2, 3):
                    _, _C, d_dim, h_dim, w_dim = imgs.shape
                    boxes_dec, scores_dec = model.decode(outputs)
                    boxes_cpu  = boxes_dec[0].cpu()
                    scores_cpu = scores_dec[0].cpu()
                    cls_pred   = int(outputs["cls_logits"][0].argmax().item())
                    pred_cls_l.append(cls_pred)

                    if has_box:
                        gt_c = gt_norm_to_corners(
                            box_single, w_dim, h_dim, d_dim)
                        ious_all = box_iou_3d(
                            boxes_cpu, gt_c.view(1,6)).squeeze(1)
                        bi = int(torch.argmax(ious_all))
                        pb = boxes_cpu[bi]
                        sc = float(scores_cpu[bi])

                        vgt_boxes.append(gt_c)
                        vp_boxes.append(pb)
                        vp_scores.append(torch.tensor(sc))
                        vp_flags.append(True)
                    else:
                        bi = int(torch.argmax(scores_cpu))
                        vp_boxes.append(boxes_cpu[bi])
                        vp_scores.append(torch.tensor(
                            float(scores_cpu[bi])))
                        vp_flags.append(False)

                    # Stage 3: collect presence probability
                    if stage == 3 and "presence_logit" in outputs:
                        pres_p = float(torch.sigmoid(
                            outputs["presence_logit"][0]).item())
                        vp_pres_scores.append(torch.tensor(pres_p))
                    else:
                        vp_pres_scores.append(vp_scores[-1])
                else:
                    pred_cls_l.append(3)  # not computed in stage 1

        # =================== METRICS =======================================
        ml  = run_loss / max(1, nb)
        mse = run_seg  / max(1, nb)
        mde = run_det  / max(1, nb)
        mce = run_cls  / max(1, nb)
        mpe = run_pres / max(1, nb)

        val_dice = compute_dice_val(pred_masks, gt_masks_v)
        metrics  = {"val_dice": val_dice}

        if stage in (2, 3) and any(vp_flags):
            is_pos = torch.tensor(vp_flags, dtype=torch.bool)
            all_pb = torch.stack(vp_boxes,  0)
            all_sc = torch.stack(vp_scores, 0)
            pos_pb = all_pb[is_pos]
            gt_pb  = (torch.stack(vgt_boxes, 0)
                      if vgt_boxes else torch.zeros(0, 6))
            metrics.update(compute_val_metrics(pos_pb, gt_pb))
            # Stage 3: use presence probability for Sens/Spec
            # Stage 2: use objectness score
            if stage == 3 and len(vp_pres_scores) == len(vp_scores):
                all_sc_for_ss = torch.stack(vp_pres_scores, 0)
                ss_score_thr  = 0.5   # presence threshold
            else:
                all_sc_for_ss = all_sc
                ss_score_thr  = args.sens_score
            ss = compute_sens_spec(
                all_pb, all_sc_for_ss, is_pos,
                gt_boxes_list = vgt_boxes,
                iou_thr       = args.sens_iou,
                score_thr     = ss_score_thr)
            metrics["Sensitivity"] = ss["Sensitivity"]
            metrics["Specificity"] = ss["Specificity"]
            metrics["TP"] = float(ss["TP"])
            metrics["FN"] = float(ss["FN"])
            metrics["FP"] = float(ss["FP"])
            metrics["TN"] = float(ss["TN"])
            metrics.update(compute_cls_accuracy(pred_cls_l, true_cls_l))

            # Stage 3: presence accuracy
            if stage == 3:
                pres_correct = 0
                pres_total   = len(true_cls_l)
                for tc, pb, sc in zip(true_cls_l,
                                      vp_boxes[:len(true_cls_l)],
                                      vp_scores[:len(true_cls_l)]):
                    # true has_lesion: any role except negative (3)
                    true_has = int(tc != 3)
                    if "presence_logit" in outputs:
                        pass  # handled per-batch above
                # Simple proxy: use TN+TP / total
                pres_correct = (metrics["TP"] + metrics["TN"])
                metrics["presence_acc"] = pres_correct / max(1, pres_total)
            else:
                metrics["presence_acc"] = 0.
        else:
            for k in ["mAP@0.2-0.5","meanIoU","Sensitivity","Specificity"]:
                metrics[k] = 0.
            metrics["cls_acc"] = 0.
            metrics["TP"] = metrics["FN"] = 0.
            metrics["FP"] = metrics["TN"] = 0.
            metrics["presence_acc"] = 0.

        score = metrics.get(monitor_key, -1.)

        # LR string
        lr_str = "  ".join(
            f"{pg.get('name','g')}={pg['lr']:.1e}"
            for pg in optimizer.param_groups)

        sens_str = (f"  Sens={metrics.get('Sensitivity',0.):.3f}"
                    f"  Spec={metrics.get('Specificity',0.):.3f}"
                    f"  TP={int(metrics.get('TP',0))}"
                    f"  FP={int(metrics.get('FP',0))}"
                    f"  TN={int(metrics.get('TN',0))}"
                    f"  FN={int(metrics.get('FN',0))}"
                    if stage == 2 else "")
        pres_str = (f"  pres={mpe:.4f}  pres_acc={metrics.get('presence_acc',0.):.3f}"
                    if stage == 3 else "")
        print(f"Epoch {epoch:03d}/{args.max_epochs} | "
              f"loss={ml:.4f} seg={mse:.4f} det={mde:.4f} "
              f"cls={mce:.4f}"
              + (f" pres={mpe:.4f}" if stage==3 else "")
              + f" | dice={val_dice:.4f}"
              + (f"  mAP={metrics.get('mAP@0.2-0.5',0.):.4f}"
                 if stage in (2,3) else "")
              + sens_str
              + f" | {lr_str}")

        # -- MLflow logging ------------------------------------------------
        if MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                "train_loss":  ml,
                "seg_loss":    mse,
                "det_loss":    mde,
                "cls_loss":    mce,
                "val_dice":    val_dice,
                "mAP":         metrics.get("mAP@0.2-0.5", 0.),
                "meanIoU":     metrics.get("meanIoU", 0.),
                "cls_acc":     metrics.get("cls_acc", 0.),
                "Sensitivity":   metrics.get("Sensitivity", 0.),
                "Specificity":   metrics.get("Specificity", 0.),
                "TP": metrics.get("TP", 0.),
                "FN": metrics.get("FN", 0.),
                "FP": metrics.get("FP", 0.),
                "TN": metrics.get("TN", 0.),
                "presence_loss": mpe,
                "presence_acc":  metrics.get("presence_acc", 0.),
                **{f"lr_{pg['name']}": pg["lr"]
                   for pg in optimizer.param_groups},
            }, step=epoch)
            # GPU memory
            if cuda:
                mlflow.log_metric(
                    "gpu_mem_MB",
                    torch.cuda.memory_allocated() / 1e6,
                    step=epoch)

        # -- CSV logging ---------------------------------------------------
        append_csv(log_csv, [
            epoch, ml, mse, mde, mce,
            val_dice,
            metrics.get("mAP@0.2-0.5", 0.),
            metrics.get("meanIoU", 0.),
            metrics.get("cls_acc", 0.),
            metrics.get("Sensitivity", 0.),
            metrics.get("Specificity", 0.),
        ])

        # -- Checkpoint -----------------------------------------------------
        state = {
            "epoch":      epoch,
            "stage":      stage,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scaler":     amp.scaler.state_dict(),
            "args":       vars(args),
            "best_score": best_score,
            "monitor":    monitor_key,
        }
        torch.save(state, ckpt_dir / "last.pt")

        if score >= 0 and score > best_score + args.min_delta:
            best_score = score; no_improve = 0
            torch.save({**state, "best_score": best_score},
                       ckpt_dir / "best.pt")
            print(f"  -> best {monitor_key}={best_score:.4f}")
            if MLFLOW_AVAILABLE:
                mlflow.log_metric("best_score", best_score, step=epoch)
        else:
            no_improve += 1

        if args.early_stop and no_improve >= args.patience:
            print(f"[EARLY STOP] {args.patience} epochs without improvement.")
            break

    # -- Restore best -------------------------------------------------------
    bp = ckpt_dir / "best.pt"
    if bp.exists():
        chk = torch.load(bp, map_location="cpu", weights_only=False)
        model.load_state_dict(chk["model"])
        print(f"[INFO] Restored best — "
              f"{monitor_key}={chk.get('best_score'):.4f}")
        if MLFLOW_AVAILABLE:
            mlflow.pytorch.log_model(model, "best_model")
            print("[MLflow] Best model logged to registry.")

    if MLFLOW_AVAILABLE:
        mlflow.end_run()
        print("[MLflow] Run ended. View at: mlflow ui --port 5000")


# =============================================================================
# CLI
# =============================================================================

def _get_parser():
    p = argparse.ArgumentParser(
        description="Segmentation-Supervised Stroke Detection")

    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config file. CLI args override config.")

    # Paths
    p.add_argument("--root",        type=str, default=None)
    p.add_argument("--split_csv",   type=str, default=None)
    p.add_argument("--output_dir",  type=str,
                   default="D:\\Stroke\\checkpoints_stroke")
    p.add_argument("--resume",      type=str, default=None)

    # MLflow
    p.add_argument("--experiment_name", type=str,
                   default="stroke-detection-isles22")
    p.add_argument("--run_name",        type=str, default="run")

    # Stage
    p.add_argument("--stage",  type=int, default=1, choices=[1, 2])

    # Data
    p.add_argument("--workers",      type=int,   default=0)
    p.add_argument("--seed",         type=int,   default=2026)
    p.add_argument("--spacing",      type=float, nargs=3,
                   default=[1.5, 1.5, 3.0])
    p.add_argument("--input_shape",  type=int,   nargs=3,
                   default=[96, 96, 64])

    # Model
    p.add_argument("--base_channels", type=int, nargs=4,
                   default=[16, 24, 32, 48])
    p.add_argument("--fpn_channels",  type=int, default=32)
    p.add_argument("--cls_hidden",    type=int, default=64)

    # Training
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--max_epochs",   type=int,   default=120)
    p.add_argument("--accum_steps",  type=int,   default=4)
    p.add_argument("--amp",          action="store_true")

    # LR groups
    p.add_argument("--backbone_lr_scale", type=float, default=1.0)
    p.add_argument("--cls_lr_scale",      type=float, default=0.3)

    # Loss
    p.add_argument("--lambda_seg",       type=float, default=1.0)
    p.add_argument("--lambda_giou",      type=float, default=2.0)
    p.add_argument("--lambda_l1",        type=float, default=5.0)
    p.add_argument("--lambda_obj_bg",    type=float, default=0.05)
    p.add_argument("--lambda_cls",       type=float, default=0.3)
    p.add_argument("--lambda_presence",  type=float, default=1.0,
                   help="Presence loss weight (stage 3 only)")
    p.add_argument("--pos_obj_weight",   type=float, default=3.0)
    p.add_argument("--giou_warmup_epochs", type=int, default=30)

    # Sampler
    p.add_argument("--use_weighted_sampler",   action="store_true")
    p.add_argument("--weight_positive_single", type=float, default=1.0)
    p.add_argument("--weight_positive_multi",  type=float, default=10.0)
    p.add_argument("--weight_embolic",         type=float, default=2.0)
    p.add_argument("--weight_negative",        type=float, default=1.0)

    # Validation
    p.add_argument("--nms_iou",    type=float, default=0.3)
    p.add_argument("--val_topk",   type=int,   default=64)
    p.add_argument("--sens_iou",   type=float, default=0.3)
    p.add_argument("--sens_score", type=float, default=0.3)
    p.add_argument("--monitor",    type=str,   default="dice",
                   choices=["dice", "mAP"])

    # Early stop
    p.add_argument("--early_stop", action="store_true")
    p.add_argument("--patience",   type=int,   default=30)
    p.add_argument("--min_delta",  type=float, default=1e-4)

    # Debug
    p.add_argument("--debug_print_epochs", type=int, default=0)

    return p


if __name__ == "__main__":
    parser = _get_parser()
    args   = parser.parse_args()

    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        args   = merge_args_config(args, config)

    # Validate required args
    if not args.root:
        parser.error("--root is required (or set in config)")
    if not args.split_csv:
        parser.error("--split_csv is required (or set in config)")

    if args.amp and not torch.cuda.is_available():
        print("[WARN] --amp ignored (no CUDA).")
        args.amp = False

    train(args)