#!/usr/bin/env python3
"""
eval_synthetic_b.py — Experiment A: Synthetic b-value generalisation.

Fixes applied after debugging:
  1. GT box: xmin/xmax in original voxel space → scaled to model space
  2. GT mask Dice: uses SAME crop start as DWI preprocessing (not GT CoM)
  3. has_gt_lesion: matches evaluate_test.py definition
     (negative role = no lesion, all others = lesion present)
  4. mAP: AP computed correctly from IoU vs threshold

Usage
-----
  python eval_synthetic_b.py ^
      --checkpoint  D:/Stroke/checkpoints_stage3/best.pt ^
      --split_csv   D:/Stroke/ISLES-2022/isles_final_split.csv ^
      --root        D:/Stroke/ISLES-2022 ^
      --synth_dir   D:/Stroke/synthetic_b ^
      --out_dir     D:/Stroke/results_exp_a ^
      --b_values    1000 1500 2000 2500
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import zoom
from nibabel.orientations import (axcodes2ornt, io_orientation,
                                   inv_ornt_aff, apply_orientation,
                                   ornt_transform)

sys.path.insert(0, str(Path(__file__).parent))
from stroke_detector.model import build_model, nms_3d
from stroke_detector.data import robust_zscore


# =============================================================================
# Geometry helpers
# =============================================================================

def box_iou_3d(a, b):
    ix0=max(a[0],b[0]); iy0=max(a[1],b[1]); iz0=max(a[2],b[2])
    ix1=min(a[3],b[3]); iy1=min(a[4],b[4]); iz1=min(a[5],b[5])
    inter=max(0,ix1-ix0)*max(0,iy1-iy0)*max(0,iz1-iz0)
    va=max(0,a[3]-a[0])*max(0,a[4]-a[1])*max(0,a[5]-a[2])
    vb=max(0,b[3]-b[0])*max(0,b[4]-b[1])*max(0,b[5]-b[2])
    union=va+vb-inter
    return inter/union if union>0 else 0.

def dice_coeff(pred, gt):
    p=(pred>0).astype(np.float32); g=(gt>0).astype(np.float32)
    inter=(p*g).sum(); denom=p.sum()+g.sum()
    return 2*inter/denom if denom>0 else float('nan')

def clamp_box(box, W, H, D):
    mins=torch.clamp(box[:3],min=0.)
    maxs=torch.clamp(box[3:],max=torch.tensor([W-1.,H-1.,D-1.]))
    return torch.cat([mins,maxs])


# =============================================================================
# Preprocessing — returns tensor AND crop metadata for GT mask alignment
# =============================================================================

def reorient_ras(arr, aff):
    ot  = ornt_transform(io_orientation(aff), axcodes2ornt(('R','A','S')))
    arr = apply_orientation(arr, ot)
    aff = aff @ inv_ornt_aff(ot, arr.shape)
    return arr, aff, ot

def com_crop_with_meta(arr, tgt_shape):
    """CoM crop — returns cropped array AND crop_start for reuse."""
    ins   = np.array(arr.shape,  dtype=int)
    tgt   = np.array(tgt_shape,  dtype=int)
    brain = arr > 0
    com   = np.argwhere(brain).mean(0).astype(int) if brain.sum()>100 else ins//2
    st    = np.clip(com-tgt//2, 0, np.maximum(ins-tgt,0))
    en    = st + tgt
    cr    = arr[st[0]:en[0], st[1]:en[1], st[2]:en[2]]
    out   = np.zeros(tuple(tgt), dtype=arr.dtype)
    cs    = np.array(cr.shape, dtype=int)
    ps    = np.clip((tgt-cs)//2, 0, tgt-cs)
    out[ps[0]:ps[0]+cs[0], ps[1]:ps[1]+cs[1], ps[2]:ps[2]+cs[2]] = cr
    return out, st   # st = crop_start in resampled space

def apply_crop(arr, crop_start, tgt_shape, pad_start=None):
    """
    Apply a pre-computed crop_start to GT mask.
    pad_start: offset within output array (same as DWI preprocessing).
    Must match the pad_start used when cropping the DWI so both arrays
    are in the same coordinate frame.
    """
    tgt   = np.array(tgt_shape,  dtype=int)
    st    = np.array(crop_start, dtype=int)
    ins   = np.array(arr.shape,  dtype=int)
    en_cl = np.minimum(st + tgt, ins)
    st_cl = np.minimum(st, ins)
    cr    = arr[st_cl[0]:en_cl[0], st_cl[1]:en_cl[1], st_cl[2]:en_cl[2]]
    out   = np.zeros(tuple(tgt), dtype=arr.dtype)
    cs    = np.array(cr.shape, dtype=int)
    if pad_start is not None:
        ps = np.array(pad_start, dtype=int)
        out[ps[0]:ps[0]+cs[0], ps[1]:ps[1]+cs[1], ps[2]:ps[2]+cs[2]] = cr
    else:
        out[:cs[0], :cs[1], :cs[2]] = cr
    return out

def preprocess_dwi_adc(dwi_path, adc_path,
                        model_shape=(96,96,64),
                        spacing=(1.5,1.5,3.0)):
    """
    Preprocess DWI + ADC to model input tensor.
    Returns (X_tensor, crop_metadata) where crop_metadata is used
    to align GT mask to the same spatial position.
    """
    def load(p):
        img=nib.load(str(p))
        return img.get_fdata(dtype=np.float32), img.affine, img.header

    dwi, dwi_aff, dwi_hdr = load(dwi_path)
    adc, adc_aff, _        = load(adc_path)

    native_z = tuple(float(z) for z in dwi_hdr.get_zooms()[:3])
    sf        = np.array(native_z) / np.array(spacing)

    # Reorient to RAS
    dwi_r, _, ot = reorient_ras(dwi, dwi_aff)
    adc_r, _,  _ = reorient_ras(adc, adc_aff)

    # Resample to canonical spacing
    dwi_rs = zoom(dwi_r, sf, order=1, mode='nearest', prefilter=False)
    adc_rs = zoom(adc_r, sf, order=1, mode='nearest', prefilter=False)

    # CoM crop — record crop_start for GT mask reuse
    dwi_m, crop_start = com_crop_with_meta(dwi_rs, model_shape)
    adc_m, _          = com_crop_with_meta(adc_rs, model_shape)

    # Exact resize to model shape
    cur = np.array(dwi_m.shape, dtype=np.float32)
    tgt = np.array(model_shape,  dtype=np.float32)
    resize_f = tgt / cur if not np.allclose(cur,tgt) else np.ones(3)
    if not np.allclose(resize_f, 1.):
        dwi_m = zoom(dwi_m, resize_f, order=1, mode='nearest', prefilter=False)
        adc_m = zoom(adc_m, resize_f, order=1, mode='nearest', prefilter=False)

    # Z-score
    bm    = dwi_m > 0
    dwi_m = robust_zscore(dwi_m, bm)
    abm   = adc_m != 0
    adc_m = robust_zscore(adc_m, abm if abm.any() else bm)

    X = np.stack([dwi_m, adc_m], 0)
    X = torch.from_numpy(X).float().permute(0,3,2,1).unsqueeze(0)

    # Also record pad_start — offset within output array when crop < model shape
    # This is applied to DWI but NOT to GT mask in apply_crop → must correct
    crop_shape = np.array(dwi_rs.shape, dtype=int)
    tgt_arr    = np.array(model_shape,  dtype=int)
    actual_end = np.minimum(crop_start + tgt_arr, crop_shape)
    actual_crop_shape = actual_end - crop_start
    pad_start  = np.clip((tgt_arr - actual_crop_shape)//2, 0, tgt_arr - actual_crop_shape)

    meta = {
        "native_z":    native_z,
        "spacing":     spacing,
        "sf":          sf,
        "ornt_transf": ot,
        "rs_shape":    dwi_rs.shape,
        "crop_start":  crop_start,
        "pad_start":   pad_start,           # offset within model array
        "crop_shape":  tuple(actual_crop_shape.tolist()),
        "resize_f":    resize_f,
        "model_shape": model_shape,
    }
    return X, meta


def preprocess_gt_mask(mask_path, meta):
    """
    Preprocess GT mask using IDENTICAL spatial transforms as the DWI.
    Uses the recorded crop_start from preprocess_dwi_adc so the mask
    and the model's seg prediction are in the same coordinate frame.
    """
    img    = nib.load(str(mask_path))
    gt     = img.get_fdata(dtype=np.float32)
    gt_aff = img.affine
    gt_hdr = img.header
    gt_z   = tuple(float(z) for z in gt_hdr.get_zooms()[:3])

    # Reorient to RAS
    ot      = meta["ornt_transf"]
    gt_r    = apply_orientation(gt, ot)

    # Resample — use GT's own voxel size
    sf_gt   = np.array(gt_z) / np.array(meta["spacing"])
    gt_rs   = zoom(gt_r, sf_gt, order=0, mode='nearest', prefilter=False)

    # Apply SAME crop_start as DWI (not GT's own CoM)
    gt_m    = apply_crop(gt_rs, meta["crop_start"], meta["model_shape"],
                         pad_start=meta.get("pad_start"))

    # Apply same resize factor as DWI
    rf = meta["resize_f"]
    if not np.allclose(rf, 1.):
        gt_m = zoom(gt_m, rf, order=0, mode='nearest', prefilter=False)

    # Return as D,H,W (same orientation as model seg output)
    return gt_m.transpose(2, 1, 0)


# =============================================================================
# GT box transform — replicates StrokeDataset.forward_bbox_native_to_model
# =============================================================================

def gt_box_native_to_model(bbox_vox, shape_native, meta):
    """
    Transform GT bounding box from native image voxel space
    to model space corners, then to normalised YOLO format.

    Replicates exactly:
      StrokeDataset.PreprocessChain.forward_bbox_native_to_model()
      StrokeDataset.PreprocessChain.normalize_model_box()

    Parameters
    ----------
    bbox_vox     : dict with xmin,xmax,ymin,ymax,zmin,zmax
    shape_native : (X,Y,Z) shape of native image
    meta         : dict from preprocess_dwi_adc()

    Returns
    -------
    gt_corners : [x0,y0,z0,x1,y1,z1] in model voxel space
    """
    xmn=float(bbox_vox["xmin"]); xmx=float(bbox_vox["xmax"])
    ymn=float(bbox_vox["ymin"]); ymx=float(bbox_vox["ymax"])
    zmn=float(bbox_vox["zmin"]); zmx=float(bbox_vox["zmax"])

    # 8 corners of the bounding box in native space
    corners = np.array([
        [xmn,ymn,zmn],[xmx,ymn,zmn],[xmn,ymx,zmn],[xmn,ymn,zmx],
        [xmx,ymx,zmn],[xmx,ymn,zmx],[xmn,ymx,zmx],[xmx,ymx,zmx],
    ], dtype=np.float64)

    # Step 1: apply orientation transform (axis permutation + flip)
    ot   = meta["ornt_transf"]    # shape (3,2): [[src_ax, flip], ...]
    nat  = np.array(shape_native, dtype=np.float64)
    out  = np.zeros_like(corners)
    for da in range(3):
        sa = int(ot[da, 0])
        if ot[da, 1] == 1.:
            out[:, da] = corners[:, sa]
        else:
            out[:, da] = (nat[sa] - 1.) - corners[:, sa]
    corners = out

    # Step 2: multiply by spacing scale (resampling factor)
    sf = np.array(meta["sf"], dtype=np.float64)
    corners *= sf

    # Step 3: subtract crop start
    cs = np.array(meta["crop_start"], dtype=np.float64)
    corners -= cs

    # Step 4: multiply by final resize factor
    rf = np.array(meta["resize_f"], dtype=np.float64)
    corners *= rf

    # Corners in model space
    mn = corners.min(axis=0)
    mx = corners.max(axis=0)

    W, H, D = meta["model_shape"]
    mn = np.maximum(mn, 0.)
    mx = np.minimum(mx, np.array([W, H, D], dtype=np.float64))

    return [mn[0], mn[1], mn[2], mx[0], mx[1], mx[2]]


# =============================================================================
# Inference
# =============================================================================

@torch.no_grad()
def infer(model, X, spacing, score_thresh=0.3, seg_thresh=0.5):
    model.eval()
    W,H,D = model.input_shape
    out   = model(X)

    boxes_dec, scores_dec = model.decode(out)
    boxes_cpu  = boxes_dec[0].cpu()
    scores_cpu = scores_dec[0].cpu()

    keep = nms_3d(boxes_cpu, scores_cpu, 0.3)
    if not keep:
        keep = [int(scores_cpu.argmax().item())]

    pred_box   = clamp_box(boxes_cpu[keep[0]], W, H, D)
    pred_score = float(scores_cpu[keep[0]].item())

    seg_prob   = torch.sigmoid(out["seg"])[0,0].cpu().numpy()
    seg_binary = (seg_prob > seg_thresh).astype(np.float32)
    seg_vol_ml = float(seg_binary.sum() *
                       spacing[0]*spacing[1]*spacing[2]/1000.)

    if model.stage==3 and "presence_logit" in out:
        pres = float(torch.sigmoid(out["presence_logit"][0]).item())
    else:
        pres = pred_score

    has_lesion = pred_score>=score_thresh and seg_vol_ml>=0.15

    return {
        "pred_box":    pred_box.tolist(),
        "pred_score":  pred_score,
        "pres":        pres,
        "has_lesion":  has_lesion,
        "seg_binary":  seg_binary,   # D,H,W
        "seg_vol_ml":  seg_vol_ml,
    }


# =============================================================================
# Evaluate one b-value across all test cases
# =============================================================================

def evaluate_b(b_val, df, root, synth_dir, model, device,
               model_shape=(96,96,64), spacing=(1.5,1.5,3.0),
               score_thresh=0.3):
    W,H,D    = model_shape
    records  = []

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  b={b_val}", leave=False):
        subject    = row["subject"]
        ses        = str(row.get("ses","ses-0001"))
        train_role = str(row.get("train_role","unknown"))

        # ── resolve paths ─────────────────────────────────────────────────
        if b_val == 1000:
            cands = [
                Path(root)/subject/ses/"dwi"/f"{subject}_{ses}_dwi.nii.gz",
                Path(root)/subject/ses/"dwi"/f"{subject}_{ses}_dwi.nii",
            ]
            dwi_path = next((p for p in cands if p.exists()), None)
        else:
            dwi_path = (Path(synth_dir)/f"b{b_val}"/
                        f"{subject}_b{b_val}_dwi.nii.gz")
            if not dwi_path.exists():
                dwi_path = None

        adc_cands = [
            Path(root)/subject/ses/"dwi"/f"{subject}_{ses}_adc.nii.gz",
            Path(root)/subject/ses/"dwi"/f"{subject}_{ses}_adc.nii",
        ]
        adc_path = next((p for p in adc_cands if p.exists()), None)

        if dwi_path is None:
            records.append({"subject":subject,
                             "error":f"DWI missing b={b_val}","b_value":b_val})
            continue
        if adc_path is None:
            records.append({"subject":subject,
                             "error":"ADC missing","b_value":b_val})
            continue

        # ── preprocess ────────────────────────────────────────────────────
        try:
            X, meta = preprocess_dwi_adc(str(dwi_path), str(adc_path),
                                          model_shape, spacing)
            X = X.to(device)
        except Exception as e:
            records.append({"subject":subject,
                             "error":f"preprocess:{e}","b_value":b_val})
            continue

        # ── inference ─────────────────────────────────────────────────────
        try:
            res = infer(model, X, spacing, score_thresh)
        except Exception as e:
            records.append({"subject":subject,
                             "error":f"inference:{e}","b_value":b_val})
            continue

        # ── GT — following evaluate_test.py exactly ───────────────────────
        # has_gt_lesion: any non-negative role has a lesion
        # has_spatial_box: only positive_single/multi have a reliable GT box
        #   (embolic lesions are scattered — no single box)
        has_gt       = (train_role != "negative")
        has_spatial_box = train_role in ("positive_single", "positive_multi")

        # GT box: load from bbox_json file → apply same transform as training
        # Only positive_single and positive_multi have reliable single GT boxes
        iou_val = 0.0
        if has_spatial_box:
            try:
                import json as _json
                bbox_json_path = str(row.get("bbox_json",""))
                shape_native   = (int(row["shape_x"]),
                                  int(row["shape_y"]),
                                  int(row["shape_z"]))
                with open(bbox_json_path) as _f:
                    bj = _json.load(_f)

                # Use largest component bbox (same as StrokeDataset)
                bv = (bj.get("bbox_vox") or
                      (bj["bboxes_vox"][0] if bj.get("bboxes_vox") else None))
                if bv is not None:
                    gt_box  = gt_box_native_to_model(bv, shape_native, meta)
                    iou_val = box_iou_3d(res["pred_box"], gt_box)
            except Exception as e:
                iou_val = 0.0

        # GT mask Dice — for all cases that have a GT mask (non-negative)
        gt_dice = float('nan')
        if has_gt and "mask_path" in row and pd.notna(row.get("mask_path","")):
            try:
                gt_dhw  = preprocess_gt_mask(str(row["mask_path"]), meta)
                gt_dice = dice_coeff(res["seg_binary"], gt_dhw)
            except Exception:
                gt_dice = float('nan')

        records.append({
            "subject":         subject,
            "train_role":      train_role,
            "b_value":         b_val,
            "has_gt_lesion":   int(has_gt),
            "has_spatial_box": int(has_spatial_box),
            "pred_has_lesion": int(res["has_lesion"]),
            "confidence":      round(res["pred_score"],4),
            "presence_conf":   round(res["pres"],4),
            "box_iou":         round(iou_val,4),
            "dice":            round(gt_dice,4) if not np.isnan(gt_dice) else float('nan'),
            "seg_vol_ml":      round(res["seg_vol_ml"],3),
            "error":           "",
        })

    return pd.DataFrame(records)


# =============================================================================
# Summary
# =============================================================================

def summarise(df_b, iou_thresholds=(0.2,0.3,0.4,0.5)):
    valid = df_b[df_b["error"]==""].copy()
    if len(valid)==0:
        return {}

    pos     = valid[valid["has_gt_lesion"]==1]    # all lesion cases (sens/spec)
    neg     = valid[valid["has_gt_lesion"]==0]    # negative cases
    pos_box = valid[valid.get("has_spatial_box", valid["has_gt_lesion"])==1]
    if "has_spatial_box" in valid.columns:
        pos_box = valid[valid["has_spatial_box"]==1]  # only single/multi for mAP

    # Sensitivity/Specificity — all lesion cases vs negative
    tp = int((pos["pred_has_lesion"]==1).sum())
    fn = int((pos["pred_has_lesion"]==0).sum())
    fp = int((neg["pred_has_lesion"]==1).sum())
    tn = int((neg["pred_has_lesion"]==0).sum())
    sens = tp/max(1,tp+fn)
    spec = tn/max(1,tn+fp)

    # mAP — only positive_single and positive_multi (have spatial GT boxes)
    ap_vals = {}
    for thr in iou_thresholds:
        rows = [(float(r["box_iou"]), float(r["confidence"]))
                for _,r in pos_box.iterrows()
                if r["error"]=="" and pd.notna(r["box_iou"])]
        rows = sorted(rows, key=lambda x:-x[1])
        tp_c=0; precs=[]
        for i,(iou,_) in enumerate(rows):
            if iou >= thr:
                tp_c += 1
            precs.append(tp_c/(i+1))
        ap_vals[f"AP@{thr}"] = round(float(np.mean(precs)) if precs else 0., 4)
    mAP = round(float(np.mean(list(ap_vals.values()))), 4)

    dice_col  = pos_box["dice"] if "has_spatial_box" in valid.columns else pos["dice"]
    dice_vals = pd.to_numeric(dice_col, errors="coerce").dropna()
    vol_vals  = pd.to_numeric(valid["seg_vol_ml"], errors="coerce").dropna()

    return {
        "b_value":         int(df_b["b_value"].iloc[0]),
        "n_cases":         len(valid),
        "n_positive":      len(pos),
        "n_negative":      len(neg),
        "mAP@0.2-0.5":    mAP,
        **ap_vals,
        "Sensitivity":     round(sens,4),
        "Specificity":     round(spec,4),
        "TP":tp,"FN":fn,"FP":fp,"TN":tn,
        "mean_Dice":       round(float(dice_vals.mean()),4) if len(dice_vals) else float('nan'),
        "std_Dice":        round(float(dice_vals.std()), 4) if len(dice_vals) else float('nan'),
        "mean_seg_vol_ml": round(float(vol_vals.mean()), 3) if len(vol_vals)  else float('nan'),
        "std_seg_vol_ml":  round(float(vol_vals.std()),  3) if len(vol_vals)  else float('nan'),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",   required=True)
    ap.add_argument("--split_csv",    required=True)
    ap.add_argument("--root",         required=True)
    ap.add_argument("--synth_dir",    required=True)
    ap.add_argument("--out_dir",      required=True)
    ap.add_argument("--b_values",     type=int, nargs="+",
                    default=[1000,1500,2000,2500])
    ap.add_argument("--split",        default="test")
    ap.add_argument("--model_shape",  type=int,   nargs=3, default=[96,96,64])
    ap.add_argument("--spacing",      type=float, nargs=3, default=[1.5,1.5,3.0])
    ap.add_argument("--score_thresh", type=float, default=0.3)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    chk        = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args  = chk.get("args", {})
    ckpt_stage = ckpt_args.get("stage", 2)
    model = build_model(
        in_ch         = 2,
        base_channels = tuple(ckpt_args.get("base_channels",[16,24,32,48])),
        fpn_channels  = ckpt_args.get("fpn_channels",32),
        input_shape   = tuple(args.model_shape),
        num_classes   = 4,
        cls_hidden    = ckpt_args.get("cls_hidden",64),
        stage         = ckpt_stage,
    ).to(device)
    model.load_state_dict(chk["model"], strict=False)
    print(f"[INFO] Stage {ckpt_stage} · "
          f"{sum(p.numel() for p in model.parameters())/1e6:.3f}M · "
          f"epoch={chk.get('epoch','?')}")

    df = pd.read_csv(args.split_csv)
    df = df[df["split"]==args.split].reset_index(drop=True)
    print(f"[INFO] Test cases: {len(df)}  "
          f"(pos={int((df.train_role!='negative').sum())}  "
          f"neg={int((df.train_role=='negative').sum())})")

    summaries = []
    for b_val in args.b_values:
        print(f"\n[EVAL] b = {b_val}"
              + (" (original ISLES)" if b_val==1000 else " (synthetic)"))
        df_b = evaluate_b(
            b_val, df,
            root         = args.root,
            synth_dir    = args.synth_dir,
            model        = model,
            device       = device,
            model_shape  = tuple(args.model_shape),
            spacing      = tuple(args.spacing),
            score_thresh = args.score_thresh,
        )
        df_b["b_value"] = b_val
        df_b.to_csv(out_dir/f"per_case_b{b_val}.csv", index=False)

        s = summarise(df_b)
        summaries.append(s)

        print(f"  mAP@0.2-0.5 = {s.get('mAP@0.2-0.5','n/a')}")
        print(f"  AP@0.2={s.get('AP@0.2','?')}  AP@0.3={s.get('AP@0.3','?')}  "
              f"AP@0.4={s.get('AP@0.4','?')}  AP@0.5={s.get('AP@0.5','?')}")
        print(f"  Sensitivity={s.get('Sensitivity','?')}  "
              f"Specificity={s.get('Specificity','?')}")
        print(f"  TP={s.get('TP',0)}  FN={s.get('FN',0)}  "
              f"FP={s.get('FP',0)}  TN={s.get('TN',0)}")
        print(f"  mean Dice   = {s.get('mean_Dice','?')} "
              f"(±{s.get('std_Dice','?')})")
        print(f"  mean vol ml = {s.get('mean_seg_vol_ml','?')} "
              f"(±{s.get('std_seg_vol_ml','?')})")

    # Summary CSV
    pd.DataFrame(summaries).to_csv(out_dir/"summary_table.csv", index=False)

    # Formatted text table for paper
    with open(out_dir/"summary_table.txt","w") as f:
        f.write("Experiment A — Synthetic b-value Generalisation\n")
        f.write("="*72+"\n")
        f.write("Model: S(b)=S(1000)*exp(-(b-1000)*ADC_mm2_s)  "
                "Ref: Bladt et al. PMC8506195\n")
        f.write("ADC units auto-detected per case\n")
        f.write("="*72+"\n\n")
        f.write(f"{'b':>6}  {'mAP':>7}  {'Sens':>6}  {'Spec':>6}  "
                f"{'Dice':>7}  {'Vol(ml)':>8}  TP  FP  TN  FN\n")
        f.write("-"*72+"\n")
        for s in summaries:
            f.write(
                f"{s.get('b_value',0):>6}  "
                f"{s.get('mAP@0.2-0.5',0):>7.4f}  "
                f"{s.get('Sensitivity',0):>6.4f}  "
                f"{s.get('Specificity',0):>6.4f}  "
                f"{s.get('mean_Dice',0):>7.4f}  "
                f"{s.get('mean_seg_vol_ml',0):>8.3f}  "
                f"{s.get('TP',0):>2}  "
                f"{s.get('FP',0):>2}  "
                f"{s.get('TN',0):>2}  "
                f"{s.get('FN',0):>2}\n")
        f.write("\n")
        f.write("b=1000: original ISLES DWI (in-distribution baseline)\n")
        f.write("b>1000: synthetic via mono-exponential decay from b=1000\n")
        f.write("NOTE: b=1000 baseline should match evaluate_test.py results\n")

    print(f"\n[DONE] {out_dir}")


if __name__ == "__main__":
    main()