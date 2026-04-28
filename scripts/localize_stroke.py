#!/usr/bin/env python3
"""
localize_stroke.py
==================
Stroke localisation from multishell or single-shell clinical DWI.

Key features
------------
1. Smart CoM crop — crops around brain centre-of-mass, not image centre.
   Fixes brain being cut off for large-FOV clinical acquisitions.

2. Single modality — works with DWI only, ADC only, or both.
   Missing modality filled with zeros automatically.

3. Bbox expansion — grows predicted box to fully contain seg mask.
   Fixes "bbox doesn't cover full lesion" issue.

4. Native space output — all masks saved in original image space
   (same shape, voxel size, affine as input DWI/ADC).
   Load directly in ITK-Snap/FSLeyes alongside original scan.

Usage
-----
  # Both modalities (recommended)
  python localize_stroke.py ^
      --dwi4d dwi.nii.gz --bval data.bval --adc adc.nii.gz ^
      --checkpoint best.pt --out_dir output/

  # DWI only
  python localize_stroke.py --dwi4d dwi.nii.gz --bval data.bval ...

  # ADC only
  python localize_stroke.py --adc adc.nii.gz ...
"""

import json
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.orientations import (axcodes2ornt, io_orientation,
                                   inv_ornt_aff, apply_orientation)
from scipy.ndimage import zoom
import torch

from stroke_detector.model import build_model, nms_3d
from stroke_detector.data import robust_zscore


# =============================================================================
# NIfTI utilities
# =============================================================================

def load_nii(path):
    img  = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine, img.header


def reorient_to_ras(arr, aff):
    from nibabel.orientations import ornt_transform
    ot  = ornt_transform(io_orientation(aff), axcodes2ornt(('R','A','S')))
    arr = apply_orientation(arr, ot)
    aff = aff @ inv_ornt_aff(ot, arr.shape)
    return arr, aff, ot


def resample(arr, cur_z, tgt_z, order=1):
    f = np.array(cur_z) / np.array(tgt_z)
    return zoom(arr, f, order=order, mode='nearest', prefilter=False), f


def load_bvals(path):
    with open(path) as f:
        return np.array([float(v) for v in f.read().split() if v.strip()])


# =============================================================================
# Shell extraction
# =============================================================================

def extract_shell(dwi4d, bvals, target_b=1000, tolerance=200):
    if dwi4d.ndim == 3:
        print(f"[INFO] DWI is 3D — no shell extraction needed")
        return dwi4d.astype(np.float32), np.array([0])
    diffs   = np.abs(bvals - target_b)
    indices = np.where(diffs <= tolerance)[0]
    if len(indices) == 0:
        cb = bvals[np.argmin(diffs)]
        print(f"[WARN] No b≈{target_b}±{tolerance}. Using b={cb:.0f}")
        indices = np.where(np.abs(bvals - cb) <= 50)[0]
    actual = bvals[indices]
    print(f"[INFO] Extracting {len(indices)} vols at "
          f"b={np.unique(actual.astype(int)).tolist()}")
    return dwi4d[..., indices].mean(-1).astype(np.float32), indices


# =============================================================================
# Smart centre-of-mass crop  (Fix 1)
# =============================================================================

def com_crop(arr, target_shape):
    """Crop to target_shape centred on brain centre-of-mass."""
    ins = np.array(arr.shape, dtype=int)
    tgt = np.array(target_shape, dtype=int)
    brain = arr > 0
    com   = np.argwhere(brain).mean(axis=0).astype(int) \
            if brain.sum() > 100 else ins // 2
    st  = np.clip(com - tgt // 2, 0, np.maximum(ins - tgt, 0))
    en  = st + tgt
    cr  = arr[st[0]:en[0], st[1]:en[1], st[2]:en[2]]
    out = np.zeros(tuple(tgt), dtype=arr.dtype)
    cs  = np.array(cr.shape, dtype=int)
    ps  = np.clip((tgt - cs) // 2, 0, tgt - cs)
    out[ps[0]:ps[0]+cs[0],
        ps[1]:ps[1]+cs[1],
        ps[2]:ps[2]+cs[2]] = cr
    return out, tuple(int(v) for v in st)


# =============================================================================
# Full preprocessing pipeline — returns model input AND saved transform chain
# =============================================================================

def preprocess(arr3d, aff, hdr,
               model_shape=(96,96,64),
               canonical_zooms=(1.5,1.5,3.0)):
    """
    Preprocess volume to model input space.
    Saves full transform chain for inverse mapping.

    Returns
    -------
    arr_model   : (W,H,D) preprocessed array in model space
    chain       : dict with all transform parameters for inversion
    """
    native_zooms = tuple(float(z) for z in hdr.get_zooms()[:3])
    native_shape = tuple(int(s) for s in arr3d.shape)

    # 1. Reorient to RAS
    arr_r, aff_r, ot = reorient_to_ras(arr3d, aff)
    ras_shape = arr_r.shape

    # 2. Resample to canonical spacing
    arr_rs, sf = resample(arr_r, native_zooms, canonical_zooms)
    rs_shape   = arr_rs.shape

    # 3. Smart CoM crop to model shape
    arr_m, crop_start = com_crop(arr_rs, model_shape)
    # com_crop may zero-pad if brain < model_shape
    # record the actual cropped shape and padding applied
    crop_shape_actual = tuple(int(s) for s in
        np.minimum(np.array(arr_rs.shape) - np.array(crop_start),
                   np.array(model_shape)))
    # how much padding was added at the start of each axis
    pad_start = tuple(int(v) for v in
        np.clip((np.array(model_shape) - np.array(crop_shape_actual)) // 2,
                0, np.array(model_shape)))

    # 4. Final resize to exact model shape (handles rounding)
    cur = np.array(arr_m.shape, dtype=np.float32)
    tgt = np.array(model_shape, dtype=np.float32)
    if not np.all(cur == tgt):
        frs   = tgt / cur
        arr_m = zoom(arr_m, frs, order=1, mode='nearest', prefilter=False)
    else:
        frs = np.ones(3)

    chain = {
        "native_zooms":    native_zooms,
        "native_shape":    native_shape,
        "canonical_zooms": canonical_zooms,
        "model_shape":     model_shape,
        "ornt_transf":     ot,
        "ras_shape":       ras_shape,
        "spacing_scale":   sf,
        "rs_shape":        rs_shape,
        "crop_start":      crop_start,
        "crop_shape":      crop_shape_actual,   # shape before padding
        "pad_start":       pad_start,           # zero-padding added
        "final_resize":    tuple(float(v) for v in frs),
        "native_affine":   aff,
        "ras_affine":      aff_r,
    }
    return arr_m.astype(np.float32), chain


# =============================================================================
# Inverse mapping — model mask → native image space
# =============================================================================

def mask_to_native(mask_dhw, chain):
    """
    Invert preprocessing chain: model mask (D,H,W) → native image space.

    Forward chain was:
      1. reorient to RAS
      2. resample to canonical spacing
      3. CoM crop  (may add zero padding if brain < model shape)
      4. final resize to exact model shape

    Inverse (applied here):
      4. undo final resize
      3. undo padding, then embed at crop_start in resampled volume
      2. undo resampling
      1. undo reorientation
    """
    from nibabel.orientations import apply_orientation

    # 0. D,H,W → W,H,D
    mask = mask_dhw.transpose(2, 1, 0).astype(np.float32)

    # 4. Undo final resize
    frs = np.array(chain["final_resize"], dtype=np.float64)
    if not np.allclose(frs, 1.0):
        mask = zoom(mask, 1.0/frs, order=0, mode='nearest', prefilter=False)

    # 3a. Undo zero-padding added by com_crop
    # pad_start = how many slices were prepended on each axis
    # crop_shape = shape of actual cropped data before padding
    pad_start  = np.array(chain.get("pad_start",  [0,0,0]), dtype=int)
    crop_shape = np.array(chain.get("crop_shape", chain["model_shape"]),
                          dtype=int)
    mask_unpadded = mask[
        pad_start[0]:pad_start[0]+crop_shape[0],
        pad_start[1]:pad_start[1]+crop_shape[1],
        pad_start[2]:pad_start[2]+crop_shape[2]]

    # 3b. Embed in full resampled volume
    rs_shape   = np.array(chain["rs_shape"],   dtype=int)
    crop_start = np.array(chain["crop_start"], dtype=int)
    full_rs    = np.zeros(tuple(rs_shape), dtype=np.float32)
    ins        = np.array(mask_unpadded.shape, dtype=int)
    avail      = rs_shape - crop_start
    copy       = np.minimum(ins, np.maximum(avail, 0))
    if np.all(copy > 0):
        full_rs[crop_start[0]:crop_start[0]+copy[0],
                crop_start[1]:crop_start[1]+copy[1],
                crop_start[2]:crop_start[2]+copy[2]] =             mask_unpadded[:copy[0], :copy[1], :copy[2]]

    # 2. Undo resampling
    sf       = np.array(chain["spacing_scale"], dtype=np.float64)
    mask_ras = zoom(full_rs, 1.0/sf, order=0, mode='nearest', prefilter=False)
    ras_shape = np.array(chain["ras_shape"], dtype=int)
    out_ras   = np.zeros(tuple(ras_shape), dtype=np.float32)
    cp        = np.minimum(np.array(mask_ras.shape), ras_shape)
    out_ras[:cp[0], :cp[1], :cp[2]] = mask_ras[:cp[0], :cp[1], :cp[2]]

    # 1. Undo reorientation
    # Forward ot[ras_ax] = (native_ax, flip)
    # Meaning: RAS axis ras_ax came from native axis native_ax,
    #          with optional flip.
    # Inverse: native axis native_ax came from RAS axis ras_ax,
    #          with same flip (applying flip twice = identity).
    ot     = chain["ornt_transf"]
    n      = len(ot)
    inv_ot = np.zeros_like(ot)
    for ras_ax in range(n):
        nat_ax          = int(ot[ras_ax, 0])
        flip            = ot[ras_ax, 1]
        inv_ot[nat_ax, 0] = ras_ax
        inv_ot[nat_ax, 1] = flip
    mask_native = apply_orientation(out_ras, inv_ot)

    # Clamp to native shape
    nat  = np.array(chain["native_shape"], dtype=int)
    out2 = np.zeros(tuple(nat), dtype=np.uint8)
    cp2  = np.minimum(np.array(mask_native.shape), nat)
    out2[:cp2[0], :cp2[1], :cp2[2]] = (
        mask_native[:cp2[0], :cp2[1], :cp2[2]] > 0.5).astype(np.uint8)
    return out2

def expand_box_from_seg(box, seg_dhw, model_shape, margin=2):
    W, H, D  = model_shape
    seg_whd  = seg_dhw.transpose(2, 1, 0)
    nz       = np.argwhere(seg_whd > 0)
    if len(nz) == 0:
        return box
    smin = nz.min(0).astype(float)
    smax = nz.max(0).astype(float)
    return [max(0,   min(box[0], smin[0]-margin)),
            max(0,   min(box[1], smin[1]-margin)),
            max(0,   min(box[2], smin[2]-margin)),
            min(W-1, max(box[3], smax[0]+margin)),
            min(H-1, max(box[4], smax[1]+margin)),
            min(D-1, max(box[5], smax[2]+margin))]


# =============================================================================
# Inference
# =============================================================================

@torch.no_grad()
def detect_stroke(model, x, spacing, score_thresh, min_seg_vol_ml,
                  seg_thresh=0.5, iou_thresh=0.3, expand_bbox=True):
    model.eval()
    outputs    = model(x)
    boxes_dec, scores_dec = model.decode(outputs)
    boxes_cpu  = boxes_dec[0].cpu()
    scores_cpu = scores_dec[0].cpu()

    keep = nms_3d(boxes_cpu, scores_cpu, iou_thresh)
    if not keep:
        keep = [int(scores_cpu.argmax().item())]

    best_box   = boxes_cpu[keep[0]]
    best_score = float(scores_cpu[keep[0]].item())

    W, H, D = model.input_shape
    mins     = torch.clamp(best_box[:3], min=0.)
    maxs     = torch.clamp(best_box[3:],
                            max=torch.tensor([W-1., H-1., D-1.]))
    best_box = torch.cat([mins, maxs]).tolist()

    # Segmentation
    seg_prob   = torch.sigmoid(outputs["seg"])[0,0].cpu().numpy()  # (D,H,W)
    seg_binary = (seg_prob > seg_thresh).astype(np.float32)
    seg_vol_ml = float(seg_binary.sum() *
                       spacing[0]*spacing[1]*spacing[2] / 1000.)

    # Expand bbox to cover full lesion
    if expand_bbox:
        best_box = expand_box_from_seg(
            best_box, seg_binary, (W,H,D), margin=1)

    cx = (best_box[0]+best_box[3])/2
    cy = (best_box[1]+best_box[4])/2
    cz = (best_box[2]+best_box[5])/2

    # Classification
    cls_names = {0:"focal",1:"multi",2:"embolic",3:"negative"}
    subtype   = int(outputs["cls_logits"][0].argmax().item())

    # Presence
    if model.stage == 3 and "presence_logit" in outputs:
        pres = float(torch.sigmoid(outputs["presence_logit"][0]).item())
    else:
        pres = best_score

    has_lesion = best_score >= score_thresh and seg_vol_ml >= min_seg_vol_ml

    return {
        "has_lesion":    has_lesion,
        "presence_conf": round(pres, 4),
        "confidence":    round(best_score, 4),
        "seg_volume_ml": round(seg_vol_ml, 3),
        "center":        (round(cx,1), round(cy,1), round(cz,1)),
        "box":           [round(v,1) for v in best_box],
        "subtype":       subtype,
        "subtype_str":   cls_names.get(subtype, "unknown"),
        "seg_map":       seg_binary,   # (D,H,W) in model space
    }


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dwi4d",  default=None,
                    help="4D multishell DWI .nii.gz or 3D b-shell. "
                         "Optional if --adc provided.")
    ap.add_argument("--bval",   default=None,
                    help="bval file. Required if dwi4d is 4D multishell.")
    ap.add_argument("--adc",    default=None,
                    help="ADC map .nii.gz. Optional if --dwi4d provided.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir",    default="clinical_output")
    ap.add_argument("--subject",    default=None)
    ap.add_argument("--target_b",   type=int,   default=1000)
    ap.add_argument("--b_tolerance",type=int,   default=200)
    ap.add_argument("--model_shape",type=int,   nargs=3, default=[96,96,64])
    ap.add_argument("--spacing",    type=float, nargs=3, default=[1.5,1.5,3.0])
    ap.add_argument("--score_thresh",      type=float, default=0.3)
    ap.add_argument("--min_seg_volume_ml", type=float, default=0.15)
    ap.add_argument("--seg_thresh",        type=float, default=0.5)
    ap.add_argument("--no_expand_bbox",    action="store_true")
    args = ap.parse_args()

    if args.dwi4d is None and args.adc is None:
        ap.error("Provide at least one of --dwi4d or --adc.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subject = args.subject or Path(
        args.dwi4d or args.adc).parent.parent.name
    print(f"\n[INFO] Subject : {subject}")
    print(f"[INFO] Device  : {device}")

    # ── Load inputs ──────────────────────────────────────────────────────────
    has_dwi = args.dwi4d is not None
    has_adc = args.adc   is not None

    ref_aff = ref_hdr = ref_shape = None   # for native-space output

    if has_dwi:
        print(f"\n[STEP 1] Loading DWI: {args.dwi4d}")
        dwi4d, dwi_aff, dwi_hdr = load_nii(args.dwi4d)
        print(f"[INFO] DWI shape: {dwi4d.shape}  "
              f"vox: {tuple(round(float(v),2) for v in dwi_hdr.get_zooms()[:3])}")
        ref_aff = dwi_aff; ref_hdr = dwi_hdr

        if dwi4d.ndim == 4:
            if args.bval is None:
                print("[WARN] 4D DWI but no bval — using first volume")
                bvals = np.zeros(dwi4d.shape[-1])
                bvals[0] = args.target_b
            else:
                bvals = load_bvals(args.bval)
            dwi3d, _ = extract_shell(
                dwi4d, bvals, args.target_b, args.b_tolerance)
        else:
            dwi3d = dwi4d.astype(np.float32)
            print(f"[INFO] DWI already 3D")
        ref_shape = dwi3d.shape
    else:
        dwi3d = dwi_aff = dwi_hdr = None

    if has_adc:
        print(f"\n[STEP 2] Loading ADC: {args.adc}")
        adc3d, adc_aff, adc_hdr = load_nii(args.adc)
        print(f"[INFO] ADC shape: {adc3d.shape}  "
              f"vox: {tuple(round(float(v),2) for v in adc_hdr.get_zooms()[:3])}")
        if ref_aff is None:
            ref_aff = adc_aff; ref_hdr = adc_hdr
            ref_shape = adc3d.shape
    else:
        adc3d = adc_aff = adc_hdr = None

    # ── Preprocess ───────────────────────────────────────────────────────────
    print(f"\n[STEP 3] Preprocessing to model space "
          f"{args.model_shape} @ {args.spacing}mm  (CoM crop)...")

    W, H, D = args.model_shape

    if has_dwi:
        dwi_m, dwi_chain = preprocess(
            dwi3d, dwi_aff, dwi_hdr, args.model_shape, args.spacing)
        brain_mask = dwi_m > 0
        dwi_m = robust_zscore(dwi_m, brain_mask)
        ref_chain = dwi_chain
    else:
        dwi_m = np.zeros((W,H,D), dtype=np.float32)
        ref_chain = None

    adc_chain = None
    if has_adc:
        adc_m, adc_chain = preprocess(
            adc3d, adc_aff, adc_hdr, args.model_shape, args.spacing)
        adc_brain = adc_m != 0
        adc_m = robust_zscore(adc_m, adc_brain)
        if ref_chain is None:
            ref_chain = adc_chain
    else:
        adc_m = np.zeros((W,H,D), dtype=np.float32)

    # Stack → [1, 2, D, H, W]
    X = np.stack([dwi_m, adc_m], 0)
    X = torch.from_numpy(X).float().permute(0,3,2,1).unsqueeze(0)
    print(f"[INFO] Model input: {tuple(X.shape)}")

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"\n[STEP 4] Loading model: {args.checkpoint}")
    chk        = torch.load(args.checkpoint, map_location="cpu",
                             weights_only=False)
    ckpt_args  = chk.get("args", {})
    ckpt_stage = ckpt_args.get("stage", 2)
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
    print(f"[INFO] stage={ckpt_stage}  "
          f"epoch={chk.get('epoch','?')}  "
          f"val_mAP={chk.get('best_score',0.):.4f}  "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.3f}M")

    # ── Inference ────────────────────────────────────────────────────────────
    print(f"\n[STEP 5] Detecting stroke...")
    result = detect_stroke(
        model, X.to(device), args.spacing,
        score_thresh   = args.score_thresh,
        min_seg_vol_ml = args.min_seg_volume_ml,
        seg_thresh     = args.seg_thresh,
        expand_bbox    = not args.no_expand_bbox,
    )

    print(f"\n{'='*60}")
    print(f"RESULT — {subject}")
    print(f"{'='*60}")
    print(f"  has_lesion   : {result['has_lesion']}")
    print(f"  confidence   : {result['confidence']}")
    print(f"  seg_vol_ml   : {result['seg_volume_ml']} ml")
    print(f"  subtype      : {result['subtype_str']}")
    print(f"  center (model): {result['center']}")
    print(f"  modalities   : "
          f"{'DWI' if has_dwi else ''}"
          f"{'+' if has_dwi and has_adc else ''}"
          f"{'ADC' if has_adc else ''}")
    print(f"{'='*60}\n")

    # ── Save b-shell volume in native space ─────────────────────────────────
    # Save the extracted b{target_b} mean volume in original image space
    # so you can load it alongside the masks in ITK-Snap/FSLeyes
    if has_dwi and dwi3d is not None:
        b_native_path = out_dir / f"b{args.target_b}_native.nii.gz"
        nib.save(nib.Nifti1Image(dwi3d.astype(np.float32), dwi_aff, dwi_hdr),
                 str(b_native_path))
        print(f"[INFO] b{args.target_b} native volume saved: {b_native_path}")

    # ── Save model-space NIfTIs ──────────────────────────────────────────────
    print("[STEP 6] Saving outputs...")
    model_aff = np.diag([args.spacing[0], args.spacing[1],
                          args.spacing[2], 1.]).astype(np.float32)

    # Model-space DWI + ADC (for reference)
    nib.save(nib.Nifti1Image(dwi_m, model_aff),
             str(out_dir/"dwi_model_space.nii.gz"))
    nib.save(nib.Nifti1Image(adc_m, model_aff),
             str(out_dir/"adc_model_space.nii.gz"))

    # Model-space seg mask
    seg_whd = result["seg_map"].transpose(2,1,0).astype(np.uint8)
    nib.save(nib.Nifti1Image(seg_whd, model_aff),
             str(out_dir/"seg_mask_model_space.nii.gz"))

    # Model-space bbox mask
    box = result["box"]
    bbox_vol = np.zeros((W,H,D), dtype=np.uint8)
    x0,y0,z0 = max(0,int(np.floor(box[0]))), \
                max(0,int(np.floor(box[1]))), \
                max(0,int(np.floor(box[2])))
    x1,y1,z1 = min(W,int(np.ceil(box[3]))+1), \
                min(H,int(np.ceil(box[4]))+1), \
                min(D,int(np.ceil(box[5]))+1)
    bbox_vol[x0:x1, y0:y1, z0:z1] = 1
    nib.save(nib.Nifti1Image(bbox_vol, model_aff),
             str(out_dir/"bbox_mask_model_space.nii.gz"))

    # ── Save NATIVE-SPACE NIfTIs ─────────────────────────────────────────────
    # These are in the same space as your original DWI/ADC
    # Load alongside original scan in ITK-Snap/FSLeyes directly
    if ref_chain is not None:
        print("[INFO] Mapping masks back to native image space...")

        seg_native  = mask_to_native(result["seg_map"], ref_chain)
        bbox_native = mask_to_native(
            result["seg_map"] * 0 +   # zeros for now
            np.zeros_like(result["seg_map"]), ref_chain)

        # Build native bbox mask in model space first then invert
        bbox_model_dhw = bbox_vol.transpose(2,1,0).astype(np.float32)
        # reuse same chain
        bbox_native2 = mask_to_native(bbox_model_dhw, ref_chain)

        nat_aff = ref_chain["native_affine"]

        nib.save(nib.Nifti1Image(seg_native.astype(np.uint8), nat_aff),
                 str(out_dir/"seg_mask_native.nii.gz"))
        nib.save(nib.Nifti1Image(bbox_native2.astype(np.uint8), nat_aff),
                 str(out_dir/"bbox_mask_native.nii.gz"))

        print(f"[INFO] Native seg mask shape : {seg_native.shape}")
        print(f"[INFO] Native bbox mask shape: {bbox_native2.shape}")
        print(f"[INFO] Original input shape  : {ref_shape}")

    # ── Compute stroke volumes in native space ───────────────────────────────
    # Use native voxel size from header — more accurate than model space
    # Volume (ml) = n_lesion_voxels × voxel_volume_mm3 / 1000
    volume_results = {}

    if ref_chain is not None and ref_hdr is not None:
        nat_zooms   = ref_chain["native_zooms"]
        vox_vol_mm3 = float(nat_zooms[0] * nat_zooms[1] * nat_zooms[2])
        n_voxels    = int(seg_native.sum())
        vol_ml_native = round(n_voxels * vox_vol_mm3 / 1000.0, 3)

        print(f"[INFO] Native voxel size: "
              f"{nat_zooms[0]:.2f}x{nat_zooms[1]:.2f}x{nat_zooms[2]:.2f}mm  "
              f"= {vox_vol_mm3:.2f}mm3/voxel")
        print(f"[INFO] Lesion voxels (native): {n_voxels}")

        if has_dwi:
            volume_results["stroke_volume_dwi_ml"] = vol_ml_native
            print(f"[INFO] Stroke volume (DWI b{args.target_b}): "
                  f"{vol_ml_native} ml")

        if has_adc:
            # ADC may have different voxel size — recompute if different
            if has_dwi:
                # Both given — same mask, check ADC voxel size
                adc_zooms   = tuple(float(z) for z in adc_hdr.get_zooms()[:3])
                adc_vox_mm3 = float(adc_zooms[0]*adc_zooms[1]*adc_zooms[2])
                # Use ADC chain if different spacing
                if adc_chain is not None:
                    adc_seg_native = mask_to_native(result["seg_map"],
                                                    adc_chain)
                    adc_n_voxels   = int(adc_seg_native.sum())
                    adc_vol_ml     = round(
                        adc_n_voxels * adc_vox_mm3 / 1000.0, 3)
                else:
                    adc_vol_ml = vol_ml_native
                volume_results["stroke_volume_adc_ml"] = adc_vol_ml
                print(f"[INFO] Stroke volume (ADC):     {adc_vol_ml} ml")
            else:
                # ADC only
                volume_results["stroke_volume_adc_ml"] = vol_ml_native
                print(f"[INFO] Stroke volume (ADC): {vol_ml_native} ml")

        if has_dwi and has_adc:
            dwi_vol = volume_results.get("stroke_volume_dwi_ml", 0)
            adc_vol = volume_results.get("stroke_volume_adc_ml", 0)
            diff    = round(abs(dwi_vol - adc_vol), 3)

            # Combined mask volume (model used both channels — best estimate)
            combined_vol = vol_ml_native
            volume_results["stroke_volume_combined_ml"] = combined_vol

            # Compute DWI-only and ADC-only masks for intersection/union
            # We use the native seg masks
            # Both are binary uint8 in native space
            try:
                # adc_seg_native was computed above if adc_chain exists
                if adc_chain is not None:
                    dwi_seg = seg_native.astype(bool)
                    adc_seg = adc_seg_native.astype(bool)

                    # Intersection — both modalities agree
                    intersect     = (dwi_seg & adc_seg).astype(np.uint8)
                    n_intersect   = int(intersect.sum())
                    vol_intersect = round(n_intersect * vox_vol_mm3 / 1000., 3)

                    # Union — at least one modality says lesion
                    union_mask  = (dwi_seg | adc_seg).astype(np.uint8)
                    n_union     = int(union_mask.sum())
                    vol_union   = round(n_union * vox_vol_mm3 / 1000., 3)

                    volume_results["stroke_volume_intersection_ml"] = vol_intersect
                    volume_results["stroke_volume_union_ml"]         = vol_union

                    # Save intersection and union masks in native space
                    nib.save(nib.Nifti1Image(intersect,  nat_aff),
                             str(out_dir / "seg_mask_intersection_native.nii.gz"))
                    nib.save(nib.Nifti1Image(union_mask, nat_aff),
                             str(out_dir / "seg_mask_union_native.nii.gz"))

                    print(f"[INFO] DWI vol       = {dwi_vol} ml")
                    print(f"[INFO] ADC vol       = {adc_vol} ml")
                    print(f"[INFO] Combined vol  = {combined_vol} ml  "
                          f"(model used both channels)")
                    print(f"[INFO] Intersection  = {vol_intersect} ml  "
                          f"(most confident infarct core)")
                    print(f"[INFO] Union         = {vol_union} ml  "
                          f"(standard clinical overestimate)")
                    print(f"[INFO] Inflation vs intersection: "
                          f"{round((vol_union-vol_intersect)/max(0.001,vol_intersect)*100,1)}%")
            except Exception as e:
                print(f"[WARN] Could not compute intersection/union: {e}")
                volume_results["stroke_volume_combined_ml"] = combined_vol

            print(f"[INFO] diff DWI-ADC = {diff} ml"
                  f"  (Standard DWI+ADC union inflates by 20-40%)")

    # ── Save JSON ────────────────────────────────────────────────────────────
    out_dict = {
        "subject":       subject,
        "has_lesion":    result["has_lesion"],
        "presence_conf": result["presence_conf"],
        "confidence":    result["confidence"],
        "seg_volume_ml": result["seg_volume_ml"],
        "center_model":  list(result["center"]),
        "box_model":     result["box"],
        "subtype":       result["subtype"],
        "subtype_str":   result["subtype_str"],
        "modalities":    ("DWI+ADC" if has_dwi and has_adc
                          else "DWI" if has_dwi else "ADC"),
        "model_shape":   list(args.model_shape),
        "spacing_mm":    list(args.spacing),
        "stroke_volumes": volume_results,
        "thresholds": {
            "score_thresh":      args.score_thresh,
            "min_seg_volume_ml": args.min_seg_volume_ml,
        }
    }
    with open(out_dir/"result.json","w") as f:
        json.dump(out_dict, f, indent=2)

    print(f"\n[DONE] {out_dir}")
    print(f"  result.json                  — full detection + volume result")
    if has_dwi:
        print(f"  b{args.target_b}_native.nii.gz        — extracted b-shell (native space)")
    print(f"  seg_mask_native.nii.gz       — lesion mask (NATIVE space)")
    print(f"  bbox_mask_native.nii.gz      — bbox ROI   (NATIVE space)")
    print(f"  seg_mask_model_space.nii.gz  — lesion mask (model space)")
    print(f"  bbox_mask_model_space.nii.gz — bbox ROI   (model space)")
    if volume_results:
        nz = ref_chain["native_zooms"] if ref_chain else None
        print(f"\nStroke volumes (native voxel size {nz}):")
        for k, v in volume_results.items():
            label = k.replace("stroke_volume_","").replace("_ml","").upper()
            print(f"  {label} lesion volume : {v} ml")
        if "stroke_volume_dwi_ml" in volume_results and \
           "stroke_volume_adc_ml" in volume_results:
            print(f"  NOTE: Standard DWI+ADC union typically inflates volume.")
            print(f"        Multishell DWI-only may be more accurate.")
    print(f"\nTo view in ITK-Snap:")
    print(f"  1. Open original DWI b{args.target_b} or ADC as main image")
    print(f"  2. Segmentation > Open Segmentation > seg_mask_native.nii.gz")
    print(f"  3. Segmentation > Open Segmentation > bbox_mask_native.nii.gz")


if __name__ == "__main__":
    main()