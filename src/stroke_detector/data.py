#!/usr/bin/env python3
"""
stroke_dataset.py
=================
Dataset for segmentation-supervised stroke detection.

Key changes from yolo3d_onfly_dataset.py
-----------------------------------------
1. Loads ground-truth mask from mask_path column in CSV.
   Applies the same reorientation + resampling + pad/crop pipeline
   as DWI/ADC so mask is pixel-aligned with the input volume.

2. Augmentation is applied consistently to image AND mask:
   - L-R flip       → mask flipped axis 0
   - S-I flip       → mask flipped axis 2
   - Axial rotation → mask rotated same angle
   - Intensity jitter, Gaussian noise, gamma → image only

3. Returns mask tensor [1, D, H, W] alongside image and box.

4. For negative/embolic cases: mask is all-zeros [1,D,H,W].

5. mask_path lookup: uses CSV column directly (full absolute path).
   Falls back to looking for *_msk.nii.gz alongside DWI file.

Return format
-------------
    image      : float32 [2, D, H, W]  z-scored DWI + ADC
    mask       : float32 [1, D, H, W]  binary lesion mask (0/1)
    box_single : float32 [6]           normalised YOLO box or zeros
    train_role : str
    has_box    : bool
    case_id    : str
    meta       : dict
"""

import os
import glob
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Any

import numpy as np
import nibabel as nib
from nibabel.orientations import (axcodes2ornt, io_orientation,
                                  inv_ornt_aff, apply_orientation)
from scipy.ndimage import zoom, rotate
import pandas as pd
import torch
from torch.utils.data import Dataset


# =============================================================================
# Utilities  (unchanged)
# =============================================================================

def one(pattern):
    h = sorted(glob.glob(pattern))
    return h[0] if h else None


def load_nii(path):
    img  = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return img, data, img.affine, img.header


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def get_train_role(row: dict) -> str:
    role = str(row.get("train_role", "")).strip().lower()
    if role in ("positive_single", "positive_multi", "embolic", "negative"):
        return role
    qc = str(row.get("qc_status", row.get("status", ""))).strip().lower()
    return "positive_single" if qc == "pass" else "negative"


# =============================================================================
# TransformChain  (unchanged)
# =============================================================================

@dataclass
class TransformChain:
    ornt_from:          Any                       = None
    ornt_to:            Tuple[str,str,str]        = ('R','A','S')
    ornt_transf:        Optional[np.ndarray]      = None
    native_zooms:       Any                       = None
    canonical_zooms:    Tuple[float,float,float]  = (1.5, 1.5, 3.0)
    spacing_scale:      Any                       = None
    pre_crop_start:     Tuple[int,int,int]        = (0, 0, 0)
    pre_crop_shape:     Any                       = None
    padded_shape:       Any                       = None
    final_resize_scale: Tuple[float,float,float]  = (1.0, 1.0, 1.0)
    model_shape:        Tuple[int,int,int]        = (96, 96, 64)

    def forward_bbox_native_to_model(self, bbox_native, shape_native):
        xmn,xmx,ymn,ymx,zmn,zmx = bbox_native
        corners = np.array([
            [xmn,ymn,zmn],[xmx,ymn,zmn],[xmn,ymx,zmn],[xmn,ymn,zmx],
            [xmx,ymx,zmn],[xmx,ymn,zmx],[xmn,ymx,zmx],[xmx,ymx,zmx],
        ], dtype=np.float32)
        if self.ornt_transf is not None:
            axes  = self.ornt_transf[:,0].astype(int)
            flips = self.ornt_transf[:,1].astype(float)
            nat   = np.array(shape_native, dtype=np.float32)
            out   = np.zeros_like(corners)
            for da in range(3):
                sa = int(axes[da])
                out[:,da] = corners[:,sa] if flips[da]==1. \
                            else (nat[sa]-1.)-corners[:,sa]
            corners = out
        corners *= np.array(self.spacing_scale,      dtype=np.float32)
        corners -= np.array(self.pre_crop_start,     dtype=np.float32)
        corners *= np.array(self.final_resize_scale, dtype=np.float32)
        return corners.min(axis=0), corners.max(axis=0)

    def normalize_model_box(self, model_min, model_max):
        W,H,D = self.model_shape
        lo = np.maximum(model_min, 0.)
        hi = np.minimum(model_max, np.array([W,H,D], dtype=np.float32))
        cx=(lo[0]+hi[0])/2/W; cy=(lo[1]+hi[1])/2/H; cz=(lo[2]+hi[2])/2/D
        sx=(hi[0]-lo[0])/W;   sy=(hi[1]-lo[1])/H;   sz=(hi[2]-lo[2])/D
        return np.array([cx,cy,cz,sx,sy,sz], dtype=np.float32)


# =============================================================================
# Image processing helpers  (unchanged)
# =============================================================================

def reorient_to_ras(arr, aff):
    ot = nib.orientations.ornt_transform(
             io_orientation(aff), axcodes2ornt(('R','A','S')))
    ax = nib.aff2axcodes(aff)
    return apply_orientation(arr, ot), aff @ inv_ornt_aff(ot, arr.shape), ot, ax


def resample_to_spacing(arr, cur_z, tgt_z, order=1):
    f = np.array(cur_z, dtype=np.float32) / np.array(tgt_z, dtype=np.float32)
    return (zoom(arr, zoom=f, order=order, mode='nearest', prefilter=False),
            tuple(float(v) for v in f))


def pad_crop_to_shape(arr, tgt):
    ins = np.array(arr.shape, dtype=int)
    tgt = np.array(tgt, dtype=int)
    st  = np.maximum((ins-tgt)//2, 0)
    en  = np.minimum(st+tgt, ins)
    cr  = arr[st[0]:en[0], st[1]:en[1], st[2]:en[2]]
    out = np.zeros(tuple(tgt), dtype=arr.dtype)
    cs  = np.array(cr.shape, dtype=int)
    ps  = ((tgt-cs)//2).astype(int)
    out[ps[0]:ps[0]+cs[0], ps[1]:ps[1]+cs[1], ps[2]:ps[2]+cs[2]] = cr
    return out, tuple(int(v) for v in st), tuple(out.shape)


def maybe_resize_to_exact(arr, exact):
    cur = np.array(arr.shape, dtype=np.float32)
    tgt = np.array(exact, dtype=np.float32)
    if np.all(cur == tgt):
        return arr, (1.,1.,1.)
    f   = tgt / cur
    out = zoom(arr, zoom=f, order=1, mode='nearest', prefilter=False)
    return out, tuple(float(v) for v in f)


def find_modalities(root, subject):
    ses = os.path.join(root, subject, "ses-0001", "dwi")
    return (one(os.path.join(ses, "*dwi*.nii*")),
            one(os.path.join(ses, "*adc*.nii*")))


def find_mask(root, subject, dwi_path=None):
    """Locate mask .nii.gz for a subject."""
    # Standard ISLES derivatives location
    deriv = os.path.join(root, "derivatives",
                         subject, "ses-0001", "dwi")
    m = one(os.path.join(deriv, "*msk*.nii*"))
    if m:
        return m
    # Alongside DWI
    if dwi_path:
        d = os.path.dirname(dwi_path)
        m = one(os.path.join(d, "*msk*.nii*"))
        if m:
            return m
    return None


def robust_zscore(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    eps  = 1e-6
    vals = arr[mask]
    if vals.size == 0:
        return arr.astype(np.float32)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals-med))*1.4826+eps)
    out = (arr-med)/mad
    out[~mask] = 0.0
    return out.astype(np.float32)


def load_bbox_json(json_path):
    if not json_path or not os.path.exists(str(json_path)):
        return None
    with open(json_path) as f:
        return json.load(f)


def bbox_vox_to_tuple(bv):
    return (int(bv["xmin"]),int(bv["xmax"]),
            int(bv["ymin"]),int(bv["ymax"]),
            int(bv["zmin"]),int(bv["zmax"]))


# =============================================================================
# Augmentation  (mask-consistent versions)
# =============================================================================

def aug_flip_lr(dwi, adc, mask, boxes, p=0.5):
    if np.random.random() >= p:
        return dwi, adc, mask, boxes
    dwi  = np.flip(dwi,  axis=0).copy()
    adc  = np.flip(adc,  axis=0).copy()
    mask = np.flip(mask, axis=0).copy()
    boxes = [np.array([1.-b[0],b[1],b[2],b[3],b[4],b[5]]) for b in boxes]
    return dwi, adc, mask, boxes


def aug_flip_si(dwi, adc, mask, boxes, p=0.5):
    if np.random.random() >= p:
        return dwi, adc, mask, boxes
    dwi  = np.flip(dwi,  axis=2).copy()
    adc  = np.flip(adc,  axis=2).copy()
    mask = np.flip(mask, axis=2).copy()
    boxes = [np.array([b[0],b[1],1.-b[2],b[3],b[4],b[5]]) for b in boxes]
    return dwi, adc, mask, boxes


def aug_rotate_axial(dwi, adc, mask, boxes, model_shape, max_angle=15.0):
    angle = np.random.uniform(-max_angle, max_angle)
    if abs(angle) < 0.5:
        return dwi, adc, mask, boxes

    kw = dict(axes=(0,1), reshape=False, order=1, mode='nearest',
              prefilter=False)
    dwi  = rotate(dwi,  angle, **kw)
    adc  = rotate(adc,  angle, **kw)
    # Nearest-neighbour for binary mask to keep 0/1 values
    mask = rotate(mask, angle, axes=(0,1), reshape=False,
                  order=0, mode='nearest', prefilter=False)

    W, H, D   = model_shape
    cx_i, cy_i = W/2., H/2.
    rad        = np.deg2rad(-angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    rotated = []
    for box in boxes:
        cx_n,cy_n,cz_n,sx_n,sy_n,sz_n = box
        cx_v,cy_v = cx_n*W, cy_n*H
        sx_v,sy_v = sx_n*W, sy_n*H
        xmn,xmx = cx_v-sx_v/2, cx_v+sx_v/2
        ymn,ymx = cy_v-sy_v/2, cy_v+sy_v/2
        corners = np.array([[xmn,ymn],[xmx,ymn],[xmn,ymx],[xmx,ymx]])
        corners -= [cx_i, cy_i]
        corners  = (np.array([[cos_a,-sin_a],[sin_a,cos_a]]) @ corners.T).T
        corners += [cx_i, cy_i]
        new_xmn = max(0., corners[:,0].min()); new_xmx = min(W, corners[:,0].max())
        new_ymn = max(0., corners[:,1].min()); new_ymx = min(H, corners[:,1].max())
        rotated.append(np.array([
            (new_xmn+new_xmx)/2/W, (new_ymn+new_ymx)/2/H, cz_n,
            (new_xmx-new_xmn)/W,   (new_ymx-new_ymn)/H,   sz_n
        ], dtype=np.float32))
    return dwi, adc, mask, rotated


def aug_intensity(dwi, adc, brain_mask):
    for arr in [dwi, adc]:
        if not brain_mask.any():
            continue
        scale = np.random.uniform(0.85, 1.15)
        shift = np.random.uniform(-0.1, 0.1)*float(arr[brain_mask].std()+1e-6)
        arr  *= scale
        arr  += shift
        arr[~brain_mask] = 0.0
    return dwi, adc


def aug_gaussian_noise(dwi, adc, brain_mask, p=0.5,
                        noise_std_range=(0.0, 0.15)):
    if np.random.random() >= p or not brain_mask.any():
        return dwi, adc
    std = np.random.uniform(*noise_std_range)
    bm  = brain_mask.astype(np.float32)
    for arr in [dwi, adc]:
        arr += np.random.normal(0., std, arr.shape).astype(np.float32) * bm
    return dwi, adc


def aug_gamma(dwi, adc, brain_mask, p=0.5, gamma_range=(0.75, 1.25)):
    if np.random.random() >= p or not brain_mask.any():
        return dwi, adc
    for arr in [dwi, adc]:
        gamma = np.random.uniform(*gamma_range)
        vals  = arr[brain_mask]
        vmin, vmax = float(vals.min()), float(vals.max())
        vrange = vmax - vmin
        if vrange < 1e-6:
            continue
        normed = np.clip((arr-vmin)/vrange, 0., 1.)
        normed[brain_mask] = np.power(normed[brain_mask], gamma)
        arr[brain_mask]    = normed[brain_mask]*vrange + vmin
    return dwi, adc


# =============================================================================
# Main Dataset
# =============================================================================

class StrokeDataset(Dataset):
    """
    3D stroke dataset with mask loading and mask-consistent augmentation.

    Returns dict:
        image      : float32 tensor [2, D, H, W]   DWI + ADC
        mask       : float32 tensor [1, D, H, W]   binary lesion mask
        box_single : float32 tensor [6]             normalised YOLO box
        train_role : str
        has_box    : bool
        case_id    : str
        meta       : dict
    """

    def __init__(self,
                 root: str,
                 split_csv: str,
                 split: str,
                 model_shape: tuple  = (96, 96, 64),
                 canonical_zooms: tuple = (1.5, 1.5, 3.0),
                 augment: bool       = False,
                 debug_print: bool   = False):
        self.root           = root
        self.model_shape    = tuple(model_shape)
        self.canonical_zooms = tuple(canonical_zooms)
        self.augment        = augment
        self.debug_print    = debug_print

        df = pd.read_csv(split_csv)
        if "subject" not in df.columns or "split" not in df.columns:
            raise ValueError("CSV needs 'subject' and 'split' columns.")
        df = df[df["split"].str.lower() == split.lower()]
        if df.empty:
            raise RuntimeError(f"No rows for split={split}")
        self.rows = df.drop_duplicates("subject").reset_index(drop=True)

        role_counts = (self.rows["train_role"].value_counts().to_dict()
                       if "train_role" in self.rows.columns else {})
        print(f"[StrokeDataset] split={split}  total={len(self.rows)}")
        print(f"  augment={augment}")
        for role, count in sorted(role_counts.items()):
            print(f"  {role:<20}: {count}")

    def __len__(self):
        return len(self.rows)

    def _load_and_preprocess_volume(self, path, chain, order=1):
        """Load a NIfTI volume and apply the same spatial transform as DWI."""
        _, arr, aff, _ = load_nii(path)
        arr_r, _, _, _ = reorient_to_ras(arr, aff)
        arr_rs, _      = resample_to_spacing(
            arr_r, chain.native_zooms, chain.canonical_zooms, order=order)
        arr_pc, _, _   = pad_crop_to_shape(arr_rs, self.model_shape)
        arr_m, _       = maybe_resize_to_exact(arr_pc, self.model_shape)
        return arr_m

    def __getitem__(self, idx):
        row    = self.rows.iloc[idx]
        row_d  = row.to_dict()
        subject = str(row["subject"])
        train_role = get_train_role(row_d)

        # ── Load DWI + ADC ────────────────────────────────────────────────
        dwi_path = str(row_d.get("dwi_path","")) if "dwi_path" in row_d else ""
        adc_path = str(row_d.get("adc_path","")) if "adc_path" in row_d else ""
        if not dwi_path or not os.path.exists(dwi_path):
            dwi_path, adc_path = find_modalities(self.root, subject)
        if not adc_path or not os.path.exists(str(adc_path)):
            _, adc_path = find_modalities(self.root, subject)
        if not dwi_path or not adc_path:
            raise FileNotFoundError(f"{subject}: DWI/ADC not found.")

        _, dwi, dwi_aff, dwi_hdr = load_nii(dwi_path)
        _, adc, adc_aff, _       = load_nii(adc_path)

        # ── Build transform chain ─────────────────────────────────────────
        chain = TransformChain()
        chain.model_shape     = self.model_shape
        chain.canonical_zooms = self.canonical_zooms
        chain.native_zooms    = tuple(float(z) for z in dwi_hdr.get_zooms()[:3])

        dwi_r, _, ot, axb = reorient_to_ras(dwi, dwi_aff)
        adc_r, _, _,  _   = reorient_to_ras(adc, adc_aff)
        chain.ornt_from    = axb
        chain.ornt_to      = ('R','A','S')
        chain.ornt_transf  = ot

        dwi_rs, sf = resample_to_spacing(dwi_r, chain.native_zooms,
                                          chain.canonical_zooms)
        adc_rs, _  = resample_to_spacing(adc_r, chain.native_zooms,
                                          chain.canonical_zooms)
        chain.spacing_scale  = sf
        chain.pre_crop_shape = tuple(int(s) for s in dwi_rs.shape)

        dwi_pc, cs, ap = pad_crop_to_shape(dwi_rs, self.model_shape)
        adc_pc, _,  _  = pad_crop_to_shape(adc_rs, self.model_shape)
        chain.pre_crop_start = cs
        chain.padded_shape   = ap

        dwi_m, frs = maybe_resize_to_exact(dwi_pc, self.model_shape)
        adc_m, _   = maybe_resize_to_exact(adc_pc, self.model_shape)
        chain.final_resize_scale = frs

        # ── Z-score ───────────────────────────────────────────────────────
        brain_mask = dwi_m > 0
        dwi_m = robust_zscore(dwi_m, brain_mask)
        adc_m = robust_zscore(adc_m, brain_mask)

        # ── Load mask ─────────────────────────────────────────────────────
        mask_path = str(row_d.get("mask_path", ""))
        mask_m    = np.zeros(self.model_shape, dtype=np.float32)
        has_mask  = False

        if mask_path and os.path.exists(mask_path):
            try:
                _, mask_raw, mask_aff, _ = load_nii(mask_path)
                # Apply same spatial transforms as DWI
                mask_raw = (mask_raw > 0).astype(np.float32)
                mask_r, _, _, _ = reorient_to_ras(mask_raw, mask_aff)
                mask_rs, _ = resample_to_spacing(
                    mask_r, chain.native_zooms, chain.canonical_zooms,
                    order=0)  # nearest-neighbour for binary mask
                mask_pc, _, _ = pad_crop_to_shape(mask_rs, self.model_shape)
                mask_m2, _    = maybe_resize_to_exact(mask_pc, self.model_shape)
                mask_m        = (mask_m2 > 0.5).astype(np.float32)
                has_mask      = mask_m.sum() > 0
            except Exception as e:
                if self.debug_print:
                    print(f"[WARN] {subject}: mask load failed — {e}")
        else:
            # Try fallback discovery
            fb = find_mask(self.root, subject, dwi_path)
            if fb:
                try:
                    _, mask_raw, mask_aff, _ = load_nii(fb)
                    mask_raw = (mask_raw > 0).astype(np.float32)
                    mask_r, _, _, _ = reorient_to_ras(mask_raw, mask_aff)
                    mask_rs, _ = resample_to_spacing(
                        mask_r, chain.native_zooms, chain.canonical_zooms,
                        order=0)
                    mask_pc, _, _ = pad_crop_to_shape(mask_rs, self.model_shape)
                    mask_m2, _    = maybe_resize_to_exact(mask_pc, self.model_shape)
                    mask_m        = (mask_m2 > 0.5).astype(np.float32)
                    has_mask      = mask_m.sum() > 0
                except Exception as e:
                    if self.debug_print:
                        print(f"[WARN] {subject}: fallback mask failed — {e}")

        # ── Load bounding box ─────────────────────────────────────────────
        boxes_norm = []
        if train_role in ("positive_single", "positive_multi"):
            json_path = str(row_d.get("bbox_json", ""))
            bi        = load_bbox_json(json_path)
            if bi and bi.get("status") == "ok":
                shape_native = tuple(int(s) for s in bi["shape"])
                for bv_key in (["bboxes_vox"] if "bboxes_vox" in bi
                               and bi["bboxes_vox"] else []):
                    for bv in bi[bv_key]:
                        mm, mx = chain.forward_bbox_native_to_model(
                            bbox_vox_to_tuple(bv), shape_native)
                        yolo = chain.normalize_model_box(mm, mx)
                        if min(yolo[3], yolo[4], yolo[5]) > 0.01:
                            boxes_norm.append(yolo)
                if not boxes_norm and bi.get("bbox_vox"):
                    mm, mx = chain.forward_bbox_native_to_model(
                        bbox_vox_to_tuple(bi["bbox_vox"]),
                        tuple(int(s) for s in bi["shape"]))
                    yolo = chain.normalize_model_box(mm, mx)
                    if min(yolo[3], yolo[4], yolo[5]) > 0.01:
                        boxes_norm.append(yolo)

            # Fall back to deriving box from mask if JSON failed
            if not boxes_norm and has_mask:
                nz = np.argwhere(mask_m > 0)
                if len(nz) > 0:
                    W, H, D = self.model_shape
                    mn, mx_ = nz.min(0), nz.max(0)
                    cx=(mn[0]+mx_[0])/2/W; cy=(mn[1]+mx_[1])/2/H
                    cz=(mn[2]+mx_[2])/2/D
                    sx=(mx_[0]-mn[0]+1)/W; sy=(mx_[1]-mn[1]+1)/H
                    sz=(mx_[2]-mn[2]+1)/D
                    boxes_norm.append(np.array(
                        [cx,cy,cz,sx,sy,sz], dtype=np.float32))

            if not boxes_norm:
                train_role = "negative"

        # ── Augmentation ──────────────────────────────────────────────────
        if self.augment:
            max_rot = 10.0 if train_role == "embolic" else 15.0
            dwi_m, adc_m, mask_m, boxes_norm = aug_flip_lr(
                dwi_m, adc_m, mask_m, boxes_norm)
            dwi_m, adc_m, mask_m, boxes_norm = aug_flip_si(
                dwi_m, adc_m, mask_m, boxes_norm)
            dwi_m, adc_m, mask_m, boxes_norm = aug_rotate_axial(
                dwi_m, adc_m, mask_m, boxes_norm,
                self.model_shape, max_angle=max_rot)
            dwi_m, adc_m = aug_intensity(dwi_m, adc_m, brain_mask)
            dwi_m, adc_m = aug_gaussian_noise(
                dwi_m, adc_m, brain_mask, p=0.5,
                noise_std_range=(0.0, 0.15))
            dwi_m, adc_m = aug_gamma(
                dwi_m, adc_m, brain_mask, p=0.5,
                gamma_range=(0.75, 1.25))

        # ── Build tensors ─────────────────────────────────────────────────
        # Image: [2, W, H, D] → permute to [2, D, H, W]
        X = torch.from_numpy(
                np.stack([dwi_m, adc_m], 0).astype(np.float32))
        X = X.permute(0, 3, 2, 1).contiguous()

        # Mask: [W, H, D] → [1, D, H, W]
        mask_t = torch.from_numpy(
            mask_m.astype(np.float32)).permute(2, 1, 0).unsqueeze(0).contiguous()

        box_single = torch.from_numpy(
            boxes_norm[0].astype(np.float32)
            if boxes_norm else np.zeros(6, dtype=np.float32))

        has_box = train_role in ("positive_single", "positive_multi")

        meta = {
            "subject":    subject,
            "train_role": train_role,
            "has_mask":   bool(has_mask),
            "n_boxes":    len(boxes_norm),
            "chain":      asdict(chain),
        }

        return {
            "image":      X,
            "mask":       mask_t,
            "box_single": box_single,
            "train_role": train_role,
            "has_box":    torch.tensor(has_box, dtype=torch.bool),
            "case_id":    subject,
            "meta":       meta,
        }


# =============================================================================
# Collate
# =============================================================================

def collate_fn(batch):
    return {
        "image":      torch.stack([b["image"]      for b in batch], 0),
        "mask":       torch.stack([b["mask"]        for b in batch], 0),
        "box_single": torch.stack([b["box_single"]  for b in batch], 0),
        "train_role": [b["train_role"]  for b in batch],
        "has_box":    torch.stack([b["has_box"]     for b in batch], 0),
        "case_id":    [b["case_id"]     for b in batch],
        "meta":       [b["meta"]        for b in batch],
    }


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",       required=True)
    ap.add_argument("--split_csv",  required=True)
    ap.add_argument("--split",      default="val")
    ap.add_argument("--n",          type=int, default=3)
    args = ap.parse_args()

    ds = StrokeDataset(args.root, args.split_csv, args.split,
                       augment=False, debug_print=True)
    print(f"\nSampling {args.n} cases...")
    for i in range(min(args.n, len(ds))):
        s = ds[i]
        print(f"\n[{i+1}] {s['case_id']}  role={s['train_role']}")
        print(f"  image : {s['image'].shape}  "
              f"min={s['image'].min():.2f}  max={s['image'].max():.2f}")
        print(f"  mask  : {s['mask'].shape}  "
              f"nonzero={s['mask'].sum().item():.0f}  "
              f"has_mask={s['meta']['has_mask']}")
        print(f"  box   : {s['box_single'].tolist()}")
        print(f"  has_box: {s['has_box'].item()}")
