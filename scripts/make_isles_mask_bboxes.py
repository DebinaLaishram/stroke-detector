#!/usr/bin/env python3
"""
make_isles_mask_bboxes.py
=========================
Generates bbox JSON sidecar files for all ISLES-2022 lesion masks.

This is the definitive version combining all fixes and design decisions:

Case classification
-------------------
  SINGLE    : 1 component OR largest > 85% of total voxels
              → 1 bbox over largest component

  DOMINANT  : largest 30-85% of total, 2nd < 30% of largest
              → 1 bbox over largest component, secondary ignored

  MULTI     : 2+ components each > 30% of largest
              → bbox_vox = largest component (for single-box head)
              → bboxes_vox = ALL significant components (for multi-box head)

  EMBOLIC   : largest <= EMBOLIC_MAX_VOXELS AND second_ratio > MULTI_RATIO_THRESH
              → no usable single bbox
              → used for EMBOLIC classification head only
              → excluded from detection training

  EMPTY     : no positive voxels
              → true negative, used for negative training

Training signal philosophy
--------------------------
  We do NOT hard-exclude cases with diffusion constraint violations
  or tiny masks. Instead we flag them so the dataset loader can
  treat them as NEGATIVES. This teaches the model:

    1. Real acute stroke:
       DWI bright + ADC dark + valid bbox → predict box

    2. Morphology present but wrong signal:
       Mask exists + ADC not dark → NOT acute stroke
       Model learns diffusion constraint is REQUIRED

    3. Tiny apparent signal:
       Mask present but < 0.3ml → noise/artifact → negative
       Model learns minimum size threshold

  This is richer than simple exclusion — it teaches clinical reasoning.

JSON fields
-----------
  status              : ok / embolic / mask_empty / too_small
  case_type           : SINGLE / DOMINANT / MULTI / EMBOLIC / EMPTY
  n_components        : total connected components
  largest_voxels      : voxels in largest component
  second_ratio        : second_largest / largest (0 if only 1 component)
  all_component_sizes : list of all component sizes, largest first
  used_largest_only   : True for SINGLE/DOMINANT/EMBOLIC
  excluded_detection  : True if should not be used for detection training
  too_small           : True if volume_ml < TOO_SMALL_ML
  diffusion_ok        : True/False/null (from QC CSV if provided)
  bbox_vox            : tight bbox over largest component {xmin,xmax,ymin,ymax,zmin,zmax}
  bboxes_vox          : list of bboxes for ALL significant components (MULTI only)
                        each entry: {xmin,xmax,ymin,ymax,zmin,zmax,voxels,rank}

Split CSV updates
-----------------
  Adds/updates columns:
    case_type         : SINGLE/DOMINANT/MULTI/EMBOLIC/EMPTY
    diffusion_ok      : from QC CSV
    too_small         : bool
    train_role        : positive / negative / embolic / excluded
  Does NOT change qc_status for diffusion-violated or small-mask cases
  Those remain qc_status=pass but train_role=negative

Run
---
  # With QC CSV (recommended — enables diffusion_ok flagging)
  python make_isles_mask_bboxes.py ^
      --root "D:\\Stroke\\ISLES-2022" ^
      --split-csv "D:\\Stroke\\ISLES-2022\\isles_final_split.csv" ^
      --qc-csv "D:\\Stroke\\ISLES-2022\\isles2022_dwi_adc_qc.csv"

  # Without QC CSV
  python make_isles_mask_bboxes.py ^
      --root "D:\\Stroke\\ISLES-2022" ^
      --split-csv "D:\\Stroke\\ISLES-2022\\isles_final_split.csv"

  # Dry run first
  python make_isles_mask_bboxes.py ^
      --root "D:\\Stroke\\ISLES-2022" ^
      --split-csv "D:\\Stroke\\ISLES-2022\\isles_final_split.csv" ^
      --dry-run
"""

import os
import glob
import json
import argparse
from datetime import datetime, timezone

import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import label as cc_label


# =============================================================================
# Tunable thresholds
# =============================================================================
MULTI_RATIO_THRESH  = 0.30   # 2nd component > 30% of largest → MULTI or EMBOLIC
EMBOLIC_MAX_VOXELS  = 800    # largest <= this AND ratio > thresh → EMBOLIC
DOMINANT_RATIO_MAX  = 0.85   # largest > 85% of total → SINGLE
TOO_SMALL_ML        = 0.30   # volume < 0.3 ml → too_small → negative training
MIN_SECONDARY_RATIO = 0.30   # minimum ratio to include component in bboxes_vox


# =============================================================================
# Helpers
# =============================================================================

def find_masks(root):
    return sorted(glob.glob(
        os.path.join(root, "derivatives",
                     "sub-strokecase*", "ses-0001", "*msk*.nii*")))


def load_nii(path):
    img  = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return img, data, img.affine, img.header


def tight_bbox(mask_bool):
    """
    Compute tight half-open bbox over True voxels.
    Returns (xmin, xmax, ymin, ymax, zmin, zmax) or None if empty.
    xmax is EXCLUSIVE (half-open interval, numpy-style).
    """
    if not mask_bool.any():
        return None
    c = np.where(mask_bool)
    return (int(c[0].min()), int(c[0].max()) + 1,
            int(c[1].min()), int(c[1].max()) + 1,
            int(c[2].min()), int(c[2].max()) + 1)


def bbox_to_dict(bbox, shape, voxels=None, rank=None):
    """
    Convert bbox tuple to dict with safety clamping.
    bbox = (xmin, xmax, ymin, ymax, zmin, zmax) half-open
    """
    xmn, xmx, ymn, ymx, zmn, zmx = bbox
    xmn = max(0, xmn); xmx = min(shape[0], xmx)
    ymn = max(0, ymn); ymx = min(shape[1], ymx)
    zmn = max(0, zmn); zmx = min(shape[2], zmx)
    d = {
        "xmin": int(xmn), "xmax": int(xmx),
        "ymin": int(ymn), "ymax": int(ymx),
        "zmin": int(zmn), "zmax": int(zmx),
    }
    if voxels is not None:
        d["voxels"] = int(voxels)
    if rank is not None:
        d["rank"] = int(rank)
    return d


def classify_case(sizes, shape_vox):
    """
    Classify case type from component sizes.

    Parameters
    ----------
    sizes     : list of voxel counts, sorted largest first
    shape_vox : 3-tuple of volume shape (unused currently, reserved)

    Returns
    -------
    case_type    : str
    second_ratio : float  (second/largest, 0 if only 1 component)
    """
    if len(sizes) == 0:
        return "EMPTY", 0.0

    largest      = sizes[0]
    total        = sum(sizes)
    largest_frac = largest / max(1, total)

    if len(sizes) == 1:
        return "SINGLE", 0.0

    second       = sizes[1]
    second_ratio = second / max(1, largest)

    if second_ratio < MULTI_RATIO_THRESH:
        # One clearly dominant component, small satellites
        if largest_frac > DOMINANT_RATIO_MAX:
            return "SINGLE", second_ratio
        else:
            return "DOMINANT", second_ratio
    else:
        # Two or more significant components
        if largest <= EMBOLIC_MAX_VOXELS:
            return "EMBOLIC", second_ratio
        else:
            return "MULTI", second_ratio


def get_component_bboxes(labeled, sizes_sorted, shape, n_components):
    """
    Get bboxes for ALL significant components, sorted by size.
    Only includes components where size >= MIN_SECONDARY_RATIO * largest.

    Returns list of bbox dicts with voxel counts and rank.
    """
    largest = sizes_sorted[0]
    bboxes  = []

    # Map size -> component label (need to find which label has which size)
    comp_sizes = [(int((labeled == c).sum()), c)
                  for c in range(1, n_components + 1)]
    comp_sizes.sort(reverse=True)

    for rank, (voxels, comp_label) in enumerate(comp_sizes, 1):
        ratio = voxels / max(1, largest)
        if ratio < MIN_SECONDARY_RATIO and rank > 1:
            break   # remaining components too small to include
        comp_mask = labeled == comp_label
        bb        = tight_bbox(comp_mask)
        if bb is None:
            continue
        bboxes.append(bbox_to_dict(bb, shape, voxels=voxels, rank=rank))

    return bboxes


# =============================================================================
# Per-mask processing
# =============================================================================

def process_mask(msk_path, diffusion_ok_map=None):
    """
    Process one lesion mask file.

    Parameters
    ----------
    msk_path         : path to *msk*.nii.gz
    diffusion_ok_map : dict {subject: bool} from QC CSV, or None

    Returns
    -------
    dict with all JSON fields
    """
    img, data, aff, hdr = load_nii(msk_path)
    shape     = data.shape
    mask_bool = data > 0.5
    voxel_mm  = [float(hdr.get_zooms()[i]) for i in range(3)]
    vol_ml    = float(mask_bool.sum() * np.prod(voxel_mm) / 1000.0)

    subject = os.path.basename(
                  os.path.dirname(os.path.dirname(msk_path)))

    # Diffusion constraint status from QC CSV
    diffusion_ok = None
    if diffusion_ok_map is not None:
        diffusion_ok = diffusion_ok_map.get(subject, None)

    base_result = {
        "subject":         subject,
        "image":           os.path.basename(msk_path),
        "image_path":      os.path.abspath(msk_path),
        "shape":           [int(shape[i]) for i in range(3)],
        "voxels_total":    int(mask_bool.sum()),
        "voxel_size_mm":   voxel_mm,
        "volume_ml":       round(vol_ml, 4),
        "diffusion_ok":    diffusion_ok,
        "created_utc":     datetime.now(timezone.utc).isoformat(
                               timespec="seconds"),
    }

    # ── Empty mask ────────────────────────────────────────────────────────
    if not mask_bool.any():
        base_result.update({
            "status":             "mask_empty",
            "case_type":          "EMPTY",
            "n_components":       0,
            "largest_voxels":     0,
            "second_ratio":       0.0,
            "all_component_sizes": [],
            "too_small":          False,
            "used_largest_only":  False,
            "excluded_detection": True,
            "train_role":         "negative",
            "bbox_vox":           None,
            "bboxes_vox":         [],
        })
        return base_result

    # ── Connected components ──────────────────────────────────────────────
    labeled, n_components = cc_label(mask_bool)

    # Sizes sorted largest first
    sizes_raw    = [(labeled == c).sum() for c in range(1, n_components + 1)]
    sizes_sorted = sorted(sizes_raw, reverse=True)

    case_type, second_ratio = classify_case(sizes_sorted, shape)

    # Largest component label
    largest_label = int(np.argmax(sizes_raw)) + 1
    largest_mask  = labeled == largest_label
    largest_vox   = int(sizes_sorted[0])

    # ── Too small check ───────────────────────────────────────────────────
    too_small = vol_ml < TOO_SMALL_ML

    # ── Determine training role ───────────────────────────────────────────
    if case_type == "EMPTY":
        train_role = "negative"
    elif case_type == "EMBOLIC":
        train_role = "embolic"
    elif too_small:
        # Too small to localise reliably → treat as negative
        # Model learns: tiny signal = noise/artifact = not acute stroke
        train_role = "negative"
    elif diffusion_ok is False:
        # Mask present but diffusion constraint violated
        # Model learns: morphology alone is not enough
        train_role = "negative"
    elif case_type in ("SINGLE", "DOMINANT"):
        train_role = "positive_single"
    elif case_type == "MULTI":
        train_role = "positive_multi"
    else:
        train_role = "positive_single"

    excluded_detection = train_role in ("negative", "embolic")

    # ── Build bbox for largest component ─────────────────────────────────
    bb_largest = tight_bbox(largest_mask)
    if bb_largest is None:
        # Should not happen but safety fallback
        bbox_vox = None
    else:
        bbox_vox = bbox_to_dict(bb_largest, shape, voxels=largest_vox, rank=1)

    # ── Build bboxes_vox for ALL significant components (MULTI only) ──────
    if case_type == "MULTI":
        bboxes_vox = get_component_bboxes(
            labeled, sizes_sorted, shape, n_components)
    else:
        # For non-MULTI cases, bboxes_vox is just the single largest bbox
        # This keeps the format consistent for the dataset loader
        bboxes_vox = [bbox_vox] if bbox_vox is not None else []

    # ── Final result ──────────────────────────────────────────────────────
    base_result.update({
        "status":              "ok" if not excluded_detection else
                               ("embolic" if case_type == "EMBOLIC" else
                                "too_small" if too_small else
                                "diffusion_violated" if diffusion_ok is False
                                else case_type.lower()),
        "case_type":           case_type,
        "n_components":        int(n_components),
        "largest_voxels":      largest_vox,
        "second_ratio":        round(float(second_ratio), 4),
        "all_component_sizes": [int(s) for s in sizes_sorted],
        "too_small":           bool(too_small),
        "used_largest_only":   case_type != "MULTI",
        "excluded_detection":  bool(excluded_detection),
        "train_role":          train_role,
        "bbox_vox":            bbox_vox,
        "bboxes_vox":          bboxes_vox,
    })
    return base_result


def sidecar_path(msk_path):
    """Same filename as mask but with .bbox.json extension."""
    base = os.path.basename(msk_path)
    stem = base[:-7] if base.endswith(".nii.gz") else base[:-4]
    return os.path.join(os.path.dirname(msk_path), stem + ".bbox.json")


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Generate bbox JSON sidecars for ISLES-2022 masks")
    ap.add_argument("--root",         required=True,
                    help="ISLES-2022 dataset root")
    ap.add_argument("--split-csv",    default=None,
                    help="Path to isles_final_split.csv")
    ap.add_argument("--qc-csv",       default=None,
                    help="Path to isles2022_dwi_adc_qc.csv "
                         "(from QC script). Enables diffusion_ok flagging.")
    ap.add_argument("--manifest-out", default=None,
                    help="Output manifest CSV path")
    ap.add_argument("--dry-run",      action="store_true",
                    help="Compute but do not write any files")
    args = ap.parse_args()

    # ── Load QC CSV for diffusion constraint status ───────────────────────
    diffusion_ok_map = None
    if args.qc_csv and os.path.exists(args.qc_csv):
        qc_df = pd.read_csv(args.qc_csv)
        # Cases where QC status is "pass" → diffusion_ok=True
        # Cases where QC status is "flag" → diffusion_ok=False
        # Cases where QC status is missing → None
        if "status" in qc_df.columns:
            diffusion_ok_map = {}
            for _, row in qc_df.iterrows():
                subj = str(row["subject"])
                st   = str(row.get("status", "")).lower().strip()
                if st == "pass":
                    diffusion_ok_map[subj] = True
                elif st == "flag":
                    diffusion_ok_map[subj] = False
                # mask_small_or_outside → None (unknown)
        print(f"Loaded QC CSV: {args.qc_csv}")
        n_pass = sum(1 for v in diffusion_ok_map.values() if v is True)
        n_flag = sum(1 for v in diffusion_ok_map.values() if v is False)
        print(f"  diffusion_ok=True:  {n_pass}")
        print(f"  diffusion_ok=False: {n_flag}")
    else:
        if args.qc_csv:
            print(f"WARNING: QC CSV not found: {args.qc_csv}")
            print("  diffusion_ok will be null for all cases")
        else:
            print("No QC CSV provided — diffusion_ok will be null for all cases")
            print("Tip: run qc_dwi_adc_first3_aligned.py first to generate QC CSV")

    # ── Find all masks ────────────────────────────────────────────────────
    masks = find_masks(args.root)
    if not masks:
        print(f"No masks found under {args.root}/derivatives/")
        return

    manifest_out = args.manifest_out or os.path.join(
        args.root, "isles_bboxes_manifest.csv")

    print(f"\nProcessing {len(masks)} masks...")
    print(f"Dry run: {args.dry_run}\n")

    rows   = []
    counts = {"SINGLE": 0, "DOMINANT": 0, "MULTI": 0,
              "EMBOLIC": 0, "EMPTY": 0, "error": 0}
    role_counts = {
        "positive_single": 0,
        "positive_multi":  0,
        "embolic":         0,
        "negative":        0,
    }

    for i, msk_path in enumerate(masks, 1):
        try:
            res  = process_mask(msk_path, diffusion_ok_map)
            ct   = res["case_type"]
            role = res["train_role"]
            counts[ct]        = counts.get(ct, 0) + 1
            role_counts[role] = role_counts.get(role, 0) + 1

            # Build print flags
            flags = ""
            if ct == "EMBOLIC":
                flags = f"  ← EMBOLIC (largest={res['largest_voxels']})"
            elif ct == "MULTI":
                n_boxes = len(res["bboxes_vox"])
                flags = (f"  ← MULTI "
                         f"({n_boxes} boxes, "
                         f"2nd={res['second_ratio']:.2f})")
            elif ct == "DOMINANT":
                flags = f"  ← DOMINANT (2nd={res['second_ratio']:.2f})"
            if res["too_small"]:
                flags += "  [TOO SMALL → negative]"
            if res["diffusion_ok"] is False:
                flags += "  [DIFFUSION VIOLATED → negative]"

            print(f"[{i:04d}/{len(masks)}] {res['subject']}"
                  f"  {ct:<10}"
                  f"  n={res['n_components']}"
                  f"  largest={res['largest_voxels']}"
                  f"  vol={res['volume_ml']:.2f}ml"
                  f"  role={role}"
                  f"{flags}")

            # Write JSON sidecar
            if not args.dry_run:
                sp = sidecar_path(msk_path)
                with open(sp, "w") as f:
                    json.dump(res, f, indent=2)

            # Manifest row
            bv = res["bbox_vox"] or {}
            rows.append({
                "subject":           res["subject"],
                "mask_path":         os.path.abspath(msk_path),
                "bbox_json":         sidecar_path(msk_path),
                "case_type":         ct,
                "train_role":        role,
                "n_components":      res["n_components"],
                "largest_voxels":    res["largest_voxels"],
                "second_ratio":      res["second_ratio"],
                "n_bboxes":          len(res["bboxes_vox"]),
                "excluded_detection":res["excluded_detection"],
                "too_small":         res["too_small"],
                "diffusion_ok":      res["diffusion_ok"],
                "volume_ml":         res["volume_ml"],
                "xmin": bv.get("xmin"), "xmax": bv.get("xmax"),
                "ymin": bv.get("ymin"), "ymax": bv.get("ymax"),
                "zmin": bv.get("zmin"), "zmax": bv.get("zmax"),
                "status":            res["status"],
            })

        except Exception as exc:
            subject = os.path.basename(
                          os.path.dirname(os.path.dirname(msk_path)))
            print(f"[ERR] {subject}: {type(exc).__name__}: {exc}")
            import traceback; traceback.print_exc()
            counts["error"] += 1
            rows.append({
                "subject":    subject,
                "status":     f"error:{type(exc).__name__}",
                "case_type":  "error",
                "train_role": "error",
            })

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("CASE TYPE SUMMARY")
    print("=" * 65)
    for k, v in counts.items():
        print(f"  {k:<12}: {v}")

    print("\nTRAINING ROLE SUMMARY")
    print("=" * 65)
    for k, v in role_counts.items():
        print(f"  {k:<20}: {v}")

    too_small_list = [r["subject"] for r in rows
                      if r.get("too_small") and r.get("status") != "error"]
    diff_viol_list = [r["subject"] for r in rows
                      if r.get("diffusion_ok") is False
                      and r.get("status") != "error"]

    if too_small_list:
        print(f"\nToo small (< {TOO_SMALL_ML}ml) → treated as negative: "
              f"{len(too_small_list)}")
        for s in too_small_list:
            print(f"  {s}")

    if diff_viol_list:
        print(f"\nDiffusion constraint violated → treated as negative: "
              f"{len(diff_viol_list)}")
        for s in diff_viol_list:
            print(f"  {s}")

    multi_list = [r["subject"] for r in rows
                  if r.get("case_type") == "MULTI"]
    if multi_list:
        print(f"\nMULTI cases ({len(multi_list)}) — multi-box head training:")
        for r in rows:
            if r.get("case_type") == "MULTI":
                print(f"  {r['subject']}  n_bboxes={r['n_bboxes']}")

    # ── Write manifest ────────────────────────────────────────────────────
    if not args.dry_run:
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(os.path.abspath(manifest_out)),
                    exist_ok=True)
        df.to_csv(manifest_out, index=False)
        print(f"\nManifest saved: {manifest_out}")

        # ── Update split CSV ──────────────────────────────────────────────
        if args.split_csv and os.path.exists(args.split_csv):
            split_df = pd.read_csv(args.split_csv)

            # Build lookup from manifest rows
            role_map      = {r["subject"]: r.get("train_role", "")
                             for r in rows}
            ct_map        = {r["subject"]: r.get("case_type", "")
                             for r in rows}
            diff_ok_map_s = {r["subject"]: r.get("diffusion_ok")
                             for r in rows}
            too_small_map = {r["subject"]: r.get("too_small", False)
                             for r in rows}

            split_df["case_type"]    = split_df["subject"].map(ct_map)
            split_df["train_role"]   = split_df["subject"].map(role_map)
            split_df["diffusion_ok"] = split_df["subject"].map(diff_ok_map_s)
            split_df["too_small"]    = split_df["subject"].map(too_small_map)

            # Update qc_status for EMBOLIC cases only
            # Diffusion violated and too_small keep qc_status=pass
            # but train_role=negative tells the loader what to do
            embolic_subjects = [r["subject"] for r in rows
                                if r.get("case_type") == "EMBOLIC"]
            if embolic_subjects:
                split_df.loc[
                    split_df["subject"].isin(embolic_subjects),
                    "qc_status"
                ] = "excluded"

            split_df.to_csv(args.split_csv, index=False)
            print(f"Updated split CSV: {args.split_csv}")
            print(f"  Added columns: case_type, train_role, "
                  f"diffusion_ok, too_small")
            print(f"  qc_status set to 'excluded' for "
                  f"{len(embolic_subjects)} embolic cases")
            print(f"  Diffusion violated cases: qc_status unchanged, "
                  f"train_role=negative")
            print(f"  Too small cases: qc_status unchanged, "
                  f"train_role=negative")

    print("\nDone.")
    print("Next step: update yolo3d_onfly_dataset.py to:")
    print("  - Read train_role from bbox JSON")
    print("  - Route MULTI cases to multi-box head using bboxes_vox")
    print("  - Treat train_role=negative as negative training examples")
    print("  - Treat train_role=embolic as embolic classification examples")


if __name__ == "__main__":
    main()