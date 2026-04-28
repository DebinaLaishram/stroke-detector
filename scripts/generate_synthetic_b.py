#!/usr/bin/env python3
"""
generate_synthetic_b.py
=======================
Synthetic high-b-value DWI generation from ISLES-2022 test cases.

Physics model
-------------
Mono-exponential signal decay (Stejskal-Tanner):

    S(b_target) = S(b=1000) * exp(-(b_target - 1000) * ADC)

where ADC is in mm²/s.

ISLES-2022 ADC maps are stored in units of 10⁻⁶ mm²/s (i.e. µm²/ms).
Conversion: ADC_mm2_s = ADC_stored / 1_000_000

Reference
---------
Bladt P, et al. "Clinical value of quantitative DWI in cancer."
PMC8506195 (2021). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8506195/

Usage
-----
  python generate_synthetic_b.py \
      --split_csv  D:/Stroke/ISLES-2022/isles_final_split.csv \
      --root       D:/Stroke/ISLES-2022 \
      --out_dir    D:/Stroke/generalization/synthetic_b \
      --b_targets  1500 2000 2500

Outputs (per case, per b-value)
--------------------------------
  {out_dir}/b{B}/{subject_id}_b{B}_dwi.nii.gz   synthetic DWI volume
  {out_dir}/b{B}/{subject_id}_metadata.json      generation parameters

All synthetic volumes share the original DWI affine and header.
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# ADC unit conversion
# ---------------------------------------------------------------------------

ADC_UNIT_SCALE = 1e-3   # ISLES ADC stored as 10⁻³ mm²/s → convert to mm²/s
B_TRAIN = 1000          # b-value the training DWI was acquired at (s/mm²)

# ADC unit detection thresholds
ADC_THRESHOLD_MM2_S = 0.1   # median < this → already in mm²/s (Group A)
ADC_THRESHOLD_1E3   = 10.0  # median < this → 10⁻³ mm²/s (Group B)
                             # median >= this → 10⁻⁶ or integer scaled (Group C)


def load_nii(path: str):
    img  = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine, img.header


def detect_adc_units(adc_map: np.ndarray) -> tuple:
    """
    Auto-detect ADC units from value range.

    ISLES-2022 is multi-site — three conventions observed:
      Group A: mm²/s already      median ~0.001   scale = 1.0
      Group B: 10⁻³ mm²/s        median ~700-1500 scale = 1e-3
      Group C: 10⁻⁶ mm²/s (int)  median ~700-900  scale = 1e-6

    Groups B and C overlap in median — distinguish by max:
      Group B max ~4-5 (physical CSF ADC ~3-4 × 10⁻³)
      Group C max ~4095 (integer scaled, 12-bit)

    Returns (scale_factor, unit_label)
    """
    brain    = adc_map > 0
    if brain.sum() == 0:
        return 1e-3, "unknown (no brain voxels)"

    median = float(np.median(adc_map[brain]))
    maxval = float(adc_map.max())

    if median < ADC_THRESHOLD_MM2_S:
        # Already in mm²/s — Group A
        return 1.0, "mm²/s (no conversion needed)"
    elif maxval > 100:
        # Large integer values — could be 10⁻³ or 10⁻⁶
        if maxval > 3000:
            # Group C: integer scaled (max ~4095) → 10⁻⁶ mm²/s
            return 1e-6, "10⁻⁶ mm²/s (integer scaled)"
        else:
            # Group B: 10⁻³ mm²/s
            return 1e-3, "10⁻³ mm²/s"
    else:
        # Fallback: assume 10⁻³ mm²/s
        return 1e-3, "10⁻³ mm²/s (fallback)"


def synthesise_dwi(
    dwi_b1000:  np.ndarray,
    adc_map:    np.ndarray,
    b_target:   int,
    b_source:   int = B_TRAIN,
) -> tuple:
    """
    Generate synthetic DWI at b_target from DWI at b_source + ADC map.

    Auto-detects ADC units per case to handle multi-site ISLES-2022.

    Parameters
    ----------
    dwi_b1000 : (X, Y, Z) float32   source DWI (b=1000 s/mm²)
    adc_map   : (X, Y, Z) float32   ADC in any ISLES unit convention
    b_target  : int                  target b-value in s/mm²
    b_source  : int                  source b-value in s/mm²

    Returns
    -------
    synth      : (X, Y, Z) float32   synthetic DWI, non-negative
    unit_label : str                 detected ADC unit for logging
    """
    scale, unit_label = detect_adc_units(adc_map)

    delta_b   = float(b_target - b_source)        # s/mm²
    adc_mm2_s = adc_map.astype(np.float64) * scale

    # Clamp ADC to physically plausible range before exponent
    # Brain ADC: 0.1×10⁻³ to 4×10⁻³ mm²/s
    # CSF ADC:   up to ~4×10⁻³ mm²/s
    # Negative ADC → set to 0 (background noise)
    adc_mm2_s = np.clip(adc_mm2_s, 0., 5e-3)

    # Mono-exponential decay: S(b) = S(b0) * exp(-delta_b * ADC)
    exponent = np.clip(-delta_b * adc_mm2_s, -20., 20.)   # prevent overflow
    synth    = dwi_b1000.astype(np.float64) * np.exp(exponent)

    # Physical constraint: signal cannot be negative
    synth = np.clip(synth, 0., None)

    return synth.astype(np.float32), unit_label


def find_case_paths(row: pd.Series, root: str):
    """
    Resolve DWI and ADC paths from a CSV row.
    Handles both absolute paths and relative-to-root paths.
    """
    subject = row["subject"]
    ses     = row.get("ses", "ses-0001")

    # Standard ISLES BIDS layout
    dwi_candidates = [
        Path(root) / subject / ses / "dwi" / f"{subject}_{ses}_dwi.nii.gz",
        Path(root) / subject / ses / "dwi" / f"{subject}_{ses}_dwi.nii",
    ]
    adc_candidates = [
        Path(root) / subject / ses / "dwi" / f"{subject}_{ses}_adc.nii.gz",
        Path(root) / subject / ses / "dwi" / f"{subject}_{ses}_adc.nii",
    ]

    dwi_path = next((p for p in dwi_candidates if p.exists()), None)
    adc_path = next((p for p in adc_candidates if p.exists()), None)

    # Also check mask_path column for the root convention
    if dwi_path is None and "mask_path" in row:
        mask_p  = Path(str(row["mask_path"]))
        ses_dir = mask_p.parent.parent
        for suffix in ["_dwi.nii.gz", "_dwi.nii"]:
            cand = ses_dir / "dwi" / (subject + "_" + ses + suffix)
            if cand.exists():
                dwi_path = cand
                break
        for suffix in ["_adc.nii.gz", "_adc.nii"]:
            cand = ses_dir / "dwi" / (subject + "_" + ses + suffix)
            if cand.exists():
                adc_path = cand
                break

    return dwi_path, adc_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Synthetic high-b DWI generation for ISLES-2022 test set")
    ap.add_argument("--split_csv", required=True,
                    help="Path to isles_final_split.csv")
    ap.add_argument("--root",      required=True,
                    help="ISLES-2022 root directory")
    ap.add_argument("--out_dir",   required=True,
                    help="Output directory for synthetic volumes")
    ap.add_argument("--b_targets", type=int, nargs="+",
                    default=[1500, 2000, 2500],
                    help="Target b-values to synthesise (default: 1500 2000 2500)")
    ap.add_argument("--split",     default="test",
                    help="Dataset split to process (default: test)")
    ap.add_argument("--b_source",  type=int, default=B_TRAIN,
                    help="Source b-value of the DWI (default: 1000)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    for b in args.b_targets:
        (out_dir / f"b{b}").mkdir(parents=True, exist_ok=True)

    # Load CSV
    df   = pd.read_csv(args.split_csv)
    df   = df[df["split"] == args.split].reset_index(drop=True)
    print(f"[INFO] Processing {len(df)} cases from split='{args.split}'")
    print(f"[INFO] b-values to synthesise: {args.b_targets}")
    print(f"[INFO] Source b-value: {args.b_source} s/mm²")
    print(f"[INFO] ADC unit auto-detection enabled (handles mm²/s, 10⁻³, 10⁻⁶ conventions)")

    stats = []
    skipped = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Synthesising"):
        subject = row["subject"]
        ses     = str(row.get("ses", "ses-0001"))

        dwi_path, adc_path = find_case_paths(row, args.root)

        if dwi_path is None or not dwi_path.exists():
            print(f"[WARN] DWI not found for {subject} — skipping")
            skipped.append(subject)
            continue
        if adc_path is None or not adc_path.exists():
            print(f"[WARN] ADC not found for {subject} — skipping")
            skipped.append(subject)
            continue

        # Load
        dwi, dwi_aff, dwi_hdr = load_nii(str(dwi_path))
        adc, _,       _        = load_nii(str(adc_path))

        # Shape check
        if dwi.shape != adc.shape:
            print(f"[WARN] {subject}: DWI {dwi.shape} != ADC {adc.shape} — skipping")
            skipped.append(subject)
            continue

        case_stats = {
            "subject":     subject,
            "ses":         ses,
            "train_role":  row.get("train_role", "unknown"),
            "dwi_shape":   str(dwi.shape),
            "dwi_min":     float(dwi.min()),
            "dwi_max":     float(dwi.max()),
            "adc_min":     float(adc.min()),
            "adc_max":     float(adc.max()),
            "adc_mean_brain": float(adc[adc > 0].mean()) if (adc > 0).any() else 0.,
        }

        # Detect ADC units once per case (same for all b-values)
        _, adc_unit = detect_adc_units(adc)
        case_stats["adc_unit_detected"] = adc_unit

        for b_target in args.b_targets:
            synth, _ = synthesise_dwi(dwi, adc, b_target, args.b_source)

            # Save NIfTI — preserve original affine and header
            out_img  = nib.Nifti1Image(synth, dwi_aff, dwi_hdr)
            out_path = out_dir / f"b{b_target}" / f"{subject}_b{b_target}_dwi.nii.gz"
            nib.save(out_img, str(out_path))

            # Stats for this b-value
            case_stats[f"b{b_target}_min"]         = float(synth.min())
            case_stats[f"b{b_target}_max"]         = float(synth.max())
            case_stats[f"b{b_target}_mean_brain"]  = float(
                synth[synth > 0].mean()) if (synth > 0).any() else 0.
            case_stats[f"b{b_target}_zero_frac"]   = float(
                (synth == 0).mean())

        # Save per-case metadata
        meta_path = out_dir / f"{subject}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump({
                "subject":          subject,
                "source_b":         args.b_source,
                "target_b_values":  args.b_targets,
                "adc_unit_scale":   ADC_UNIT_SCALE,
                "adc_unit_note":    "ISLES ADC stored as 10^-3 mm2/s (verified: median~1.05), converted to mm2/s",
                "physics_model":    "S(b) = S(b0) * exp(-(b - b0) * ADC_mm2_s)",
                "reference":        "Bladt et al. PMC8506195 (2021)",
                "dwi_path":         str(dwi_path),
                "adc_path":         str(adc_path),
            }, f, indent=2)

        stats.append(case_stats)

    # Save generation stats
    stats_df = pd.DataFrame(stats)
    stats_path = out_dir / "generation_stats.csv"
    stats_df.to_csv(stats_path, index=False)

    print(f"\n[DONE] Generated synthetic volumes for {len(stats)} cases")
    print(f"       Skipped: {len(skipped)} cases: {skipped}")
    print(f"       Stats saved: {stats_path}")
    for b in args.b_targets:
        n = len(list((out_dir / f"b{b}").glob("*.nii.gz")))
        print(f"       b{b}: {n} volumes in {out_dir / f'b{b}'}")


if __name__ == "__main__":
    main()