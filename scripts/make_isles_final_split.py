#!/usr/bin/env python3
"""
Subject-level 70/15/15 split for ISLES with double stratification:
- Primary stratifier: status_class (positive if qc_status=='pass', else negative)
- Secondary stratifier: lesion volume bins (from bbox manifest volume_ml)

Inputs:
  --root           : ISLES root (base path)
  --qc-csv         : Path to QC CSV (contains 'subject','status')
  --bbox-manifest  : Path to bbox manifest CSV created in the bbox step
  --out            : Path to write the final split CSV (default: <root>/isles_final_split.csv)

Behavior:
  - Merges QC with bbox manifest on 'subject'
  - QC column 'status' is renamed to 'qc_status' to avoid collision with manifest 'status'
  - Reduces to one row per subject (keeps max volume_ml for binning)
  - For each stratum (status_class, volume_bin):
      * deterministic shuffle (seeded)
      * allocate 70% train, 15% val, 15% test (rounding; remainder to test)
  - Assigns split to all rows of that subject in the merged table
  - Writes final CSV with columns incl. subject/ses/mask_path/bbox_json/qc_status/status_class/annotation_type/volume_ml/volume_bin/split

Run example (Windows):
  python make_isles_final_split.py ^
    --root "D:\\Stroke\\ISLES-2022" ^
    --qc-csv "D:\\Stroke\\ISLES-2022\\isles2022_dwi_adc_qc.csv" ^
    --bbox-manifest "D:\\Stroke\\ISLES-2022\\isles_bboxes_manifest.csv" ^
    --out "D:\\Stroke\\ISLES-2022\\isles_final_split.csv" ^
    --use-volume-bins
"""

import os
import argparse
import numpy as np
import pandas as pd

DEFAULT_BINS = [-np.inf, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0, np.inf]
DEFAULT_BIN_LABELS = ["<0.5", "0.5–1", "1–3", "3–10", "10–30", "30–100", ">100"]


def read_qc(qc_csv: str) -> pd.DataFrame:
    df = pd.read_csv(qc_csv)
    if "subject" not in df.columns or "status" not in df.columns:
        raise ValueError("QC CSV must contain columns: 'subject', 'status'")
    df["subject"] = df["subject"].astype(str)
    df = df[["subject", "status"]].copy()
    # Rename to avoid collision with manifest's 'status'
    df = df.rename(columns={"status": "qc_status"})
    return df


def read_manifest(manifest_csv: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    if "subject" not in df.columns:
        raise ValueError("Manifest CSV must contain 'subject'.")
    # Ensure expected columns exist or fill if missing
    for col in ["ses", "mask_path", "bbox_json", "volume_ml", "shape_x", "shape_y", "shape_z"]:
        if col not in df.columns:
            df[col] = np.nan
    df["subject"] = df["subject"].astype(str)
    return df


def make_status_class(qc_status: str) -> str:
    # positive if qc_status=='pass', else negative
    return "positive" if (isinstance(qc_status, str) and qc_status.lower() == "pass") else "negative"


def bin_volumes(vol_ml: pd.Series,
                bins=DEFAULT_BINS,
                labels=DEFAULT_BIN_LABELS) -> pd.Series:
    # values NaN -> 'unknown'
    binned = pd.cut(vol_ml, bins=bins, labels=labels, right=False)
    binned = binned.astype(object)
    binned[pd.isna(binned)] = "unknown"
    return binned


def deterministic_split(items, train_frac=0.70, val_frac=0.15, seed=2026):
    """
    Deterministically split a list of unique items into train/val/test
    with rounding per stratum and remainder assigned to test.
    Returns three python sets.
    """
    rng = np.random.default_rng(seed)
    items = list(items)
    items = sorted(items)        # stable order
    rng.shuffle(items)           # deterministic shuffle

    n = len(items)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_test = n - n_train - n_val

    train_set = set(items[:n_train])
    val_set = set(items[n_train:n_train + n_val])
    test_set = set(items[n_train + n_val:])
    return train_set, val_set, test_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="ISLES dataset root")
    ap.add_argument("--qc-csv", required=True, help="Path to QC CSV (with 'subject','status')")
    ap.add_argument("--bbox-manifest", required=True, help="Path to bbox manifest CSV")
    ap.add_argument("--out", default=None, help="Final split CSV path (default: <root>/isles_final_split.csv)")
    ap.add_argument("--seed", type=int, default=2026, help="Random seed for determinism")
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--use-volume-bins", action="store_true", help="Enable secondary stratification by volume bins")
    ap.add_argument("--bins", type=float, nargs="*", default=DEFAULT_BINS, help="Volume bin edges (ml)")
    args = ap.parse_args()

    # Resolve output path
    out_csv = args.out or os.path.join(args.root, "isles_final_split.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    qc = read_qc(args.qc_csv)
    man = read_manifest(args.bbox_manifest)

    # Merge manifest with QC on subject (now 'qc_status' avoids collision)
    merged = pd.merge(man, qc, on="subject", how="inner")

    # Inform about non-overlapping subjects
    missing_in_qc = sorted(set(man["subject"]) - set(qc["subject"]))
    missing_in_manifest = sorted(set(qc["subject"]) - set(man["subject"]))
    if missing_in_qc:
        print(f"[WARN] {len(missing_in_qc)} subjects present in manifest but missing in QC. They will be skipped.")
    if missing_in_manifest:
        print(f"[WARN] {len(missing_in_manifest)} subjects present in QC but missing in manifest. They will be skipped.")

    if merged.empty:
        raise RuntimeError("No overlapping subjects between QC and manifest.")

    # Primary stratifier from QC status
    merged["status_class"] = merged["qc_status"].apply(make_status_class)

    # Reduce to one row per subject for stratification (use max volume_ml for binning)
    sub_agg = merged.groupby("subject", as_index=False).agg({
        "status_class": "first",
        "volume_ml": "max"
    })

    # Secondary stratifier: volume bins
    if args.use_volume_bins:
        # Build dynamic labels if custom bins provided
        if list(args.bins) != list(DEFAULT_BINS):
            edges = args.bins
            labels = []
            for i in range(len(edges) - 1):
                a, b = edges[i], edges[i + 1]
                if np.isneginf(a):
                    label = f"<{b}"
                elif np.isposinf(b):
                    label = f">{a}"
                else:
                    label = f"{a}–{b}"
                labels.append(label)
        else:
            labels = DEFAULT_BIN_LABELS
        sub_agg["volume_bin"] = bin_volumes(sub_agg["volume_ml"], bins=args.bins, labels=labels)
    else:
        sub_agg["volume_bin"] = "all"

    # Build strata keys: (status_class, volume_bin)
    sub_agg["stratum"] = sub_agg.apply(lambda r: (r["status_class"], r["volume_bin"]), axis=1)

    # Split deterministically within each stratum
    strata = sub_agg.groupby("stratum")["subject"].apply(list)

    train_subjects, val_subjects, test_subjects = set(), set(), set()
    per_stratum_counts = []

    for stratum, subjects in strata.items():
        tr, va, te = deterministic_split(subjects,
                                         train_frac=args.train_frac,
                                         val_frac=args.val_frac,
                                         seed=args.seed)
        train_subjects |= tr
        val_subjects |= va
        test_subjects |= te
        per_stratum_counts.append((stratum, len(subjects), len(tr), len(va), len(te)))

    # Assign split back to all rows for each subject
    def subj_to_split(s):
        if s in train_subjects:
            return "train"
        if s in val_subjects:
            return "val"
        if s in test_subjects:
            return "test"
        return "unsplit"  # should not happen

    merged["split"] = merged["subject"].apply(subj_to_split)

    # Annotation type for training: positives for pass, negatives otherwise (from QC)
    merged["annotation_type"] = np.where(
        merged["qc_status"].astype(str).str.lower() == "pass",
        "positive",
        "negative"
    )

    # Carry volume_bin to output (merge back from subject aggregation)
    merged = merged.merge(sub_agg[["subject", "volume_bin"]], on="subject", how="left")

    # Save final CSV
    merged.to_csv(out_csv, index=False)
    print(f"\nFinal split saved: {out_csv}")

    # --------- Summary ----------
    def summarize(df):
        out = {}
        out["subjects"] = len(df["subject"].unique())
        out["rows"] = len(df)
        out["positives"] = int((df["annotation_type"] == "positive").sum())
        out["negatives"] = int((df["annotation_type"] == "negative").sum())
        return out

    print("\n=== Split Summary (unique subjects) ===")
    for sp in ["train", "val", "test"]:
        sub = merged[merged["split"] == sp]
        s = summarize(sub)
        print(f"{sp:5s}: subjects={s['subjects']:4d} rows={s['rows']:5d} "
              f"pos={s['positives']:4d} neg={s['negatives']:4d}")

    # Volume-bin distribution by split & class (subjects)
    print("\n=== Volume-bin distribution per split (subjects) ===")
    sub_only = merged.drop_duplicates(["subject", "split"]).copy()
    if "volume_bin" in sub_only.columns:
        tab = (sub_only
               .groupby(["split", "status_class", "volume_bin"])["subject"]
               .nunique()
               .unstack(fill_value=0))
        print(tab)
    else:
        print("(no volume bins column)")

    # Per-stratum allocation recap
    print("\n=== Per-stratum allocation (subjects) ===")
    for (stratum, n, t, v, te) in per_stratum_counts:
        print(f"{stratum}: total={n:3d} -> train={t:3d}, val={v:3d}, test={te:3d}")


if __name__ == "__main__":
    main()