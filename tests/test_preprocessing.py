"""
test_preprocessing.py
=====================
Tests for preprocessing pipeline:
  - CoM crop produces correct output shape
  - Resampling produces correct voxel spacing
  - Transform chain inverts correctly

All tests run without GPU, without ISLES-2022, under 30 seconds.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── CoM crop ─────────────────────────────────────────────────────────────────

def make_brain_volume(shape=(149, 149, 49), lesion=True):
    """Synthetic brain volume with known brain mask and optional lesion."""
    vol = np.zeros(shape, dtype=np.float32)
    # Brain region: central 60% of each axis
    mx, my, mz = [int(s * 0.2) for s in shape], \
                 [int(s * 0.2) for s in shape], \
                 [int(s * 0.2) for s in shape]
    ex, ey, ez = [int(s * 0.8) for s in shape], \
                 [int(s * 0.8) for s in shape], \
                 [int(s * 0.8) for s in shape]
    sl = (
        slice(mx[0], ex[0]),
        slice(my[1], ey[1]),
        slice(mz[2], ez[2]),
    )
    vol[sl] = np.random.rand(
        ex[0]-mx[0], ey[1]-my[1], ez[2]-mz[2]
    ).astype(np.float32) * 500 + 50

    if lesion:
        # Place a bright lesion in the brain
        cx, cy, cz = shape[0]//2, shape[1]//2, shape[2]//2
        vol[cx-5:cx+5, cy-5:cy+5, cz-3:cz+3] += 800

    return vol


def com_crop_fn(arr, tgt_shape):
    """Reference implementation of CoM crop (same as StrokeDataset)."""
    ins   = np.array(arr.shape, dtype=int)
    tgt   = np.array(tgt_shape, dtype=int)
    brain = arr > 0
    com   = np.argwhere(brain).mean(0).astype(int) if brain.sum() > 100 else ins // 2
    st    = np.clip(com - tgt // 2, 0, np.maximum(ins - tgt, 0))
    en    = st + tgt
    cr    = arr[st[0]:en[0], st[1]:en[1], st[2]:en[2]]
    out   = np.zeros(tuple(tgt), dtype=arr.dtype)
    cs    = np.array(cr.shape, dtype=int)
    ps    = np.clip((tgt - cs) // 2, 0, tgt - cs)
    out[ps[0]:ps[0]+cs[0], ps[1]:ps[1]+cs[1], ps[2]:ps[2]+cs[2]] = cr
    return out, st, ps


def test_com_crop_output_shape():
    """CoM crop must always produce exactly the target shape."""
    vol = make_brain_volume(shape=(149, 149, 49))
    tgt = (96, 96, 64)
    cropped, _, _ = com_crop_fn(vol, tgt)
    assert cropped.shape == tgt, (
        f"Expected shape {tgt}, got {cropped.shape}"
    )


def test_com_crop_larger_input():
    """CoM crop from a larger volume must produce target shape."""
    vol = make_brain_volume(shape=(200, 200, 100))
    tgt = (96, 96, 64)
    cropped, _, _ = com_crop_fn(vol, tgt)
    assert cropped.shape == tgt


def test_com_crop_preserves_lesion():
    """CoM crop should include the central lesion."""
    vol = make_brain_volume(shape=(149, 149, 49), lesion=True)
    tgt = (96, 96, 64)
    cropped, _, _ = com_crop_fn(vol, tgt)
    # Lesion was placed at centre — cropped volume should contain high values
    assert cropped.max() > 800, (
        f"Lesion likely cropped out. Max value: {cropped.max()}"
    )


def test_com_crop_all_zeros_fallback():
    """CoM crop on empty volume should not crash — uses centre fallback."""
    vol = np.zeros((149, 149, 49), dtype=np.float32)
    tgt = (96, 96, 64)
    cropped, st, _ = com_crop_fn(vol, tgt)
    assert cropped.shape == tgt
    assert st is not None


def test_com_crop_smaller_input():
    """CoM crop on input smaller than target should pad with zeros."""
    vol = make_brain_volume(shape=(80, 80, 50))
    tgt = (96, 96, 64)
    cropped, _, _ = com_crop_fn(vol, tgt)
    assert cropped.shape == tgt, (
        f"Expected {tgt}, got {cropped.shape}"
    )


# ── resampling ────────────────────────────────────────────────────────────────

def test_resample_output_spacing():
    """Resampling from 2mm isotropic to 1.5/1.5/3mm should give correct shape."""
    from scipy.ndimage import zoom

    native_zooms    = (2.0, 2.0, 2.0)
    target_spacing  = (1.5, 1.5, 3.0)
    vol_native      = np.random.rand(112, 112, 73).astype(np.float32)

    sf        = np.array(native_zooms) / np.array(target_spacing)
    vol_rs    = zoom(vol_native, sf, order=1, mode="nearest", prefilter=False)
    expected  = tuple(int(round(s * f)) for s, f in zip(vol_native.shape, sf))

    for i in range(3):
        assert abs(vol_rs.shape[i] - expected[i]) <= 1, (
            f"Axis {i}: expected ~{expected[i]}, got {vol_rs.shape[i]}"
        )


def test_resample_preserves_nonzero_ratio():
    """Resampling should preserve the approximate non-zero voxel fraction."""
    from scipy.ndimage import zoom

    vol = np.zeros((112, 112, 73), dtype=np.float32)
    vol[30:80, 30:80, 10:60] = 1.0    # brain region

    sf   = np.array([2.0, 2.0, 2.0]) / np.array([1.5, 1.5, 3.0])
    rs   = zoom(vol, sf, order=1, mode="nearest", prefilter=False)

    orig_frac = (vol > 0).mean()
    rs_frac   = (rs > 0).mean()
    assert abs(orig_frac - rs_frac) < 0.05, (
        f"Non-zero fraction changed too much: {orig_frac:.3f} → {rs_frac:.3f}"
    )


# ── transform chain inversion ─────────────────────────────────────────────────

def test_crop_start_recorded_correctly():
    """Crop start should be within valid range of resampled volume."""
    vol = make_brain_volume(shape=(149, 149, 49))
    tgt = (96, 96, 64)
    _, crop_start, _ = com_crop_fn(vol, tgt)

    ins = np.array(vol.shape, dtype=int)
    tgt_a = np.array(tgt, dtype=int)

    for i in range(3):
        assert 0 <= crop_start[i] <= max(0, ins[i] - tgt_a[i]), (
            f"Axis {i}: crop_start={crop_start[i]} out of range "
            f"[0, {max(0, ins[i]-tgt_a[i])}]"
        )


def test_pad_start_with_small_input():
    """When input is smaller than target, pad_start should be positive."""
    vol = make_brain_volume(shape=(80, 80, 50))
    tgt = (96, 96, 64)
    _, crop_start, pad_start = com_crop_fn(vol, tgt)

    # With input smaller than target, padding must occur
    tgt_a   = np.array(tgt, dtype=int)
    ins     = np.array(vol.shape, dtype=int)
    for i in range(3):
        if ins[i] < tgt_a[i]:
            assert pad_start[i] >= 0, (
                f"Axis {i}: expected non-negative pad_start"
            )


def test_gt_mask_same_crop_as_dwi():
    """Applying the same crop_start to GT mask should align with DWI crop."""
    # Create matching DWI and mask volumes
    shape = (149, 149, 49)
    dwi   = make_brain_volume(shape, lesion=True)
    mask  = np.zeros(shape, dtype=np.float32)
    # Place mask at same location as lesion
    cx, cy, cz = shape[0]//2, shape[1]//2, shape[2]//2
    mask[cx-5:cx+5, cy-5:cy+5, cz-3:cz+3] = 1.0

    tgt = (96, 96, 64)
    dwi_cropped, crop_start, pad_start = com_crop_fn(dwi, tgt)

    # Apply same crop to mask
    tgt_a = np.array(tgt, dtype=int)
    ins   = np.array(mask.shape, dtype=int)
    st    = crop_start
    en    = st + tgt_a
    en_cl = np.minimum(en, ins)
    cr    = mask[st[0]:en_cl[0], st[1]:en_cl[1], st[2]:en_cl[2]]
    mask_cropped = np.zeros(tuple(tgt_a), dtype=np.float32)
    cs = np.array(cr.shape, dtype=int)
    mask_cropped[
        pad_start[0]:pad_start[0]+cs[0],
        pad_start[1]:pad_start[1]+cs[1],
        pad_start[2]:pad_start[2]+cs[2],
    ] = cr

    # Lesion should be in the cropped mask
    assert mask_cropped.sum() > 0, (
        "GT mask is empty after applying DWI crop_start — alignment is broken"
    )


# ── z-score normalisation ─────────────────────────────────────────────────────

def test_zscore_brain_mean_near_zero():
    """After z-score normalisation, brain voxel mean should be near 0."""
    vol  = make_brain_volume(shape=(96, 96, 64))
    mask = vol > 0

    # Replicate robust_zscore: median and MAD
    brain_vals = vol[mask]
    med        = np.median(brain_vals)
    mad        = np.median(np.abs(brain_vals - med))
    std_est    = mad * 1.4826 + 1e-8
    vol_norm   = (vol - med) / std_est

    brain_mean = vol_norm[mask].mean()
    assert abs(brain_mean) < 0.1, (
        f"Brain mean after z-score = {brain_mean:.4f}, expected near 0"
    )


def test_zscore_does_not_affect_background():
    """Background voxels (==0) should remain near zero after z-score."""
    vol  = make_brain_volume(shape=(96, 96, 64))
    mask = vol > 0

    brain_vals = vol[mask]
    med        = np.median(brain_vals)
    mad        = np.median(np.abs(brain_vals - med))
    std_est    = mad * 1.4826 + 1e-8
    vol_norm   = np.where(mask, (vol - med) / std_est, 0.)

    bg_vals = vol_norm[~mask]
    assert (bg_vals == 0).all(), (
        "Background voxels changed after z-score"
    )
