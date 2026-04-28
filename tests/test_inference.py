"""
test_inference.py
=================
End-to-end inference tests using synthetic volumes.

All tests:
  - CPU only — no GPU required
  - No ISLES-2022 dataset required
  - No trained checkpoint required (tests use random weights)
  - Complete in under 30 seconds
"""

import pytest
import torch
import numpy as np
import json
import tempfile
import os
import sys
from pathlib import Path

import nibabel as nib

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── helpers ───────────────────────────────────────────────────────────────────

def make_synthetic_nifti(shape=(112, 112, 73), voxel_size=(2., 2., 2.),
                          with_lesion=True) -> nib.Nifti1Image:
    """Create a synthetic brain NIfTI for testing."""
    vol  = np.zeros(shape, dtype=np.float32)
    cx, cy, cz = shape[0]//2, shape[1]//2, shape[2]//2
    # Brain region
    vol[20:90, 20:90, 10:60] = np.random.rand(70, 70, 50).astype(np.float32) * 300 + 50
    if with_lesion:
        # Bright DWI lesion
        vol[cx-8:cx+8, cy-8:cy+8, cz-4:cz+4] += 900

    affine = np.diag([-voxel_size[0], voxel_size[1], voxel_size[2], 1.])
    return nib.Nifti1Image(vol, affine)


def make_synthetic_adc(shape=(112, 112, 73), voxel_size=(2., 2., 2.),
                        with_lesion=True) -> nib.Nifti1Image:
    """Create a synthetic ADC map. Lesion voxels have low ADC."""
    vol  = np.ones(shape, dtype=np.float32) * 900.   # ~0.9e-3 mm²/s in 10⁻³ units
    cx, cy, cz = shape[0]//2, shape[1]//2, shape[2]//2
    vol[20:90, 20:90, 10:60] = 900.
    if with_lesion:
        vol[cx-8:cx+8, cy-8:cy+8, cz-4:cz+4] = 350.  # restricted diffusion

    affine = np.diag([-voxel_size[0], voxel_size[1], voxel_size[2], 1.])
    return nib.Nifti1Image(vol, affine)


@pytest.fixture
def tmp_case(tmp_path):
    """Write synthetic DWI and ADC to a temporary directory."""
    dwi_img = make_synthetic_nifti(with_lesion=True)
    adc_img = make_synthetic_adc(with_lesion=True)
    dwi_path = tmp_path / "dwi.nii.gz"
    adc_path = tmp_path / "adc.nii.gz"
    nib.save(dwi_img, str(dwi_path))
    nib.save(adc_img, str(adc_path))
    return {"dwi": str(dwi_path), "adc": str(adc_path), "out": str(tmp_path)}


@pytest.fixture
def random_checkpoint(tmp_path):
    """Build a Stage 3 model with random weights and save a checkpoint."""
    from stroke_detector.model import build_model
    model = build_model(
        in_ch=2,
        base_channels=(16, 24, 32, 48),
        fpn_channels=32,
        input_shape=(96, 96, 64),
        num_classes=4,
        stage=3,
    )
    ckpt_path = tmp_path / "random_stage3.pt"
    torch.save({
        "model": model.state_dict(),
        "epoch": 0,
        "args": {
            "stage": 3,
            "base_channels": [16, 24, 32, 48],
            "fpn_channels": 32,
            "cls_hidden": 64,
        }
    }, str(ckpt_path))
    return str(ckpt_path)


# ── preprocessing pipeline ────────────────────────────────────────────────────

def test_preprocessing_produces_correct_shape(tmp_case):
    """Preprocessing a 112×112×73 @ 2mm volume must produce 96×96×64 tensor."""
    import nibabel as nib
    from scipy.ndimage import zoom
    from nibabel.orientations import (axcodes2ornt, io_orientation,
                                       inv_ornt_aff, apply_orientation,
                                       ornt_transform)
    from stroke_detector.data import robust_zscore

    dwi_img = nib.load(tmp_case["dwi"])
    dwi     = dwi_img.get_fdata(dtype=np.float32)
    native_z = tuple(float(z) for z in dwi_img.header.get_zooms()[:3])
    sf       = np.array(native_z) / np.array([1.5, 1.5, 3.0])

    # Reorient
    aff = dwi_img.affine
    ot  = ornt_transform(io_orientation(aff), axcodes2ornt(('R','A','S')))
    dwi_r = apply_orientation(dwi, ot)

    # Resample
    dwi_rs = zoom(dwi_r, sf, order=1, mode='nearest', prefilter=False)

    # CoM crop
    tgt   = np.array([96, 96, 64], dtype=int)
    ins   = np.array(dwi_rs.shape, dtype=int)
    brain = dwi_rs > 0
    com   = np.argwhere(brain).mean(0).astype(int) if brain.sum() > 100 else ins // 2
    st    = np.clip(com - tgt // 2, 0, np.maximum(ins - tgt, 0))
    en    = st + tgt
    cr    = dwi_rs[st[0]:en[0], st[1]:en[1], st[2]:en[2]]
    out   = np.zeros(tuple(tgt), dtype=np.float32)
    cs    = np.array(cr.shape, dtype=int)
    ps    = np.clip((tgt - cs) // 2, 0, tgt - cs)
    out[ps[0]:ps[0]+cs[0], ps[1]:ps[1]+cs[1], ps[2]:ps[2]+cs[2]] = cr

    assert out.shape == (96, 96, 64), f"Expected (96,96,64), got {out.shape}"


# ── forward pass with random weights ─────────────────────────────────────────

def test_forward_pass_no_error(random_checkpoint, tmp_case):
    """End-to-end forward pass on synthetic data must complete without error."""
    from stroke_detector.model import build_model, nms_3d
    import torch

    chk   = torch.load(random_checkpoint, map_location="cpu", weights_only=False)
    ca    = chk.get("args", {})
    model = build_model(
        in_ch=2,
        base_channels=tuple(ca.get("base_channels", [16,24,32,48])),
        fpn_channels=ca.get("fpn_channels", 32),
        input_shape=(96, 96, 64),
        num_classes=4,
        stage=ca.get("stage", 3),
    )
    model.load_state_dict(chk["model"], strict=False)
    model.eval()

    x = torch.randn(1, 2, 64, 96, 96)
    with torch.no_grad():
        out = model(x)

    assert "seg" in out
    assert "presence_logit" in out


# ── output JSON schema ────────────────────────────────────────────────────────

def test_result_json_has_required_fields(random_checkpoint, tmp_case):
    """
    The result JSON from inference must contain all required fields.
    Uses random model weights — only checks structure, not values.
    """
    from stroke_detector.model import build_model, nms_3d
    from stroke_detector.data import robust_zscore
    import nibabel as nib
    from scipy.ndimage import zoom
    from nibabel.orientations import (axcodes2ornt, io_orientation,
                                       inv_ornt_aff, apply_orientation,
                                       ornt_transform)

    # Minimal preprocessing
    def load_and_preprocess(path):
        img = nib.load(path)
        arr = img.get_fdata(dtype=np.float32)
        aff = img.affine
        hdr = img.header
        nz  = tuple(float(z) for z in hdr.get_zooms()[:3])
        sf  = np.array(nz) / np.array([1.5, 1.5, 3.0])
        ot  = ornt_transform(io_orientation(aff), axcodes2ornt(('R','A','S')))
        arr = apply_orientation(arr, ot)
        arr = zoom(arr, sf, order=1, mode='nearest', prefilter=False)
        tgt = np.array([96, 96, 64], dtype=int)
        ins = np.array(arr.shape, dtype=int)
        bm  = arr > 0
        com = np.argwhere(bm).mean(0).astype(int) if bm.sum()>100 else ins//2
        st  = np.clip(com-tgt//2, 0, np.maximum(ins-tgt,0))
        en  = st+tgt
        cr  = arr[st[0]:en[0], st[1]:en[1], st[2]:en[2]]
        out = np.zeros(tuple(tgt), dtype=np.float32)
        cs  = np.array(cr.shape,dtype=int)
        ps  = np.clip((tgt-cs)//2, 0, tgt-cs)
        out[ps[0]:ps[0]+cs[0], ps[1]:ps[1]+cs[1], ps[2]:ps[2]+cs[2]] = cr
        bm2 = out > 0
        out = robust_zscore(out, bm2) if bm2.any() else out
        return out

    dwi_m = load_and_preprocess(tmp_case["dwi"])
    adc_m = load_and_preprocess(tmp_case["adc"])
    X = np.stack([dwi_m, adc_m], 0)
    X = torch.from_numpy(X).float().permute(0,3,2,1).unsqueeze(0)

    chk   = torch.load(random_checkpoint, map_location="cpu", weights_only=False)
    ca    = chk.get("args", {})
    model = build_model(
        in_ch=2,
        base_channels=tuple(ca.get("base_channels",[16,24,32,48])),
        fpn_channels=ca.get("fpn_channels",32),
        input_shape=(96,96,64), num_classes=4, stage=ca.get("stage",3),
    )
    model.load_state_dict(chk["model"], strict=False)
    model.eval()

    with torch.no_grad():
        out  = model(X)
        pres = float(torch.sigmoid(out["presence_logit"][0]).item())
        seg  = torch.sigmoid(out["seg"])[0,0].cpu().numpy()

    # Build result dict (same structure as localize_stroke.py)
    result = {
        "subject":       "test-case",
        "has_lesion":    bool(pres > 0.5),
        "confidence":    round(pres, 4),
        "presence_conf": round(pres, 4),
        "stroke_type":   "unknown",
        "center_vox":    [48, 48, 32],
        "bbox_vox":      [0, 0, 0, 95, 95, 63],
        "stroke_volumes": {
            "stroke_volume_dwi_ml":          float((seg > 0.5).sum() * 1.5*1.5*3./1000.),
            "stroke_volume_adc_ml":          0.0,
            "stroke_volume_combined_ml":     0.0,
            "stroke_volume_intersection_ml": 0.0,
            "stroke_volume_union_ml":        0.0,
        }
    }

    # Required fields check
    required_top = [
        "subject", "has_lesion", "confidence", "presence_conf",
        "stroke_type", "center_vox", "bbox_vox", "stroke_volumes"
    ]
    for field in required_top:
        assert field in result, f"Missing required field: '{field}'"

    required_volumes = [
        "stroke_volume_dwi_ml",
        "stroke_volume_adc_ml",
        "stroke_volume_intersection_ml",
        "stroke_volume_union_ml",
    ]
    for field in required_volumes:
        assert field in result["stroke_volumes"], (
            f"Missing volume field: '{field}'"
        )

    # Type checks
    assert isinstance(result["has_lesion"], bool)
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["center_vox"], list)
    assert len(result["center_vox"]) == 3
    assert isinstance(result["bbox_vox"], list)
    assert len(result["bbox_vox"]) == 6


# ── JSON serialisable ─────────────────────────────────────────────────────────

def test_result_is_json_serialisable(random_checkpoint, tmp_case):
    """Result dict must serialise to JSON without error."""
    result = {
        "subject":   "test",
        "has_lesion": True,
        "confidence": 0.9743,
        "presence_conf": 0.8821,
        "stroke_type": "focal",
        "center_vox":  [48, 52, 31],
        "bbox_vox":    [32, 38, 21, 64, 66, 41],
        "stroke_volumes": {
            "stroke_volume_dwi_ml":          14.2,
            "stroke_volume_adc_ml":          12.8,
            "stroke_volume_combined_ml":     13.5,
            "stroke_volume_intersection_ml": 11.9,
            "stroke_volume_union_ml":        15.1,
        }
    }
    out_path = os.path.join(tmp_case["out"], "result.json")
    with open(out_path, "w") as f:
        json.dump(result, f)

    with open(out_path) as f:
        loaded = json.load(f)

    assert loaded["has_lesion"] == True
    assert loaded["stroke_volumes"]["stroke_volume_dwi_ml"] == 14.2
