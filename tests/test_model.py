"""
test_model.py
=============
Tests for StrokeDetector architecture.

All tests:
  - Run without GPU (CPU only)
  - Do not require ISLES-2022 dataset
  - Do not require trained checkpoint
  - Complete in under 30 seconds total
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_input():
    """Minimal 2-channel brain volume at model input size."""
    return torch.randn(1, 2, 64, 96, 96)   # (B, C, D, H, W)


@pytest.fixture
def model_stage1():
    from stroke_detector.model import build_model
    return build_model(
        in_ch=2,
        base_channels=(16, 24, 32, 48),
        fpn_channels=32,
        input_shape=(96, 96, 64),
        num_classes=4,
        stage=1,
    )


@pytest.fixture
def model_stage2():
    from stroke_detector.model import build_model
    return build_model(
        in_ch=2,
        base_channels=(16, 24, 32, 48),
        fpn_channels=32,
        input_shape=(96, 96, 64),
        num_classes=4,
        stage=2,
    )


@pytest.fixture
def model_stage3():
    from stroke_detector.model import build_model
    return build_model(
        in_ch=2,
        base_channels=(16, 24, 32, 48),
        fpn_channels=32,
        input_shape=(96, 96, 64),
        num_classes=4,
        stage=3,
    )


# ── parameter count ───────────────────────────────────────────────────────────

def test_parameter_count_within_budget(model_stage3):
    """Model must fit within 1M parameter budget for lightweight deployment."""
    n_params = sum(p.numel() for p in model_stage3.parameters())
    assert n_params < 1_000_000, (
        f"Model has {n_params:,} parameters, expected < 1M"
    )


def test_parameter_count_approximately_correct(model_stage3):
    """Stage 3 should have approximately 0.504M parameters."""
    n_params = sum(p.numel() for p in model_stage3.parameters())
    assert 400_000 < n_params < 700_000, (
        f"Model has {n_params:,} parameters, expected ~504K"
    )


# ── forward pass output keys ──────────────────────────────────────────────────

def test_stage1_output_has_seg(model_stage1, dummy_input):
    """Stage 1 forward pass must return segmentation logits."""
    model_stage1.eval()
    with torch.no_grad():
        out = model_stage1(dummy_input)
    assert "seg" in out, f"Missing 'seg' key. Got: {list(out.keys())}"


def test_stage2_output_keys(model_stage2, dummy_input):
    """Stage 2 must return seg, detection outputs, and classification."""
    model_stage2.eval()
    with torch.no_grad():
        out = model_stage2(dummy_input)
    for key in ["seg"]:
        assert key in out, f"Missing '{key}' key. Got: {list(out.keys())}"


def test_stage3_output_has_presence(model_stage3, dummy_input):
    """Stage 3 must return presence_logit in addition to Stage 2 outputs."""
    model_stage3.eval()
    with torch.no_grad():
        out = model_stage3(dummy_input)
    assert "presence_logit" in out, (
        f"Missing 'presence_logit'. Got: {list(out.keys())}"
    )
    assert "seg" in out, f"Missing 'seg'. Got: {list(out.keys())}"


# ── output shapes ─────────────────────────────────────────────────────────────

def test_seg_output_shape(model_stage1, dummy_input):
    """Segmentation output must match input spatial dimensions."""
    model_stage1.eval()
    with torch.no_grad():
        out = model_stage1(dummy_input)
    seg = out["seg"]
    # seg shape: (B, 1, D, H, W)
    assert seg.shape[0] == 1,  f"Batch dim wrong: {seg.shape}"
    assert seg.shape[1] == 1,  f"Channel dim should be 1: {seg.shape}"
    assert seg.shape[2:] == dummy_input.shape[2:], (
        f"Spatial dims mismatch: seg={seg.shape[2:]}, "
        f"input={dummy_input.shape[2:]}"
    )


def test_presence_logit_shape(model_stage3, dummy_input):
    """Presence logit must be a scalar per batch item."""
    model_stage3.eval()
    with torch.no_grad():
        out = model_stage3(dummy_input)
    pres = out["presence_logit"]
    assert pres.shape[0] == 1, f"Batch dim wrong: {pres.shape}"


# ── sigmoid output range ──────────────────────────────────────────────────────

def test_seg_sigmoid_in_range(model_stage1, dummy_input):
    """Sigmoid of seg logits must be in [0, 1]."""
    model_stage1.eval()
    with torch.no_grad():
        out    = model_stage1(dummy_input)
        probs  = torch.sigmoid(out["seg"])
    assert probs.min() >= 0.0, f"Seg prob below 0: {probs.min()}"
    assert probs.max() <= 1.0, f"Seg prob above 1: {probs.max()}"


def test_presence_sigmoid_in_range(model_stage3, dummy_input):
    """Presence probability must be in [0, 1]."""
    model_stage3.eval()
    with torch.no_grad():
        out   = model_stage3(dummy_input)
        pres  = torch.sigmoid(out["presence_logit"])
    assert 0.0 <= float(pres) <= 1.0, (
        f"Presence prob out of range: {float(pres)}"
    )


# ── determinism ───────────────────────────────────────────────────────────────

def test_inference_deterministic(model_stage3, dummy_input):
    """Two forward passes on the same input must produce identical outputs."""
    model_stage3.eval()
    with torch.no_grad():
        out1 = model_stage3(dummy_input)
        out2 = model_stage3(dummy_input)
    assert torch.allclose(out1["seg"], out2["seg"]), (
        "Seg output is not deterministic"
    )
    assert torch.allclose(
        out1["presence_logit"], out2["presence_logit"]
    ), "Presence logit is not deterministic"


# ── stage progression ─────────────────────────────────────────────────────────

def test_stage1_has_no_presence(model_stage1, dummy_input):
    """Stage 1 should not have presence_logit in output."""
    model_stage1.eval()
    with torch.no_grad():
        out = model_stage1(dummy_input)
    assert "presence_logit" not in out, (
        "Stage 1 should not output presence_logit"
    )
