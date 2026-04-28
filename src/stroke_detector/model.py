#!/usr/bin/env python3
"""
model_stroke.py
===============
Segmentation-Supervised Detection model for ischemic stroke localization.

Design rationale
----------------
Previous YOLO-only approach hit a mAP ceiling (~0.42) because:
  1. Detection heads had no idea what a lesion looks like internally --
     only where the box centre is and how big the box is.
  2. Classification head used global average pool on P5, collapsing to
     predicting one class every run.
  3. Ground-truth masks were never used -- only bbox JSON.

This model fixes all three:

Stage 1 — Segmentation pretraining
  Backbone → FPN → SegDecoder → binary mask
  Loss: Dice + BCE
  Goal: backbone learns rich lesion appearance features from pixel-level
  supervision before any detection training begins.

Stage 2 — Detection fine-tuning
  Load Stage 1 encoder+FPN weights.
  Backbone → FPN → DetectionHead → box + objectness  (primary)
                 → SegDecoder    → mask               (auxiliary, λ=0.3)
                 → ClassHead     → 4-class subtype     (masked pool)
  Loss: GIoU + L1 + obj + λ_seg×Dice + λ_cls×CE

Key novelty vs prior work
--------------------------
  - Every top ISLES method outputs masks only and stops there.
  - This model outputs bounding box + center + subtype + confidence
    for cross-domain deployment on clinical multishell DWI data.
  - Segmentation branch is training-time only -- inference outputs
    boxes, which are far more robust to domain shift than voxel masks.
  - ClassificationHead uses lesion-masked pooling not global pooling,
    so it actually sees lesion features rather than background.

Output at inference (Stage 2)
------------------------------
  {
    "has_lesion":  bool,
    "center":      (cx, cy, cz),   # voxel coords in model space
    "box":         (x0,y0,z0,x1,y1,z1),
    "subtype":     0-3,            # focal/multi/embolic/negative
    "confidence":  float,
    "seg_mask":    optional [1,W,H,D]  # if return_seg=True
  }

Architecture
------------
  Backbone : ResNetFPN3DBackbone  (unchanged from v1)
  FPN      : FPN3D                (unchanged from v1)
  SegDecoder: 4× transposed conv upsample from P2 → [1,W,H,D]
  DetHead  : YOLOHead3D on P2..P5 (single-box, anchor-free)
  ClsHead  : MaskedPool(P5, pred_mask) → Linear → 4 classes
  MultiHead: REMOVED (too few positive_multi cases, 8 train)

Parameters: ~0.85M (was 0.72M, extra ~0.13M from SegDecoder)
"""

import math
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 3-D IoU / GIoU / NMS  (unchanged)
# =============================================================================

def box_cxcycz_whd_to_xyzxyz(box: torch.Tensor) -> torch.Tensor:
    c    = box[..., :3]
    s    = torch.clamp(box[..., 3:], min=1e-6)
    half = 0.5 * s
    return torch.cat([c - half, c + half], dim=-1)


def box_xyzxyz_to_cxcycz_whd(box: torch.Tensor) -> torch.Tensor:
    mins = box[..., :3]; maxs = box[..., 3:]
    return torch.cat([(mins + maxs) * 0.5,
                      torch.clamp(maxs - mins, min=1e-6)], dim=-1)


def box_iou_3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    N, M = a.shape[0], b.shape[0]
    if N == 0 or M == 0:
        return a.new_zeros((N, M))
    max_min = torch.maximum(a[:, None, :3], b[None, :, :3])
    min_max = torch.minimum(a[:, None, 3:], b[None, :, 3:])
    inter   = torch.clamp(min_max - max_min, min=0).prod(dim=-1)
    vol_a   = torch.clamp(a[:, 3:] - a[:, :3], min=0).prod(dim=-1)
    vol_b   = torch.clamp(b[:, 3:] - b[:, :3], min=0).prod(dim=-1)
    union   = vol_a[:, None] + vol_b[None, :] - inter
    return inter / torch.clamp(union, min=1e-6)


def box_giou_3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape
    N = a.shape[0]
    if N == 0:
        return a.new_zeros((0,))
    iou     = box_iou_3d(a, b).diag()
    enc_min = torch.minimum(a[:, :3], b[:, :3])
    enc_max = torch.maximum(a[:, 3:], b[:, 3:])
    enc_vol = torch.clamp(enc_max - enc_min, min=0).prod(dim=-1)
    vol_a   = torch.clamp(a[:, 3:] - a[:, :3], min=0).prod(dim=-1)
    vol_b   = torch.clamp(b[:, 3:] - b[:, :3], min=0).prod(dim=-1)
    inter   = torch.clamp(
        torch.minimum(a[:, 3:], b[:, 3:]) -
        torch.maximum(a[:, :3], b[:, :3]), min=0).prod(dim=-1)
    union = vol_a + vol_b - inter
    return iou - (enc_vol - union) / torch.clamp(enc_vol, min=1e-6)


def nms_3d(boxes: torch.Tensor, scores: torch.Tensor,
           iou_thresh: float) -> List[int]:
    if boxes.numel() == 0:
        return []
    order = scores.argsort(descending=True)
    keep: List[int] = []
    while order.numel() > 0:
        i = int(order[0].item()); keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = box_iou_3d(boxes[i:i+1], boxes[rest]).squeeze(0)
        order = rest[ious <= iou_thresh]
    return keep


# =============================================================================
# Dice loss  (NEW — for segmentation branch)
# =============================================================================

def dice_loss(pred: torch.Tensor,
              target: torch.Tensor,
              smooth: float = 1.0) -> torch.Tensor:
    """
    Soft Dice loss for binary segmentation.
    pred   : (B, 1, D, H, W) sigmoid probabilities
    target : (B, 1, D, H, W) binary float mask
    """
    pred   = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    inter  = (pred * target).sum(dim=1)
    denom  = pred.sum(dim=1) + target.sum(dim=1)
    d      = 1.0 - (2.0 * inter + smooth) / (denom + smooth)
    return d.mean()


# =============================================================================
# GroupNorm helper
# =============================================================================

def make_norm(num_ch: int, num_groups: int = 8) -> nn.Module:
    g = min(num_groups, num_ch)
    while num_ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, num_ch)


# =============================================================================
# Backbone building blocks  (unchanged)
# =============================================================================

class ConvBNAct3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 k: int = 3, s: int = 1, p: int = 1,
                 num_groups: int = 8):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = make_norm(out_ch, num_groups)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock3d(nn.Module):
    def __init__(self, ch: int, bottleneck_ratio: float = 0.5,
                 num_groups: int = 8):
        super().__init__()
        h       = max(1, int(ch * bottleneck_ratio))
        self.c1 = ConvBNAct3d(ch, h, 1, 1, 0, num_groups)
        self.c2 = ConvBNAct3d(h,  h, 3, 1, 1, num_groups)
        self.c3 = nn.Conv3d(h, ch, 1, bias=False)
        self.bn3 = make_norm(ch, num_groups)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn3(self.c3(self.c2(self.c1(x)))) + x)


class DownBlock3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 num_res: int = 1, num_groups: int = 8):
        super().__init__()
        self.down = ConvBNAct3d(in_ch, out_ch, 3, 2, 1, num_groups)
        self.res  = nn.Sequential(
            *[ResidualBlock3d(out_ch, num_groups=num_groups)
              for _ in range(num_res)])

    def forward(self, x):
        return self.res(self.down(x))


# =============================================================================
# Backbone + FPN  (unchanged)
# =============================================================================

class ResNetFPN3DBackbone(nn.Module):
    def __init__(self, in_ch: int = 2,
                 base_channels: Tuple = (16, 24, 32, 48),
                 num_groups: int = 8):
        super().__init__()
        c2, c3, c4, c5 = base_channels
        self.stem   = nn.Sequential(
            ConvBNAct3d(in_ch, c2, 3, 2, 1, num_groups),
            ConvBNAct3d(c2,    c2, 3, 1, 1, num_groups))
        self.stage2 = nn.Sequential(
            ResidualBlock3d(c2, num_groups=num_groups),
            ResidualBlock3d(c2, num_groups=num_groups))
        self.stage3 = DownBlock3d(c2, c3, 2, num_groups)
        self.stage4 = DownBlock3d(c3, c4, 2, num_groups)
        self.stage5 = DownBlock3d(c4, c5, 2, num_groups)
        self.out_channels = {"C2": c2, "C3": c3, "C4": c4, "C5": c5}

    def forward(self, x) -> Dict[str, torch.Tensor]:
        x  = self.stem(x)
        c2 = self.stage2(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return {"C2": c2, "C3": c3, "C4": c4, "C5": c5}


class FPN3D(nn.Module):
    def __init__(self, in_channels: Dict[str, int],
                 out_ch: int = 32, num_groups: int = 8):
        super().__init__()
        c2, c3, c4, c5 = (in_channels[k] for k in ("C2","C3","C4","C5"))
        self.lat5 = nn.Conv3d(c5, out_ch, 1, bias=False)
        self.lat4 = nn.Conv3d(c4, out_ch, 1, bias=False)
        self.lat3 = nn.Conv3d(c3, out_ch, 1, bias=False)
        self.lat2 = nn.Conv3d(c2, out_ch, 1, bias=False)
        self.sm5  = ConvBNAct3d(out_ch, out_ch, num_groups=num_groups)
        self.sm4  = ConvBNAct3d(out_ch, out_ch, num_groups=num_groups)
        self.sm3  = ConvBNAct3d(out_ch, out_ch, num_groups=num_groups)
        self.sm2  = ConvBNAct3d(out_ch, out_ch, num_groups=num_groups)

    def forward(self, f: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        c2, c3, c4, c5 = f["C2"], f["C3"], f["C4"], f["C5"]
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(
             p5, c4.shape[2:], mode="trilinear", align_corners=False)
        p3 = self.lat3(c3) + F.interpolate(
             p4, c3.shape[2:], mode="trilinear", align_corners=False)
        p2 = self.lat2(c2) + F.interpolate(
             p3, c2.shape[2:], mode="trilinear", align_corners=False)
        return {"P2": self.sm2(p2), "P3": self.sm3(p3),
                "P4": self.sm4(p4), "P5": self.sm5(p5)}


# =============================================================================
# Segmentation Decoder  (NEW)
# =============================================================================

class SegDecoder3D(nn.Module):
    """
    Lightweight segmentation decoder.

    Takes P2 (highest-resolution FPN feature, stride=2 from input) and
    upsamples back to full input resolution via 4 transposed conv stages.

    Architecture
    ------------
    P2 [fpn_ch, D/2, H/2, W/2]
      → ConvBNAct (fpn_ch → 16)
      → Upsample ×2  (trilinear)
      → ConvBNAct (16 → 8)
      → Upsample ×2  (if needed for exact shape)
      → Conv3d (8 → 1) + Sigmoid

    Output: (B, 1, D, H, W) — lesion probability map in [0, 1]

    Why this design
    ---------------
    - Keeps parameter count low (~0.03M extra)
    - P2 already has the spatial resolution closest to the input
    - Simple enough to train stably with 175 cases
    - Sigmoid output compatible with Dice + BCE loss
    """

    def __init__(self, in_ch: int = 32, num_groups: int = 8):
        super().__init__()
        self.conv1 = ConvBNAct3d(in_ch, 16, num_groups=min(8, 16))
        self.conv2 = ConvBNAct3d(16, 8,  num_groups=min(8, 8))
        self.out   = nn.Conv3d(8, 1, 1)

    def forward(self, p2: torch.Tensor,
                target_shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        p2           : (B, C, D/2, H/2, W/2)
        target_shape : (D, H, W) — full input resolution
        Returns      : (B, 1, D, H, W) sigmoid probabilities
        """
        x = self.conv1(p2)
        x = F.interpolate(x, scale_factor=2,
                          mode="trilinear", align_corners=False)
        x = self.conv2(x)
        # Final resize to exact target shape (handles any rounding)
        if x.shape[2:] != target_shape:
            x = F.interpolate(x, size=target_shape,
                              mode="trilinear", align_corners=False)
        return self.out(x)   # raw logits — sigmoid applied in loss/inference


# =============================================================================
# Detection head  (unchanged)
# =============================================================================

class YOLOHead3D(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int = 64,
                 num_out: int = 7, num_groups: int = 8):
        super().__init__()
        self.convs = nn.Sequential(
            ConvBNAct3d(in_ch, hidden_ch, num_groups=num_groups),
            ConvBNAct3d(hidden_ch, hidden_ch, num_groups=num_groups))
        self.pred  = nn.Conv3d(hidden_ch, num_out, 1)

    def forward(self, x):
        return self.pred(self.convs(x))


# =============================================================================
# Classification head  (NEW — masked pooling replaces global pool)
# =============================================================================

class MaskedClassificationHead(nn.Module):
    """
    4-class stroke subtype head using lesion-masked pooling.

    Why masked pooling
    ------------------
    Previous ClassificationHead used GlobalAvgPool3d(P5) which pooled
    the entire brain volume including skull, CSF, and healthy tissue.
    This drowned out lesion-specific features and caused the head to
    collapse to predicting one class every run.

    Masked pooling restricts attention to the predicted lesion region:
      1. Downsample predicted seg mask to P5 resolution
      2. Softmax-weight P5 features by mask probability
      3. Sum → compact lesion feature vector
      4. Linear → 4-class logits

    If mask is empty (negative case), falls back to mean pooling.

    Classes
    -------
      0 = focal    (positive_single)
      1 = multi    (positive_multi)
      2 = embolic  (embolic)
      3 = negative (negative)
    """

    def __init__(self, in_ch: int = 32, num_classes: int = 4,
                 hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_ch, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, p5: torch.Tensor,
                seg_mask: torch.Tensor) -> torch.Tensor:
        """
        p5       : (B, C, D5, H5, W5)  FPN P5 features
        seg_mask : (B, 1, D,  H,  W)   sigmoid segmentation output
        Returns  : (B, num_classes)    raw logits
        """
        # Downsample mask to P5 resolution — apply sigmoid since seg outputs logits
        mask_ds = torch.sigmoid(F.interpolate(seg_mask, size=p5.shape[2:],
                                mode="trilinear",
                                align_corners=False))   # (B,1,D5,H5,W5)

        # Flatten spatial dims
        B, C, D5, H5, W5 = p5.shape
        p5_flat   = p5.view(B, C, -1)          # (B, C, N)
        mask_flat = mask_ds.view(B, 1, -1)     # (B, 1, N)

        # Softmax attention weights from mask
        attn = mask_flat + 1e-6
        attn_sum = attn.sum(dim=-1, keepdim=True)

        # Fall back to uniform if mask is empty
        empty = (attn_sum < 1e-3).squeeze(-1).squeeze(-1)   # (B,)
        if empty.any():
            uniform = torch.ones_like(attn) / (D5 * H5 * W5)
            attn[empty] = uniform[empty]
            attn_sum[empty] = 1.0

        attn = attn / attn_sum                 # (B, 1, N)

        # Weighted sum → lesion feature vector
        feat = (p5_flat * attn).sum(dim=-1)    # (B, C)

        return self.head(feat)




# =============================================================================
# Global Presence Head  (NEW — Stage 3)
# =============================================================================

class GlobalPresenceHead(nn.Module):
    """
    Binary stroke presence classifier.

    Answers: "Is there a stroke anywhere in this volume?"
    Completely separate from the localisation objectness heads.

    Why this fixes Specificity=0
    ----------------------------
    The objectness heads learn "fire here = lesion location".
    They are trained with strong positive detection loss which
    overwhelms the suppression loss on negative cases.

    This head operates at the whole-volume level using two signals:
      1. Global average pool of P5 features (semantic context)
      2. Max pool of predicted seg mask (is anything activated?)

    These two signals together give the head enough information to
    distinguish "brain with diffusion restriction" from "normal brain"
    without competing with the localisation gradient.

    Architecture
    ------------
    P5 → GlobalAvgPool → flatten  → [fpn_ch]
    seg_logits → sigmoid → MaxPool3d(whole vol) → [1]
    concat → [fpn_ch + 1]
    → Linear → ReLU → Dropout → Linear(1)
    → raw logit (sigmoid applied in loss/inference)

    Training
    --------
    Label: 1 if role in (positive_single, positive_multi, embolic)
           0 if role == negative
    Loss:  BCEWithLogitsLoss with pos_weight to handle imbalance
    """

    def __init__(self, fpn_ch: int = 32, hidden: int = 32,
                 dropout: float = 0.3):
        super().__init__()
        self.gap  = nn.AdaptiveAvgPool3d(1)   # global avg pool on P5
        self.head = nn.Sequential(
            nn.Linear(fpn_ch + 1, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, 1),             # raw logit
        )

    def forward(self, p5: torch.Tensor,
                seg_logits: torch.Tensor) -> torch.Tensor:
        """
        p5         : (B, C, D5, H5, W5)  FPN P5 features
        seg_logits : (B, 1, D,  H,  W)   raw seg logits (before sigmoid)
        Returns    : (B, 1)  raw presence logit
        """
        # Global average pool of P5 → semantic volume summary
        gap_feat = self.gap(p5).view(p5.shape[0], -1)        # (B, C)

        # Max activation in seg map → "is anything lit up?"
        seg_prob = torch.sigmoid(seg_logits)                  # (B,1,D,H,W)
        seg_max  = seg_prob.view(seg_prob.shape[0], -1).max(dim=1,
                   keepdim=True)[0]                           # (B, 1)

        feat = torch.cat([gap_feat, seg_max], dim=1)          # (B, C+1)
        return self.head(feat)                                 # (B, 1)

# =============================================================================
# Class constants
# =============================================================================

CLS_FOCAL    = 0
CLS_MULTI    = 1
CLS_EMBOLIC  = 2
CLS_NEGATIVE = 3

ROLE_TO_CLS = {
    "positive_single": CLS_FOCAL,
    "positive_multi":  CLS_MULTI,
    "embolic":         CLS_EMBOLIC,
    "negative":        CLS_NEGATIVE,
}


# =============================================================================
# Main detector
# =============================================================================

class StrokeDetector(nn.Module):
    """
    Segmentation-Supervised Stroke Detector.

    forward() returns dict:
        "seg"           : (B,1,D,H,W)  sigmoid mask  [all stages]
        "P2".."P5"      : detection feature maps      [stage 2+]
        "cls_logits"    : (B,4) classification logits [stage 2+]
        "presence_logit": (B,1) stroke presence logit [stage 3]
    """

    def __init__(self,
                 in_ch: int = 2,
                 base_channels: Tuple = (16, 24, 32, 48),
                 fpn_channels: int = 32,
                 input_shape: Tuple = (96, 96, 64),
                 num_classes: int = 4,
                 num_groups: int = 8,
                 cls_hidden: int = 64,
                 stage: int = 1):
        """
        stage : 1 = segmentation only (pretraining)
                2 = detection + aux segmentation (fine-tuning)
                3 = add global presence head (fixes Specificity)
        """
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.stage       = stage

        # Shared backbone + FPN
        self.backbone = ResNetFPN3DBackbone(in_ch, base_channels, num_groups)
        self.fpn      = FPN3D(self.backbone.out_channels,
                              out_ch=fpn_channels,
                              num_groups=num_groups)

        # Segmentation decoder (both stages)
        self.seg_decoder = SegDecoder3D(in_ch=fpn_channels,
                                        num_groups=num_groups)

        if stage == 2:
            # Detection heads (stage 2 only)
            hc = max(32, fpn_channels)
            self.det_heads = nn.ModuleDict({
                k: YOLOHead3D(fpn_channels, hc, 7, num_groups)
                for k in ["P2", "P3", "P4", "P5"]})

            # Classification head with masked pooling (stage 2 only)
            self.cls_head = MaskedClassificationHead(
                in_ch=fpn_channels,
                num_classes=num_classes,
                hidden=cls_hidden,
                dropout=0.3)

            self.levels = ["P2", "P3", "P4", "P5"]
            self._init_det_biases()

        if stage == 3:
            # Stage 3 inherits everything from stage 2
            hc = max(32, fpn_channels)
            self.det_heads = nn.ModuleDict({
                k: YOLOHead3D(fpn_channels, hc, 7, num_groups)
                for k in ["P2", "P3", "P4", "P5"]})
            self.cls_head = MaskedClassificationHead(
                in_ch=fpn_channels,
                num_classes=num_classes,
                hidden=cls_hidden,
                dropout=0.3)
            self.levels = ["P2", "P3", "P4", "P5"]
            self._init_det_biases()

            # Global presence head — new in stage 3
            self.presence_head = GlobalPresenceHead(
                fpn_ch  = fpn_channels,
                hidden  = 32,
                dropout = 0.3)

    def _init_det_biases(self):
        with torch.no_grad():
            for head in self.det_heads.values():
                if head.pred.bias is not None:
                    head.pred.bias[0]   = -2.0
                    head.pred.bias[1:4] = 0.0
                    head.pred.bias[4:7] = 0.0

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x : (B, 2, D, H, W)
        """
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        target_shape = (D, H, W)

        feats = self.fpn(self.backbone(x))

        # Segmentation output (always)
        seg = self.seg_decoder(feats["P2"], target_shape)
        out = {"seg": seg}

        if self.stage in (2, 3):
            # Detection heads
            for k in self.levels:
                out[k] = self.det_heads[k](feats[k])

            # Classification with masked pooling
            out["cls_logits"] = self.cls_head(feats["P5"], seg)

        if self.stage == 3:
            # Global presence head — is there a stroke at all?
            out["presence_logit"] = self.presence_head(feats["P5"], seg)

        return out

    # -- Decode detection heads (stage 2 inference) ------------------------

    @staticmethod
    def _decode_level(pred: torch.Tensor,
                      input_shape: Tuple[int, int, int]):
        B, C, D_l, H_l, W_l = pred.shape
        W_in, H_in, D_in    = input_shape
        obj = torch.sigmoid(pred[:, 0:1])
        reg = pred[:, 1:]

        zz, yy, xx = torch.meshgrid(
            torch.arange(D_l, device=pred.device),
            torch.arange(H_l, device=pred.device),
            torch.arange(W_l, device=pred.device),
            indexing="ij")
        xx = xx[None, None]; yy = yy[None, None]; zz = zz[None, None]

        sx_s = W_in / W_l; sy_s = H_in / H_l; sz_s = D_in / D_l

        cx = (torch.sigmoid(reg[:, 0:1]) + xx) * sx_s
        cy = (torch.sigmoid(reg[:, 1:2]) + yy) * sy_s
        cz = (torch.sigmoid(reg[:, 2:3]) + zz) * sz_s

        reg_sz = torch.clamp(reg[:, 3:6], -6., 6.)
        w = torch.exp(reg_sz[:, 0:1]) * sx_s
        h = torch.exp(reg_sz[:, 1:2]) * sy_s
        d = torch.exp(reg_sz[:, 2:3]) * sz_s

        box_cs = torch.cat([cx, cy, cz, w, h, d], 1) \
                         .permute(0, 2, 3, 4, 1).contiguous().view(B, -1, 6)
        scores = obj.permute(0, 2, 3, 4, 1).contiguous().view(B, -1)
        return box_cxcycz_whd_to_xyzxyz(box_cs), scores

    @torch.no_grad()
    def decode(self, outputs: Dict[str, torch.Tensor]):
        all_boxes = []; all_scores = []
        for k in self.levels:
            bx, sc = self._decode_level(outputs[k], self.input_shape)
            all_boxes.append(bx); all_scores.append(sc)
        return torch.cat(all_boxes, 1), torch.cat(all_scores, 1)


# =============================================================================
# Combined loss
# =============================================================================

class StrokeLoss(nn.Module):
    """
    Stage-aware loss function.

    Stage 1 (segmentation pretraining)
    -----------------------------------
      L = Dice(pred_mask, gt_mask) + BCE(pred_mask, gt_mask)
      Applied to ALL cases that have a mask (positive + embolic).
      Negative cases: L = BCE(pred_mask, zeros) — suppress false positives.

    Stage 2 (detection fine-tuning)
    ---------------------------------
      L = L_det + lambda_seg × L_seg + lambda_cls × L_cls

      L_det : bce_pos + lambda_obj_bg × bce_bg + lambda_giou × giou + lambda_l1 × l1
      L_seg : Dice + BCE (auxiliary, keeps backbone grounded)
      L_cls : CrossEntropy (4-class subtype)

    GIoU warmup
    -----------
      giou_w = min(1, epoch / giou_warmup_epochs)
      Prevents GIoU from dominating before L1 has pulled boxes near GT.
    """

    def __init__(self,
                 stage: int             = 1,
                 lambda_giou: float     = 2.0,
                 lambda_l1: float       = 5.0,
                 lambda_obj_bg: float   = 0.05,
                 lambda_cls: float      = 0.3,
                 lambda_seg: float      = 1.0,
                 pos_obj_weight: float  = 3.0,
                 large_box_thresh: float = 0.7,
                 giou_warmup_epochs: int = 30,
                 lambda_presence: float  = 1.0,
                 return_components: bool = True):
        super().__init__()
        self.stage               = stage
        self.lambda_giou         = lambda_giou
        self.lambda_l1           = lambda_l1
        self.lambda_obj_bg       = lambda_obj_bg
        self.lambda_cls          = lambda_cls
        self.lambda_seg          = lambda_seg
        self.lambda_presence     = lambda_presence
        self.large_box_thresh    = large_box_thresh
        self.giou_warmup_epochs  = giou_warmup_epochs
        self.return_components   = return_components
        self._epoch              = 0

        self.register_buffer("pos_weight",
                             torch.tensor([float(pos_obj_weight)]))
        self.bce    = nn.BCEWithLogitsLoss(reduction="mean")
        self.l1     = nn.SmoothL1Loss(reduction="mean")
        self.cls_ce = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 4.0, 0.3, 2.0]),  # up-weight multi+negative, reduce embolic
            reduction="mean")

        # Presence loss — stage 3
        # pos_weight=2.0: slightly upweight positive presence
        # (there are more negative cases after rebalancing)
        self.presence_bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([2.0]),
            reduction="mean")

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    # -- Segmentation loss -------------------------------------------------

    def _seg_loss(self,
                  pred_mask: torch.Tensor,
                  gt_mask: torch.Tensor,
                  has_lesion: bool) -> torch.Tensor:
        """
        pred_mask : (1,1,D,H,W) sigmoid
        gt_mask   : (1,1,D,H,W) binary float
        has_lesion: True if case has a lesion mask
        """
        if has_lesion:
            return dice_loss(torch.sigmoid(pred_mask), gt_mask) + \
                   F.binary_cross_entropy_with_logits(
                       pred_mask, gt_mask, reduction="mean")
        else:
            return F.binary_cross_entropy_with_logits(
                pred_mask, torch.zeros_like(pred_mask), reduction="mean")

    # -- Detection loss (stage 2) ------------------------------------------

    @staticmethod
    def _level_from_size(norm_size: torch.Tensor,
                         fmap_shapes: List[Tuple]) -> int:
        best = 0; best_cost = 1e9
        for li, (D_l, H_l, W_l) in enumerate(fmap_shapes):
            cells = torch.tensor(
                [norm_size[0]*W_l, norm_size[1]*H_l, norm_size[2]*D_l],
                device=norm_size.device, dtype=norm_size.dtype)
            cost = float(torch.abs(cells.max() - 4.0))
            if cost < best_cost:
                best_cost = cost; best = li
        return best

    def _det_loss_one(self,
                      outputs: Dict[str, torch.Tensor],
                      gt_norm: torch.Tensor,
                      fmap_shapes: List[Tuple],
                      levels: List[str]) -> Dict[str, torch.Tensor]:
        li   = self._level_from_size(gt_norm[3:], fmap_shapes)
        k    = levels[li]
        pred = outputs[k]
        _, _, D, H, W = pred.shape

        cx, cy, cz, sx, sy, sz = gt_norm
        ci = torch.clamp((cx*W).long(), 0, W-1)
        cj = torch.clamp((cy*H).long(), 0, H-1)
        ck = torch.clamp((cz*D).long(), 0, D-1)

        obj_logits = pred[:, 0:1]
        pos_logit  = obj_logits[0, 0, ck, cj, ci].view(1)
        bce_pos    = F.binary_cross_entropy_with_logits(
            pos_logit, torch.ones_like(pos_logit),
            reduction="mean") * self.pos_weight[0]

        bg_mask = torch.ones_like(obj_logits, dtype=torch.bool)
        bg_mask[0, 0, ck, cj, ci] = False
        bce_bg  = self.bce(obj_logits[bg_mask],
                           torch.zeros_like(obj_logits[bg_mask]))

        reg  = pred[:, 1:, ck:ck+1, cj:cj+1, ci:ci+1].view(1, 6)
        cx_p = (torch.sigmoid(reg[:, 0]) + ci.float()) / W
        cy_p = (torch.sigmoid(reg[:, 1]) + cj.float()) / H
        cz_p = (torch.sigmoid(reg[:, 2]) + ck.float()) / D
        sx_p = torch.exp(torch.clamp(reg[:, 3], -6., 6.)) / W
        sy_p = torch.exp(torch.clamp(reg[:, 4], -6., 6.)) / H
        sz_p = torch.exp(torch.clamp(reg[:, 5], -6., 6.)) / D
        pn   = torch.stack([cx_p, cy_p, cz_p, sx_p, sy_p, sz_p], -1).squeeze(0)

        l1_w = 0.1 if float(max(sx, sy, sz)) > self.large_box_thresh else 1.0
        l1   = self.l1(pn, gt_norm) * l1_w

        scale    = gt_norm.new_tensor([W, H, D, W, H, D])
        gt_abs   = box_cxcycz_whd_to_xyzxyz((gt_norm * scale).unsqueeze(0))
        pred_abs = box_cxcycz_whd_to_xyzxyz((pn * scale).unsqueeze(0))
        giou     = (1.0 - box_giou_3d(pred_abs, gt_abs)).mean()
        giou_w   = min(1.0, self._epoch / max(1, self.giou_warmup_epochs))

        return {"bce_pos": bce_pos, "bce_bg": bce_bg,
                "giou": giou * giou_w, "l1": l1}

    def _suppress_loss(self,
                       outputs: Dict[str, torch.Tensor],
                       levels: List[str]) -> torch.Tensor:
        total = outputs[levels[0]].new_tensor(0.)
        for k in levels:
            total = total + self.bce(outputs[k][:, 0:1],
                                     torch.zeros_like(outputs[k][:, 0:1]))
        return total / len(levels)

    # -- Main forward ------------------------------------------------------

    def forward(self,
                outputs: Dict[str, torch.Tensor],
                gt_mask: torch.Tensor,
                box_single: torch.Tensor,
                train_role: str) -> Tuple[torch.Tensor, Dict]:

        device     = outputs["seg"].device
        pred_mask  = outputs["seg"]
        has_lesion = train_role in ("positive_single", "positive_multi",
                                    "embolic")
        gt_mask    = gt_mask.to(device)

        comps = {}

        # ── Stage 1: segmentation only ────────────────────────────────────
        if self.stage == 1:
            l_seg = self._seg_loss(pred_mask, gt_mask, has_lesion)
            comps["seg"] = l_seg.detach()
            loss = l_seg
            comps["total"] = loss.detach()
            if not self.return_components:
                return loss, {}
            return loss, comps

        # ── Stage 2: detection + aux seg + cls ────────────────────────────
        levels      = ["P2", "P3", "P4", "P5"]
        fmap_shapes = []
        for k in levels:
            _, _, D, H, W = outputs[k].shape
            fmap_shapes.append((D, H, W))

        box_single  = box_single.to(device)
        cls_logits  = outputs["cls_logits"]
        cls_target  = torch.tensor(
            [ROLE_TO_CLS.get(train_role, CLS_NEGATIVE)],
            device=device, dtype=torch.long)

        # Segmentation auxiliary loss
        l_seg = self._seg_loss(pred_mask, gt_mask, has_lesion)
        comps["seg"] = l_seg.detach()

        # Classification loss
        l_cls = self.cls_ce(cls_logits, cls_target)
        comps["cls"] = l_cls.detach()

        # Detection loss
        l_det = box_single.new_tensor(0.)

        if train_role in ("positive_single", "positive_multi"):
            dc    = self._det_loss_one(outputs, box_single,
                                       fmap_shapes, levels)
            l_det = (dc["bce_pos"]
                     + self.lambda_obj_bg * dc["bce_bg"]
                     + self.lambda_giou   * dc["giou"]
                     + self.lambda_l1     * dc["l1"])
            comps.update({f"det_{k}": v.detach() for k, v in dc.items()})

        elif train_role in ("embolic", "negative"):
            l_det = self._suppress_loss(outputs, levels)
            comps["suppress"] = l_det.detach()

        # Stage 3: presence loss
        l_presence = box_single.new_tensor(0.)
        if self.stage == 3 and "presence_logit" in outputs:
            presence_logit = outputs["presence_logit"]           # (B,1)
            # Label: 1 if has lesion (positive/embolic), 0 if negative
            has_lesion_lbl = torch.tensor(
                [[1.0]] if train_role in
                ("positive_single","positive_multi","embolic")
                else [[0.0]],
                device=device)
            l_presence = self.presence_bce(presence_logit,
                                           has_lesion_lbl)
            comps["presence"] = l_presence.detach()

        loss = (l_det
                + self.lambda_seg      * l_seg
                + self.lambda_cls      * l_cls
                + self.lambda_presence * l_presence)

        comps["total"] = loss.detach()
        if not self.return_components:
            return loss, {}
        return loss, comps


# =============================================================================
# Inference utility
# =============================================================================

def _clamp_boxes(boxes: torch.Tensor,
                 input_shape: Tuple) -> torch.Tensor:
    W, H, D = input_shape
    mins = torch.maximum(boxes[..., :3], boxes.new_tensor([0., 0., 0.]))
    maxs = torch.minimum(boxes[..., 3:],
                         boxes.new_tensor([float(W-1),
                                           float(H-1),
                                           float(D-1)]))
    return torch.cat([mins, maxs], dim=-1)


@torch.no_grad()
def predict(model: StrokeDetector,
            x: torch.Tensor,
            iou_thresh: float = 0.3,
            topk: int = 1,
            seg_thresh: float = 0.5,
            return_seg: bool = False) -> Dict:
    """
    Run inference on a single volume.

    Parameters
    ----------
    model      : StrokeDetector in stage 2
    x          : (1, 2, D, H, W) tensor
    iou_thresh : NMS IoU threshold
    topk       : max boxes to return
    seg_thresh : threshold for binary mask
    return_seg : if True, include predicted mask in output

    Returns
    -------
    dict with keys:
        has_lesion  : bool
        confidence  : float  (max objectness score)
        center      : (cx, cy, cz) in voxel coords
        box         : (x0, y0, z0, x1, y1, z1) voxel coords
        subtype     : int  0=focal 1=multi 2=embolic 3=negative
        subtype_str : str
        seg_mask    : (1,1,D,H,W) optional
    """
    assert model.stage in (2, 3), "predict() requires stage=2 or stage=3 model"
    model.eval()
    outputs       = model(x)
    boxes, scores = model.decode(outputs)
    boxes         = _clamp_boxes(boxes[0], model.input_shape)  # (N, 6)
    scores        = scores[0]                                    # (N,)

    subtype_names = {0: "focal", 1: "multi", 2: "embolic", 3: "negative"}
    subtype       = int(outputs["cls_logits"][0].argmax().item())

    keep = nms_3d(boxes, scores, iou_thresh)
    if not keep:
        keep = [int(scores.argmax().item())]
    keep = keep[:topk]

    best_box   = boxes[keep[0]]                    # (6,) xyzxyz
    best_score = float(scores[keep[0]].item())

    # Center from box
    cx = float((best_box[0] + best_box[3]) / 2)
    cy = float((best_box[1] + best_box[4]) / 2)
    cz = float((best_box[2] + best_box[5]) / 2)

    # Stage 3: use presence head for has_lesion (fixes Specificity)
    # Stage 2: fall back to objectness score threshold
    if model.stage == 3 and "presence_logit" in outputs:
        presence_prob = float(torch.sigmoid(
            outputs["presence_logit"][0]).item())
        has_lesion = presence_prob > 0.5
    else:
        presence_prob = best_score
        has_lesion    = best_score > 0.3

    result = {
        "has_lesion":    has_lesion,
        "presence_conf": round(presence_prob, 4),
        "confidence":    best_score,
        "center":        (cx, cy, cz),
        "box":           tuple(best_box.tolist()),
        "subtype":       subtype,
        "subtype_str":   subtype_names.get(subtype, "unknown"),
    }

    if return_seg:
        seg_binary = (torch.sigmoid(outputs["seg"]) > seg_thresh).float()
        result["seg_mask"] = seg_binary.cpu()

    return result


# =============================================================================
# Factory
# =============================================================================

def build_model(in_ch: int = 2,
                base_channels: Tuple = (16, 24, 32, 48),
                fpn_channels: int = 32,
                input_shape: Tuple = (96, 96, 64),
                num_classes: int = 4,
                cls_hidden: int = 64,
                stage: int = 1) -> StrokeDetector:
    """Build model for given stage (1=seg pretrain, 2=det finetune, 3=presence)."""
    return StrokeDetector(
        in_ch         = in_ch,
        base_channels = base_channels,
        fpn_channels  = fpn_channels,
        input_shape   = input_shape,
        num_classes   = num_classes,
        cls_hidden    = cls_hidden,
        stage         = stage,
    )


# =============================================================================
# Quick sanity check
# =============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for stage in [1, 2, 3]:
        print(f"\n=== Stage {stage} ===")
        model = build_model(stage=stage).to(device)
        total = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total/1e6:.3f}M")

        x    = torch.randn(1, 2, 64, 96, 96, device=device)
        mask = torch.randint(0, 2, (1, 1, 64, 96, 96),
                             dtype=torch.float32, device=device)
        out  = model(x)
        print("Output keys:", list(out.keys()))
        print("seg shape:", out["seg"].shape)

        loss_fn = StrokeLoss(stage=stage)
        loss_fn.set_epoch(10)

        box = torch.tensor([0.5, 0.5, 0.5, 0.2, 0.2, 0.15], device=device)

        for role in ["positive_single", "embolic", "negative"]:
            model.zero_grad()
            out2 = model(x)
            loss, comps = loss_fn(out2, mask, box, role)
            print(f"  {role:<20} loss={loss.item():.4f}  "
                  f"seg={comps.get('seg', torch.tensor(0.)).item():.4f}")
            loss.backward()

    print("\nAll checks passed.")