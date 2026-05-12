# StrokeDetector

A lightweight three-stage deep learning pipeline for acute ischaemic stroke
localisation, subtype classification, and infarct volume estimation from
diffusion-weighted MRI (DWI) and ADC maps.

**Single inference pass produces five outputs:**
bounding box · presence flag · subtype · segmentation mask · infarct volume in ml

All outputs are in **native image space** — no co-registration required.

---

## Design rationale

StrokeDetector was explicitly designed for the 4 GB VRAM constraint typical of
entry-level clinical workstation GPUs, using GroupNorm instead of BatchNorm,
gradient accumulation to simulate larger batch sizes, and a 0.504M-parameter
architecture significantly smaller than transformer-based alternatives. The
pipeline uses DWI and ADC as the sole input modalities; FLAIR imaging, while
useful for chronic infarct characterisation, does not reliably show acute
ischaemia at stroke onset and was therefore excluded.

---

## Installation

Requires Python 3.9+ and PyTorch 2.0+.

```bash
git clone https://github.com/DebinaLaishram/stroke-detector.git
cd stroke-detector
pip install -e .
```

Verify installation:

```bash
python -c "import stroke_detector; print(stroke_detector.__version__)"
```

---

## Download the Stage 3 checkpoint

The trained Stage 3 checkpoint is required for inference and benchmark
reproduction. Download from the GitHub Releases page:

```bash
# Linux:
wget https://github.com/DebinaLaishram/stroke-detector/releases/download/v0.1.0/stage3_best.pt \
     -O checkpoints_stage3/best.pt

# macOS:
curl -L https://github.com/DebinaLaishram/stroke-detector/releases/download/v0.1.0/stage3_best.pt \
     -o checkpoints_stage3/best.pt
```

On Windows download manually from:
https://github.com/DebinaLaishram/stroke-detector/releases/tag/v0.1.0

---

## Running the tests

Tests run without GPU and without the ISLES-2022 dataset:

```bash
pytest tests/ -v
```

Expected: 27 passed in under 30 seconds.

---

## Quick start

```python
from stroke_detector.inference import run_inference

result = run_inference(
    dwi_path  = "path/to/dwi.nii.gz",
    adc_path  = "path/to/adc.nii.gz",
    ckpt_path = "checkpoints_stage3/best.pt",
    out_dir   = "output/",
    subject   = "sub-001",
)

print(result["has_lesion"])
print(result["subtype_str"])
print(result["stroke_volumes"]["stroke_volume_dwi_ml"])
```

---

## Command-line usage

### 3D DWI input

```bash
python scripts/localize_stroke.py \
    --dwi        path/to/dwi.nii.gz \
    --adc        path/to/adc.nii.gz \
    --checkpoint checkpoints_stage3/best.pt \
    --out_dir    output/ \
    --subject    sub-001
```

### 4D DWI input

```bash
python scripts/localize_stroke.py \
    --dwi4d      path/to/dwi_4d.nii.gz \
    --bval       path/to/dwi.bval \
    --adc        path/to/adc.nii.gz \
    --checkpoint checkpoints_stage3/best.pt \
    --out_dir    output/ \
    --subject    sub-001 \
    --target_b   1200
```

### All flags

| Flag | Description | Default |
|------|-------------|---------|
| `--dwi` | 3D DWI NIfTI path | — |
| `--dwi4d` | 4D DWI NIfTI path | — |
| `--bval` | bval sidecar file | — |
| `--adc` | ADC map NIfTI path | — |
| `--checkpoint` | Path to Stage 3 checkpoint | — |
| `--out_dir` | Output directory | — |
| `--subject` | Subject ID string | `subject` |
| `--target_b` | Target b-value for shell extraction | `1000` |
| `--score_thresh` | Objectness confidence threshold | `0.3` |
| `--seg_thresh` | Segmentation probability threshold | `0.5` |

---

## Output format

### JSON result file: `result.json`

```json
{
  "subject":       "sub-001",
  "has_lesion":    true,
  "confidence":    0.9998,
  "presence_conf": 0.8743,
  "subtype":       0,
  "subtype_str":   "focal",
  "center_model":  [48.0, 52.0, 31.0],
  "box_model":     [32.0, 38.0, 21.0, 64.0, 66.0, 41.0],
  "modalities":    "DWI+ADC",
  "seg_volume_ml": 14.2,
  "stroke_volumes": {
    "stroke_volume_dwi_ml":          14.2,
    "stroke_volume_adc_ml":          12.8,
    "stroke_volume_combined_ml":     13.5,
    "stroke_volume_intersection_ml": 11.9,
    "stroke_volume_union_ml":        15.1
  }
}
```

`subtype_str` values: `focal`, `multi`, `embolic`, `negative`. The
`subtype` field is the integer ID corresponding to that label
(0=focal, 1=multi, 2=embolic, 3=negative).

### NIfTI output files

| File | Description |
|------|-------------|
| `{subject}_seg_mask_native.nii.gz` | Lesion segmentation in native image space |
| `{subject}_bbox_mask_native.nii.gz` | Bounding box ROI in native image space |

**Native space** means the output NIfTI shares the exact shape, voxel size,
and affine of the input DWI. Open DWI and mask together directly in ITK-Snap
or FSLeyes — no registration required.

---

## Benchmark results

Evaluated on the 36-case held-out ISLES-2022 test set.

| Metric | Value |
|--------|-------|
| mAP@0.2-0.5 | 0.798 |
| AP@0.2 | 1.000 |
| AP@0.3 | 0.952 |
| AP@0.5 | 0.476 |
| Sensitivity | 0.905 |
| Specificity | 0.200 |
| Mean Dice (positive cases) | 0.634 |
| Mean Dice (all 36 cases) | 0.549 |
| F1 Score | 0.731 |
| TP / FN / FP / TN | 19 / 2 / 12 / 3 |
| Model parameters | 0.504M |
| Inference time | < 10 s (T400 4 GB GPU) |

---

## Reproducing benchmarks

Requires the ISLES-2022 dataset. Register and download from:
https://isles-22.grand-challenge.org/

Expected directory layout after extraction:

```
ISLES-2022/
├── sub-strokecase0001/
│   └── ses-0001/
│       └── dwi/
│           ├── sub-strokecase0001_ses-0001_dwi.nii.gz
│           └── sub-strokecase0001_ses-0001_adc.nii.gz
├── sub-strokecase0002/
│   └── ...
└── derivatives/
    └── sub-strokecase0001/
        └── ses-0001/
            └── sub-strokecase0001_ses-0001_msk.nii.gz
```

Generate the deterministic split (seed=2026):

```bash
python scripts/make_isles_final_split.py \
    --root     path/to/ISLES-2022 \
    --out_csv  path/to/ISLES-2022/isles_final_split.csv \
    --seed     2026
```

Run evaluation:

```bash
python scripts/evaluate_primary.py \
    --checkpoint checkpoints_stage3/best.pt \
    --split_csv  path/to/ISLES-2022/isles_final_split.csv \
    --root       path/to/ISLES-2022 \
    --out_dir    results/primary/
```

Expected output: mAP@0.2-0.5 = 0.798, Sensitivity = 0.905, Specificity = 0.200

---

## Training from scratch

Training proceeds in three sequential stages. Each stage loads the
previous stage checkpoint.

```bash
# Stage 1 — segmentation pretraining
python scripts/train_stroke.py --config configs/stage1.yaml

# Stage 2 — detection fine-tuning
python scripts/train_stroke.py --config configs/stage2.yaml

# Stage 3 — presence head
python scripts/train_stroke.py --config configs/stage3.yaml
```

Edit the `root`, `split_csv`, and `output_dir` fields in each YAML config
to match your local paths before running.

MLflow tracking is enabled by default. Run `mlflow ui` to inspect
training curves.

---

## Synthetic b-value utility

This utility is for research on model behaviour under b-value protocol
variation. It is not a validated generalisation capability.

The pipeline is trained on ISLES-2022 b=1000 DWI only. Clinical protocols commonly use b=1200 or higher — this utility helps researchers evaluate how the model behaves under that shift. Preliminary results
show that sensitivity approaches 1.0 while specificity drops toward 0.0 at
higher synthetic b-values, consistent with increased signal contrast
reducing both missed lesions and discriminability of negative cases.

```bash
# Step 1 — generate synthetic DWI at b=1500, 2000, 2500
python scripts/generate_synthetic_b.py \
    --split_csv path/to/ISLES-2022/isles_final_split.csv \
    --root      path/to/ISLES-2022 \
    --out_dir   data/synthetic_b/ \
    --b_targets 1500 2000 2500

# Step 2 — evaluate
python scripts/evaluate_synthetic_bvalue.py \
    --checkpoint checkpoints_stage3/best.pt \
    --split_csv  path/to/ISLES-2022/isles_final_split.csv \
    --root       path/to/ISLES-2022 \
    --synth_dir  data/synthetic_b/ \
    --out_dir    results/synthetic_b/ \
    --b_values   1000 1500 2000 2500
```

---

## Limitations

- **Specificity (0.200):** The pipeline has a high false positive rate on
  negative cases. Detection heads fire on any restricted diffusion pattern
  including T2 shine-through and susceptibility artefacts. All results
  require radiologist review.
- **Training data:** Trained on ISLES-2022 (175 cases, b=1000 only).
  Performance on other protocols, field strengths, or scanner manufacturers
  has not been systematically validated.
- **Subtype imbalance:** Only 8 training cases for the multi-territorial
  subtype. Classification performance for this subtype is limited.
- **No external validation:** Benchmark numbers are from internal
  ISLES-2022 evaluation only.

---

## Issues and support

Bug reports and feature requests are welcome via
[GitHub Issues](https://github.com/DebinaLaishram/stroke-detector/issues).

---

## Citation

If you use StrokeDetector in your research please cite:

```bibtex
@software{strokedetector,
  author  = {Laishram, Debina},
  title   = {StrokeDetector: A lightweight multi-output deep learning
             pipeline for ischaemic stroke analysis on diffusion-weighted MRI},
  year    = {2026},
  url     = {https://github.com/DebinaLaishram/stroke-detector},
  version = {0.1.0}
}
```

---

## License

MIT License. See [LICENSE.txt](LICENSE.txt) for full text.
