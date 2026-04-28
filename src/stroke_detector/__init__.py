"""
stroke_detector
===============
Lightweight three-stage deep learning pipeline for acute ischaemic stroke
localisation, subtype classification, and infarct volume estimation from
diffusion-weighted MRI.

Modules
-------
model     : StrokeDetector architecture (backbone, FPN, detection heads)
data      : StrokeDataset, preprocessing, augmentation
inference : run_inference() for single-case inference

Quick start
-----------
>>> from stroke_detector.inference import run_inference
>>> result = run_inference(
...     dwi_path  = "dwi.nii.gz",
...     adc_path  = "adc.nii.gz",
...     ckpt_path = "checkpoint_stage3.pt",
...     out_dir   = "output/",
...     subject   = "sub-001",
... )
"""

__version__ = "0.1.0"
__author__  = "Debina Laishram"