# Attribution and scope

- DFormerV2: https://github.com/VCIP-RGBD/DFormer, paper https://arxiv.org/abs/2504.04701.
  `vendor/dformerv2_backbone.py` and `vendor/mlp_decoder.py` derive from that project.
  The upstream MIT license is retained verbatim in `vendor/LICENSE`; original encoder attribution is retained.
  Local import replacements use `vendor/torch_compat.py` instead of the upstream utility packages.
  The reconstruction reuses the upstream attention-prior equations, with normalized thermal intensity as the auxiliary input.
- MFNet: https://github.com/haqishen/MFNet-pytorch. Obtain images, labels, and splits from the upstream source under its terms.
  The dataset is not redistributed here. Included PNGs are this project's predicted class-ID masks, not source photographs or ground-truth labels.
- DejaVu fonts: redistributed unmodified for the plotting scripts, with their complete license at `fonts/LICENSE_DEJAVU`.

This is a 2026 public-data reconstruction by Qindong Gan illustrating an approach associated with his 2025 SERES internship.
It does not contain company data, original company code, or company-trained weights. The historical prior/fusion edits were not fully recovered.
DFormerV2 and its prior equations are upstream contributions, not a new architecture claimed by this project.
The local reconstruction adds the RGB-T data handling, task-specific model wrapper and training/evaluation pipeline, and independently retrained RGB-only control.
LLM tools assisted the reconstruction and release preparation; this repository is not evidence of unaided implementation.
