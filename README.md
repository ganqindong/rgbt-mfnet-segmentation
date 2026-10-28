# RGB-Thermal Segmentation: MFNet Reconstruction

A public-data engineering reconstruction by Qindong Gan, using a DFormerV2-S encoder with thermal-guided attention and a matched, independently trained RGB-only control.

**Scope:** this is the 2026 MFNet reconstruction shown on [the project website](https://ganqindong.github.io/projects/rgb-thermal.html). It illustrates an approach associated with a 2025 SERES internship. It is not the original company implementation, historical company results, or a deployed autonomous-driving system. The exact historical attention-prior/fusion edits were not recovered. No company data are included.

## Read the code

1. `dataset.py`: four-channel MFNet input, synchronized augmentation, label handling, and thermal normalization.
2. `model.py`: task wrapper, pretrained encoder loading, multiscale decoder, and constant-auxiliary RGB-only control.
3. `vendor/dformerv2_backbone.py` and `vendor/mlp_decoder.py`: attributed upstream DFormer components; see `THIRD_PARTY_NOTICES.md`.
4. `train_demo.py`: bounded CUDA fine-tuning, validation-based checkpoint selection, and held-out mask export.
5. `compare_runs.py`: group-level confusion matrices and paired result summaries from exported masks.

The upstream depth-conditioned attention-prior equations receive normalized thermal intensities. Thermal intensity is not metric depth or calibrated temperature. This reconstruction does not claim a new attention formula or superiority over other RGB-T fusion methods.

## Included results

Both variants were trained independently for 20 epochs and 3,920 steps, with one shared initialization seed and the same architecture, input resolution, data splits, optimizer, losses, and schedule. Each selected its checkpoint independently by validation foreground mIoU. RGB-only replaces the auxiliary map with a constant during both training and inference; the spatial prior remains.

| Evaluation | RGB-only | RGB-T | Gain (percentage points) |
| --- | ---: | ---: | ---: |
| All 393 test images: 9-class mIoU | 49.89% | 51.32% | +1.43 |
| All 393: foreground 8-class mIoU | 43.97% | 45.56% | +1.59 |
| All 393: Person IoU | 57.35% | 63.06% | +5.71 |
| Nighttime 188: Person IoU | 48.90% | 58.20% | +9.30 |

Sources: `output/comparison/comparison.json`, `metrics.csv`, and `confusion_matrices.json`.
Nine-class mIoU includes Unlabeled; foreground mIoU excludes it. Aggregate metrics use global pixel confusion matrices, not averages of image-level IoUs. RGB-T Guardrail IoU remained low (4.40%); Color cone IoU declined from 47.33% to 46.87%.

These are descriptive, single-seed pilot results. cuDNN benchmark was enabled; bitwise determinism is not guaranteed. The fixed qualitative examples were selected for RGB-T quality before the control finished, not sampled randomly. They do not establish robustness, statistical significance, general superiority, or performance of the original internship system.

## Quick checks: no training, data download, or model weights

From the repository root, Python 3.10+ can verify hashes and recompute the aggregate metrics using only the standard library:

```bash
python3 verify_results.py
```

This checks the unchanged source/artifact manifest, recomputes metrics from stored confusion matrices, and compares them with the published summaries and run reports. It does not rerun network inference or independently verify the ground-truth pixels.

For CPU input/model checks and validation of all exported mask shapes and class IDs:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The model check uses random parameters and synthetic inputs. It verifies finite output and the intended dependence/independence on thermal input; it does not reproduce the trained checkpoints' accuracy. The recorded original training environment was PyTorch 2.3.0 / CUDA 12.1, NumPy 1.26.4, and Pillow 10.2.0; `requirements.txt` gives compatible ranges, not an exact historical environment lock.

## Recompute metrics from the existing masks

Obtain MFNet from [its upstream source](https://github.com/haqishen/MFNet-pytorch), under its terms. Place `images/`, `labels/`, `train.txt`, `val.txt`, and `test.txt` under `data/MFNet/`. Input PNGs must retain their original four channels: RGB channels 0–2, thermal channel 3. Do not alpha-blend the thermal channel. Labels must be indexed masks with IDs 0–8.

The code deduplicates original IDs, strips `_flip`, and performs joint flipping on the fly. The retained split has 784 training, 392 validation, and 393 test images. For mask-only metric reconstruction, `compare_runs.py` needs `test.txt` and the corresponding labels, not the source images:

```bash
python compare_runs.py --data data/MFNet \
  --rgbt output/run --rgb-only output/rgb_only_run \
  --output analysis_out/comparison
```

`output/run/predictions/` and `output/rgb_only_run/predictions/` contain all 786 exported predicted class-ID masks. They are model outputs, not ground-truth labels. Existing reports/configurations and histories are preserved. `SOURCE_ARTIFACTS.json` records their hashes and the unchanged source files; `provenance.json` retains input hashes while omitting machine-local paths.

## Training and checkpoint-based checks

The original `train_demo.py` requires a suitable NVIDIA CUDA GPU and bfloat16 autocast support. It is not a CPU trainer. Review resource availability and choose a **new output directory** before running it; this retained script does not protect an existing run directory from overwrite. A local `STOP_REQUESTED` file requests this job to yield. Site-specific server/container launchers are not included.

Obtain `DFormerv2_Small_NYU.pth` via the [DFormer upstream checkpoint instructions](https://github.com/VCIP-RGBD/DFormer), and place it under `weights/`. The retained initialization file had SHA-256 `2efc0267818ea358f795c253924464695742d60812a5f489b9fc8ea5441f47e2`. The loader requires all encoder tensors after matching names and shapes. Neither this initialization nor the trained weights are redistributed here.

Optional eight-step GPU smoke (loads the dataset and encoder; performs a small amount of training):

```bash
python train_demo.py --mode smoke --variant rgbt \
  --root data/MFNet --encoder weights/DFormerv2_Small_NYU.pth \
  --output runs/new_smoke
```

Full pilot training uses separate output directories for the independent models:

```bash
python train_demo.py --mode train --variant rgbt \
  --root data/MFNet --encoder weights/DFormerv2_Small_NYU.pth \
  --output runs/new_rgbt
python train_demo.py --mode train --variant rgb-only \
  --root data/MFNet --encoder weights/DFormerv2_Small_NYU.pth \
  --output runs/new_rgb_only
```

Training is bounded by 20 epochs or 1,500 seconds by default. The retained runs reached 20 epochs; a slower machine may hit the time limit first. New training does not guarantee the same numbers. Replacing thermal at inference in an RGB-T model is not a substitute for independently training the RGB-only control.

`check_result.py` requires an actual trained checkpoint, the test images/labels, and the corresponding run report/masks. `render_demo.py` and `render_comparison.py` require MFNet source images/labels plus the exported masks; bundled DejaVu fonts retain their license. Run `--help` for their path arguments. No model checkpoint is needed for plotting existing masks.

## Validation and omissions

Release preparation checks are recorded in `VALIDATION.md`. Training and full checkpoint-based inference were not rerun for this release. Original weights remain outside this repository. Dataset images, labels, private work notes, server addresses, and the original company code are excluded.

## Sources and attribution

- DFormerV2: [paper](https://arxiv.org/abs/2504.04701), [upstream code](https://github.com/VCIP-RGBD/DFormer).
- MFNet: [dataset and original implementation](https://github.com/haqishen/MFNet-pytorch).
- [Project description and technical report](https://ganqindong.github.io/projects/rgb-thermal.html).

Qindong Gan's reconstruction covers task data handling, model wrapping, training/evaluation integration, and the matched control. LLM tools assisted the reconstruction and release preparation. See `THIRD_PARTY_NOTICES.md` and the retained licenses for upstream contributions. The project is an engineering demonstration, not a peer-reviewed publication.
