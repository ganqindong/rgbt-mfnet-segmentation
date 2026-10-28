# Public release validation

Release preparation date: 2026-10-28.

## Checks performed

- `python3 verify_results.py`: verified SHA-256 and byte counts for 814 unchanged source/artifact files; recomputed all/day/night metrics from retained confusion matrices; checked the matched run reports and metric deltas.
- `python -m unittest discover -s tests -v`: three tests passed, covering all 786 predicted masks and their report IDs, four-channel input/label handling (including constant thermal and invalid labels), and CPU forward behavior with synthetic data.
- `compare_runs.py` was run against the 393 original public MFNet test labels retained locally. The resulting comparison JSON and confusion matrices exactly matched the included reference JSON. These dataset labels are not shipped in this repository.
- Documentation local-file links were checked. Public text was scanned for machine-local paths, server addresses, and common credential patterns; none were found.
- Copied model/data/training source files were not changed during release preparation; their hashes remain in `SOURCE_ARTIFACTS.json`.

CPU checks used the existing Python 3.11 environment with PyTorch 2.13.0, NumPy 1.26.4, and Pillow 12.3.0. This was not a fresh dependency installation. The original recorded training environment was PyTorch 2.3.0 / CUDA 12.1, NumPy 1.26.4, and Pillow 10.2.0. The upstream encoder emits a `torch.meshgrid` future warning in the CPU check; the check still passes.

## Checks not performed

No training, GPU smoke, full checkpoint inference, or new experiments were run for this release. The CPU model test uses random parameters, not the retained trained weights. Hash/metric reconstruction and exported-mask validation do not establish out-of-sample accuracy beyond the original single-seed pilot.

Source dataset images/labels, initialization/trained weights, server configuration, and private notes are not redistributed. Downloading data and obtaining the initialization checkpoint are prerequisites for training, checkpoint-based checks, and figures that require source images.
