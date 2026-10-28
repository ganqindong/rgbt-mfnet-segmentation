"""Verify released files and recompute aggregate metrics without ML dependencies."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = ['Unlabeled', 'Car', 'Person', 'Bicycle', 'Curve', 'Car stop', 'Guardrail', 'Color cone', 'Bump']


def scores(matrix):
    assert len(matrix) == 9 and all(len(row) == 9 for row in matrix)
    assert all(isinstance(v, int) and v >= 0 for row in matrix for v in row)
    iou = []
    for k in range(9):
        denominator = sum(matrix[k]) + sum(row[k] for row in matrix) - matrix[k][k]
        iou.append(matrix[k][k] / denominator if denominator else None)
    mean = lambda values: sum(v for v in values if v is not None) / sum(v is not None for v in values)
    return {'miou_9_classes': mean(iou), 'foreground_miou_8_classes': mean(iou[1:]),
            'pixel_accuracy': sum(matrix[k][k] for k in range(9)) / sum(map(sum, matrix)),
            'class_iou': dict(zip(NAMES, iou))}


def close(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            close(actual[key], expected[key])
    elif actual is None or expected is None:
        assert actual is expected
    else:
        assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), (actual, expected)


def main():
    manifest = json.loads((ROOT / 'SOURCE_ARTIFACTS.json').read_text())
    for name, metadata in manifest.items():
        content = (ROOT / name).read_bytes()
        assert len(content) == metadata['bytes'], name
        assert hashlib.sha256(content).hexdigest() == metadata['sha256'], name
    comparison = json.loads((ROOT / 'output/comparison/comparison.json').read_text())
    histograms = json.loads((ROOT / 'output/comparison/confusion_matrices.json').read_text())
    for variant, groups in histograms.items():
        for group, matrix in groups.items():
            close(scores(matrix), comparison['groups'][group][variant])
        for i in range(9):
            for j in range(9):
                assert groups['all'][i][j] == groups['day'][i][j] + groups['night'][i][j]
        folder = 'run' if variant == 'rgbt' else 'rgb_only_run'
        report = json.loads((ROOT / 'output' / folder / 'report.json').read_text())
        close(scores(groups['all']), report['test'])
        assert report['completed_epochs'] == 20 and report['training_steps'] == 3920
    for group, data in comparison['groups'].items():
        for metric in ('miou_9_classes', 'foreground_miou_8_classes', 'pixel_accuracy'):
            close(100 * (data['rgbt'][metric] - data['rgb_only'][metric]), data['delta_percentage_points'][metric])
        assert data['test_samples'] == {'all': 393, 'day': 205, 'night': 188}[group]
    print(f'PASS: {len(manifest)} unchanged source/artifact hashes, all/day/night metrics, and matched-run reports.')


if __name__ == '__main__':
    main()
