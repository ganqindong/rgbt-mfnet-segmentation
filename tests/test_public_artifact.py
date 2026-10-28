import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from dataset import MFNetDataset, original_ids
from model import RGBTSegmenter

ROOT = Path(__file__).resolve().parents[1]


class PublicArtifactTests(unittest.TestCase):
    def test_all_prediction_masks_match_report_ids_and_classes(self):
        for folder in ('run', 'rgb_only_run'):
            run = ROOT / 'output' / folder
            report = json.loads((run / 'report.json').read_text())
            expected = {row['id'] for row in report['per_image']}
            masks = list((run / 'predictions').glob('*.png'))
            self.assertEqual(len(masks), 393)
            self.assertEqual({p.stem for p in masks}, expected)
            for path in masks:
                mask = np.asarray(Image.open(path))
                self.assertEqual(mask.shape, (480, 640))
                self.assertGreaterEqual(int(mask.min()), 0)
                self.assertLessEqual(int(mask.max()), 8)

    def test_input_channels_constant_thermal_and_invalid_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'images').mkdir()
            (root / 'labels').mkdir()
            (root / 'test.txt').write_text('sample\nsample_flip\n')
            raw = np.zeros((480, 640, 4), dtype=np.uint8)
            raw[:, :, 0] = 255
            raw[:, :, 3] = 127
            Image.fromarray(raw, mode='RGBA').save(root / 'images/sample.png')
            Image.fromarray(np.full((480, 640), 2, np.uint8)).save(root / 'labels/sample.png')
            self.assertEqual(original_ids(root, 'test'), ['sample'])
            rgb, thermal, label, _ = MFNetDataset(root, 'test', size=(64, 64))[0]
            self.assertEqual(tuple(rgb.shape), (3, 64, 64))
            self.assertTrue(torch.isfinite(rgb).all())
            self.assertEqual(float(thermal.abs().max()), 0.0)
            self.assertTrue((label == 2).all())
            Image.fromarray(np.full((480, 640), 9, np.uint8)).save(root / 'labels/sample.png')
            with self.assertRaises(ValueError):
                MFNetDataset(root, 'test', size=(64, 64))[0]

    def test_model_thermal_dependence_and_rgb_only_invariance(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        model = RGBTSegmenter('rgbt').eval()
        rgb, thermal = torch.randn(1, 3, 64, 64), torch.rand(1, 1, 64, 64)
        with torch.no_grad():
            actual = model(rgb, thermal)
            constant = model(rgb, torch.full_like(thermal, 0.5))
            self.assertEqual(tuple(actual.shape), (1, 9, 64, 64))
            self.assertTrue(torch.isfinite(actual).all())
            self.assertGreater(float((actual - constant).abs().max()), 0.0)
            model.variant = 'rgb-only'
            self.assertTrue(torch.equal(model(rgb, thermal), model(rgb, torch.zeros_like(thermal))))


if __name__ == '__main__':
    unittest.main()
