"""Check the completed artifact and that the thermal input reaches the model."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from dataset import MFNetDataset
from model import RGBTSegmenter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/demo_v1")
    parser.add_argument("--data", default="data/MFNet")
    parser.add_argument("--checkpoint", help="Optional checkpoint outside the run directory")
    args = parser.parse_args()
    run = Path(args.run)
    report = json.loads((run / "report.json").read_text())
    variant = report["config"].get("variant", "rgbt")
    data = MFNetDataset(args.data, "test", tuple(report["model_input_resolution"]))
    prediction_ids = {p.stem for p in (run / "predictions").glob("*.png")}
    if prediction_ids != set(data.ids):
        raise ValueError("Prediction export does not exactly match the test split")
    for sid in data.ids:
        mask = np.array(Image.open(run / "predictions" / f"{sid}.png"))
        if mask.shape != (480, 640) or mask.min() < 0 or mask.max() > 8:
            raise ValueError(f"Invalid exported mask {sid}")
    torch.set_num_threads(2)
    model = RGBTSegmenter(variant).eval()
    checkpoint = torch.load(args.checkpoint or run / "best.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    rgb, thermal, _, sid = data[0]
    with torch.no_grad():
        actual = model(rgb[None], thermal[None])
        constant = model(rgb[None], torch.full_like(thermal[None], 0.5))
    delta = float((actual - constant).abs().mean())
    if not np.isfinite(delta) or (variant == "rgbt" and delta <= 1e-6) or (variant == "rgb-only" and delta != 0):
        raise ValueError("Thermal-input sanity check failed")
    result = {"variant": variant, "test_predictions_verified": len(prediction_ids), "all_masks_have_valid_shape_and_class_ids": True,
              "thermal_sanity_sample": sid, "mean_absolute_logit_change_when_thermal_is_constant": delta,
              "interpretation": ("Verifies that thermal input affects computation; does not establish an accuracy gain" if variant == "rgbt"
                                 else "Verifies that this retrained RGB-only control is insensitive to thermal input")}
    (run / "artifact_check.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
