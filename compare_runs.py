"""Aggregate exact exported masks for matched RGB-only / RGB-T test comparisons."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from dataset import CLASS_NAMES, original_ids


def scores(hist):
    den = hist.sum(0) + hist.sum(1) - hist.diagonal()
    iou = np.divide(hist.diagonal(), den, out=np.full(9, np.nan), where=den > 0)
    return {"miou_9_classes": float(np.nanmean(iou)),
            "foreground_miou_8_classes": float(np.nanmean(iou[1:])),
            "pixel_accuracy": float(hist.trace() / max(hist.sum(), 1)),
            "class_iou": {name: float(v) if np.isfinite(v) else None for name, v in zip(CLASS_NAMES, iou)}}


def deltas(a, b):
    # Percentage points, not relative percent improvement.
    result = {k: 100 * (b[k] - a[k]) for k in ("miou_9_classes", "foreground_miou_8_classes", "pixel_accuracy")}
    result["class_iou"] = {k: 100 * (b["class_iou"][k] - a["class_iou"][k])
                           if a["class_iou"][k] is not None and b["class_iou"][k] is not None else None
                           for k in CLASS_NAMES}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/MFNet")
    parser.add_argument("--rgbt", default="runs/demo_v1")
    parser.add_argument("--rgb-only", default="runs/rgb_only_v1")
    parser.add_argument("--output", default="comparison")
    args = parser.parse_args()
    data, out = Path(args.data), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"rgb_only": Path(args.rgb_only), "rgbt": Path(args.rgbt)}
    reports = {name: json.loads((path / "report.json").read_text()) for name, path in paths.items()}
    ignored = {"variant", "output", "run_tag"}
    configs = {name: {k: v for k, v in report["config"].items() if k not in ignored}
               for name, report in reports.items()}
    if configs["rgb_only"] != configs["rgbt"]:
        raise ValueError("Training/evaluation command configurations are not matched")
    if reports["rgb_only"]["config"].get("variant") != "rgb-only":
        raise ValueError("Expected a retrained RGB-only control")
    if any(report["completed_epochs"] != 20 or report["training_steps"] != 3920 or report["stop_reason"] is not None
           for report in reports.values()):
        raise ValueError("One run did not complete the matched 20-epoch budget")
    ids = original_ids(data, "test")
    for path in paths.values():
        if {p.stem for p in (path / "predictions").glob("*.png")} != set(ids):
            raise ValueError("Exported predictions do not exactly match the test IDs")
    hist = {name: {scene: np.zeros((9, 9), dtype=np.int64) for scene in ("all", "day", "night")}
            for name in paths}
    rows, counts = [], {"all": len(ids), "day": 0, "night": 0}
    for sid in ids:
        scene = "night" if sid.endswith("N") else "day"
        counts[scene] += 1
        gt = np.array(Image.open(data / "labels" / f"{sid}.png"))
        record = {"id": sid, "scene": scene, "foreground_pixels": int((gt > 0).sum())}
        for name, path in paths.items():
            pred = np.array(Image.open(path / "predictions" / f"{sid}.png"))
            if pred.shape != (480, 640) or gt.shape != pred.shape or pred.max() > 8 or gt.max() > 8:
                raise ValueError(f"Invalid class mask: {name}/{sid}")
            h = np.bincount((9 * gt.astype(np.int64) + pred).ravel(), minlength=81).reshape(9, 9)
            hist[name]["all"] += h
            hist[name][scene] += h
            record[name] = scores(h)
        record["delta_percentage_points"] = deltas(record["rgb_only"], record["rgbt"])
        rows.append(record)
    groups = {}
    for scene in counts:
        a, b = scores(hist["rgb_only"][scene]), scores(hist["rgbt"][scene])
        groups[scene] = {"test_samples": counts[scene], "rgb_only": a, "rgbt": b,
                         "delta_percentage_points": deltas(a, b)}
    for name in paths:
        measured = groups["all"][name]
        if abs(measured["miou_9_classes"] - reports[name]["test"]["miou_9_classes"]) > 1e-12:
            raise ValueError("Mask recomputation differs from the original run report")
    # These IDs were fixed before the baseline finished; retain the original
    # RGB-T showcase selection rather than picking the largest baseline gains.
    fixed = ["01464D", "01335N"]
    result = {"scope": "2026 public MFNet pilot; not archived internship results",
              "comparison": "Retrained DFormerV2-S RGB-only control versus thermal-intensity attention prior",
              "matched_config": configs["rgbt"],
              "matching": {"same_architecture_and_parameter_shapes": True, "same_initial_encoder_checkpoint": True,
                           "seed": 20261006, "seeds_per_variant": 1,
                           "same_decoder_initialization_seed": True, "same_data_loss_optimizer_and_schedule": True,
                           "checkpoint_selection": "Each variant independently selects maximum validation foreground mIoU",
                           "cudnn_benchmark": True, "bitwise_determinism_guaranteed": False},
              "runs": {name: {"selected_epoch": report["selected_epoch"], "completed_epochs": report["completed_epochs"],
                              "training_steps": report["training_steps"], "elapsed_seconds": report["elapsed_seconds"]}
                       for name, report in reports.items()},
              "metric_definition": "Global pixel confusion matrix per group; mIoU9 includes Unlabeled. Foreground mIoU8 excludes it.",
              "groups": groups,
              "fixed_figure_examples": [next(row for row in rows if row["id"] == sid) for sid in fixed],
              "figure_selection": "Previously chosen high-quality RGB-T examples, fixed before RGB-only results; not random or representative of all test images",
              "limitations": ["One seed and a short 20-epoch budget; no statistical or general superiority claim",
                              "NYU RGB-D pretrained initialization in both variants",
                              "Isolates the thermal-input difference; does not compare thermal-prior fusion against another RGB-T fusion method",
                              "The missing historical prior/fusion edits were not recovered"],
              "per_image": rows}
    (out / "comparison.json").write_text(json.dumps(result, indent=2))
    (out / "confusion_matrices.json").write_text(json.dumps({name: {scene: h.tolist() for scene, h in groups_.items()}
                                                            for name, groups_ in hist.items()}, indent=2))
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "test_images", "metric", "rgb_only_percent", "rgbt_percent", "delta_percentage_points"])
        for scene, group in groups.items():
            for metric in ("miou_9_classes", "foreground_miou_8_classes", "pixel_accuracy"):
                writer.writerow([scene, group["test_samples"], metric, 100 * group["rgb_only"][metric],
                                 100 * group["rgbt"][metric], group["delta_percentage_points"][metric]])
            for name in CLASS_NAMES:
                a, b = group["rgb_only"]["class_iou"][name], group["rgbt"]["class_iou"][name]
                writer.writerow([scene, group["test_samples"], f"IoU_{name}",
                                 100 * a if a is not None else "", 100 * b if b is not None else "",
                                 group["delta_percentage_points"]["class_iou"][name]])
    print(json.dumps({"groups": groups, "runs": result["runs"]}, indent=2))


if __name__ == "__main__":
    main()
