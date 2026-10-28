"""Bounded single-GPU MFNet fine-tuning and held-out prediction export."""
import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, MFNetDataset, verify_dataset
from model import RGBTSegmenter


class GPUYieldRequested(Exception):
    pass


def emit(kind, **data):
    print(json.dumps({"event": kind, "utc": datetime.now(timezone.utc).isoformat(), **data}), flush=True)


def score_hist(hist):
    denominator = hist.sum(0) + hist.sum(1) - hist.diagonal()
    iou = np.divide(hist.diagonal(), denominator, out=np.full(9, np.nan), where=denominator > 0)
    return {"miou_9_classes": float(np.nanmean(iou)),
            "foreground_miou_8_classes": float(np.nanmean(iou[1:])),
            "pixel_accuracy": float(hist.trace() / max(hist.sum(), 1)),
            "class_iou": {name: (float(v) if np.isfinite(v) else None) for name, v in zip(CLASS_NAMES, iou)}}


@torch.no_grad()
def evaluate(model, loader, device, save_root=None):
    model.eval()
    hist = np.zeros((9, 9), dtype=np.int64)
    rows = []
    for rgb, thermal, gt, ids in loader:
        if Path("STOP_REQUESTED").exists():
            raise GPUYieldRequested(Path("STOP_REQUESTED").read_text())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(rgb.to(device), thermal.to(device))
        if save_root:
            # Export at original image resolution, with unmodified class argmax.
            logits = F.interpolate(logits.float(), size=(480, 640), mode="bilinear", align_corners=False)
            labels = np.stack([np.array(Image.open(loader.dataset.root / "labels" / f"{sid}.png")) for sid in ids])
        else:
            labels = gt.numpy()
        predictions = logits.argmax(1).cpu().numpy()
        for prediction, label, sid in zip(predictions, labels, ids):
            h = np.bincount((9 * label + prediction).ravel(), minlength=81).reshape(9, 9)
            hist += h
            if save_root:
                Image.fromarray(prediction.astype(np.uint8)).save(save_root / f"{sid}.png")
                rows.append({"id": sid, "split": "test", "scene": "night" if sid.endswith("N") else "day",
                             **score_hist(h), "foreground_pixels": int((label > 0).sum()),
                             "person_pixels": int((label == 2).sum()), "car_pixels": int((label == 1).sum())})
    return score_hist(hist), rows


def loss_fn(logits, gt, weights):
    ce = F.cross_entropy(logits.float(), gt, weight=weights)
    probabilities = logits.float().softmax(1)
    target = F.one_hot(gt, 9).permute(0, 3, 1, 2).float()
    intersection = (probabilities * target).sum((0, 2, 3))
    denominator = probabilities.sum((0, 2, 3)) + target.sum((0, 2, 3))
    present = target.sum((0, 2, 3)) > 0
    present[0] = False
    dice = (1 - (2 * intersection + 1) / (denominator + 1))[present].mean() if present.any() else ce * 0
    return ce + 0.3 * dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "train", "export"), default="smoke")
    parser.add_argument("--variant", choices=("rgbt", "rgb-only"), default="rgbt")
    parser.add_argument("--root", default="data/MFNet")
    parser.add_argument("--encoder", default="weights/DFormerv2_Small_NYU.pth")
    parser.add_argument("--output", default="runs/demo_v1")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-seconds", type=int, default=1500)
    parser.add_argument("--run-tag", default="rgbt20261006")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "container_pid.txt").write_text(str(os.getpid()))
    torch.set_num_threads(2)
    torch.manual_seed(20261006)
    random.seed(20261006)
    np.random.seed(20261006)
    torch.cuda.set_per_process_memory_fraction(0.72, 0)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    dataset_info = verify_dataset(args.root)
    (out / "dataset_check.json").write_text(json.dumps(dataset_info, indent=2))
    model = RGBTSegmenter(args.variant)
    loading = model.load_encoder(args.encoder)
    model.to(device)
    emit("initialized", variant=args.variant, loading=loading, dataset=dataset_info["split_counts"],
         train_size=[args.height, args.width], batch_size=args.batch_size,
         gpu=torch.cuda.get_device_name(), parameters=sum(p.numel() for p in model.parameters()))
    train_set = MFNetDataset(args.root, "train", (args.height, args.width), augment=True)
    val_set = MFNetDataset(args.root, "val", (args.height, args.width))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=2, pin_memory=True)
    counts = np.array(dataset_info["counts"], dtype=np.float64)
    weights = 1 / np.sqrt(np.maximum(counts / counts.sum(), 1e-6))
    weights = np.clip(weights / np.median(weights[1:]), 0.1, 3)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 3e-5},
        {"params": model.decode_head.parameters(), "lr": 8e-4},
    ], weight_decay=0.01)
    initial_lrs = [g["lr"] for g in optimizer.param_groups]
    stop_file = Path("STOP_REQUESTED")
    started = time.monotonic()
    best_score, steps, completed_epochs, stop_reason = -1.0, 0, 0, None

    def save(name, epoch, metrics):
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics,
                    "config": vars(args), "dataset": dataset_info, "reconstruction_year": 2026}, out / name)

    if args.mode == "export":
        checkpoint = torch.load(out / "best.pth", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)
        completed_epochs = checkpoint["epoch"]
        best_score = checkpoint["metrics"]["foreground_miou_8_classes"]
    else:
        epoch_count = 1 if args.mode == "smoke" else args.epochs
        for epoch in range(1, epoch_count + 1):
            # Brief decoder warm-up, then fine-tune the complete pretrained encoder.
            frozen = args.mode == "train" and epoch <= 2
            for parameter in model.backbone.parameters():
                parameter.requires_grad_(not frozen)
            model.train()
            losses = []
            epoch_start = time.monotonic()
            for rgb, thermal, gt, _ in train_loader:
                if stop_file.exists():
                    stop_reason = stop_file.read_text()
                    break
                if args.mode == "train" and time.monotonic() - started > args.max_seconds:
                    stop_reason = "Reached the bounded training time budget"
                    break
                fraction = min(steps / max(len(train_loader) * args.epochs, 1), 1)
                for group, lr in zip(optimizer.param_groups, initial_lrs):
                    group["lr"] = lr * (1 - fraction) ** 0.9
                rgb, thermal, gt = rgb.to(device), thermal.to(device), gt.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(rgb, thermal)
                    loss = loss_fn(logits, gt, weights)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                losses.append(float(loss.detach()))
                steps += 1
                if args.mode == "smoke" and steps >= 8:
                    break
                if steps % 50 == 0:
                    emit("progress", epoch=epoch, steps=steps, loss=float(np.mean(losses[-50:])),
                         elapsed_seconds=round(time.monotonic() - started, 1))
            torch.cuda.synchronize()
            if args.mode == "smoke":
                model.eval()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    actual = model(rgb[:1], thermal[:1]).float()
                    constant = model(rgb[:1], torch.full_like(thermal[:1], 0.5)).float()
                delta = float((actual - constant).abs().mean())
                if args.variant == "rgb-only" and delta != 0:
                    raise RuntimeError("RGB-only control still depends on thermal input")
                stats = {"steps": steps, "mean_loss": float(np.mean(losses)),
                         "seconds_per_step": (time.monotonic() - epoch_start) / max(steps, 1),
                         "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
                         "variant": args.variant, "thermal_perturbation_logit_delta": delta}
                (out / "smoke.json").write_text(json.dumps(stats, indent=2))
                emit("smoke_complete", **stats)
                return
            if losses:
                completed_epochs = epoch
            if stop_file.exists():
                save("last.pth", completed_epochs, {})
                emit("yielded_gpu", reason=stop_reason)
                return
            save("last.pth", completed_epochs, {})
            metrics, _ = evaluate(model, val_loader, device)
            emit("epoch_complete", epoch=epoch, loss=float(np.mean(losses)) if losses else None,
                 validation=metrics, elapsed_seconds=round(time.monotonic() - started, 1),
                 peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024**3)
            with (out / "history.jsonl").open("a") as handle:
                handle.write(json.dumps({"epoch": epoch, "validation": metrics,
                                         "loss": float(np.mean(losses)) if losses else None}) + "\n")
            if metrics["foreground_miou_8_classes"] > best_score:
                best_score = metrics["foreground_miou_8_classes"]
                save("best.pth", epoch, metrics)
            save("last.pth", completed_epochs, metrics)
            if stop_reason:
                break
        checkpoint = torch.load(out / "best.pth", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)
    predictions = out / "predictions"
    predictions.mkdir(exist_ok=True)
    test_loader = DataLoader(MFNetDataset(args.root, "test", (args.height, args.width)),
                             batch_size=args.batch_size, num_workers=2, pin_memory=True)
    metrics, rows = evaluate(model, test_loader, device, predictions)
    report = {"status": "completed", "config": vars(args), "completed_epochs": completed_epochs,
              "selected_epoch": checkpoint["epoch"], "validation_selection": "foreground mIoU on the validation split",
              "training_steps": steps, "elapsed_seconds": time.monotonic() - started,
              "stop_reason": stop_reason, "test_samples": len(test_loader.dataset),
              "test_resolution": [480, 640], "model_input_resolution": [args.height, args.width],
              "test": metrics, "per_image": rows,
              "scope": "2026 MFNet reconstruction, not archived SERES company results"}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    emit("export_complete", test=metrics, test_samples=len(rows), selected_epoch=checkpoint["epoch"])


if __name__ == "__main__":
    try:
        main()
    except GPUYieldRequested as error:
        emit("yielded_gpu", reason=str(error))
