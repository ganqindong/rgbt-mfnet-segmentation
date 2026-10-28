"""Scientific comparison panels from unchanged exported prediction masks."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from dataset import CLASS_NAMES
from render_demo import COLORS, font, overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", default="comparison/comparison.json")
    parser.add_argument("--rgbt", default="runs/demo_v1")
    parser.add_argument("--rgb-only", default="runs/rgb_only_v1")
    parser.add_argument("--data", default="data/MFNet")
    parser.add_argument("--output", default="figures")
    args = parser.parse_args()
    result = json.loads(Path(args.comparison).read_text())
    out, data = Path(args.output), Path(args.data)
    out.mkdir(parents=True, exist_ok=True)
    width, height = 2720, 1260
    canvas = Image.new("RGB", (width, height), "#F8FAFC")
    draw = ImageDraw.Draw(canvas)
    ink, muted = "#162235", "#546377"
    draw.rounded_rectangle((60, 40, 380, 82), radius=21, fill="#E0F2EF")
    draw.text((78, 49), "MULTIMODAL PERCEPTION", fill="#087F70", font=font(20, True))
    draw.text((60, 110), "What changes when thermal guides attention?", fill=ink, font=font(50, True))
    draw.text((63, 179), "DFormerV2-S  /  Matched 20-epoch pilot  /  MFNet public-data reconstruction (2026)", fill=muted, font=font(25))
    panel_w, panel_h, gap, left = 500, 375, 25, 60
    headers = ["RGB input", "Thermal input", "RGB-only control", "RGB + thermal prior", "Ground truth"]
    used = set()
    for row, record in enumerate(result["fixed_figure_examples"]):
        sid = record["id"]
        raw = np.array(Image.open(data / "images" / f"{sid}.png"))
        rgb, t = raw[:, :, :3], raw[:, :, 3].astype(float)
        gt = np.array(Image.open(data / "labels" / f"{sid}.png"))
        a = np.array(Image.open(Path(args.rgb_only) / "predictions" / f"{sid}.png"))
        b = np.array(Image.open(Path(args.rgbt) / "predictions" / f"{sid}.png"))
        used.update(np.unique(np.stack([a, b, gt])).tolist())
        lo, hi = np.percentile(t, [1, 99])
        t = np.uint8(np.clip((t - lo) / max(hi - lo, 1) * 255, 0, 255))
        panels = [Image.fromarray(rgb), Image.fromarray(t).convert("RGB"), overlay(rgb, a), overlay(rgb, b), overlay(rgb, gt)]
        y_header, y_image = 265 + row * 440, 305 + row * 440
        for column, (title, panel) in enumerate(zip(headers, panels)):
            x = left + column * (panel_w + gap)
            draw.text((x, y_header), title, fill=ink, font=font(24, True))
            canvas.paste(panel.resize((panel_w, panel_h), Image.Resampling.LANCZOS), (x, y_image))
            draw.rectangle((x, y_image, x + panel_w - 1, y_image + panel_h - 1), outline="#D8E0E9", width=2)
        draw.text((left + 10, y_image + panel_h - 40), f"{record['scene'].upper()}  /  MFNet {sid}",
                  fill="#FFFFFF", stroke_fill=ink, stroke_width=2, font=font(20, True))
    group = result["groups"]["all"]
    a = group["rgb_only"]["miou_9_classes"] * 100
    b = group["rgbt"]["miou_9_classes"] * 100
    d = group["delta_percentage_points"]["miou_9_classes"]
    draw.text((60, 1150), f"Full test set ({group['test_samples']} images): mIoU (9 classes)  {a:.2f}% RGB-only  /  {b:.2f}% RGB-T  /  {d:+.2f} pp",
              fill=ink, font=font(25, True))
    x, y = 65, 1201
    for cls in sorted(used - {0}):
        draw.rounded_rectangle((x, y, x + 20, y + 20), radius=4, fill=tuple(int(c) for c in COLORS[cls]))
        draw.text((x + 29, y - 2), CLASS_NAMES[cls], fill=muted, font=font(19))
        x += 29 + int(draw.textlength(CLASS_NAMES[cls], font=font(19))) + 31
    draw.text((60, 1237), "Selected qualitative examples; one seed per variant. Raw predictions; unlabeled pixels transparent. Thermal contrast adjusted for display.",
              fill=muted, font=font(17))
    canvas.save(out / "rgbt-mfnet-comparison.png", optimize=True)
    web = canvas.resize((2040, 945), Image.Resampling.LANCZOS)
    web.save(out / "rgbt-mfnet-comparison.webp", quality=91, method=6)
    print(json.dumps({"figure": str(out / "rgbt-mfnet-comparison.png"), "samples": [r["id"] for r in result["fixed_figure_examples"]]}))


if __name__ == "__main__":
    main()
