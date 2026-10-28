"""Build a portfolio figure from real model predictions and public MFNet inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dataset import CLASS_NAMES

COLORS = np.array([[0, 0, 0], [56, 139, 253], [244, 108, 79], [166, 111, 240],
                   [29, 182, 154], [239, 178, 56], [102, 166, 89], [247, 133, 191], [151, 121, 90]], dtype=np.uint8)


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(Path(__file__).parent / "fonts" / name), size)


def overlay(rgb, mask):
    result = rgb.astype(np.float32).copy()
    foreground = mask > 0
    result[foreground] = 0.35 * result[foreground] + 0.65 * COLORS[mask[foreground]]
    return Image.fromarray(np.uint8(np.clip(result, 0, 255)))


def choose(rows, scene):
    candidates = [row for row in rows if row["scene"] == scene and row["car_pixels"] > 1000 and row["person_pixels"] > 200]
    if not candidates:
        candidates = [row for row in rows if row["scene"] == scene and row["foreground_pixels"] > 1000]
    def rank(row):
        iou = row["class_iou"]
        car, person = iou["Car"] or 0, iou["Person"] or 0
        return 0.45 * car + 0.4 * person + 0.15 * row["foreground_miou_8_classes"]
    return max(candidates, key=rank)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/demo_v1")
    parser.add_argument("--data", default="data/MFNet")
    parser.add_argument("--output", default="figures")
    parser.add_argument("--day")
    parser.add_argument("--night")
    args = parser.parse_args()
    run, data, out = Path(args.run), Path(args.data), Path(args.output)
    out.mkdir(exist_ok=True)
    report = json.loads((run / "report.json").read_text())
    selected = [next(row for row in report["per_image"] if row["id"] == sid) if sid else choose(report["per_image"], scene)
                for sid, scene in ((args.day, "day"), (args.night, "night"))]
    width, height = 2624, 1396
    canvas = Image.new("RGB", (width, height), "#F8FAFC")
    draw = ImageDraw.Draw(canvas)
    ink, secondary = "#162235", "#546377"
    draw.rounded_rectangle((66, 56, 390, 104), radius=24, fill="#E0F2EF")
    draw.text((87, 66), "MULTIMODAL PERCEPTION", fill="#087F70", font=font(20, True))
    draw.text((66, 132), "RGB + Thermal Semantic Segmentation", fill=ink, font=font(56, True))
    draw.text((69, 214), "DFormerV2-S  /  Thermal-guided attention  /  MFNet public-data reconstruction", fill=secondary, font=font(27))
    draw.text((2110, 70), "SELECTED TEST EXAMPLES", fill=secondary, font=font(20, True))
    panel_w, panel_h, gap, left = 600, 450, 28, 66
    headers = ["RGB input", "Thermal input", "Prediction overlay", "Ground truth overlay"]
    used_classes = set()
    for row_index, record in enumerate(selected):
        sid = record["id"]
        y_header = 307 + row_index * 507
        y_image = y_header + 43
        raw = np.array(Image.open(data / "images" / f"{sid}.png"))
        rgb, thermal = raw[:, :, :3], raw[:, :, 3].astype(float)
        label = np.array(Image.open(data / "labels" / f"{sid}.png"))
        prediction = np.array(Image.open(run / "predictions" / f"{sid}.png"))
        used_classes.update(np.unique(label).tolist())
        used_classes.update(np.unique(prediction).tolist())
        lo, hi = np.percentile(thermal, [1, 99])
        thermal = np.uint8(np.clip((thermal - lo) / max(hi - lo, 1) * 255, 0, 255))
        inputs = [Image.fromarray(rgb), Image.fromarray(thermal).convert("RGB"), overlay(rgb, prediction), overlay(rgb, label)]
        for column, (title, image) in enumerate(zip(headers, inputs)):
            x = left + column * (panel_w + gap)
            header = f"{record['scene'].upper()}  /  {title}" if column == 0 else title
            draw.text((x, y_header), header, fill=ink, font=font(24, True))
            canvas.paste(image.resize((panel_w, panel_h), Image.Resampling.LANCZOS), (x, y_image))
            draw.rectangle((x, y_image, x + panel_w - 1, y_image + panel_h - 1), outline="#D8E0E9", width=2)
        draw.text((left + 10, y_image + panel_h - 40), f"MFNet  {sid}", fill="#FFFFFF", stroke_fill="#162235", stroke_width=2, font=font(20, True))
    x, y = 68, 1323
    for cls in sorted(used_classes - {0}):
        color = tuple(int(c) for c in COLORS[cls])
        draw.rounded_rectangle((x, y, x + 22, y + 22), radius=5, fill=color)
        draw.text((x + 34, y - 2), CLASS_NAMES[cls], fill=secondary, font=font(20))
        x += 34 + int(draw.textlength(CLASS_NAMES[cls], font=font(20))) + 39
    draw.text((68, 1363), "Raw model predictions. Unlabeled pixels remain transparent. Thermal contrast adjusted for display. Reconstructed in 2026.",
              fill=secondary, font=font(18))
    canvas.save(out / "rgbt-mfnet-demo.png", optimize=True)
    web = canvas.resize((1968, 1047), Image.Resampling.LANCZOS)
    web.save(out / "rgbt-mfnet-demo.webp", quality=91, method=6)
    web.save(out / "rgbt-mfnet-demo.jpg", quality=93, optimize=True)
    (out / "selected_examples.json").write_text(json.dumps({"selected": selected, "selection": "High-quality qualitative examples selected by car/person performance on RGB-T test predictions; not a random or representative sample", "checkpoint_epoch": report["selected_epoch"]}, indent=2))
    for record in selected:
        sid = record["id"]
        raw = np.array(Image.open(data / "images" / f"{sid}.png"))
        prediction = np.array(Image.open(run / "predictions" / f"{sid}.png"))
        Image.fromarray(raw[:, :, :3]).save(out / f"{sid}-rgb.png")
        overlay(raw[:, :, :3], prediction).save(out / f"{sid}-prediction-overlay.png")
    print(json.dumps({"figure": str(out / "rgbt-mfnet-demo.png"), "ids": [r["id"] for r in selected]}, indent=2))


if __name__ == "__main__":
    main()
