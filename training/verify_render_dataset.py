"""Check every YOLO annotation and make a rendered sample sheet."""

import argparse
from collections import Counter
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


TRAINING = Path(__file__).resolve().parent
DEFAULT_OUTPUT = TRAINING / "data" / "tts-synthetic"
COLORS = ("#42f5da", "#ffee70", "#ff99db", "#79bcff", "#ff935a")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def listed_images(root, split):
    return [path.removeprefix("./") for path in (root / f"{split}.txt").read_text(encoding="utf-8").splitlines()]


def annotations(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def verify(root):
    cfg, summary = load(root / "dataset.yaml"), load(root / "summary.json")
    names = (root / "labels.txt").read_text(encoding="utf-8").splitlines()
    catalog = load(TRAINING.parent / "Assets" / "WarhammerAssistant" / "Units.json")
    units = {row["label"]: row["name"] for row in catalog["units"]}
    if names != cfg["names"] or len(names) != cfg["nc"] or len(set(names)) != len(names) or not set(names) <= units.keys():
        raise ValueError("Class IDs disagree with the Unity catalog")
    image_list = {name: listed_images(root, name) for name in ("train", "val")}
    split_by_image = {image: split for split, paths in image_list.items() for image in paths}
    if len(split_by_image) != sum(map(len, image_list.values())):
        raise ValueError("Duplicate image entries in dataset lists")
    ids = {name: index for index, name in enumerate(names)}
    size = summary["image_size"]
    represented = {"train": set(), "val": set()}
    geometry = {"train": set(), "val": set()}
    multi = Counter()
    boxes = Counter()
    seen = set()
    singles_for_preview = []
    scenes_for_preview = []
    for item in annotations(root / "annotations.jsonl"):
        image = item["image"]
        if image in seen:
            raise ValueError(f"Repeated annotation: {image}")
        seen.add(image)
        split = split_by_image[image]
        with Image.open(root / image) as pixels:
            if pixels.size != (size, size) or pixels.mode != "RGB":
                raise ValueError(f"Unexpected render size or color: {image}")
            pixels.verify()
        label_file = root / "labels" / split / Path(image).with_suffix(".txt").name
        lines = label_file.read_text(encoding="ascii").splitlines()
        if not item["boxes"] or len(lines) != len(item["boxes"]):
            raise ValueError(f"Missing labels: {image}")
        multi[split] += item["kind"] == "multi"
        if item["kind"] == "multi" and len(lines) < 2:
            raise ValueError(f"Multi-model image has fewer than two boxes: {image}")
        for line, box in zip(lines, item["boxes"]):
            class_id, x, y, w, h = line.split()
            x1, y1, x2, y2 = box["xyxy"]
            subject = box["label"]
            xywh = ((x1+x2)/(2*size), (y1+y2)/(2*size), (x2-x1)/size, (y2-y1)/size)
            if int(class_id) != ids[subject] or any(abs(float(p) - expected) > 1e-7
                                                    for p, expected in zip((x, y, w, h), xywh)):
                raise ValueError(f"Inaccurate label coordinates: {image}")
            if not (0 <= x1 < x2 <= size and 0 <= y1 < y2 <= size):
                raise ValueError(f"Box outside image: {image}")
            boxes[split] += 1
            geometry[split].add(box["geometry_group"])
            represented[split].add(subject)
        if item["kind"] == "single" and items_with_other_classes(item, singles_for_preview):
            singles_for_preview.append(item)
        if item["kind"] == "multi" and len(item["boxes"]) >= 3 and items_with_other_classes(item, scenes_for_preview):
            scenes_for_preview.append(item)
    if seen != split_by_image.keys() or geometry["train"] & geometry["val"]:
        raise ValueError("Image list mismatch or pack geometry shared across splits")
    for split, paths in image_list.items():
        if (len(paths) != summary["images"][split] or boxes[split] != summary["boxes"][split]
                or multi[split] != summary["multi_model_images"][split]):
            raise ValueError(f"Dataset summary mismatch: {split}")
    if len(represented["train"]) != len(names) or len(represented["val"]) != summary["labels_in_validation"]:
        raise ValueError("Dataset class coverage mismatch")
    preview(root, (singles_for_preview + scenes_for_preview)[:9], units, size)
    print(f"Verified {len(seen):,} JPEG images, {sum(boxes.values()):,} YOLO boxes, "
          f"{len(names):,} Unity classes, zero source-geometry overlap.")


def items_with_other_classes(item, selected):
    if len(selected) >= (3 if item["kind"] == "single" else 6):
        return False
    previous = {box["label"] for row in selected for box in row["boxes"]}
    return any(box["label"] not in previous for box in item["boxes"])


def preview(root, samples, units, size):
    tile, gap = 340, 10
    width = 3 * tile + 4 * gap
    height = ((len(samples) + 2) // 3) * (tile + gap) + gap
    board = Image.new("RGB", (width, height), "#172331")
    font = ImageFont.load_default(size=14)
    for i, sample in enumerate(samples):
        with Image.open(root / sample["image"]) as source:
            pixels = source.resize((tile, tile), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(pixels)
        for index, box in enumerate(sample["boxes"]):
            x1, y1, x2, y2 = [round(value * tile / size) for value in box["xyxy"]]
            color = COLORS[index % len(COLORS)]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
            text = units[box["label"]]
            length = int(font.getlength(text)) + 8
            tx, ty = min(x1, max(0, tile-length)), max(0, y1-18)
            draw.rectangle((tx, ty, tx+length, ty+18), fill=color)
            draw.text((tx+4, ty+1), text, fill="#0e1923", font=font)
        board.paste(pixels, (gap + i % 3 * (tile + gap), gap + i // 3 * (tile + gap)))
    board.save(root / "preview.jpg", quality=91)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    verify(parser.parse_args().output.resolve())
