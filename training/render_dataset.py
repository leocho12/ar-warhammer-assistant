"""Render a resumable YOLO dataset from labelled Tabletop Simulator assemblies."""

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import json
from math import ceil, cos, sin, tau
from pathlib import Path

import cv2
import numpy as np

from project import _project_lock, _write_json
from tts_render_assets import assembly_surfaces, asset_nodes, catalog_bundles
from tts_renderer import Renderer


TRAINING = Path(__file__).resolve().parent
SOURCE = TRAINING / "data" / "tts-models" / "saved-factions"
OUTPUT = TRAINING / "data" / "tts-synthetic"
VERSION = 1
REPAIR_LEVEL = 1
ELEVATIONS = (23, 51, 35, 74, 28, 58)
WORKER_RENDERER = None


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def seed_for(*values):
    key = "/".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(sha256(key).digest()[:8], "little")


def geometry_group(tree):
    keys = {asset["sha256"] for node in asset_nodes(tree) for asset in node["assets"]
            if asset["kind"] in {"mesh", "bundle"}}
    if not keys:
        raise ValueError("Assembly has no mesh or prefab reference")
    return sha256("".join(sorted(keys)).encode("ascii")).hexdigest()


def selected_samples(source, index, labels, faction, limit):
    groups = defaultdict(deque)
    for row in index:
        if row["assets_status"] != "complete" or row["label"] not in labels:
            continue
        if faction and row["faction"] not in faction:
            continue
        groups[row["label"]].append(row)
    if limit is None:
        selected = [row for group in groups.values() for row in group]
    else:
        queue = deque(groups.values())
        selected = []
        while queue and len(selected) < limit:
            group = queue.popleft()
            selected.append(group.popleft())
            if group:
                queue.append(group)
    for row in selected:
        relative = Path(row["assembly"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe assembly path from source index: {relative}")
        assembly = read_json(source / relative)
        yield {
            "id": row["id"], "name": row["name"], "assembly": relative.as_posix(), "group": geometry_group(assembly["tree"]),
            "label": row["label"], "faction": row["faction"], "match": row["match"],
            "proxy_or_conversion": row["proxy_or_conversion"],
        }


def create_batches(samples, fraction, batch_size, seed):
    counts = defaultdict(Counter)
    labels_in_group = defaultdict(set)
    for sample in samples:
        counts[sample["label"]][sample["group"]] += 1
        labels_in_group[sample["group"]].add(sample["label"])
    # A model and paint variants with identical source geometry always share a split.
    validation_groups = set()
    train_group_count = {label: len(groups) for label, groups in counts.items()}
    for label, groups in sorted(counts.items(), key=lambda pair: len(pair[1])):
        threshold = max(1, round(sum(groups.values()) * fraction)) if fraction else 0
        already = sum(amount for key, amount in groups.items() if key in validation_groups)
        choices = sorted(groups, key=lambda key: (groups[key], seed_for(seed, key)))
        for key in choices:
            if already >= threshold:
                break
            if key in validation_groups or any(train_group_count[label] < 2 for label in labels_in_group[key]):
                continue
            validation_groups.add(key)
            already += groups[key]
            for affected in labels_in_group[key]:
                train_group_count[affected] -= 1
    sides = {"train": defaultdict(list), "val": defaultdict(list)}
    for sample in samples:
        split = "val" if sample["group"] in validation_groups else "train"
        sides[split][sample["faction"]].append(sample)
    batches = []
    for split, side in sides.items():
        buckets = deque()
        for faction, entries in sorted(side.items()):
            entries.sort(key=lambda item: (item["label"], item["group"], item["id"]))
            buckets.append((faction, deque(entries)))
        while buckets:
            # Keep some parts in cache across variants, but make mixed-faction scenes as well.
            pile = []
            for _ in range(len(buckets)):
                faction, entries = buckets.popleft()
                amount = min(len(entries), ceil(batch_size / 2), batch_size - len(pile))
                pile.extend(entries.popleft() for _ in range(amount))
                if entries:
                    buckets.append((faction, entries))
                if len(pile) >= batch_size:
                    break
            batches.append({"split": split, "samples": pile})
    return batches


def write_image(output, split, stem, pixels):
    path = output / "images" / split / f"{stem}.jpg"
    ok, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("JPEG encoding failed")
    temp = path.with_suffix(".tmp")
    temp.write_bytes(encoded.tobytes())
    temp.replace(path)
    return path.relative_to(output).as_posix()


def singles(renderer, subject, views, seed):
    rng = np.random.default_rng(seed_for(seed, subject.sample["id"]))
    initial_yaw = rng.uniform(0, 360)
    for index in range(views):
        azimuth = (index * 360 / views + initial_yaw + rng.uniform(-8, 8)) % 360
        elevation = float(ELEVATIONS[index % len(ELEVATIONS)] + rng.uniform(-6, 6))
        distance = rng.uniform(1.9, 2.8)
        fov = rng.uniform(37, 48)
        target = np.array([rng.uniform(-0.12, 0.12), subject.dimensions[1]/2, rng.uniform(-0.12, 0.12)])
        placement = [(subject, (0, 0), 0, 1.)]
        pixels, boxes = renderer.render(placement, target, azimuth, elevation, distance, fov, rng)
        if pixels is not None:
            yield index, pixels, boxes, [round(azimuth, 2), round(elevation, 2)]


def multi(renderer, subjects, rng):
    count = int(rng.integers(2, min(5, len(subjects)) + 1))
    candidates = rng.choice(len(subjects), size=count, replace=False)
    selection = [subjects[index] for index in candidates]
    extents = np.array([subject.original_extent for subject in selection])
    sizes = np.clip((extents / np.median(extents)) ** 0.55, 0.47, 1.55)
    yaw = rng.uniform(0, tau)
    elevation = rng.uniform(22, 74)
    radius = rng.uniform(0.52, 0.92) * (count / 2) ** 0.5
    distance = rng.uniform(2.85, 3.72) + 0.29 * (count - 2)
    azimuth, fov = rng.uniform(0, 360), rng.uniform(39, 49)
    target = np.array([rng.uniform(-0.16, 0.16), 0.28, rng.uniform(-0.16, 0.16)])
    for spread in (1.0, 1.3):
        placement = []
        for index, subject in enumerate(selection):
            angle = yaw + tau * index / count + rng.uniform(-0.26, 0.26)
            r = radius * spread * rng.uniform(0.80, 1.18)
            placement.append((subject, (r*cos(angle), r*sin(angle)), rng.uniform(0, 360), float(sizes[index])))
        pixels, boxes = renderer.render(placement, target, azimuth, elevation, distance * spread, fov, rng)
        if pixels is not None:
            return pixels, boxes, [round(azimuth, 2), round(elevation, 2)]
    return None


def render_batch(args, renderer, batch, batch_number):
    samples = batch["samples"]
    split = batch["split"]
    frames = []
    subjects = []
    failures = []
    for sample in samples:
        assembly = read_json(args.source / sample["assembly"])
        try:
            subject = renderer.subject(sample, assembly_surfaces(args.source, assembly["tree"], renderer))
        except (ValueError, KeyError, OSError, RuntimeError, IndexError, TypeError, AttributeError, AssertionError) as exc:
            failures.append({**sample, "error": f"{type(exc).__name__}: {exc}"})
            continue
        before = len(frames)
        for index, pixels, boxes, camera in singles(renderer, subject, args.views, args.seed):
            stem = f'{sample["id"]}_{index:02}'
            frames.append({"image": write_image(args.output, split, stem, pixels), "kind": "single",
                           "camera_deg": camera, "boxes": boxes})
        if len(frames) == before:
            failures.append({**sample, "error": "No single-model renders contained a visible silhouette"})
            continue
        subjects.append(subject)
    rng = np.random.default_rng(seed_for(args.seed, batch_number, "multi"))
    if len(subjects) >= 2:
        for index in range(ceil(len(subjects) * args.multi)):
            result = multi(renderer, subjects, rng)
            if result is None:
                continue
            pixels, boxes, camera = result
            stem = f"scene_{batch_number:04}_{index:03}"
            frames.append({"image": write_image(args.output, split, stem, pixels), "kind": "multi",
                           "camera_deg": camera, "boxes": boxes})
    return {"split": split, "frames": frames, "failures": failures}


def start_worker(size, library):
    global WORKER_RENDERER
    cv2.setNumThreads(1)
    WORKER_RENDERER = Renderer(size, library)


def save_batch(args, batch, number, checkpoint, previous):
    renderer = WORKER_RENDERER
    try:
        record = render_batch(args, renderer, batch, number)
        record["repair_level"] = REPAIR_LEVEL
        _write_json(checkpoint, record)
        if previous:
            current = {frame["image"] for frame in record["frames"]}
            for image in {frame["image"] for frame in previous["frames"]} - current:
                relative = Path(image)
                if relative.parts[:2] != ("images", record["split"]) or len(relative.parts) != 3:
                    raise ValueError(f"Unsafe checkpoint image path: {image}")
                (args.output / relative).unlink(missing_ok=True)
                (args.output / "labels" / record["split"] / relative.with_suffix(".txt").name).unlink(missing_ok=True)
    finally:
        renderer.clear_assets()
    return number, record


def needs_repair(record, image_size):
    if record.get("repair_level", 0) >= REPAIR_LEVEL:
        return False
    if record["failures"]:
        return True
    largest = Counter()
    for frame in record["frames"]:
        if frame["kind"] == "single":
            box = frame["boxes"][0]
            x1, y1, x2, y2 = box["xyxy"]
            largest[box["assembly"]] = max(largest[box["assembly"]], x2-x1, y2-y1)
    # A misplaced attachment can shrink every view of a subject to a few pixels.
    return any(edge < image_size * 0.20 for edge in largest.values())


def yolo_line(box, classes, size):
    x1, y1, x2, y2 = box["xyxy"]
    if not (0 <= x1 < x2 <= size and 0 <= y1 < y2 <= size):
        raise ValueError(f"Out-of-frame bounding box: {box}")
    xywh = [(x1 + x2) / (2*size), (y1 + y2) / (2*size), (x2 - x1) / size, (y2 - y1) / size]
    return f'{classes[box["label"]]} ' + " ".join(f"{value:.8f}" for value in xywh)


def finalize(output, records, size, plan):
    seen = {"train": Counter(), "val": Counter()}
    images = {"train": [], "val": []}
    groups = {"train": set(), "val": set()}
    frames = Counter()
    assemblies = set()
    failures = []
    omitted_validation = 0
    names = sorted({box["label"] for record in records if record["split"] == "train"
                    for frame in record["frames"] for box in frame["boxes"]})
    ids = {label: index for index, label in enumerate(names)}
    with (output / "annotations.jsonl").open("w", encoding="utf-8") as annotated:
        for record in records:
            split = record["split"]
            failures.extend(record["failures"])
            for frame in record["frames"]:
                boxes = frame["boxes"]
                if any(box["label"] not in ids for box in boxes):
                    omitted_validation += 1
                    continue
                image = frame["image"]
                if not (output / image).is_file():
                    raise FileNotFoundError(f"Missing rendered image: {image}")
                label = Path(image).with_suffix(".txt")
                label_path = output / "labels" / label.parent.name / label.name
                label_path.write_text("\n".join(yolo_line(box, ids, size) for box in boxes) + "\n", encoding="ascii")
                images[split].append("./" + image)
                seen[split].update(box["label"] for box in boxes)
                groups[split].update(box["geometry_group"] for box in boxes)
                assemblies.update(box["assembly"] for box in boxes)
                frames[split, frame["kind"]] += 1
                annotated.write(json.dumps(frame, ensure_ascii=False) + "\n")
    if groups["train"] & groups["val"]:
        raise ValueError("Geometry leaked between training and validation")
    if not images["train"] or not images["val"]:
        raise ValueError("Dataset needs both training and validation views")
    for split, paths in images.items():
        (output / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    (output / "labels.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    _write_json(output / "dataset.yaml", {
        "path": output.as_posix(), "train": "train.txt", "val": "val.txt", "nc": len(names), "names": names,
    })
    _write_json(output / "skipped-models.json", failures)
    by_label = {label: {split: seen[split][label] for split in images} for label in names}
    used_samples = [sample for batch in plan["batches"] for sample in batch["samples"] if sample["assembly"] in assemblies]
    summary = {
        "input_assemblies": sum(len(batch["samples"]) for batch in plan["batches"]),
        "rendered_assemblies": len(assemblies), "skipped_assemblies": len(failures),
        "images": {split: len(paths) for split, paths in images.items()},
        "multi_model_images": {split: frames[split, "multi"] for split in images},
        "boxes": {split: sum(counts.values()) for split, counts in seen.items()}, "labels": len(names),
        "labels_in_validation": len(seen["val"]), "train_only_labels": sorted(set(names) - set(seen["val"])),
        "label_box_counts": by_label,
        "faction_assemblies": dict(sorted(Counter(sample["faction"] for sample in used_samples).items())),
        "named_proxy_or_conversion_assemblies": sum(sample["proxy_or_conversion"] for sample in used_samples),
        "validation_views_without_training_labels": omitted_validation, "image_size": size,
        "split_strategy": "set of source mesh and primary bundle hashes",
    }
    _write_json(output / "summary.json", summary)
    print({key: value for key, value in summary.items() if key not in {"label_box_counts", "train_only_labels"}}, flush=True)


def generate(args):
    catalog_file = TRAINING.parent / "Assets" / "WarhammerAssistant" / "Units.json"
    index = (args.source / "index.json").read_bytes()
    catalog = catalog_file.read_bytes()
    config = {
        "version": VERSION, "source": args.source.as_posix(), "index": sha256(index).hexdigest(),
        "catalog": sha256(catalog).hexdigest(), "views": args.views, "size": args.size,
        "batch_size": args.batch_size, "multi": args.multi, "validation": args.validation, "seed": args.seed,
        "faction": sorted(args.faction or []), "limit": args.limit,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with _project_lock(args.output):
        plan_path = args.output / "render-plan.json"
        if plan_path.exists():
            plan = read_json(plan_path)
            if plan["config"] != config:
                raise ValueError("Existing output uses different inputs or settings. Choose a new output directory.")
        else:
            labels = {row["label"] for row in json.loads(catalog)["units"]}
            samples = list(selected_samples(args.source, json.loads(index), labels, args.faction, args.limit))
            batches = create_batches(samples, args.validation, args.batch_size, args.seed)
            if not samples or not any(batch["split"] == "val" for batch in batches):
                raise ValueError("Need labelled assemblies and more than one geometry group for at least one label")
            plan = {"config": config, "batches": batches}
            _write_json(plan_path, plan)
        checkpoints = args.output / ".batches"
        checkpoints.mkdir(exist_ok=True)
        for split in ("train", "val"):
            for category in ("images", "labels"):
                (args.output / category / split).mkdir(parents=True, exist_ok=True)
        batch_count = len(plan["batches"])
        print(f"{batch_count} batches, {args.workers} renderer workers", flush=True)
        records = {}
        pending = []
        for number, batch in enumerate(plan["batches"]):
            path = checkpoints / f"{number:04}.json"
            if path.exists():
                record = read_json(path)
                if args.retry_skipped and needs_repair(record, args.size):
                    pending.append((args, batch, number, path, record))
                else:
                    records[number] = record
            else:
                pending.append((args, batch, number, path, None))
        print(f"{len(pending)} batches pending", flush=True)
        library = catalog_bundles(args.source) if pending else {}

        def accept(result):
            number, record = result
            records[number] = record
            print(f'{len(records)}/{batch_count} (batch {number + 1}, {record["split"]}): '
                  f'{len(record["frames"])} images, {len(record["failures"])} skipped', flush=True)

        if args.workers == 1 and pending:
            start_worker(args.size, library)
            for task in pending:
                accept(save_batch(*task))
        elif pending:
            with ProcessPoolExecutor(args.workers, initializer=start_worker, initargs=(args.size, library)) as pool:
                jobs = [pool.submit(save_batch, *task) for task in pending]
                for result in as_completed(jobs):
                    accept(result.result())
        finalize(args.output, [records[number] for number in range(batch_count)], args.size, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--faction", action="append")
    parser.add_argument("--limit", type=int, help="Small stratified test run; default processes all inputs.")
    parser.add_argument("--views", type=int, default=6)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--retry-skipped", action="store_true", help="Refresh older batches with failed renders.")
    parser.add_argument("--multi", type=float, default=0.65, help="Number of multi-model scenes per usable assembly.")
    parser.add_argument("--validation", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (args.views < 1 or args.size < 64 or args.batch_size < 2 or args.workers < 1 or args.multi < 0
            or not 0 < args.validation < 1 or args.limit is not None and args.limit < 2):
        parser.error("Invalid render settings")
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    if args.source == args.output or args.source in args.output.parents:
        parser.error("Output must be outside the source assembly directory")
    generate(args)


if __name__ == "__main__":
    main()
