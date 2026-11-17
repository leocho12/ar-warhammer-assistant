"""Persistent, normalized annotations and reproducible YOLO dataset snapshots."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import tempfile

from PIL import Image, ImageOps


class ProjectError(ValueError):
    """An actionable project or annotation error."""


@contextmanager
def _project_lock(root: Path):
    # OS locks are released even if the application terminates unexpectedly.
    with (root / ".project.lock").open("a+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            if stream.read(1) == b"":
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise ProjectError("다른 창에서 프로젝트를 저장 중입니다. 잠시 후 다시 시도하세요.") from exc
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _write_json(path: Path, value: dict):
    """Replace metadata only after the complete new document reaches disk."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _image_digest(image: Image.Image) -> str:
    digest = hashlib.sha256(f"RGB:{image.width}:{image.height}:".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _safe_image_path(root: Path, record: dict) -> Path:
    identifier = record.get("id")
    if not isinstance(identifier, str) or re.fullmatch(r"[0-9a-f]{64}", identifier) is None:
        raise ProjectError("프로젝트에 올바르지 않은 사진 ID가 있습니다.")
    if record.get("filename") != f"images/{identifier}.png":
        raise ProjectError("프로젝트의 사진 경로가 올바르지 않습니다.")
    path = (root / record["filename"]).resolve()
    if not path.is_relative_to(root):
        raise ProjectError("프로젝트 바깥을 가리키는 사진 경로는 사용할 수 없습니다.")
    return path


def _validate_boxes(boxes: list, class_count: int):
    if not isinstance(boxes, list):
        raise ProjectError("사진의 박스 목록이 올바르지 않습니다.")
    for box in boxes:
        if not isinstance(box, dict):
            raise ProjectError("박스 데이터가 올바르지 않습니다.")
        class_id = box.get("class_id")
        if type(class_id) is not int or not 0 <= class_id < class_count:
            raise ProjectError("박스에 지정된 유닛 종류가 없습니다.")
        coordinates = [box.get(key) for key in ("x1", "y1", "x2", "y2")]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in coordinates):
            raise ProjectError("박스 좌표는 유한한 숫자여야 합니다.")
        x1, y1, x2, y2 = coordinates
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ProjectError("박스는 사진 안에 있어야 하며 너비와 높이가 0보다 커야 합니다.")


class Project:
    def __init__(self, root: Path, data: dict):
        self.root = root.resolve()
        self._data = data
        self._validate()

    @property
    def name(self) -> str:
        return self._data["name"]

    @property
    def classes(self) -> list[str]:
        return self._data["classes"]

    @property
    def images(self) -> list[dict]:
        return self._data["images"]

    @classmethod
    def create(cls, root: Path, name: str) -> "Project":
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        project = cls(root, {"version": 1, "revision": 0, "name": name.strip(), "classes": [], "images": []})
        with _project_lock(root):
            if (root / "project.json").exists():
                raise ProjectError("이 폴더에는 이미 프로젝트가 있습니다. 프로젝트 열기를 사용하세요.")
            project._data["revision"] = 1
            _write_json(root / "project.json", project._data)
        return project

    @classmethod
    def load(cls, root: Path) -> "Project":
        root = Path(root).resolve()
        try:
            data = json.loads((root / "project.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ProjectError(f"프로젝트를 읽을 수 없습니다. {exc}") from exc
        project = cls(root, data)
        for record in project.images:
            if not _safe_image_path(root, record).is_file():
                raise ProjectError(f"사진 파일이 없습니다. {record['filename']}")
        return project

    def _validate(self):
        data = self._data
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ProjectError("지원하지 않는 프로젝트 형식입니다.")
        if type(data.get("revision")) is not int or data["revision"] < 0:
            raise ProjectError("프로젝트 저장 버전이 올바르지 않습니다.")
        if not isinstance(data.get("name"), str) or not data["name"].strip():
            raise ProjectError("프로젝트 이름을 입력하세요.")
        classes = data.get("classes")
        if not isinstance(classes, list) or any(
            not isinstance(name, str) or not name.strip() or "\n" in name or "\r" in name
            for name in classes
        ):
            raise ProjectError("유닛 이름은 비어 있지 않은 한 줄이어야 합니다.")
        if len(set(classes)) != len(classes):
            raise ProjectError("유닛 이름이 중복되어 있습니다.")
        if not isinstance(data.get("images"), list):
            raise ProjectError("프로젝트의 사진 목록이 올바르지 않습니다.")
        identifiers = set()
        for record in self.images:
            if not isinstance(record, dict):
                raise ProjectError("사진 데이터가 올바르지 않습니다.")
            _safe_image_path(self.root, record)
            if record["id"] in identifiers:
                raise ProjectError("프로젝트에 중복 사진이 있습니다.")
            identifiers.add(record["id"])
            if any(type(record.get(key)) is not int or record[key] <= 0 for key in ("width", "height")):
                raise ProjectError("사진 크기가 올바르지 않습니다.")
            if type(record.get("reviewed")) is not bool or record.get("split") not in ("", "train", "val"):
                raise ProjectError("사진의 검토 또는 데이터 분할 설정이 올바르지 않습니다.")
            _validate_boxes(record.get("boxes"), len(classes))

    def save(self):
        self._validate()
        path = self.root / "project.json"
        with _project_lock(self.root):
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ProjectError("저장된 프로젝트를 확인할 수 없습니다. 프로젝트를 다시 열어주세요.") from exc
            if current.get("revision") != self._data["revision"]:
                raise ProjectError("다른 창에서 프로젝트가 변경되었습니다. 프로젝트를 다시 열어주세요.")
            old_classes = current.get("classes", [])
            if self.classes[: len(old_classes)] != old_classes:
                raise ProjectError("저장된 유닛 종류의 순서를 바꾸거나 삭제할 수 없습니다.")
            saved = deepcopy(self._data)
            saved["revision"] += 1
            _write_json(path, saved)
            self._data["revision"] = saved["revision"]

    def add_class(self, name: str) -> int:
        name = name.strip()
        if name in self.classes:
            return self.classes.index(name)
        self.classes.append(name)
        try:
            self.save()
        except Exception:
            self.classes.pop()
            raise
        return len(self.classes) - 1

    def import_images(self, paths) -> dict:
        result = {"added": 0, "duplicates": 0, "errors": []}
        image_directory = (self.root / "images").resolve()
        if not image_directory.is_relative_to(self.root):
            raise ProjectError("사진 폴더가 프로젝트 바깥을 가리키고 있습니다.")
        image_directory.mkdir(exist_ok=True)
        for source in paths:
            source = Path(source)
            try:
                with Image.open(source) as original:
                    image = ImageOps.exif_transpose(original).convert("RGB")
                identifier = _image_digest(image)
                if any(record["id"] == identifier for record in self.images):
                    result["duplicates"] += 1
                    continue
                filename = f"images/{identifier}.png"
                path = self.root / filename
                with tempfile.NamedTemporaryFile(dir=image_directory, suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                try:
                    image.save(temporary, format="PNG")
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
                record = {
                    "id": identifier, "filename": filename, "source_name": source.name,
                    "width": image.width, "height": image.height,
                    "reviewed": False, "split": "", "boxes": [],
                }
                self.images.append(record)
                try:
                    self.save()
                except Exception:
                    self.images.pop()
                    raise
                result["added"] += 1
            except (OSError, ValueError) as exc:
                result["errors"].append(f"{source.name} — {exc}")
        return result

    def update_image(self, image_id: str, boxes: list, reviewed: bool, split: str):
        for index, record in enumerate(self.images):
            if record["id"] == image_id:
                updated = {**record, "boxes": deepcopy(boxes), "reviewed": reviewed, "split": split}
                self.images[index] = updated
                try:
                    self.save()
                except Exception:
                    self.images[index] = record
                    raise
                return
        raise ProjectError("사진을 찾을 수 없습니다. 프로젝트를 다시 열어주세요.")

    def _split(self, records: list[dict], val_fraction: float, seed: int) -> dict[str, str]:
        labels = {record["id"]: {box["class_id"] for box in record["boxes"]} for record in records}
        fixed_train = {record["id"] for record in records if record["split"] == "train"}
        fixed_val = {record["id"] for record in records if record["split"] == "val"}
        automatic = [record["id"] for record in records if not record["split"]]
        all_ids = set(labels)
        all_classes = {class_id for values in labels.values() for class_id in values}
        for class_id in sorted(all_classes):
            name = self.classes[class_id]
            examples = {identifier for identifier, values in labels.items() if class_id in values}
            if len(examples) < 2:
                raise ProjectError(f"'{name}'의 검토 완료 사진이 최소 2장 필요합니다. 학습과 검증에 각각 사용합니다.")
            if not examples - fixed_val or not examples - fixed_train:
                raise ProjectError(f"'{name}' 사진이 한쪽 분할에만 지정되어 있습니다. 학습과 검증에 각각 사진을 배정하세요.")
        randomizer = random.Random(seed)
        target = max(1, min(len(records) - 1, round(len(records) * val_fraction)))
        for attempt in range(64):
            validation = set(fixed_val)
            train_counts = Counter(class_id for identifier in all_ids - validation for class_id in labels[identifier])
            missing = all_classes - {class_id for identifier in validation for class_id in labels[identifier]}
            candidates = automatic.copy()
            randomizer.shuffle(candidates)

            def eligible(identifier):
                return all(train_counts[class_id] > 1 for class_id in labels[identifier])

            while missing:
                choices = [identifier for identifier in candidates if eligible(identifier) and labels[identifier] & missing]
                if not choices:
                    break
                if attempt == 0:
                    chosen = max(choices, key=lambda identifier: len(labels[identifier] & missing))
                else:
                    chosen = randomizer.choice(choices)
                candidates.remove(chosen)
                validation.add(chosen)
                train_counts.subtract(labels[chosen])
                missing -= labels[chosen]
            if missing:
                continue
            for identifier in candidates:
                if len(validation) >= target:
                    break
                if eligible(identifier):
                    validation.add(identifier)
                    train_counts.subtract(labels[identifier])
            if validation and all_ids - validation:
                return {identifier: "val" if identifier in validation else "train" for identifier in all_ids}
        raise ProjectError("모든 유닛을 학습과 검증에 포함하는 자동 분할을 찾지 못했습니다. 사진별 분할을 직접 지정하거나 사진을 추가하세요.")

    def build_dataset(self, output_dir: Path, val_fraction: float = 0.2, seed: int = 42) -> Path:
        self._validate()
        if not 0 < val_fraction < 1:
            raise ProjectError("검증 비율은 0과 1 사이여야 합니다.")
        if not self.classes:
            raise ProjectError("유닛 종류를 먼저 추가하세요.")
        records = deepcopy([record for record in self.images if record["reviewed"]])
        if len(records) < 2:
            raise ProjectError("검토 완료 사진이 최소 2장 필요합니다. 모든 대상에 박스를 그리고 검토 완료를 표시하세요.")
        class_ids = sorted({box["class_id"] for record in records for box in record["boxes"]})
        if not class_ids:
            raise ProjectError("검토 완료 사진에 유닛 박스가 없습니다. 학습할 유닛에 박스를 그리고 완료 표시를 해주세요.")
        model_classes = [self.classes[class_id] for class_id in class_ids]
        model_class_ids = {class_id: model_id for model_id, class_id in enumerate(class_ids)}
        assignment = self._split(records, val_fraction, seed)
        output_dir = Path(output_dir).resolve()
        if output_dir.exists():
            raise ProjectError("내보낼 폴더가 이미 있습니다. 새 폴더를 선택하세요.")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".dataset-", dir=output_dir.parent))
        try:
            for split in ("train", "val"):
                (staging / "images" / split).mkdir(parents=True)
                (staging / "labels" / split).mkdir(parents=True)
            for record in records:
                path = _safe_image_path(self.root, record)
                with Image.open(path) as image:
                    decoded = image.convert("RGB")
                    if decoded.size != (record["width"], record["height"]) or _image_digest(decoded) != record["id"]:
                        raise ProjectError(f"가져온 사진 파일이 변경되었습니다. {record.get('source_name', record['filename'])}")
                split = assignment[record["id"]]
                shutil.copyfile(path, staging / "images" / split / path.name)
                lines = []
                for box in record["boxes"]:
                    center_x = (box["x1"] + box["x2"]) / 2
                    center_y = (box["y1"] + box["y2"]) / 2
                    width, height = box["x2"] - box["x1"], box["y2"] - box["y1"]
                    model_id = model_class_ids[box["class_id"]]
                    lines.append(f"{model_id} {center_x:.10f} {center_y:.10f} {width:.10f} {height:.10f}\n")
                (staging / "labels" / split / f"{record['id']}.txt").write_text("".join(lines), encoding="utf-8")
            # JSON is valid YAML, including Unicode names and Windows paths.
            _write_json(staging / "dataset.yaml", {
                "path": output_dir.as_posix(), "train": "images/train", "val": "images/val",
                "names": model_classes, "nc": len(model_classes),
            })
            _write_json(staging / "manifest.json", {
                "project": self.name, "revision": self._data["revision"], "seed": seed,
                "requested_val_fraction": val_fraction, "classes": self.classes.copy(),
                "model_classes": model_classes,
                "class_id_mapping": [
                    {"project_class_id": class_id, "model_class_id": model_id, "label": self.classes[class_id]}
                    for class_id, model_id in model_class_ids.items()
                ],
                # Preserve project annotation IDs so this snapshot remains auditable
                # even when the exported model uses only a subset of the catalog.
                "images": [{**record, "split": assignment[record["id"]]} for record in records],
            })
            # Rename is atomic on the same volume. Windows refuses an existing target.
            if output_dir.exists():
                raise ProjectError("내보낼 폴더가 이미 있습니다. 새 폴더를 선택하세요.")
            staging.rename(output_dir)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return output_dir / "dataset.yaml"
