import json
from pathlib import Path
import sys

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import project as project_module
from project import Project, ProjectError


def import_picture(project, tmp_path, name, color):
    source = tmp_path / name
    Image.new("RGB", (80, 50), color).save(source)
    assert project.import_images([source])["added"] == 1
    return project.images[-1]


def box(class_id=0, x1=0.1, x2=0.4):
    return {"class_id": class_id, "x1": x1, "y1": 0.2, "x2": x2, "y2": 0.8}


def populated(tmp_path):
    project = Project.create(tmp_path / "project", "미니어처")
    project.add_class("intercessor")
    for index, color in enumerate(("red", "blue", "green", "yellow")):
        record = import_picture(project, tmp_path, f"{index}.png", color)
        project.update_image(record["id"], [box()], True, "")
    return project


def test_roundtrip_multiple_boxes_and_stable_classes(tmp_path):
    project = Project.create(tmp_path / "project", "Miniatures")
    assert project.add_class("intercessor") == 0
    assert project.add_class("terminator") == 1
    assert project.add_class("intercessor") == 0
    record = import_picture(project, tmp_path, "photo.png", "blue")
    boxes = [box(), box(1, 0.5, 0.9)]
    project.update_image(record["id"], boxes, True, "train")
    restored = Project.load(project.root)
    assert restored.classes == ["intercessor", "terminator"]
    assert restored.images[0]["boxes"] == boxes
    assert restored.images[0]["reviewed"]
    assert restored.images[0]["split"] == "train"


def test_duplicates_are_decoded_and_exif_normalized(tmp_path):
    project = Project.create(tmp_path / "project", "Miniatures")
    image = Image.new("RGB", (3, 2))
    image.putdata([(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14, 15), (16, 17, 18)])
    image.save(tmp_path / "original.png")
    rotated = image.transpose(Image.Transpose.ROTATE_90)
    exif = rotated.getexif()
    exif[274] = 6
    rotated.save(tmp_path / "rotated.png", exif=exif)
    result = project.import_images([tmp_path / "original.png", tmp_path / "rotated.png"])
    assert result == {"added": 1, "duplicates": 1, "errors": []}
    assert (project.images[0]["width"], project.images[0]["height"]) == (3, 2)
    assert (tmp_path / "rotated.png").exists()


def test_snapshot_disjoint_reproducible_and_ignores_unreviewed(tmp_path):
    project = populated(tmp_path)
    pending = import_picture(project, tmp_path, "pending.png", "purple")
    background = import_picture(project, tmp_path, "background.png", "white")
    project.update_image(background["id"], [], True, "val")
    first = project.build_dataset(tmp_path / "first")
    second = project.build_dataset(tmp_path / "second")
    data = json.loads(first.read_text(encoding="utf-8"))
    assert data["names"] == ["intercessor"]
    assert data["path"] == first.parent.as_posix()
    train_ids = {path.stem for path in (first.parent / "images/train").glob("*.png")}
    val_ids = {path.stem for path in (first.parent / "images/val").glob("*.png")}
    assert train_ids and val_ids and train_ids.isdisjoint(val_ids)
    assert pending["id"] not in train_ids | val_ids
    assert background["id"] in val_ids
    assert (first.parent / "labels/val" / f"{background['id']}.txt").read_text() == ""
    assert train_ids == {path.stem for path in (second.parent / "images/train").glob("*.png")}
    manifest = json.loads((first.parent / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["images"]) == 5
    for split in ("train", "val"):
        assert any(record["boxes"] for record in manifest["images"] if record["split"] == split)
    with pytest.raises(ProjectError, match="이미"):
        project.build_dataset(first.parent)


def test_yolo_coordinates_and_explicit_splits(tmp_path):
    project = populated(tmp_path)
    train, val = project.images[:2]
    project.update_image(train["id"], [box()], True, "train")
    project.update_image(val["id"], [box()], True, "val")
    dataset = project.build_dataset(tmp_path / "dataset")
    contents = (dataset.parent / "labels/train" / f"{train['id']}.txt").read_text().split()
    assert contents[0] == "0"
    assert [float(value) for value in contents[1:]] == pytest.approx([0.25, 0.5, 0.3, 0.6])
    assert (dataset.parent / "images/val" / f"{val['id']}.png").exists()


def test_split_errors_are_actionable_and_no_partial_dataset(tmp_path):
    project = populated(tmp_path)
    for record in project.images.copy():
        project.update_image(record["id"], record["boxes"], True, "train")
    with pytest.raises(ProjectError, match="학습과 검증"):
        project.build_dataset(tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()
    project.add_class("ork")
    with pytest.raises(ProjectError):
        project.build_dataset(tmp_path / "dataset")


def test_corrupt_metadata_paths_and_boxes_rejected(tmp_path):
    project = populated(tmp_path)
    record = project.images[0]
    for bad_box in (box(9), box(x1=0.8, x2=0.4), box(x1=float("nan"))):
        with pytest.raises(ProjectError):
            project.update_image(record["id"], [bad_box], True, "")
    assert project.images[0]["boxes"] == [box()]
    metadata = json.loads((project.root / "project.json").read_text(encoding="utf-8"))
    metadata["images"][0]["filename"] = "../outside.png"
    (project.root / "project.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ProjectError, match="경로"):
        Project.load(project.root)


def test_conflicting_writers_do_not_overwrite(tmp_path):
    first = Project.create(tmp_path / "project", "Miniatures")
    second = Project.load(first.root)
    first.add_class("intercessor")
    with pytest.raises(ProjectError, match="다른 창"):
        second.add_class("ork")
    assert second.classes == []
    assert Project.load(first.root).classes == ["intercessor"]


def test_atomic_save_failure_preserves_disk_and_rolls_back_update(tmp_path, monkeypatch):
    project = populated(tmp_path)
    original = (project.root / "project.json").read_bytes()

    def fail_replace(source, destination):
        raise OSError("disk write failed")

    monkeypatch.setattr(project_module.os, "replace", fail_replace)
    with pytest.raises(OSError):
        project.update_image(project.images[0]["id"], [], True, "")
    assert (project.root / "project.json").read_bytes() == original
    assert project.images[0]["boxes"] == [box()]
    assert not list(project.root.glob("*.tmp"))


def test_changed_image_rejects_snapshot_and_cleans_staging(tmp_path):
    project = populated(tmp_path)
    Image.new("RGB", (80, 50), "black").save(project.root / project.images[0]["filename"])
    with pytest.raises(ProjectError, match="변경"):
        project.build_dataset(tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()
    assert not list(tmp_path.glob(".dataset-*"))


def test_bad_file_does_not_prevent_other_imports(tmp_path):
    project = Project.create(tmp_path / "project", "Miniatures")
    image = tmp_path / "image.png"
    Image.new("RGB", (40, 40), "green").save(image)
    bad = tmp_path / "broken.png"
    bad.write_text("broken")
    result = project.import_images([bad, image])
    assert result["added"] == 1
    assert len(result["errors"]) == 1


def test_unused_catalog_classes_do_not_require_photos(tmp_path):
    project = populated(tmp_path)
    project.add_class("ork_boy")
    pending = import_picture(project, tmp_path, "pending_ork.png", "purple")
    project.update_image(pending["id"], [box(1)], False, "")
    dataset = project.build_dataset(tmp_path / "dataset")
    data = json.loads(dataset.read_text(encoding="utf-8"))
    assert data["names"] == ["intercessor"]
    assert data["nc"] == 1
    manifest = json.loads((dataset.parent / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["classes"] == ["intercessor", "ork_boy"]
    assert manifest["model_classes"] == ["intercessor"]
    assert manifest["class_id_mapping"] == [
        {"project_class_id": 0, "model_class_id": 0, "label": "intercessor"}
    ]
    assert pending["id"] not in {record["id"] for record in manifest["images"]}


def test_nonconsecutive_ids_remap_without_changing_project_annotations(tmp_path):
    project = Project.create(tmp_path / "project", "Space Marines")
    for name in ("unused_captain", "terminator", "unused_scout", "intercessor", "unused_dreadnought"):
        project.add_class(name)
    annotations = [box(3), box(1, 0.5, 0.9)]
    for name, color, split in (("train.png", "red", "train"), ("val.png", "blue", "val")):
        record = import_picture(project, tmp_path, name, color)
        project.update_image(record["id"], annotations, True, split)
    metadata_before = (project.root / "project.json").read_bytes()
    dataset = project.build_dataset(tmp_path / "dataset")
    data = json.loads(dataset.read_text(encoding="utf-8"))
    assert data["names"] == ["terminator", "intercessor"]
    assert data["nc"] == 2
    for split in ("train", "val"):
        label_file = next((dataset.parent / "labels" / split).glob("*.txt"))
        model_ids = [int(line.split()[0]) for line in label_file.read_text().splitlines()]
        assert model_ids == [1, 0]
    manifest = json.loads((dataset.parent / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["classes"] == project.classes
    assert manifest["model_classes"] == data["names"]
    assert manifest["class_id_mapping"] == [
        {"project_class_id": 1, "model_class_id": 0, "label": "terminator"},
        {"project_class_id": 3, "model_class_id": 1, "label": "intercessor"},
    ]
    assert all(record["boxes"] == annotations for record in manifest["images"])
    assert all(record["boxes"] == annotations for record in project.images)
    assert (project.root / "project.json").read_bytes() == metadata_before
    reloaded = Project.load(project.root)
    assert reloaded.classes == project.classes
    second = reloaded.build_dataset(tmp_path / "reopened_dataset")
    assert json.loads(second.read_text(encoding="utf-8"))["names"] == data["names"]
    assert reloaded.add_class("new_unit") == 5
    assert reloaded.images[0]["boxes"] == annotations


def test_background_only_reviewed_dataset_rejects_without_output(tmp_path):
    project = Project.create(tmp_path / "project", "Space Marines")
    project.add_class("intercessor")
    for name, color in (("background_a.png", "white"), ("background_b.png", "black")):
        record = import_picture(project, tmp_path, name, color)
        project.update_image(record["id"], [], True, "")
    pending = import_picture(project, tmp_path, "unreviewed_unit.png", "blue")
    project.update_image(pending["id"], [box()], False, "")
    with pytest.raises(ProjectError, match="유닛 박스가 없습니다"):
        project.build_dataset(tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()


def test_observed_classes_still_require_train_validation_coverage(tmp_path):
    project = populated(tmp_path)
    project.add_class("unused_catalog_entry")
    project.add_class("ork_boy")
    record = import_picture(project, tmp_path, "ork.png", "purple")
    project.update_image(record["id"], [box(2)], True, "")
    with pytest.raises(ProjectError, match="'ork_boy'.*최소 2장"):
        project.build_dataset(tmp_path / "dataset")
