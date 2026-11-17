from copy import deepcopy
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catalog import sync_projects
from project import Project, ProjectError


def catalogue():
    return {
        "source": {"url": "https://wahapedia.ru/wh40k11ed/", "edition": "11", "retrieved_on": "2026-11-17"},
        "factions": [
            {"id": "space-marines", "name": "Space Marines", "source_url": "https://example.test/marines",
             "labels": ["zeta_unit", "alpha_unit"]},
            {"id": "orks", "name": "Orks", "labels": ["boy"]},
        ],
        "units": [
            {"label": "zeta_unit", "name": "Zeta Unit", "faction": "Space Marines"},
            {"label": "alpha_unit", "name": "Alpha Unit", "faction": "Space Marines"},
            {"label": "boy", "name": "Boy", "faction": "Orks"},
        ],
    }


def write_catalog(tmp_path, contents):
    path = tmp_path / "Units.json"
    path.write_text(json.dumps(contents), encoding="utf-8")
    return path


def test_create_factions_in_display_order_with_shared_labels_and_metadata(tmp_path):
    contents = catalogue()
    contents["factions"][1]["labels"].append("alpha_unit")
    source = write_catalog(tmp_path, contents)
    summaries = sync_projects(source, tmp_path / "projects")
    assert [(item["id"], item["unit_count"], item["added"]) for item in summaries] == [
        ("space-marines", 2, 2), ("orks", 2, 2),
    ]
    marines = Project.load(Path(summaries[0]["path"]))
    assert marines.classes == ["alpha_unit", "zeta_unit"]
    assert marines._data["class_display_names"] == {"alpha_unit": "Alpha Unit", "zeta_unit": "Zeta Unit"}
    assert marines._data["faction_id"] == "space-marines"
    assert marines._data["catalog_source"]["edition"] == "11"
    assert marines._data["catalog_source"]["faction_url"] == "https://example.test/marines"
    assert Project.load(Path(summaries[1]["path"])).classes == ["alpha_unit", "boy"]
    index = json.loads((tmp_path / "projects/index.json").read_text(encoding="utf-8"))
    assert len(index["projects"]) == 2
    assert index["source"] == contents["source"]


def test_rerun_preserves_class_ids_images_custom_labels_and_revision(tmp_path):
    contents = catalogue()
    source = write_catalog(tmp_path, contents)
    output = tmp_path / "projects"
    sync_projects(source, output)
    project = Project.load(output / "space-marines")
    project.add_class("custom_unit")
    photo = tmp_path / "photo.png"
    Image.new("RGB", (20, 20), "blue").save(photo)
    project.import_images([photo])
    record = project.images[0]
    boxes = [{"class_id": 1, "x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9}]
    project.update_image(record["id"], boxes, True, "train")
    before = deepcopy(project._data)
    index_time = (output / "index.json").stat().st_mtime_ns
    summaries = sync_projects(source, output)
    assert all(summary["added"] == 0 for summary in summaries)
    assert Project.load(project.root)._data == before
    assert (output / "index.json").stat().st_mtime_ns == index_time
    contents["units"].append({"label": "aardvark", "name": "Aardvark"})
    contents["factions"][0]["labels"].append("aardvark")
    contents["units"][0]["name"] = "Zeta Unit Revised"
    write_catalog(tmp_path, contents)
    summary = sync_projects(source, output)[0]
    updated = Project.load(project.root)
    assert summary["added"] == 1
    assert updated.classes == ["alpha_unit", "zeta_unit", "custom_unit", "aardvark"]
    assert updated.images == before["images"]
    assert updated._data["revision"] == before["revision"] + 1
    assert updated._data["class_display_names"]["zeta_unit"] == "Zeta Unit Revised"
    stable_revision = updated._data["revision"]
    sync_projects(source, output)
    assert Project.load(project.root)._data["revision"] == stable_revision


def test_legends_display_name_preserves_canonical_label_and_name(tmp_path):
    contents = catalogue()
    contents["units"][0]["display_name"] = "Zeta Unit [Legends]"
    source = write_catalog(tmp_path, contents)
    output = tmp_path / "projects"
    sync_projects(source, output)
    project = Project.load(output / "space-marines")
    assert project.classes == ["alpha_unit", "zeta_unit"]
    assert project._data["class_display_names"]["zeta_unit"] == "Zeta Unit [Legends]"
    assert json.loads(source.read_text())["units"][0]["name"] == "Zeta Unit"


@pytest.mark.parametrize("invalid", ["case_collision", "unsafe_id", "reserved_id", "missing_label", "duplicate_label"])
def test_invalid_catalogue_is_rejected_before_any_project_writes(tmp_path, invalid):
    contents = catalogue()
    if invalid == "case_collision":
        contents["units"].append({"label": "ALPHA_UNIT", "name": "Other"})
    elif invalid == "unsafe_id":
        contents["factions"][1]["id"] = "../escape"
    elif invalid == "reserved_id":
        contents["factions"][1]["id"] = "con"
    elif invalid == "missing_label":
        contents["factions"][1]["labels"].append("missing")
    else:
        contents["factions"][1]["labels"].append("boy")
    source = write_catalog(tmp_path, contents)
    output = tmp_path / "projects"
    with pytest.raises(ProjectError):
        sync_projects(source, output)
    assert not output.exists()


def test_existing_project_collision_preflight_leaves_other_projects_unchanged(tmp_path):
    contents = catalogue()
    source = write_catalog(tmp_path, contents)
    output = tmp_path / "projects"
    marines = Project.create(output / "space-marines", "My custom name")
    orks = Project.create(output / "orks", "Orks")
    orks.add_class("BOY")
    before = deepcopy(marines._data)
    with pytest.raises(ProjectError, match="BOY"):
        sync_projects(source, output)
    assert Project.load(marines.root)._data == before
    assert not (output / "index.json").exists()


def test_save_failure_rolls_back_in_memory_changes_and_keeps_index(tmp_path, monkeypatch):
    contents = catalogue()
    source = write_catalog(tmp_path, contents)
    output = tmp_path / "projects"
    sync_projects(source, output)
    before = Project.load(output / "space-marines")._data
    index_bytes = (output / "index.json").read_bytes()
    contents["units"].append({"label": "new_unit", "name": "New Unit"})
    contents["factions"][0]["labels"].append("new_unit")
    write_catalog(tmp_path, contents)
    attempted = []

    def fail_save(self):
        attempted.append(self)
        raise OSError("simulated disk failure")

    monkeypatch.setattr(Project, "save", fail_save)
    with pytest.raises(OSError):
        sync_projects(source, output)
    assert attempted[0]._data == before
    assert Project.load(output / "space-marines")._data == before
    assert (output / "index.json").read_bytes() == index_bytes
