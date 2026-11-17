"""Regression checks through the real Tk canvas and project persistence."""

from pathlib import Path
import json
import sys
import time

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as app_module
from app import TrainerApp
from project import Project


@pytest.fixture(scope="module")
def tk_window(tmp_path_factory):
    # Use one Tcl interpreter, as the desktop app does. Repeated creation and
    # destruction of Tk interpreters is unreliable in the Windows Tk runtime.
    original_local = app_module.LOCAL_DIR
    app_module.LOCAL_DIR = tmp_path_factory.mktemp("tk-settings")
    window = TrainerApp()
    window.update()
    yield window
    for timer in window.tk.call("after", "info"):
        window.after_cancel(timer)
    window.destroy()
    app_module.LOCAL_DIR = original_local


@pytest.fixture
def desktop(tk_window, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "LOCAL_DIR", tmp_path / "settings")
    tk_window.test_errors = []
    tk_window.test_callbacks = []
    tk_window.error = lambda error: tk_window.test_errors.append(str(error))
    tk_window.report_callback_exception = lambda kind, error, trace: tk_window.test_callbacks.append(error)
    tk_window.editor.mode.set("draw")
    return tk_window


def project_with_photo(tmp_path, class_name="intercessor"):
    project = Project.create(tmp_path / "miniatures", "미니어처")
    if class_name:
        project.add_class(class_name)
    source = tmp_path / "tabletop.png"
    Image.new("RGB", (300, 180), "#283a46").save(source)
    project.import_images([source])
    return project


def drag(window, start, end):
    editor = window.editor
    canvas = editor.canvas

    def pixel(point):
        return (
            round(editor.origin[0] + point[0] * editor.picture.width * editor.scale - canvas.canvasx(0)),
            round(editor.origin[1] + point[1] * editor.picture.height * editor.scale - canvas.canvasy(0)),
        )

    x1, y1 = pixel(start)
    x2, y2 = pixel(end)
    canvas.event_generate("<ButtonPress-1>", x=x1, y=y1)
    canvas.event_generate("<B1-Motion>", x=x2, y=y2)
    canvas.event_generate("<ButtonRelease-1>", x=x2, y=y2)
    window.update()


def test_drag_normalizes_and_persists_all_boxes(desktop, tmp_path):
    project = project_with_photo(tmp_path)
    desktop.set_project(project)
    desktop.update()
    drag(desktop, (0.1, 0.2), (0.4, 0.8))
    drag(desktop, (0.9, 0.9), (0.6, 0.3))
    assert not desktop.test_callbacks
    assert not desktop.test_errors
    saved = Project.load(project.root).images[0]
    assert len(saved["boxes"]) == 2
    for box, expected in zip(saved["boxes"], ([0.1, 0.2, 0.4, 0.8], [0.6, 0.3, 0.9, 0.9])):
        assert [box[key] for key in ("x1", "y1", "x2", "y2")] == pytest.approx(expected, abs=0.006)
    desktop.load_project(project.root)
    assert desktop.editor.boxes == saved["boxes"]
    assert desktop.box_list.size() == 2


def test_new_class_is_immediately_drawable(desktop, tmp_path, monkeypatch):
    project = project_with_photo(tmp_path, class_name=None)
    desktop.set_project(project)
    desktop.update()
    monkeypatch.setattr(app_module.simpledialog, "askstring", lambda *args, **kwargs: "intercessor")
    desktop.add_class()
    drag(desktop, (0.1, 0.2), (0.4, 0.8))
    assert not desktop.test_callbacks
    assert not desktop.test_errors
    assert Project.load(project.root).images[0]["boxes"][0]["class_id"] == 0
    assert desktop.editor.names == ["intercessor"]


def test_catalog_display_names_keep_stable_annotation_ids(desktop, tmp_path, monkeypatch):
    project = project_with_photo(tmp_path)
    project.add_class("space_marines_captain")
    path = project.root / "project.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["class_display_names"] = {"intercessor": "Intercessor Squad", "space_marines_captain": "Captain"}
    path.write_text(json.dumps(metadata), encoding="utf-8")
    project = Project.load(project.root)

    desktop.set_project(project)
    desktop.update()
    assert tuple(desktop.class_combo["values"]) == ("Intercessor Squad", "Captain")
    assert desktop.editor.names == ["Intercessor Squad", "Captain"]
    desktop.class_combo.current(1)
    desktop.class_combo.event_generate("<<ComboboxSelected>>")
    desktop.update()
    assert desktop.active_class() == 1
    assert desktop.class_label.get() == "space_marines_captain"
    drag(desktop, (0.1, 0.2), (0.4, 0.8))
    assert desktop.box_list.get(0) == "1. Captain"
    stored = Project.load(project.root)
    assert stored.classes == ["intercessor", "space_marines_captain"]
    assert stored.images[0]["boxes"][0]["class_id"] == 1

    monkeypatch.setattr(app_module.simpledialog, "askstring", lambda *args, **kwargs: "custom_unit")
    desktop.add_class()
    assert desktop.class_combo.get() == "custom_unit"
    assert desktop.class_label.get() == "custom_unit"
    assert desktop.editor.names == ["Intercessor Squad", "Captain", "custom_unit"]
    desktop.load_project(project.root)
    assert desktop.editor.names == ["Intercessor Squad", "Captain", "custom_unit"]
    assert not desktop.test_callbacks
    assert not desktop.test_errors


@pytest.mark.parametrize("mapping", [[], "invalid", {"intercessor": None}, {"intercessor": "line\nbreak"}])
def test_invalid_catalog_display_metadata_falls_back_to_label(desktop, tmp_path, mapping):
    project = project_with_photo(tmp_path)
    path = project.root / "project.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["class_display_names"] = mapping
    path.write_text(json.dumps(metadata), encoding="utf-8")
    desktop.set_project(project)
    assert desktop.class_combo.get() == "intercessor"
    assert desktop.class_label.get() == "intercessor"
    assert desktop.editor.names == ["intercessor"]


def test_open_project_starts_in_imported_catalog_folder(desktop, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "APP_DIR", tmp_path)
    catalog_directory = tmp_path / "projects" / "wahapedia-11e"
    catalog_directory.mkdir(parents=True)
    dialog_options = {}

    def choose_folder(**options):
        dialog_options.update(options)
        return ""

    monkeypatch.setattr(app_module.filedialog, "askdirectory", choose_folder)
    desktop.open_project()
    assert dialog_options["initialdir"] == str(catalog_directory)


def test_annotation_controls_remain_visible_at_minimum_window_size(desktop):
    previous_geometry = desktop.geometry()
    try:
        desktop.geometry("1120x760")
        desktop.update()
        sidebar = desktop.class_combo.master
        for control in sidebar.winfo_children():
            assert control.winfo_ismapped()
            assert control.winfo_y() + control.winfo_height() <= sidebar.winfo_height()
        assert desktop.box_list.winfo_height() >= 35
        assert desktop.editor.canvas.winfo_width() > 250
    finally:
        desktop.geometry(previous_geometry)
        desktop.update()


def test_review_flag_resets_on_edit_and_undo_restores_it(desktop, tmp_path):
    project = project_with_photo(tmp_path)
    desktop.set_project(project)
    desktop.update()
    drag(desktop, (0.1, 0.2), (0.4, 0.8))
    desktop.mark_reviewed()
    assert Project.load(project.root).images[0]["reviewed"] is True
    drag(desktop, (0.6, 0.3), (0.9, 0.9))
    assert Project.load(project.root).images[0]["reviewed"] is False
    assert desktop.reviewed.get() is False
    desktop.undo()
    restored = Project.load(project.root).images[0]
    assert restored["reviewed"] is True
    assert len(restored["boxes"]) == 1
    assert desktop.reviewed.get() is True


def test_explicit_background_review_is_saved(desktop, tmp_path):
    project = project_with_photo(tmp_path)
    desktop.set_project(project)
    desktop.mark_reviewed()
    stored = Project.load(project.root).images[0]
    assert stored["boxes"] == []
    assert stored["reviewed"] is True


def test_project_switch_clears_previous_result_and_annotation_state(desktop, tmp_path):
    first = project_with_photo(tmp_path)
    desktop.set_project(first)
    desktop.mark_reviewed()
    desktop.split.set("검증용")
    desktop.save_split()
    desktop.job_result = {"best": str(first.root / "old-model.pt")}
    desktop.job_error = "old error"
    desktop.job_kind = "train"
    desktop.progress["value"] = 100
    desktop.metrics.set("old metrics")
    second = Project.create(tmp_path / "empty", "새 프로젝트")
    desktop.set_project(second)
    assert desktop.current_image() is None
    assert desktop.editor.picture is None
    assert desktop.editor.boxes == []
    assert desktop.job_result is None
    assert desktop.job_error is None
    assert desktop.job_kind is None
    assert desktop.class_combo.get() == ""
    assert desktop.reviewed.get() is False
    assert desktop.split.get() == "자동 분할"
    assert float(desktop.progress["value"]) == 0
    assert desktop.metrics.get() == ""


def test_dataset_preparation_error_restores_editing(desktop, tmp_path, monkeypatch):
    project = project_with_photo(tmp_path)
    desktop.set_project(project)

    def failed_export(*args, **kwargs):
        raise OSError("dataset disk error")

    monkeypatch.setattr(project, "build_dataset", failed_export)
    desktop.start_training()
    deadline = time.monotonic() + 3
    while desktop.io_busy and time.monotonic() < deadline:
        desktop.update()
        time.sleep(0.01)
    assert desktop.io_busy is False
    assert desktop.editor.enabled is True
    assert desktop.process is None
    assert desktop.test_errors == ["dataset disk error"]
    assert str(desktop.train_button["state"]) != "disabled"


def test_split_save_error_restores_displayed_split(desktop, tmp_path, monkeypatch):
    project = project_with_photo(tmp_path)
    desktop.set_project(project)

    def failed_save(*args, **kwargs):
        raise OSError("disk error")

    monkeypatch.setattr(project, "update_image", failed_save)
    desktop.split.set("검증용")
    desktop.save_split()
    assert desktop.test_errors == ["disk error"]
    assert desktop.split.get() == "자동 분할"
    assert Project.load(project.root).images[0]["split"] == ""


def test_move_and_resize_keep_boxes_inside_picture(desktop, tmp_path):
    project = project_with_photo(tmp_path)
    desktop.set_project(project)
    desktop.update()
    drag(desktop, (0.1, 0.2), (0.4, 0.8))
    desktop.editor.mode.set("select")
    drag(desktop, (0.25, 0.5), (1.0, 1.0))
    moved = Project.load(project.root).images[0]["boxes"][0]
    assert moved["x2"] == pytest.approx(1)
    assert moved["y2"] == pytest.approx(1)
    assert moved["x1"] == pytest.approx(0.7, abs=0.006)
    assert moved["y1"] == pytest.approx(0.4, abs=0.006)
    drag(desktop, (moved["x1"], moved["y1"]), (0.5, 0.2))
    resized = Project.load(project.root).images[0]["boxes"][0]
    assert resized["x1"] == pytest.approx(0.5, abs=0.006)
    assert resized["y1"] == pytest.approx(0.2, abs=0.006)
    assert resized["x2"] == pytest.approx(1)
    assert resized["y2"] == pytest.approx(1)
    assert not desktop.test_callbacks
