"""Create or update faction annotation projects from the shared Unity unit catalogue."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import sys

from project import Project, ProjectError, _write_json


TRAINING_DIR = Path(__file__).resolve().parent
DEFAULT_CATALOG = TRAINING_DIR.parent / "Assets" / "WarhammerAssistant" / "Units.json"
DEFAULT_OUTPUT = TRAINING_DIR / "projects" / "wahapedia-11e"
RESERVED_FOLDERS = {"con", "prn", "aux", "nul"} | {
    f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(1, 10)
}


def _text(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip() or any(
        character in value for character in ("\n", "\r", "\0")
    ):
        raise ProjectError(f"카탈로그의 {field} 값은 앞뒤 공백이 없는 한 줄이어야 합니다.")
    return value


def _read_catalog(path: Path) -> tuple[dict, list[dict], dict[str, dict]]:
    try:
        catalogue = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ProjectError(f"유닛 카탈로그를 읽을 수 없습니다. {error}") from error
    if not isinstance(catalogue, dict):
        raise ProjectError("유닛 카탈로그는 JSON 객체여야 합니다.")
    units, factions = catalogue.get("units"), catalogue.get("factions")
    source = catalogue.get("source", {})
    if not isinstance(units, list) or not isinstance(factions, list) or not isinstance(source, dict):
        raise ProjectError("카탈로그에 units와 factions 목록, source 객체가 필요합니다.")
    try:
        json.dumps(source, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ProjectError("카탈로그의 source에 올바르지 않은 값이 있습니다.") from error
    unit_map = {}
    folded_labels = set()
    for unit in units:
        if not isinstance(unit, dict):
            raise ProjectError("유닛 항목은 JSON 객체여야 합니다.")
        label = _text(unit.get("label"), "유닛 label")
        _text(unit.get("name"), f"{label}의 name")
        if "display_name" in unit:
            _text(unit["display_name"], f"{label}의 display_name")
        if label.casefold() in folded_labels:
            raise ProjectError(f"대소문자를 제외하면 중복되는 유닛 label이 있습니다. {label}")
        folded_labels.add(label.casefold())
        unit_map[label] = unit
    faction_ids = set()
    for faction in factions:
        if not isinstance(faction, dict):
            raise ProjectError("진영 항목은 JSON 객체여야 합니다.")
        identifier = _text(faction.get("id"), "진영 id")
        if re.fullmatch(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", identifier) is None or identifier in RESERVED_FOLDERS:
            raise ProjectError(f"진영 id는 폴더로 사용할 수 있는 영문 소문자·숫자·하이픈·밑줄이어야 합니다. {identifier}")
        if identifier in faction_ids:
            raise ProjectError(f"진영 id가 중복되었습니다. {identifier}")
        faction_ids.add(identifier)
        _text(faction.get("name"), f"{identifier}의 name")
        labels = faction.get("labels")
        if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
            raise ProjectError(f"{identifier}의 labels는 문자열 목록이어야 합니다.")
        if len(set(labels)) != len(labels):
            raise ProjectError(f"{identifier}의 유닛 목록에 중복 label이 있습니다.")
        missing = [label for label in labels if label not in unit_map]
        if missing:
            raise ProjectError(f"{identifier}가 카탈로그에 없는 유닛을 참조합니다. {', '.join(missing[:5])}")
        if "source_url" in faction and not isinstance(faction["source_url"], str):
            raise ProjectError(f"{identifier}의 source_url은 문자열이어야 합니다.")
    return source, factions, unit_map


def sync_projects(catalog_path: Path, output_root: Path) -> list[dict]:
    """Append catalogue labels without changing existing annotation class IDs.

    All source records and existing projects are validated before any project is
    written. Each project save is atomic. A failed multi-project sync can be retried.
    The index is published only after all projects have been synchronized.
    """
    catalog_path, output_root = Path(catalog_path).resolve(), Path(output_root).resolve()
    source, factions, units = _read_catalog(catalog_path)
    plans = []
    for faction in factions:
        root = (output_root / faction["id"]).resolve()
        if not root.is_relative_to(output_root):
            raise ProjectError(f"진영 폴더가 출력 폴더 바깥을 가리킵니다. {faction['id']}")
        project = Project.load(root) if (root / "project.json").exists() else None
        labels = sorted(faction["labels"], key=lambda label: (units[label]["name"].casefold(), label))
        existing = project.classes if project else []
        folded = {}
        for label in existing:
            if label.casefold() in folded:
                raise ProjectError(f"{faction['name']} 프로젝트에 대소문자만 다른 label이 있습니다. {label}")
            folded[label.casefold()] = label
        for label in labels:
            matched = folded.get(label.casefold())
            if matched is not None and matched != label:
                raise ProjectError(f"{faction['name']}의 기존 label '{matched}'와 카탈로그 '{label}'가 충돌합니다.")
        if project:
            assigned_faction = project._data.get("faction_id")
            if assigned_faction is not None and assigned_faction != faction["id"]:
                raise ProjectError(f"다른 진영에 연결된 프로젝트 폴더입니다. {root}")
            display_names = project._data.get("class_display_names", {})
            if not isinstance(display_names, dict) or any(
                not isinstance(key, str) or not isinstance(value, str) for key, value in display_names.items()
            ):
                raise ProjectError(f"유닛 표시 이름 정보가 올바르지 않습니다. {root}")
        plans.append((faction, root, project, labels))

    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for faction, root, project, labels in plans:
        if project is None:
            project = Project.create(root, faction["name"])
        before = deepcopy(project._data)
        new_labels = [label for label in labels if label not in project.classes]
        project.classes.extend(new_labels)
        display_names = dict(project._data.get("class_display_names", {}))
        display_names.update({label: units[label].get("display_name", units[label]["name"]) for label in labels})
        catalog_source = deepcopy(source)
        if faction.get("source_url"):
            catalog_source["faction_url"] = faction["source_url"]
        project._data.update(class_display_names=display_names, faction_id=faction["id"],
                             catalog_source=catalog_source)
        if project._data != before:
            try:
                project.save()
            except Exception:
                project._data = before
                raise
        summaries.append({"id": faction["id"], "name": faction["name"], "path": str(root),
                          "unit_count": len(labels), "added": len(new_labels)})

    index = {"source": source, "projects": [
        {key: value for key, value in summary.items() if key != "added"} for summary in summaries
    ]}
    index_path = output_root / "index.json"
    previous_index = None
    if index_path.is_file():
        try:
            previous_index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    if index != previous_index:
        _write_json(index_path, index)
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="공유 Unity Units.json 경로")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="진영 프로젝트를 저장할 상위 폴더")
    arguments = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    try:
        summaries = sync_projects(arguments.catalog, arguments.output)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    for summary in summaries:
        print(f"{summary['name']} · {summary['unit_count']}종 · 추가 {summary['added']}종 · {summary['path']}")
    print(f"진영 {len(summaries)}개 동기화 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
