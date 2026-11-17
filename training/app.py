"""Local miniature annotation, detector training, and evaluation desktop app."""

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText
import uuid

from PIL import Image, ImageTk

from canvas import BoxCanvas
from project import Project


APP_DIR = Path(__file__).resolve().parent
LOCAL_DIR = APP_DIR / ".local"
SPLITS = {"자동 분할": "", "학습용": "train", "검증용": "val"}


def unique_name(prefix):
    return f"{prefix}-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


class TrainerApp(tk.Tk):
    def __init__(self, project_path=None):
        super().__init__()
        self.title("Miniature Trainer")
        self.geometry("1380x900")
        self.minsize(1120, 760)
        self.configure(background="#f0f3f6")
        self.project = None
        self.class_display_names = {}
        self.class_names = []
        self.image_index = -1
        self.selected_box = -1
        self.undo_stack = []
        self.events = queue.Queue()
        self.process = None
        self.job_kind = None
        self.job_dir = None
        self.job_result = None
        self.job_error = None
        self.cancel_file = None
        self.cancelled_checkpoint = None
        self.stopping = False
        self.io_busy = False
        self.preview_picture = None
        self.preview_photo = None
        self.status = tk.StringVar(value="새 프로젝트를 만들거나 기존 프로젝트 폴더를 열어 주세요.")
        self.project_title = tk.StringVar(value="Miniature Trainer")
        self.summary = tk.StringVar(value="유닛 사진 라벨링 · 객체 탐지 학습")
        self.class_label = tk.StringVar()
        self.split = tk.StringVar(value="자동 분할")
        self.reviewed = tk.BooleanVar(value=False)
        self.epochs = tk.StringVar(value="50")
        self.imgsz = tk.StringVar(value="640")
        self.batch = tk.StringVar(value="4")
        self.val_percent = tk.StringVar(value="20")
        self.device = tk.StringVar(value="auto")
        self.checkpoint = tk.StringVar()
        self.test_image = tk.StringVar()
        self.confidence = tk.StringVar(value="0.25")
        self.job_status = tk.StringVar(value="학습 대기 중")
        self.metrics = tk.StringVar()
        self.preview_status = tk.StringVar(value="학습 모델과 사진을 선택한 뒤 탐지를 실행하세요.")
        self._style()
        self._build()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Control-s>", lambda event: self.mark_reviewed())
        self.bind("<Control-z>", self.undo_shortcut)
        self.after(100, self.poll)
        if project_path:
            self.after(50, lambda: self.load_project(Path(project_path)))
        else:
            try:
                previous = json.loads((LOCAL_DIR / "state.json").read_text(encoding="utf-8"))
                root = Path(previous["project"])
                if (root / "project.json").is_file():
                    self.after(50, lambda: self.load_project(root))
            except (OSError, ValueError, KeyError):
                pass

    def _style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        self.option_add("*Font", ("맑은 고딕", 10))
        style.configure(".", font=("맑은 고딕", 10), background="#f0f3f6", foreground="#172331")
        style.configure("TButton", padding=(12, 7))
        style.configure("Accent.TButton", background="#087e75", foreground="white", padding=(14, 9))
        style.map("Accent.TButton", background=[("active", "#096b65"), ("disabled", "#a6bbb9")])
        style.configure("TNotebook.Tab", padding=(22, 10))
        style.configure("TLabelframe", padding=14)
        style.configure("TLabelframe.Label", font=("맑은 고딕", 11, "bold"))
        style.configure("Muted.TLabel", foreground="#596b7b")
        style.configure("Treeview", rowheight=30, fieldbackground="white", background="white")
        style.configure("TProgressbar", background="#087e75")

    def _build(self):
        header = tk.Frame(self, bg="#172331", padx=24, pady=18)
        header.pack(fill="x")
        tk.Label(header, textvariable=self.project_title, bg="#172331", fg="white",
                 font=("맑은 고딕", 20, "bold")).pack(side="left")
        ttk.Button(header, text="프로젝트 열기", command=self.open_project).pack(side="right", padx=(8, 0))
        ttk.Button(header, text="새 프로젝트", command=self.new_project).pack(side="right")
        ttk.Label(self, textvariable=self.summary, style="Muted.TLabel", padding=(24, 12)).pack(fill="x")
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=20)
        self.annotation_tab = ttk.Frame(self.tabs, padding=12)
        self.training_tab = ttk.Frame(self.tabs, padding=18)
        self.preview_tab = ttk.Frame(self.tabs, padding=18)
        self.tabs.add(self.annotation_tab, text="사진 · 라벨링")
        self.tabs.add(self.training_tab, text="모델 학습")
        self.tabs.add(self.preview_tab, text="탐지 시험 · 내보내기")
        self._annotation_ui()
        self._training_ui()
        self._preview_ui()
        ttk.Label(self, textvariable=self.status, style="Muted.TLabel", padding=(24, 12)).pack(fill="x")

    def _annotation_ui(self):
        frame = self.annotation_tab
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(1, weight=1)
        toolbar = ttk.Frame(frame)
        toolbar.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 12))
        ttk.Button(toolbar, text="사진 추가", command=self.import_images, style="Accent.TButton").pack(side="left")
        ttk.Label(toolbar, text="드래그로 박스 지정 · 휠로 확대", style="Muted.TLabel").pack(side="left", padx=16)
        self.editor = BoxCanvas(frame, self.commit_boxes, self.select_box, self.active_class)
        ttk.Radiobutton(toolbar, text="박스 그리기", variable=self.editor.mode, value="draw").pack(side="left", padx=8)
        ttk.Radiobutton(toolbar, text="선택 / 이동 / 크기", variable=self.editor.mode, value="select").pack(side="left", padx=8)
        ttk.Button(toolbar, text="화면에 맞춤", command=self.editor.fit).pack(side="right")
        left = ttk.Frame(frame, width=230)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        left.rowconfigure(1, weight=1)
        ttk.Label(left, text="프로젝트 사진", font=("맑은 고딕", 11, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.image_list = tk.Listbox(left, width=26, borderwidth=0, highlightthickness=1,
                                    highlightbackground="#d7dfe5", exportselection=False, activestyle="none",
                                    selectbackground="#d7eeeb", selectforeground="#123d38")
        self.image_list.grid(row=1, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(left, command=self.image_list.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        self.image_list.configure(yscrollcommand=scroll.set)
        self.image_list.bind("<<ListboxSelect>>", self.image_selected)
        self.editor.grid(row=1, column=1, sticky="nsew")
        right = ttk.Frame(frame, width=250)
        right.grid(row=1, column=2, sticky="nsew", padx=(14, 0))
        ttk.Label(right, text="유닛 종류", font=("맑은 고딕", 11, "bold")).pack(anchor="w")
        self.class_combo = ttk.Combobox(right, state="readonly", width=25)
        self.class_combo.pack(fill="x", pady=4)
        self.class_combo.bind("<<ComboboxSelected>>", self.update_class_label)
        label_row = ttk.Frame(right)
        label_row.pack(fill="x", pady=(0, 4))
        ttk.Label(label_row, text="Unity label", style="Muted.TLabel").pack(side="left", padx=(0, 6))
        ttk.Entry(label_row, textvariable=self.class_label, state="readonly", width=16,
                  font=("Consolas", 9)).pack(side="left", fill="x", expand=True)
        ttk.Button(right, text="유닛 종류 추가", command=self.add_class).pack(fill="x")
        ttk.Label(right, text="사진 속 모든 유닛을 하나씩 표시하세요.", style="Muted.TLabel",
                  wraplength=240).pack(anchor="w", pady=4)
        ttk.Label(right, text="이 사진의 박스", font=("맑은 고딕", 11, "bold")).pack(anchor="w")
        self.box_list = tk.Listbox(right, height=3, exportselection=False, borderwidth=0,
                                  selectbackground="#d7eeeb", selectforeground="#123d38")
        self.box_list.pack(fill="both", expand=True, pady=4)
        self.box_list.bind("<<ListboxSelect>>", self.box_selected)
        buttons = ttk.Frame(right)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="삭제", command=self.delete_box).pack(side="left")
        ttk.Button(buttons, text="종류 변경", command=self.relabel_box).pack(side="left", padx=4)
        ttk.Button(right, text="되돌리기  Ctrl+Z", command=self.undo).pack(fill="x", pady=4)
        ttk.Label(right, text="데이터 용도").pack(anchor="w")
        split_combo = ttk.Combobox(right, textvariable=self.split, values=list(SPLITS), state="readonly", width=25)
        split_combo.pack(fill="x", pady=4)
        split_combo.bind("<<ComboboxSelected>>", lambda event: self.save_split())
        ttk.Checkbutton(right, text="라벨링 완료 (학습에 포함)", variable=self.reviewed,
                        command=self.save_reviewed).pack(anchor="w")
        ttk.Label(right, text="박스 0개로 완료하면 배경 사진입니다.", style="Muted.TLabel").pack(anchor="w", pady=4)
        ttk.Button(right, text="완료 저장  Ctrl+S", style="Accent.TButton", command=self.mark_reviewed).pack(fill="x")
        nav = ttk.Frame(frame)
        nav.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        ttk.Button(nav, text="이전 사진", command=lambda: self.navigate(-1)).pack(side="left")
        ttk.Button(nav, text="다음 사진", command=lambda: self.navigate(1)).pack(side="left", padx=6)
        ttk.Label(nav, text="변경 내용은 자동 저장됩니다. 박스를 수정하면 완료 표시를 다시 확인하세요.",
                  style="Muted.TLabel").pack(side="left", padx=12)

    def _training_ui(self):
        controls = ttk.LabelFrame(self.training_tab, text="학습 설정")
        controls.pack(fill="x")
        fields = [("반복 횟수 (epoch)", self.epochs), ("입력 크기 (32의 배수)", self.imgsz),
                  ("배치 크기", self.batch), ("자동 검증 비율 (%)", self.val_percent)]
        for column, (label, variable) in enumerate(fields):
            ttk.Label(controls, text=label).grid(row=0, column=column, sticky="w", padx=(0, 22))
            ttk.Entry(controls, textvariable=variable, width=16).grid(row=1, column=column, sticky="w", pady=(6, 12))
        ttk.Label(controls, text="실행 장치").grid(row=0, column=4, sticky="w")
        ttk.Combobox(controls, textvariable=self.device, values=["auto", "0", "cpu"], state="readonly",
                     width=12).grid(row=1, column=4, sticky="w", pady=(6, 12))
        ttk.Label(controls, text="auto는 사용 가능한 NVIDIA GPU를 선택합니다. CPU도 지원합니다.",
                  style="Muted.TLabel").grid(row=2, column=0, columnspan=5, sticky="w")
        ttk.Label(self.training_tab, text="완료한 사진으로 데이터셋을 만들고 YOLOv8n 사전학습 모델을 미세 조정합니다.\n"
                  "검증용 사진은 다른 촬영 세션에서 선택하세요. 첫 실행에는 모델 다운로드가 필요합니다.",
                  style="Muted.TLabel").pack(anchor="w", pady=16)
        row = ttk.Frame(self.training_tab)
        row.pack(fill="x")
        self.train_button = ttk.Button(row, text="학습 시작", style="Accent.TButton", command=self.start_training)
        self.train_button.pack(side="left")
        self.resume_button = ttk.Button(row, text="중단한 학습 이어하기", command=self.resume_training)
        self.resume_button.pack(side="left", padx=8)
        self.stop_button = ttk.Button(row, text="작업 중단", command=self.stop_job, state="disabled")
        self.stop_button.pack(side="left")
        ttk.Button(row, text="결과 폴더 열기", command=self.open_results).pack(side="right")
        ttk.Label(self.training_tab, textvariable=self.job_status, font=("맑은 고딕", 12, "bold")).pack(anchor="w", pady=(18, 8))
        self.progress = ttk.Progressbar(self.training_tab, maximum=100)
        self.progress.pack(fill="x")
        ttk.Label(self.training_tab, textvariable=self.metrics, style="Muted.TLabel", wraplength=1000).pack(anchor="w", pady=8)
        self.log = ScrolledText(self.training_tab, height=12, background="#172331", foreground="#d9e6ee",
                                insertbackground="white", font=("Consolas", 10), relief="flat", state="disabled")
        self.log.pack(fill="both", expand=True)

    def _path_row(self, parent, label, variable, command):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(0, 8))
        ttk.Label(row, text=label, width=12).pack(side="left")
        ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row, text="찾아보기", command=command).pack(side="right")

    def _preview_ui(self):
        self._path_row(self.preview_tab, "학습 모델 (.pt)", self.checkpoint, self.choose_checkpoint)
        self._path_row(self.preview_tab, "시험할 사진", self.test_image, self.choose_test_image)
        row = ttk.Frame(self.preview_tab)
        row.pack(fill="x", pady=(4, 12))
        ttk.Label(row, text="신뢰도 기준").pack(side="left")
        ttk.Entry(row, textvariable=self.confidence, width=8).pack(side="left", padx=8)
        self.predict_button = ttk.Button(row, text="사진에서 탐지", command=self.predict, style="Accent.TButton")
        self.predict_button.pack(side="left", padx=4)
        self.export_button = ttk.Button(row, text="Unity ONNX 내보내기", command=self.export)
        self.export_button.pack(side="right")
        ttk.Label(self.preview_tab, textvariable=self.preview_status, wraplength=1050,
                  style="Muted.TLabel").pack(anchor="w", pady=(0, 10))
        self.preview = tk.Canvas(self.preview_tab, bg="#172331", highlightthickness=0)
        self.preview.pack(fill="both", expand=True)
        self.preview.bind("<Configure>", lambda event: self.draw_preview())

    def error(self, error):
        self.status.set(str(error))
        messagebox.showerror("작업을 완료하지 못했습니다", str(error), parent=self)

    def require_project(self):
        if self.io_busy:
            self.status.set("현재 사진 또는 데이터셋을 처리하고 있습니다.")
            return False
        if self.project is None:
            messagebox.showinfo("프로젝트 선택", "먼저 새 프로젝트를 만들거나 기존 폴더를 열어 주세요.", parent=self)
            return False
        return True

    def can_switch(self):
        if self.process is not None or self.io_busy:
            self.status.set("진행 중인 작업이 끝난 뒤 프로젝트를 전환할 수 있습니다.")
            return False
        return True

    def new_project(self):
        if not self.can_switch():
            return
        name = simpledialog.askstring("새 프로젝트", "프로젝트 이름", parent=self)
        if not name or not name.strip():
            return
        parent = filedialog.askdirectory(title="프로젝트를 저장할 상위 폴더", parent=self)
        if not parent:
            return
        try:
            # Keep display names unrestricted while creating a safe directory name.
            folder = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name.strip()).strip(". ")
            root = Path(parent) / (folder or "miniatures")
            self.set_project(Project.create(root, name.strip()))
        except Exception as exc:
            self.error(exc)

    def open_project(self):
        if not self.can_switch():
            return
        catalog_directory = APP_DIR / "projects" / "wahapedia-11e"
        options = {"initialdir": str(catalog_directory)} if catalog_directory.is_dir() else {}
        path = filedialog.askdirectory(title="project.json이 있는 프로젝트 폴더", parent=self, **options)
        if path:
            self.load_project(Path(path))

    def load_project(self, root):
        try:
            self.set_project(Project.load(root))
        except Exception as exc:
            self.error(exc)

    def set_project(self, project):
        self.project = project
        self.class_display_names = {}
        try:
            metadata = json.loads((project.root / "project.json").read_text(encoding="utf-8"))
            names = metadata.get("class_display_names", {}) if isinstance(metadata, dict) else {}
            if isinstance(names, dict):
                self.class_display_names = {
                    label: name.strip() for label, name in names.items()
                    if isinstance(name, str) and name.strip() and "\n" not in name and "\r" not in name
                }
        except (OSError, ValueError):
            pass
        self.project_title.set(project.name)
        self.title(f"{project.name} — Miniature Trainer")
        self.image_index = -1
        self.selected_box = -1
        self.undo_stack = []
        self.refresh_classes()
        self.refresh_images()
        self.editor.set_boxes([], self.class_names)
        self.editor.set_image()
        self.box_list.delete(0, "end")
        self.reviewed.set(False)
        self.split.set("자동 분할")
        self.checkpoint.set("")
        self.test_image.set("")
        self.preview_picture = None
        self.draw_preview()
        self.job_dir = None
        self.job_result = None
        self.job_error = None
        self.job_kind = None
        self.stopping = False
        self.cancel_file = None
        self.cancelled_checkpoint = None
        self.progress["value"] = 0
        self.metrics.set("")
        self.job_status.set("학습 대기 중")
        self.preview_status.set("학습 모델과 사진을 선택한 뒤 탐지를 실행하세요.")
        self.status.set(str(project.root))
        LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        (LOCAL_DIR / "state.json").write_text(json.dumps({"project": str(project.root)}), encoding="utf-8")
        if project.images:
            self.show_image(0)

    def refresh_classes(self):
        previous = self.class_combo.current()
        self.class_names = [self.class_display_names.get(label, label) for label in self.project.classes]
        self.class_combo["values"] = self.class_names
        if self.project.classes:
            self.class_combo.current(max(0, min(previous, len(self.project.classes) - 1)))
        else:
            self.class_combo.set("")
        self.update_class_label()

    def update_class_label(self, event=None):
        index = self.class_combo.current()
        label = self.project.classes[index] if self.project is not None and index >= 0 else ""
        self.class_label.set(label)

    def refresh_images(self):
        self.image_list.delete(0, "end")
        for index, item in enumerate(self.project.images):
            marker = "✓" if item["reviewed"] else "·"
            split = {"": "자동", "train": "학습", "val": "검증"}[item["split"]]
            display = item.get("source_name", Path(item["filename"]).name)
            self.image_list.insert("end", f"{marker} {index + 1:03d}  {display}  [{split}]")
        if self.image_index >= 0:
            self.image_list.selection_set(self.image_index)
            self.image_list.see(self.image_index)
        complete = sum(item["reviewed"] for item in self.project.images)
        boxes = sum(len(item["boxes"]) for item in self.project.images)
        self.summary.set(f"사진 {len(self.project.images)}장    완료 {complete}장    유닛 종류 {len(self.project.classes)}개    박스 {boxes}개")

    def import_images(self):
        if not self.require_project():
            return
        paths = filedialog.askopenfilenames(title="미니어처 사진 선택", parent=self,
                                           filetypes=[("사진", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"), ("모든 파일", "*.*")])
        if not paths:
            return
        self.io_busy = True
        self.editor.enabled = False
        self.status.set(f"사진 {len(paths)}장을 가져오고 있습니다…")

        def import_task():
            try:
                self.events.put(("imported", self.project.import_images([Path(p) for p in paths])))
            except Exception as exc:
                self.events.put(("io_error", str(exc)))

        threading.Thread(target=import_task, daemon=True).start()

    def add_class(self):
        if not self.require_project():
            return
        name = simpledialog.askstring("유닛 종류 추가", "유닛 식별 이름 (예: intercessor)\nUnity Units.json의 label과 맞춰 주세요.", parent=self)
        if name:
            try:
                index = self.project.add_class(name.strip())
                self.refresh_classes()
                self.class_combo.current(index)
                self.update_class_label()
                self.refresh_images()
                self.refresh_boxes()
            except Exception as exc:
                self.error(exc)

    def active_class(self):
        index = self.class_combo.current()
        if index < 0:
            self.status.set("유닛 종류를 먼저 추가해 주세요.")
        return index

    def current_image(self):
        if self.project is None or self.image_index < 0:
            return None
        return self.project.images[self.image_index]

    def image_selected(self, event):
        selection = self.image_list.curselection()
        if self.io_busy:
            self.image_list.selection_clear(0, "end")
            if self.image_index >= 0:
                self.image_list.selection_set(self.image_index)
            return
        if selection and selection[0] != self.image_index:
            self.show_image(selection[0])

    def show_image(self, index):
        self.image_index = index
        item = self.current_image()
        self.undo_stack = []
        self.selected_box = -1
        self.reviewed.set(item["reviewed"])
        self.split.set(next(label for label, value in SPLITS.items() if value == item["split"]))
        self.editor.set_boxes(item["boxes"], self.class_names)
        try:
            self.editor.set_image(self.project.root / item["filename"])
        except (OSError, ValueError) as exc:
            self.editor.set_image()
            self.error(exc)
        self.refresh_boxes()
        self.image_list.selection_clear(0, "end")
        self.image_list.selection_set(index)
        self.image_list.see(index)
        self.status.set(f'{index + 1} / {len(self.project.images)}  ·  {item["width"]} × {item["height"]}  ·  {item.get("source_name", item["filename"])}')

    def navigate(self, step):
        if self.project and self.project.images and not self.io_busy:
            self.show_image(max(0, min(len(self.project.images) - 1, self.image_index + step)))

    def refresh_boxes(self):
        self.box_list.delete(0, "end")
        item = self.current_image()
        if item is None:
            return
        for index, box in enumerate(item["boxes"]):
            self.box_list.insert("end", f'{index + 1}. {self.class_names[box["class_id"]]}')
        self.editor.set_boxes(item["boxes"], self.class_names, self.selected_box)
        if self.selected_box >= 0:
            self.box_list.selection_set(self.selected_box)

    def select_box(self, index):
        self.selected_box = index
        self.box_list.selection_clear(0, "end")
        if index >= 0:
            self.box_list.selection_set(index)

    def box_selected(self, event):
        selection = self.box_list.curselection()
        if selection:
            self.selected_box = selection[0]
            self.editor.selected = self.selected_box
            self.editor.draw_boxes()

    def commit_boxes(self, boxes, selected=-1):
        item = self.current_image()
        if item is None or self.io_busy:
            return
        previous = (deepcopy(item["boxes"]), item["reviewed"])
        try:
            self.project.update_image(item["id"], boxes, False, item["split"])
            self.undo_stack.append(previous)
            self.undo_stack = self.undo_stack[-40:]
            self.selected_box = selected
            self.reviewed.set(False)
            self.refresh_images()
            self.status.set("박스를 자동 저장했습니다. 모든 유닛을 표시한 뒤 완료 저장을 누르세요.")
        except Exception as exc:
            self.error(exc)
        self.refresh_boxes()

    def delete_box(self):
        item = self.current_image()
        if item and 0 <= self.selected_box < len(item["boxes"]):
            boxes = deepcopy(item["boxes"])
            del boxes[self.selected_box]
            self.commit_boxes(boxes)

    def relabel_box(self):
        item = self.current_image()
        class_id = self.active_class()
        if item and class_id >= 0 and 0 <= self.selected_box < len(item["boxes"]):
            boxes = deepcopy(item["boxes"])
            boxes[self.selected_box]["class_id"] = class_id
            self.commit_boxes(boxes, self.selected_box)

    def undo_shortcut(self, event):
        if self.tabs.select() == str(self.annotation_tab) and not isinstance(self.focus_get(), (ttk.Entry, tk.Entry)):
            self.undo()

    def undo(self):
        if not self.undo_stack or self.io_busy:
            return
        item = self.current_image()
        boxes, reviewed = self.undo_stack[-1]
        try:
            self.project.update_image(item["id"], boxes, reviewed, item["split"])
            self.undo_stack.pop()
            self.selected_box = -1
            self.reviewed.set(reviewed)
            self.refresh_boxes()
            self.refresh_images()
        except Exception as exc:
            self.error(exc)

    def save_reviewed(self):
        item = self.current_image()
        if item is None:
            return
        if self.io_busy:
            self.reviewed.set(item["reviewed"])
            return
        try:
            self.project.update_image(item["id"], item["boxes"], self.reviewed.get(), item["split"])
            self.refresh_images()
            self.status.set("라벨링 완료 상태를 저장했습니다.")
        except Exception as exc:
            self.reviewed.set(item["reviewed"])
            self.error(exc)

    def mark_reviewed(self):
        if self.current_image() is not None and not self.io_busy:
            self.reviewed.set(True)
            self.save_reviewed()

    def save_split(self):
        item = self.current_image()
        if item is None:
            return
        if self.io_busy:
            self.split.set(next(label for label, value in SPLITS.items() if value == item["split"]))
            return
        try:
            self.project.update_image(item["id"], item["boxes"], item["reviewed"], SPLITS[self.split.get()])
            self.refresh_images()
        except Exception as exc:
            self.split.set(next(label for label, value in SPLITS.items() if value == item["split"]))
            self.error(exc)

    def training_options(self):
        epochs, size, batch = int(self.epochs.get()), int(self.imgsz.get()), int(self.batch.get())
        fraction = float(self.val_percent.get()) / 100
        if epochs < 1 or batch < 1 or size < 64 or size > 2048 or size % 32 or not 0 < fraction < 1:
            raise ValueError("반복 횟수와 배치는 1 이상, 입력 크기는 64~2048 사이 32의 배수, 검증 비율은 0~100 사이여야 합니다.")
        return dict(epochs=epochs, imgsz=size, batch=batch, device=self.device.get()), fraction

    def start_training(self):
        if not self.require_project() or self.process is not None:
            return
        try:
            options, fraction = self.training_options()
        except ValueError as exc:
            self.error(exc)
            return
        root = self.project.root / "runs" / unique_name("train")
        self.io_busy = True
        self.editor.enabled = False
        self.status.set("학습·검증 데이터셋을 준비하고 있습니다…")
        self.job_status.set("데이터셋 준비 중")

        def prepare():
            try:
                data = self.project.build_dataset(root / "dataset", val_fraction=fraction)
                self.events.put(("dataset_ready", dict(kind="train", data=str(data), output=str(root / "model"), **options)))
            except Exception as exc:
                self.events.put(("io_error", str(exc)))

        threading.Thread(target=prepare, daemon=True).start()

    def resume_training(self):
        if not self.require_project() or self.process is not None:
            return
        checkpoint = filedialog.askopenfilename(title="중단된 학습의 last.pt 선택", filetypes=[("체크포인트", "*.pt")], parent=self)
        if checkpoint:
            try:
                self.launch_job(dict(kind="train", resume=checkpoint, device=self.device.get()))
            except Exception as exc:
                self.error(exc)

    def launch_job(self, job):
        if self.process is not None:
            return
        self.job_dir = self.project.root / "jobs" / unique_name(job["kind"])
        self.job_dir.mkdir(parents=True)
        self.cancel_file = self.job_dir / "cancel.request"
        job["cancel_file"] = str(self.cancel_file)
        job_file = self.job_dir / "job.json"
        job_file.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        cache = LOCAL_DIR / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                   YOLO_CONFIG_DIR=str(LOCAL_DIR / "ultralytics"), YOLO_AUTOINSTALL="false")
        self.job_kind = job["kind"]
        self.job_result = None
        self.job_error = None
        self.cancelled_checkpoint = None
        self.stopping = False
        try:
            self.process = subprocess.Popen([sys.executable, "-u", str(APP_DIR / "worker.py"), "--job", str(job_file)],
                                            cwd=cache, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            text=True, encoding="utf-8", errors="replace",
                                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except OSError as exc:
            self.error(exc)
            return
        self.set_job_controls(True)
        self.progress["value"] = 0
        self.metrics.set("")
        self.job_status.set({"train": "모델 준비 및 학습 중", "predict": "사진 탐지 중", "export": "ONNX 변환 및 검증 중"}[self.job_kind])
        self.status.set("진행 상황은 모델 학습 탭에서 확인할 수 있습니다.")
        self.append_log(f"\n--- {datetime.now():%H:%M:%S} {self.job_kind} ---\n")
        process = self.process
        log_file = self.job_dir / "worker.log"

        def consume():
            with log_file.open("w", encoding="utf-8") as log:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    self.events.put(("line", line))
            code = process.wait()
            self.events.put(("exited", code))

        threading.Thread(target=consume, daemon=True).start()

    def set_job_controls(self, running):
        for button in (self.train_button, self.resume_button, self.predict_button, self.export_button):
            button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")

    def stop_job(self):
        if self.process is None or self.stopping:
            return
        self.stopping = True
        self.job_status.set("중단 요청 · 현재 epoch의 체크포인트 저장 후 종료합니다" if self.job_kind == "train"
                            else "작업을 중단하고 있습니다")
        self.stop_button.configure(state="disabled")
        process = self.process
        if self.job_kind == "train":
            try:
                self.cancel_file.touch()
            except OSError as exc:
                self.stopping = False
                self.stop_button.configure(state="normal")
                self.error(exc)
            return

        def stop():
            if process.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    process.terminate()

        threading.Thread(target=stop, daemon=True).start()

    def choose_checkpoint(self):
        path = filedialog.askopenfilename(title="학습 모델 선택", filetypes=[("PyTorch 모델", "*.pt")], parent=self)
        if path:
            self.checkpoint.set(path)

    def choose_test_image(self):
        path = filedialog.askopenfilename(title="탐지할 사진 선택", filetypes=[("사진", "*.jpg *.jpeg *.png *.bmp *.webp")], parent=self)
        if path:
            self.test_image.set(path)
            try:
                self.load_preview(Path(path))
            except OSError as exc:
                self.error(exc)

    def predict(self):
        if not self.require_project() or self.process is not None:
            return
        try:
            model, image = Path(self.checkpoint.get()), Path(self.test_image.get())
            confidence = float(self.confidence.get())
            options, _ = self.training_options()
            if not model.is_file() or not image.is_file() or not 0 < confidence <= 1:
                raise ValueError("모델과 사진 파일을 선택하고 신뢰도 기준을 0 초과 1 이하로 입력하세요.")
            output = self.project.root / "predictions" / (unique_name("prediction") + ".png")
            self.preview_status.set("탐지 중…")
            self.launch_job(dict(kind="predict", model=str(model.resolve()), image=str(image.resolve()),
                                 output=str(output), confidence=confidence, imgsz=options["imgsz"], device=self.device.get()))
        except (ValueError, OSError) as exc:
            self.error(exc)

    def export(self):
        if not self.require_project() or self.process is not None:
            return
        try:
            model = Path(self.checkpoint.get())
            options, _ = self.training_options()
            if not model.is_file():
                raise ValueError("내보낼 학습 모델 (.pt)을 먼저 선택하세요.")
            output = self.project.root / "exports" / unique_name("unity")
            self.launch_job(dict(kind="export", model=str(model.resolve()), output=str(output), imgsz=options["imgsz"]))
        except (ValueError, OSError) as exc:
            self.error(exc)

    def load_preview(self, path):
        with Image.open(path) as source:
            self.preview_picture = source.convert("RGB")
        self.draw_preview()

    def draw_preview(self):
        self.preview.delete("all")
        if self.preview_picture is None:
            return
        size = (max(1, self.preview.winfo_width() - 24), max(1, self.preview.winfo_height() - 24))
        picture = self.preview_picture.copy()
        picture.thumbnail(size, Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(picture)
        self.preview.create_image(self.preview.winfo_width() / 2, self.preview.winfo_height() / 2, image=self.preview_photo)

    def open_results(self):
        if not self.require_project():
            return
        path = self.project.root
        if self.cancelled_checkpoint:
            path = Path(self.cancelled_checkpoint).parent
        if self.job_result:
            result_path = self.job_result.get("best") or self.job_result.get("output") or self.job_result.get("onnx")
            if result_path:
                candidate = Path(result_path)
                path = candidate if candidate.is_dir() else candidate.parent
        if os.name == "nt":
            os.startfile(path)
        else:
            self.status.set(str(path))

    def append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        if int(self.log.index("end-1c").split(".")[0]) > 2000:
            self.log.delete("1.0", "501.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def process_event(self, event):
        kind = event.get("event")
        if kind == "progress":
            epoch, epochs = event.get("epoch", 0), event.get("epochs", 1)
            self.progress["value"] = 100 * epoch / max(1, epochs)
            if not self.stopping:
                device = event.get("device", "")
                self.job_status.set(f"학습 중 · {epoch} / {epochs} epoch · 장치 {device}")
            metrics = event.get("metrics", {})
            self.metrics.set("    ".join(f"{key} = {value:.4f}" for key, value in metrics.items()
                                         if isinstance(value, (int, float))))
        elif kind == "result":
            self.job_result = event
        elif kind == "error":
            self.job_error = event.get("message", str(event))
        elif kind == "cancelled":
            self.cancelled_checkpoint = event.get("last")

    def job_finished(self, code):
        self.process = None
        self.set_job_controls(False)
        if self.stopping and self.job_result is None and self.job_error is None:
            if self.cancelled_checkpoint:
                self.job_status.set("학습을 중단했습니다. 저장한 last.pt로 이어할 수 있습니다.")
                self.checkpoint.set(self.cancelled_checkpoint)
            else:
                self.job_status.set("작업을 중단했습니다.")
            self.status.set(f"작업 로그 · {self.job_dir}")
            return
        if code != 0 or self.job_result is None:
            self.job_status.set("작업 실패 · 아래 로그를 확인하세요")
            self.error(self.job_error or f"작업이 종료 코드 {code}로 끝났습니다. 로그 위치: {self.job_dir}")
            return
        self.progress["value"] = 100
        self.job_status.set("작업 완료")
        result = self.job_result
        if self.job_kind == "train":
            self.checkpoint.set(result["best"])
            self.status.set(f'학습 완료 · {result["best"]}')
            self.preview_status.set("학습 모델을 불러왔습니다. 새 사진으로 탐지를 시험하세요.")
        elif self.job_kind == "predict":
            self.load_preview(Path(result["output"]))
            detections = result.get("detections", [])
            descriptions = ", ".join(f'{box.get("name", box.get("class", ""))} {box["score"]:.0%}' for box in detections[:12])
            self.preview_status.set(f"탐지 {len(detections)}개  ·  {descriptions}" if detections else "신뢰도 기준을 넘는 유닛이 없습니다.")
            self.status.set(f'탐지 결과 저장 · {result["output"]}')
        else:
            path = result.get("onnx") or result.get("output")
            self.preview_status.set(f"Unity용 ONNX와 레이블 내보내기 완료 · {path}")
            self.status.set(str(path))

    def poll(self):
        # Bound work per tick so logs cannot starve pointer input.
        for _ in range(100):
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            try:
                if kind == "line":
                    if payload.startswith("@@"):
                        self.process_event(json.loads(payload[2:]))
                    else:
                        self.append_log(payload)
                elif kind == "exited":
                    self.job_finished(payload)
                elif kind == "imported":
                    self.io_busy = False
                    self.editor.enabled = True
                    self.refresh_images()
                    if self.image_index < 0 and self.project.images:
                        self.show_image(0)
                    self.status.set(f'추가 {payload["added"]}장 · 중복 {payload["duplicates"]}장 · 실패 {len(payload["errors"])}장')
                    if payload["errors"]:
                        self.error("\n".join(payload["errors"][:10]))
                elif kind == "dataset_ready":
                    self.io_busy = False
                    self.editor.enabled = True
                    self.launch_job(payload)
                elif kind == "io_error":
                    self.io_busy = False
                    self.editor.enabled = True
                    self.job_status.set("대기 중")
                    self.error(payload)
            except Exception as exc:
                self.error(exc)
        self.after(100, self.poll)

    def close(self):
        if self.io_busy:
            self.status.set("사진 또는 데이터셋 저장이 끝나면 창을 닫을 수 있습니다.")
            return
        if self.process is not None:
            if messagebox.askyesno("작업 진행 중", "진행 중인 작업을 중단하고 종료할까요?", parent=self):
                self.stop_job()
                self.after(200, self.close_when_stopped)
            return
        self.destroy()

    def close_when_stopped(self):
        if self.process is None:
            self.destroy()
        else:
            self.after(200, self.close_when_stopped)


def main():
    parser = argparse.ArgumentParser(description="Miniature Trainer desktop application")
    parser.add_argument("--project", type=Path)
    args = parser.parse_args()
    app = TrainerApp(args.project)
    app.mainloop()


if __name__ == "__main__":
    main()
