"""Image-space bounding box editor shared by the desktop annotation view."""

from copy import deepcopy
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk


COLORS = ("#10b9a1", "#ffb454", "#69a9ff", "#dd8df5", "#ff7e87", "#bbd76b")


class BoxCanvas(ttk.Frame):
    def __init__(self, parent, on_change, on_select, get_class):
        super().__init__(parent)
        self.on_change = on_change
        self.on_select = on_select
        self.get_class = get_class
        self.mode = tk.StringVar(value="draw")
        self.picture = None
        self.photo = None
        self.boxes = []
        self.names = []
        self.selected = -1
        self.zoom = 1.0
        self.scale = 1.0
        self.origin = (0, 0)
        self.gesture = None
        self.enabled = True
        self.canvas = tk.Canvas(self, background="#172331", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        horizontal = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        vertical = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        horizontal.grid(row=1, column=0, sticky="ew")
        vertical.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.canvas.bind("<Configure>", lambda event: self.render())
        self.canvas.bind("<ButtonPress-1>", self.press)
        self.canvas.bind("<B1-Motion>", self.motion)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.canvas.bind("<MouseWheel>", self.wheel)
        self.canvas.bind("<Escape>", self.cancel)

    def set_image(self, path=None):
        self.picture = None
        if path is not None:
            with Image.open(path) as source:
                self.picture = source.convert("RGB")
        self.zoom = 1.0
        self.gesture = None
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self.render()

    def set_boxes(self, boxes, names, selected=-1):
        self.boxes = deepcopy(boxes)
        self.names = list(names)
        self.selected = selected
        self.draw_boxes()

    def fit(self):
        self.zoom = 1.0
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self.render()

    def wheel(self, event):
        self.zoom = max(1.0, min(5.0, self.zoom * (1.2 if event.delta > 0 else 1 / 1.2)))
        self.render()

    def render(self):
        self.canvas.delete("all")
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        if self.picture is None:
            self.canvas.create_text(width / 2, height / 2, text="사진을 추가하면 이곳에서 박스를 그릴 수 있습니다",
                                    fill="#bac9d8", font=("맑은 고딕", 12))
            return
        fit = min(max(1, width - 32) / self.picture.width, max(1, height - 32) / self.picture.height)
        self.scale = fit * self.zoom
        size = (max(1, round(self.picture.width * self.scale)), max(1, round(self.picture.height * self.scale)))
        self.origin = (max(16, (width - size[0]) / 2), max(16, (height - size[1]) / 2))
        self.photo = ImageTk.PhotoImage(self.picture.resize(size, Image.Resampling.LANCZOS))
        self.canvas.create_image(*self.origin, image=self.photo, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, max(width, size[0] + 32), max(height, size[1] + 32)))
        self.draw_boxes()

    def pixel_box(self, box):
        left, top = self.origin
        width = self.picture.width * self.scale
        height = self.picture.height * self.scale
        return (left + box["x1"] * width, top + box["y1"] * height,
                left + box["x2"] * width, top + box["y2"] * height)

    def draw_boxes(self):
        self.canvas.delete("box")
        if self.picture is None:
            return
        for index, box in enumerate(self.boxes):
            x1, y1, x2, y2 = self.pixel_box(box)
            color = COLORS[box["class_id"] % len(COLORS)]
            selected = index == self.selected
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=3 if selected else 2, tags="box")
            label = f'{index + 1}  {self.names[box["class_id"]]}'
            text = self.canvas.create_text(x1 + 5, y1 + 3, text=label, anchor="nw", fill="#10202b",
                                          font=("맑은 고딕", 10, "bold"), tags="box")
            bounds = self.canvas.bbox(text)
            rect = self.canvas.create_rectangle(bounds[0] - 3, bounds[1] - 2, bounds[2] + 3, bounds[3] + 2,
                                                fill=color, outline=color, tags="box")
            self.canvas.tag_lower(rect, text)
            if selected:
                for x, y in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
                    self.canvas.create_rectangle(x - 4, y - 4, x + 4, y + 4, fill="white", outline=color, tags="box")

    def point(self, event):
        x = (self.canvas.canvasx(event.x) - self.origin[0]) / (self.picture.width * self.scale)
        y = (self.canvas.canvasy(event.y) - self.origin[1]) / (self.picture.height * self.scale)
        return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))

    def press(self, event):
        if self.picture is None or not self.enabled:
            return
        self.canvas.focus_set()
        x, y = self.point(event)
        if self.mode.get() == "draw":
            class_id = self.get_class()
            if class_id < 0:
                return
            self.gesture = {"kind": "new", "start": (x, y), "original": deepcopy(self.boxes)}
            self.boxes.append(dict(class_id=class_id, x1=x, y1=y, x2=x, y2=y))
            self.selected = len(self.boxes) - 1
        else:
            # Test corner handles before interiors so small boxes can still be resized.
            if 0 <= self.selected < len(self.boxes):
                box = self.boxes[self.selected]
                for x_key, y_key in (("x1", "y1"), ("x2", "y1"), ("x1", "y2"), ("x2", "y2")):
                    if (abs(x - box[x_key]) * self.picture.width * self.scale < 9
                            and abs(y - box[y_key]) * self.picture.height * self.scale < 9):
                        self.gesture = dict(kind="resize", start=(x, y), original=deepcopy(self.boxes),
                                            keys=(x_key, y_key))
                        return
            self.selected = next((i for i in reversed(range(len(self.boxes)))
                                  if self.boxes[i]["x1"] <= x <= self.boxes[i]["x2"]
                                  and self.boxes[i]["y1"] <= y <= self.boxes[i]["y2"]), -1)
            self.on_select(self.selected)
            if self.selected >= 0:
                self.gesture = dict(kind="move", start=(x, y), original=deepcopy(self.boxes))
        self.draw_boxes()

    def motion(self, event):
        if self.gesture is None:
            return
        x, y = self.point(event)
        box = self.boxes[self.selected]
        start_x, start_y = self.gesture["start"]
        kind = self.gesture["kind"]
        if kind == "new":
            box.update(x1=min(x, start_x), y1=min(y, start_y), x2=max(x, start_x), y2=max(y, start_y))
        elif kind == "resize":
            x_key, y_key = self.gesture["keys"]
            box[x_key], box[y_key] = x, y
        else:
            original = self.gesture["original"][self.selected]
            dx = max(-original["x1"], min(1 - original["x2"], x - start_x))
            dy = max(-original["y1"], min(1 - original["y2"], y - start_y))
            box.update(x1=original["x1"] + dx, x2=original["x2"] + dx,
                       y1=original["y1"] + dy, y2=original["y2"] + dy)
        self.draw_boxes()

    def release(self, event):
        if self.gesture is None:
            return
        self.motion(event)
        original = self.gesture["original"]
        self.gesture = None
        box = self.boxes[self.selected]
        box["x1"], box["x2"] = sorted((box["x1"], box["x2"]))
        box["y1"], box["y2"] = sorted((box["y1"], box["y2"]))
        if ((box["x2"] - box["x1"]) * self.picture.width < 2
                or (box["y2"] - box["y1"]) * self.picture.height < 2):
            self.boxes = original
            self.selected = -1
            self.draw_boxes()
            return
        if self.boxes != original:
            self.on_change(deepcopy(self.boxes), self.selected)

    def cancel(self, event=None):
        if self.gesture is not None:
            self.boxes = self.gesture["original"]
            self.gesture = None
            self.selected = -1
            self.draw_boxes()
