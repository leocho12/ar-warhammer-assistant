"""Background model jobs for Miniature Trainer. Run with --job <JSON file>."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback


APP_DIR = Path(__file__).resolve().parent


class TrainingCancelled(Exception):
    def __init__(self, last: str | None = None):
        self.last = last
        super().__init__("학습을 중단했습니다.")


def emit(event: str, **values) -> None:
    # A leading newline separates events from Ultralytics' carriage-return progress bars.
    print("\n@@" + json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def positive_integer(job: dict, key: str, default: int) -> int:
    value = job.get(key, default)
    if type(value) is not int or value < 1:
        raise ValueError(f"{key} 값은 1 이상의 정수여야 합니다.")
    return value


def image_size(job: dict) -> int:
    size = positive_integer(job, "imgsz", 640)
    if size % 32:
        raise ValueError("입력 크기는 32의 배수여야 합니다.")
    return size


def absolute_path(job: dict, key: str, must_exist: bool = False) -> Path:
    value = job.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} 경로를 지정해 주세요.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{key} 경로는 절대 경로여야 합니다.")
    if must_exist and not path.is_file():
        raise ValueError(f"파일을 찾을 수 없습니다: {path}")
    return path.resolve()


def select_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if requested not in ("cpu", "0"):
        raise ValueError("학습 장치는 auto, cpu, 0 중 하나여야 합니다.")
    if requested == "0" and not torch.cuda.is_available():
        raise ValueError("CUDA GPU를 사용할 수 없습니다. CPU 또는 자동 선택을 사용하거나 GPU용 PyTorch를 설치해 주세요.")
    return requested


def finite_metrics(metrics: dict) -> dict:
    result = {}
    for key, value in metrics.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result[str(key)] = number
    return result


def train(job: dict) -> None:
    from ultralytics import YOLO

    device = select_device(job.get("device", "auto"))
    cancel_file = absolute_path(job, "cancel_file") if job.get("cancel_file") else None
    if cancel_file and cancel_file.exists():
        raise TrainingCancelled()
    resume = job.get("resume")
    if resume:
        model_path = absolute_path(job, "resume", must_exist=True)
    else:
        data = absolute_path(job, "data", must_exist=True)
        output = absolute_path(job, "output")
        epochs = positive_integer(job, "epochs", 50)
        batch = positive_integer(job, "batch", 8)
        size = image_size(job)
        if output.exists():
            raise ValueError("학습 결과 폴더가 이미 있습니다. 새 결과 폴더를 지정해 주세요.")
        model_name = job.get("model", "yolov8n.pt")
        if model_name == "yolov8n.pt":
            cache = APP_DIR / ".cache"
            cache.mkdir(parents=True, exist_ok=True)
            model_path = cache / model_name
        else:
            model_path = absolute_path(job, "model", must_exist=True)
        output.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    if model.task != "detect":
        raise ValueError("객체 탐지 모델만 학습할 수 있습니다.")
    if resume:
        checkpoint = model.ckpt or {}
        saved = checkpoint.get("train_args", {})
        if checkpoint.get("epoch", -1) < 0 or checkpoint.get("optimizer") is None:
            raise ValueError("이 체크포인트의 학습은 이미 완료되었습니다. 새 학습의 시작 모델로 선택해 주세요.")
        original_data = Path(str(saved.get("data", "")))
        if not original_data.is_absolute() or not original_data.is_file():
            raise ValueError("체크포인트의 원래 학습 데이터셋을 찾을 수 없습니다. 원래 데이터셋 폴더를 복원해 주세요.")
        output = Path(str(saved.get("save_dir", "")))
        if not output.is_absolute() or output.resolve() != model_path.parent.parent:
            raise ValueError("원래 학습 결과 폴더의 weights/last.pt를 선택해 주세요.")
        epochs = int(saved["epochs"])
        options = {"resume": True, "device": device, "workers": 0}
    else:
        options = {
            "data": str(data), "epochs": epochs, "imgsz": size, "batch": batch,
            "device": device, "workers": 0, "cache": False,
            # Ultralytics reserves name="model". A dot under the requested project
            # directory keeps every valid output basename literal, including "model".
            "project": str(output), "name": ".", "exist_ok": False,
            "seed": 42, "deterministic": True, "save": True, "plots": True,
            "amp": device != "cpu",
        }
    emit("progress", epoch=0, epochs=epochs, metrics={}, device=device,
         message="모델과 학습 데이터를 준비하고 있습니다.")
    if cancel_file and cancel_file.exists():
        raise TrainingCancelled(str(model_path) if resume else None)

    last_reported_epoch = -1

    def epoch_finished(trainer):
        nonlocal last_reported_epoch
        # Ultralytics invokes this hook again after final validation, after stripping
        # optimizer state. Only the first invocation has a resumable checkpoint.
        if trainer.epoch == last_reported_epoch:
            return
        last_reported_epoch = trainer.epoch
        metrics = finite_metrics(trainer.metrics)
        metrics.update(finite_metrics(trainer.label_loss_items(trainer.tloss, prefix="train")))
        emit("progress", epoch=trainer.epoch + 1, epochs=trainer.epochs,
             metrics=metrics, device=device)
        if cancel_file and cancel_file.exists():
            # This hook runs after checkpoint save, before final_eval strips optimizer state.
            raise TrainingCancelled(str(Path(trainer.last).resolve()))

    model.add_callback("on_fit_epoch_end", epoch_finished)
    model.train(**options)
    actual_output = Path(model.trainer.save_dir).resolve()
    if actual_output != output.resolve():
        raise RuntimeError(f"학습 결과가 지정한 폴더와 다른 위치에 저장되었습니다: {actual_output}")
    best = actual_output / "weights" / "best.pt"
    last = actual_output / "weights" / "last.pt"
    if not best.is_file() or not last.is_file():
        raise RuntimeError("학습이 종료되었지만 모델 체크포인트를 찾을 수 없습니다.")
    emit("result", kind="train", output=str(actual_output), best=str(best), last=str(last),
         results_csv=str(actual_output / "results.csv"),
         metrics=finite_metrics(model.trainer.metrics))


def predict(job: dict) -> None:
    from PIL import Image, ImageOps
    from ultralytics import YOLO

    checkpoint = absolute_path(job, "model", must_exist=True)
    source = absolute_path(job, "image", must_exist=True)
    output = absolute_path(job, "output")
    size = image_size(job)
    confidence = job.get("confidence", 0.25)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 < confidence <= 1:
        raise ValueError("신뢰도 기준은 0보다 크고 1 이하여야 합니다.")
    if output.exists():
        raise ValueError("예측 결과 파일이 이미 있습니다. 새 파일명을 지정해 주세요.")
    if output.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        raise ValueError("예측 결과 파일은 PNG 또는 JPG여야 합니다.")
    device = select_device(job.get("device", "auto"))
    model = YOLO(str(checkpoint))
    if model.task != "detect":
        raise ValueError("객체 탐지 모델을 선택해 주세요.")
    with Image.open(source) as original:
        picture = ImageOps.exif_transpose(original).convert("RGB")
    result = model.predict(picture, imgsz=size, conf=float(confidence), device=device, verbose=False)[0]
    detections = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        detections.append({"class_id": class_id, "name": model.names[class_id],
                           "score": float(box.conf.item()), "xyxy": box.xyxy[0].tolist()})
    output.parent.mkdir(parents=True, exist_ok=True)
    # plot() returns BGR. Pillow writes Unicode Windows paths reliably.
    preview = Image.fromarray(result.plot()[:, :, ::-1])
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=output.suffix, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        preview.save(temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    emit("result", kind="predict", output=str(output), detections=detections,
         width=picture.width, height=picture.height)


def adapt_onnx(raw_path: Path, destination: Path, class_count: int) -> None:
    """Decode YOLOv8's [1, 4+C, N] output to the sample's three output tensors."""
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(raw_path))
    if len(model.graph.output) != 1:
        raise ValueError("출력이 하나인 YOLOv8 탐지 ONNX 모델이 필요합니다.")
    raw_output = model.graph.output[0]
    shape = [dimension.dim_value for dimension in raw_output.type.tensor_type.shape.dim]
    if len(shape) != 3 or shape[0] != 1 or shape[1] != 4 + class_count or shape[2] <= 0:
        raise ValueError(f"지원하지 않는 탐지 출력 형태입니다: {shape}")
    count = shape[2]
    prefix = "miniature_decode_"

    def constant(name, dtype, values):
        tensor_name = prefix + name
        model.graph.initializer.append(helper.make_tensor(tensor_name, dtype, [len(values)], values))
        return tensor_name

    axes = constant("batch_axis", TensorProto.INT64, [0])
    head_sizes = constant("head_sizes", TensorProto.INT64, [4, class_count])
    coordinate_sizes = constant("coordinate_sizes", TensorProto.INT64, [2, 2])
    divisor = constant("divisor", TensorProto.FLOAT, [2.0])
    node_specs = [
        ("Squeeze", [raw_output.name, axes], [prefix + "head"], {}),
        ("Split", [prefix + "head", head_sizes], [prefix + "coords", prefix + "classes"], {"axis": 0}),
        ("Transpose", [prefix + "coords"], [prefix + "xywh"], {"perm": [1, 0]}),
        ("Transpose", [prefix + "classes"], [prefix + "class_scores"], {"perm": [1, 0]}),
        ("Split", [prefix + "xywh", coordinate_sizes], [prefix + "xy", prefix + "wh"], {"axis": 1}),
        ("Div", [prefix + "wh", divisor], [prefix + "half_wh"], {}),
        ("Sub", [prefix + "xy", prefix + "half_wh"], [prefix + "minimum"], {}),
        ("Add", [prefix + "xy", prefix + "half_wh"], [prefix + "maximum"], {}),
        ("Concat", [prefix + "minimum", prefix + "maximum"], ["boxes"], {"axis": 1}),
        ("ArgMax", [prefix + "class_scores"], [prefix + "class_ids64"], {"axis": 1, "keepdims": 0}),
        ("Cast", [prefix + "class_ids64"], ["class_ids"], {"to": TensorProto.INT32}),
        ("ReduceMax", [prefix + "class_scores"], ["scores"], {"axes": [1], "keepdims": 0}),
    ]
    for index, (operation, inputs, outputs, attributes) in enumerate(node_specs):
        model.graph.node.append(helper.make_node(operation, inputs, outputs,
                                                name=prefix + str(index), **attributes))
    del model.graph.output[:]
    model.graph.output.extend([
        helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [count, 4]),
        helper.make_tensor_value_info("class_ids", TensorProto.INT32, [count]),
        helper.make_tensor_value_info("scores", TensorProto.FLOAT, [count]),
    ])
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(destination))


def verify_export(raw_path: Path, destination: Path, size: int) -> None:
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    raw = ort.InferenceSession(str(raw_path), sess_options=options, providers=["CPUExecutionProvider"])
    decoded = ort.InferenceSession(str(destination), sess_options=options, providers=["CPUExecutionProvider"])
    sample = np.random.default_rng(42).random((1, 3, size, size), dtype=np.float32)
    prediction = raw.run(None, {raw.get_inputs()[0].name: sample})[0][0].T
    center, half_size = prediction[:, :2], prediction[:, 2:4] / 2
    expected_boxes = np.concatenate([center - half_size, center + half_size], axis=1)
    boxes, classes, scores = decoded.run(None, {decoded.get_inputs()[0].name: sample})
    np.testing.assert_allclose(boxes, expected_boxes, rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(classes, prediction[:, 4:].argmax(axis=1))
    np.testing.assert_allclose(scores, prediction[:, 4:].max(axis=1), rtol=1e-4, atol=1e-5)
    if classes.dtype != np.int32:
        raise RuntimeError("Unity 출력의 클래스 ID가 int32가 아닙니다.")


def export(job: dict) -> None:
    from ultralytics import YOLO

    checkpoint = absolute_path(job, "model", must_exist=True)
    destination = absolute_path(job, "output")
    size = image_size(job)
    if destination.exists():
        raise ValueError("내보내기 폴더가 이미 있습니다. 새 폴더를 지정해 주세요.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".export-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        copied_checkpoint = staging / "source.pt"
        shutil.copy2(checkpoint, copied_checkpoint)
        model = YOLO(str(copied_checkpoint))
        if model.task != "detect":
            raise ValueError("YOLOv8 객체 탐지 모델을 선택해 주세요.")
        labels = [model.names[index] for index in range(len(model.names))]
        raw_path = Path(model.export(format="onnx", imgsz=size, batch=1, dynamic=False,
                                     simplify=False, opset=15, nms=False, half=False, device="cpu"))
        final_model = staging / "miniature_detector.onnx"
        adapt_onnx(raw_path, final_model, len(labels))
        verify_export(raw_path, final_model, size)
        (staging / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
        manifest = {
            "format": "Unity Inference Engine detection tensors",
            "source_checkpoint": str(checkpoint), "input_shape": [1, 3, size, size],
            "input_dtype": "float32", "input_color": "RGB", "input_range": [0, 1],
            "outputs": [
                {"name": "boxes", "dtype": "float32", "shape": ["N", 4], "coordinates": "xMin,yMin,xMax,yMax in input pixels"},
                {"name": "class_ids", "dtype": "int32", "shape": ["N"]},
                {"name": "scores", "dtype": "float32", "shape": ["N"]},
            ],
            "labels": labels, "opset": 15, "nms_in_model": False,
            "preprocessing": "YOLO training/prediction uses aspect-preserving letterbox resize. Match preprocessing and inverse box mapping in Unity before deployment.",
            "verification": "ONNX checker and ONNX Runtime decode equivalence passed. Unity/device execution has not been tested.",
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        copied_checkpoint.unlink()
        raw_path.unlink()
        # Publish only after conversion and numeric validation both succeed.
        os.replace(staging, destination)
    emit("result", kind="export", output=str(destination),
         model=str(destination / "miniature_detector.onnx"), labels=str(destination / "labels.txt"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    arguments = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(APP_DIR / ".cache" / "ultralytics"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    try:
        job = json.loads(arguments.job.read_text(encoding="utf-8"))
        if not isinstance(job, dict):
            raise ValueError("작업 파일은 JSON 객체여야 합니다.")
        handlers = {"train": train, "predict": predict, "export": export}
        if job.get("kind") not in handlers:
            raise ValueError("지원하지 않는 작업 종류입니다.")
        handlers[job["kind"]](job)
        return 0
    except TrainingCancelled as cancelled:
        emit("cancelled", message=str(cancelled), last=cancelled.last)
        return 0
    except Exception as error:
        traceback.print_exc()
        emit("error", message=str(error), error_type=type(error).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
