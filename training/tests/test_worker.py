"""Checks the exported tensor contract and worker failure reporting."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


WORKER_PATH = Path(__file__).resolve().parents[1] / "worker.py"
SPEC = importlib.util.spec_from_file_location("training_worker", WORKER_PATH)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class WorkerTests(unittest.TestCase):
    def test_validation_and_json_events(self):
        self.assertEqual(worker.image_size({"imgsz": 640}), 640)
        for invalid in (0, -32, 33, 64.0, True):
            with self.assertRaises(ValueError):
                worker.image_size({"imgsz": invalid})
        with self.assertRaises(ValueError):
            worker.absolute_path({"output": "relative"}, "output")
        self.assertEqual(worker.finite_metrics({"good": 0.5, "nan": float("nan"), "bad": None}), {"good": 0.5})
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            worker.emit("progress", message="학습", epoch=1)
        self.assertEqual(json.loads(stream.getvalue().strip()[2:])["message"], "학습")

    def test_failed_job_has_nonzero_exit_and_error_event(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary) / "job.json"
            job.write_text('{"kind": "unknown"}', encoding="utf-8")
            process = subprocess.run([sys.executable, str(WORKER_PATH), "--job", str(job)],
                                     capture_output=True, text=True, encoding="utf-8")
        self.assertNotEqual(process.returncode, 0)
        events = [json.loads(line[2:]) for line in process.stdout.splitlines() if line.startswith("@@")]
        self.assertEqual(events[-1]["event"], "error")

    def test_resume_uses_checkpoint_dataset_without_new_data_argument(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            data = root / "dataset.yaml"
            data.write_text("{}")
            run = root / "run"
            weights = run / "weights"
            weights.mkdir(parents=True)
            last = weights / "last.pt"
            last.touch()
            (weights / "best.pt").touch()
            called = {}
            model = SimpleNamespace(
                task="detect",
                ckpt={"epoch": 0, "optimizer": {}, "train_args": {
                    "data": str(data), "save_dir": str(run), "epochs": 3,
                }},
                trainer=SimpleNamespace(save_dir=run, metrics={}),
                add_callback=lambda name, callback: None,
                train=lambda **options: called.update(options),
            )
            with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=lambda path: model)}), \
                    patch.object(worker, "select_device", return_value="cpu"), \
                    contextlib.redirect_stdout(io.StringIO()):
                worker.train({"resume": str(last), "device": "cpu"})
            self.assertEqual(called, {"resume": True, "device": "cpu", "workers": 0})

    def test_resume_rejects_finished_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary).resolve() / "last.pt"
            checkpoint.touch()
            model = SimpleNamespace(task="detect", ckpt={"epoch": -1, "optimizer": None})
            with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=lambda path: model)}), \
                    patch.object(worker, "select_device", return_value="cpu"), \
                    self.assertRaises(ValueError):
                worker.train({"resume": str(checkpoint)})

    def test_export_decodes_coordinates_scores_and_int32_ids(self):
        import numpy as np
        import onnx
        import onnxruntime as ort
        from onnx import TensorProto, helper

        graph = helper.make_graph(
            [helper.make_node("Identity", ["input"], ["raw_output"])], "test_yolo_head",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 6, 3])],
            [helper.make_tensor_value_info("raw_output", TensorProto.FLOAT, [1, 6, 3])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
        model.ir_version = 9
        raw = np.array([[[10, 100, 0], [20, 200, 0], [4, 20, 2], [8, 40, 2],
                         [0.8, 0.1, 0.3], [0.2, 0.9, 0.3]]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "raw.onnx"
            exported = Path(temporary) / "model.onnx"
            onnx.save(model, source)
            worker.adapt_onnx(source, exported, 2)
            session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
            boxes, classes, scores = session.run(None, {"input": raw})
            with self.assertRaises(ValueError):
                worker.adapt_onnx(source, exported, 3)
        np.testing.assert_allclose(boxes, [[8, 16, 12, 24], [90, 180, 110, 220], [-1, -1, 1, 1]])
        np.testing.assert_array_equal(classes, [0, 1, 0])
        np.testing.assert_allclose(scores, [0.8, 0.9, 0.3])
        self.assertEqual(classes.dtype, np.int32)


if __name__ == "__main__":
    unittest.main()
