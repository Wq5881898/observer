import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from observer.object_detection import (
    YoloOnnxBackend,
    detector_runtime_report,
    is_phone_class_name,
)


class _FakeSession:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output

    def get_inputs(self):
        return [SimpleNamespace(name="images")]

    def run(self, _outputs, _feed):
        return [self.output]


class ObjectDetectionTests(unittest.TestCase):
    def test_single_phone_model_does_not_advertise_person_detection(self) -> None:
        backend = YoloOnnxBackend(
            Path('unused.onnx'), session=_FakeSession(np.zeros((1, 5, 1))),
            single_class_name='cell_phone',
        )
        self.assertFalse(backend.supports_person_detection)
        coco = YoloOnnxBackend(
            Path('unused.onnx'), session=_FakeSession(np.zeros((1, 84, 1))),
        )
        self.assertTrue(coco.supports_person_detection)

    def test_single_class_prefilter_matches_multiclass_reference(self) -> None:
        rng = np.random.default_rng(42)
        rows = np.zeros((8400, 5), dtype=np.float32)
        rows[:, :2] = rng.uniform(0, 640, (8400, 2))
        rows[:, 2:4] = rng.uniform(5, 200, (8400, 2))
        rows[:, 4] = rng.uniform(0, 0.09, 8400)
        rows[:4, 4] = (0.25, 0.5, 0.5, 0.9)
        # Six-column output bypasses the single-class prefilter while
        # exercising the same decoder, coordinate clipping and NMS.
        reference = np.column_stack((rows, np.zeros(8400, dtype=np.float32)))
        backend = YoloOnnxBackend(
            Path('unused.onnx'), session=_FakeSession(rows),
            single_class_name='cell_phone',
        )
        backend.class_names = {0: 'cell_phone'}
        for threshold in (0.25, 0.1, float(np.float32(0.1)) + 1e-10):
            rows[4, 4] = np.float32(0.1)
            reference[:, :5] = rows
            backend.score_threshold = threshold
            expected = backend._decode(reference, 640, 480, 1.0, 0.0, 80.0)
            for raw in (rows, rows.T[None]):
                self.assertEqual(
                    backend._decode(raw, 640, 480, 1.0, 0.0, 80.0), expected
                )

    def test_phone_class_aliases_are_normalized(self) -> None:
        self.assertTrue(is_phone_class_name("cell phone"))
        self.assertTrue(is_phone_class_name("cell_phone"))
        self.assertFalse(is_phone_class_name("person"))
        self.assertFalse(is_phone_class_name(None))

    def test_yolo_common_channel_first_output_decodes_cell_phone(self) -> None:
        output = np.zeros((1, 84, 2), dtype=np.float32)
        # candidate 0: center box, COCO class 67 = cell phone
        output[0, 0:4, 0] = (320, 320, 160, 160)
        output[0, 67 + 4, 0] = 0.91
        # candidate 1 is lower confidence and should be filtered
        output[0, 0:4, 1] = (100, 100, 50, 50)
        output[0, 67 + 4, 1] = 0.10

        backend = YoloOnnxBackend(
            Path("unused.onnx"),
            score_threshold=0.25,
            session=_FakeSession(output),
        )
        detections = backend.detect(np.zeros((640, 640, 3), dtype=np.uint8), 123)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_name, "cell phone")
        self.assertAlmostEqual(detections[0].confidence, 0.91, places=4)
        self.assertAlmostEqual(detections[0].box.x, 0.375, places=4)
        self.assertGreaterEqual(detections[0].latency_ms, 0.0)

    def test_yolo_output_suppresses_overlapping_same_class(self) -> None:
        output = np.zeros((1, 2, 84), dtype=np.float32)
        output[0, 0, 0:4] = (320, 320, 200, 200)
        output[0, 0, 4 + 67] = 0.90
        output[0, 1, 0:4] = (322, 322, 200, 200)
        output[0, 1, 4 + 67] = 0.80

        backend = YoloOnnxBackend(
            Path("unused.onnx"),
            session=_FakeSession(output),
        )
        detections = backend.detect(np.zeros((640, 640, 3), dtype=np.uint8), 123)
        self.assertEqual(len(detections), 1)

    def test_yolo_single_class_output_decodes_with_explicit_name(self) -> None:
        output = np.zeros((1, 5, 1), dtype=np.float32)
        output[0, 0:4, 0] = (320, 320, 160, 160)
        output[0, 4, 0] = 0.88

        backend = YoloOnnxBackend(
            Path("unused.onnx"),
            score_threshold=0.25,
            session=_FakeSession(output),
            single_class_name="cell phone",
        )
        detections = backend.detect(np.zeros((640, 640, 3), dtype=np.uint8), 123)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_name, "cell phone")
        self.assertAlmostEqual(detections[0].confidence, 0.88, places=4)

    def test_yolo_single_class_output_decodes_with_project_label(self) -> None:
        output = np.zeros((1, 5, 1), dtype=np.float32)
        output[0, 0:4, 0] = (320, 320, 160, 160)
        output[0, 4, 0] = 0.88

        backend = YoloOnnxBackend(
            Path("unused.onnx"),
            score_threshold=0.25,
            session=_FakeSession(output),
            single_class_name="cell_phone",
        )
        detections = backend.detect(np.zeros((640, 640, 3), dtype=np.uint8), 123)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_name, "cell_phone")
        self.assertAlmostEqual(detections[0].confidence, 0.88, places=4)

    def test_yolo_single_class_output_without_name_is_not_mislabelled(self) -> None:
        output = np.zeros((1, 5, 1), dtype=np.float32)
        output[0, 0:4, 0] = (320, 320, 160, 160)
        output[0, 4, 0] = 0.88

        backend = YoloOnnxBackend(
            Path("unused.onnx"),
            score_threshold=0.25,
            session=_FakeSession(output),
        )
        detections = backend.detect(np.zeros((640, 640, 3), dtype=np.uint8), 123)

        self.assertEqual(detections, ())

    def test_detector_runtime_report_is_side_effect_free(self) -> None:
        report = detector_runtime_report(Path("does-not-exist.onnx"))
        self.assertEqual(report["backend"], "yolo_onnx")
        self.assertFalse(report["model"]["exists"])
        self.assertFalse(report["ready"])


if __name__ == "__main__":
    unittest.main()
