import unittest
from pathlib import Path
from unittest.mock import patch

from observer.cli import build_parser, _vision_smoke


class NaturalBackendTests(unittest.TestCase):
    def test_yolo_options_parse_without_changing_default(self):
        parser = build_parser()
        self.assertEqual(parser.parse_args(['vision-smoke']).detector_backend, 'efficientdet')
        args = parser.parse_args(['vision-smoke', '--detector-backend', 'yolo',
                                 '--yolo-model', 'candidate.onnx', '--seconds', '300'])
        self.assertEqual(args.yolo_model, Path('candidate.onnx'))
        self.assertEqual(args.seconds, 300)

    def test_missing_yolo_model_fails_before_camera_access(self):
        with patch('observer.cli.CameraWorker') as camera:
            with self.assertRaises(ValueError):
                _vision_smoke(None, Path('models'), 30, detector_backend='yolo')
            camera.assert_not_called()

    def test_event_commands_accept_the_frozen_yolo_backend(self):
        parser = build_parser()
        common = [
            '--profile', 'data/calibration/default.json', '--role', 'tuning',
            '--detector-backend', 'yolo', '--yolo-model',
            'models/yolo/observer-phone-s304-e2b.onnx',
        ]
        away = parser.parse_args(['validate-away-events', *common])
        phone = parser.parse_args([
            'validate-phone-events', *common,
            '--phone-score-threshold', '0.25', '--phone-retain-seconds', '1.5',
            '--phone-center-distance', '0.18',
        ])
        self.assertEqual(away.detector_backend, 'yolo')
        self.assertEqual(phone.yolo_model, Path('models/yolo/observer-phone-s304-e2b.onnx'))
        self.assertEqual(phone.phone_retain_seconds, 1.5)
