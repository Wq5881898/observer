import unittest

from observer.config import AppConfig, load_config


class ConfigTests(unittest.TestCase):
    def test_default_config_is_valid(self) -> None:
        config = load_config()
        self.assertEqual(config.camera.width, 640)
        self.assertEqual(config.camera.height, 480)
        self.assertLess(config.inference.object_fps, config.inference.face_fps)
        self.assertEqual(config.resources.max_inference_workers, 3)
        self.assertEqual(len(config.rules), 4)
        phone = next(rule for rule in config.rules if rule.rule_id == "phone_held")
        self.assertEqual(phone.trigger_after_seconds, 10.0)

    def test_unknown_section_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AppConfig.from_mapping({"experimental": {}})

    def test_model_rate_cannot_exceed_capture_rate(self) -> None:
        with self.assertRaises(ValueError):
            AppConfig.from_mapping(
                {
                    "camera": {"capture_fps": 5.0},
                    "inference": {"face_fps": 8.0},
                }
            )

    def test_duplicate_rule_ids_are_rejected(self) -> None:
        rule = {
            "rule_id": "duplicate",
            "signal_name": "phone_held",
            "trigger_after_seconds": 1.0,
        }
        with self.assertRaises(ValueError):
            AppConfig.from_mapping({"rules": [rule, rule]})


if __name__ == "__main__":
    unittest.main()
