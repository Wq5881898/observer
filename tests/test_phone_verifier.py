import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from observer.object_detection import ObjectDetection
from observer.observations import NormalizedBox
from observer.phone_verifier import PhoneProposalVerifier, VerifiedPhoneBackend


class PhoneVerifierTests(unittest.TestCase):
    def test_filters_phone_only_and_preserves_overlapping_person(self):
        box = NormalizedBox(.2, .2, .1, .1)
        phone = ObjectDetection('cell_phone', .9, box, 'base')
        person = ObjectDetection('person', .8, box, 'base')
        retained = ObjectDetection('cell phone', .6, NormalizedBox(.7, .4, .1, .2), 'base')
        source = Mock(model_id='base', supports_person_detection=True)
        source.detect.return_value = (phone, person, retained)
        verifier = Mock(sha256='a'*64)
        verifier.reject.side_effect = (True, False)
        backend = VerifiedPhoneBackend(source, verifier)
        frame = np.zeros((480, 640, 3), np.uint8)
        actual = backend.detect(frame, 123)
        self.assertEqual([(d.class_name, d.confidence, d.box) for d in actual],
                         [('person', .8, box), ('cell phone', .6, retained.box)])
        self.assertEqual(verifier.reject.call_count, 2)
        source.detect.assert_called_once_with(frame, 123)
        self.assertTrue(backend.supports_person_detection)
        backend.close()
        source.close.assert_called_once_with()

    def test_changed_weights_rejected_before_opencv_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.xml'
            path.write_bytes(b'changed-model')
            expected = hashlib.sha256(b'original-model').hexdigest()
            with self.assertRaisesRegex(ValueError, '校验失败'):
                PhoneProposalVerifier(path, expected)

    def test_empty_or_zero_area_crop_retains_unknown_proposal(self):
        # No model call should turn an invalid crop into evidence of non-phone.
        verifier = PhoneProposalVerifier.__new__(PhoneProposalVerifier)
        frame = np.zeros((480, 640, 3), np.uint8)
        self.assertFalse(verifier.reject(frame, NormalizedBox(.4, .3, 0, .1)))
        self.assertFalse(verifier.reject(frame, NormalizedBox(1, 1, 0, 0)))


if __name__ == '__main__':
    unittest.main()
