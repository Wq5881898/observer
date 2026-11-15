"""Conservative, locally trained phone-proposal verification.

This stage can only remove phone proposals. It never masks a screen region or
changes another detector class, the YOLO threshold, or the tracking rules.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time

from observer.models import sha256_file
from observer.object_detection import is_phone_class_name


class PhoneProposalVerifier:
    padding = 0.12
    reject_margin = 0.5

    def __init__(self, path: Path, expected_sha256: str):
        import cv2

        if sha256_file(path).lower() != expected_sha256.lower():
            raise ValueError('手机复核模型校验失败；未打开摄像头')
        self.sha256 = expected_sha256.lower()
        self._hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
        self._svm = cv2.ml.SVM_load(str(path))
        if (not self._svm.isTrained() or self._svm.getVarCount() != 1764
                or self._svm.getType() != cv2.ml.SVM_C_SVC):
            raise ValueError('手机复核模型格式不匹配；未打开摄像头')

    def feature(self, frame, box):
        import cv2
        import numpy as np

        h, w = frame.shape[:2]
        x, y, bw, bh = box.x, box.y, box.width, box.height
        pad = self.padding
        x1, y1 = max(0, int((x-pad*bw)*w)), max(0, int((y-pad*bh)*h))
        x2 = min(w, int((x+(1+pad)*bw)*w+1))
        y2 = min(h, int((y+(1+pad)*bh)*h+1))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or bw <= 0 or bh <= 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        scale = min(64/gray.shape[1], 64/gray.shape[0])
        nw, nh = max(1, round(gray.shape[1]*scale)), max(1, round(gray.shape[0]*scale))
        canvas = np.full((64, 64), 127, np.uint8)
        canvas[(64-nh)//2:(64-nh)//2+nh, (64-nw)//2:(64-nw)//2+nw] = cv2.resize(gray, (nw, nh))
        feature = self._hog.compute(canvas).ravel()
        return feature / (np.linalg.norm(feature)+1e-8)

    def reject(self, frame, box) -> bool:
        import cv2

        feature = self.feature(frame, box)
        # An unusable crop is not evidence of a non-phone.
        if feature is None:
            return False
        _, raw = self._svm.predict(feature.reshape(1, -1), flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
        # This model's labels are -1=non-phone, +1=phone; raw sign was checked
        # against the training labels before export. The digest pins that model.
        return float(raw[0, 0]) >= self.reject_margin


class VerifiedPhoneBackend:
    def __init__(self, backend, verifier: PhoneProposalVerifier):
        self._backend, self.verifier = backend, verifier
        self.model_id = backend.model_id + '+proposal-' + verifier.sha256[:12]
        self.last_latency_ms = 0.0

    @property
    def supports_person_detection(self):
        return self._backend.supports_person_detection

    def detect(self, bgr_frame, timestamp_ms):
        started = time.perf_counter()
        proposals = self._backend.detect(bgr_frame, timestamp_ms)
        kept = tuple(d for d in proposals
                     if not is_phone_class_name(d.class_name)
                     or not self.verifier.reject(bgr_frame, d.box))
        self.last_latency_ms = (time.perf_counter()-started)*1000
        return tuple(replace(d, model_id=self.model_id, latency_ms=self.last_latency_ms) for d in kept)

    def close(self):
        self._backend.close()
