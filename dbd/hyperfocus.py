from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional
import threading
import time
import numpy as np
import cv2

try:
    import pytesseract
    _HAS_TESS = True
except Exception:
    _HAS_TESS = False

Slot = int
ROI = Tuple[int, int, int, int]  # x,y,w,h Full-HD

@dataclass
class HyperfocusConfig:
    enabled: bool = True
    slot: Slot = 2
    use_full_frame_for_ocr: bool = False
    rois: Dict[Slot, ROI] = None

    def __post_init__(self):
        if self.rois is None:
            # Standard-Quadrate um HUD (Beispielwerte, Full-HD)
            self.rois = {
                1: (1745, 860, 25, 25),
                2: (1815, 920, 25, 25),
                3: (1745, 990, 25, 25),
                4: (1675, 920, 25, 25),
            }

class Hyperfocus:
    def __init__(self, cfg: Optional[HyperfocusConfig] = None) -> None:
        self.cfg = cfg or HyperfocusConfig()
        self.tokens_live: int = 0
        self._last_full_from_app: Optional[np.ndarray] = None
        self._dbg_pre: Optional[np.ndarray] = None
        self._dbg_roi_orig: Optional[np.ndarray] = None   # NEU: Original-ROI
        self._worker_th: Optional[threading.Thread] = None
        self._worker_run: bool = False
        self._worker_hz: float = 10.0

    def from_settings(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        self.cfg.enabled = bool(data.get("enabled", self.cfg.enabled))
        self.cfg.slot = int(data.get("slot", self.cfg.slot))
        self.cfg.use_full_frame_for_ocr = bool(data.get("use_full_frame_for_ocr", self.cfg.use_full_frame_for_ocr))
        ro = data.get("rois")
        if isinstance(ro, dict):
            try:
                for k, roi in ro.items():
                    slot = int(k)
                    x, y, w, h = [int(v) for v in roi]
                    self.cfg.rois[int(slot)] = (x, y, w, h)
            except Exception:
                pass

    def adjust_delay_ms(self, base_ms: float) -> float:
        t = max(0, int(self.tokens_live))
        return float(base_ms) / (1.0 + 0.04 * t)  # ~4% pro Token

    @staticmethod
    def _safe_crop(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        if img is None or getattr(img, "size", 0) == 0:
            return np.zeros((1,1,3), np.uint8)
        H, W = img.shape[:2]
        if w <= 0 or h <= 0 or H <= 0 or W <= 0:
            return np.zeros((1,1,3), np.uint8)
        x0 = max(0, min(W-1, int(x)))
        y0 = max(0, min(H-1, int(y)))
        x1 = max(0, min(W, x0 + int(w)))
        y1 = max(0, min(H, y0 + int(h)))
        if x1 <= x0 or y1 <= y0:
            return np.zeros((1,1,3), np.uint8)
        return np.ascontiguousarray(img[y0:y1, x0:x1]).copy()

    @staticmethod
    def _preprocess(crop: np.ndarray) -> np.ndarray:
        """Minimaler Preprocess für Ziffern-OCR."""
        if crop is None or getattr(crop, "size", 0) == 0:
            return np.zeros((1,1), np.uint8)
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        g = cv2.resize(g, (g.shape[1]*2, g.shape[0]*2), interpolation=cv2.INTER_LINEAR)
        g = cv2.GaussianBlur(g, (3,3), 0)
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        return np.ascontiguousarray(bw)

    def _ocr_digit(self, bw_img: np.ndarray) -> Optional[int]:
        if not _HAS_TESS or bw_img is None or bw_img.size == 0:
            return None
        cfg = (
            "--psm 10 --oem 3 "
            "-c tessedit_char_whitelist=0123456 "
            "-c load_system_dawg=0 -c load_freq_dawg=0 "
            "-c classify_bln_numeric_mode=1"
        )
        try:
            import pytesseract
            txt = pytesseract.image_to_string(bw_img, config=cfg)
            for ch in txt:
                if ch in "0123456":
                    return int(ch)
            return None
        except Exception:
            return None

    def _ocr_once(self) -> tuple[str, Optional[np.ndarray]]:
        if not self.cfg.enabled:
            self.tokens_live = 0
            self._dbg_pre = None
            self._dbg_roi_orig = None  # NEU: reset
            return "Detected: 0", None
        src = self._last_full_from_app
        if src is None or getattr(src, "size", 0) == 0:
            self.tokens_live = 0
            self._dbg_pre = None
            self._dbg_roi_orig = None  # NEU: reset
            return "Detected: 0", None
        roi = self.cfg.rois.get(int(self.cfg.slot), (0, 0, 0, 0))
        crop = self._safe_crop(src, *roi)
        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            self.tokens_live = 0
            self._dbg_pre = None
            self._dbg_roi_orig = None  # NEU: reset
            return f"OCR error: empty ROI for slot {self.cfg.slot}.", None

        # NEU: Original-ROI sichern (genau dieses Bild sieht Tesseract, nur noch ungefiltert)
        self._dbg_roi_orig = np.ascontiguousarray(crop).copy()

        pre = self._preprocess(crop)
        self._dbg_pre = pre
        val = self._ocr_digit(pre)
        if val is None:
            self.tokens_live = 0
            return "Detected: 0", pre
        self.tokens_live = max(0, min(6, int(val)))
        return f"Detected: {self.tokens_live}", pre

    def maybe_update_tokens(self) -> None:
        if self._worker_run:
            return
        if not self.cfg.enabled:
            self.tokens_live = 0
            return
        self._ocr_once()

    def start_worker(self, hz: float = 10.0) -> None:
        self._worker_hz = max(1.0, float(hz))
        if self._worker_run:
            return
        self._worker_run = True

        def _loop():
            period = 1.0 / self._worker_hz
            next_run = time.perf_counter()
            while self._worker_run:
                try:
                    self._ocr_once()
                except Exception:
                    pass
                next_run += period
                dt = next_run - time.perf_counter()
                if dt > 0:
                    time.sleep(dt)
                else:
                    next_run = time.perf_counter() + period

        th = threading.Thread(target=_loop, name="HyperfocusWorker", daemon=True)
        th.start()
        self._worker_th = th

    def stop_worker(self) -> None:
        self._worker_run = False
        th = self._worker_th
        self._worker_th = None
        if th and th.is_alive():
            try:
                th.join(timeout=0.1)
            except Exception:
                pass

    def push_full_frame(self, frame_bgr: Optional[np.ndarray]) -> None:
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            self._last_full_from_app = None
            return
        self._last_full_from_app = np.ascontiguousarray(frame_bgr).copy()

    def preview_roi(self) -> Optional[np.ndarray]:
        """Nur Vorschau-Overlay (farbig, Slot-Balken mit Text)."""
        src = self._last_full_from_app
        if src is None:
            return None
        r = self.cfg.rois.get(int(self.cfg.slot), (0, 0, 0, 0))
        c = self._safe_crop(src, *r)
        if c is None or getattr(c, "size", 0) == 0:
            return None
        c = np.ascontiguousarray(c).copy()
        h, w = c.shape[:2]
        if h < 2 or w < 2:
            return c
        bar_h = max(1, min(18, h - 1))
        cv2.rectangle(c, (0, 0), (w - 1, bar_h), (0, 0, 0), -1)
        cv2.putText(c, f"slot {self.cfg.slot}", (4, min(14, h - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return c
