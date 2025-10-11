import os
import json
import time
import threading
import numpy as np
import cv2
import sys
from typing import Tuple, Union, List, Optional

def _log(msg):
    print(f"[HDMI] {msg}", file=sys.stderr, flush=True)

def _try_load_x1300_env(path="/tmp/x1300_env"):
    """Optional: /tmp/x1300_env laden (vom setup-Script erzeugt)."""
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln or ln.startswith("#") or "=" not in ln:
                        continue
                    k, v = ln.split("=", 1)
                    k = k.strip().upper(); v = v.strip()
                    if k == "VIDEO_NODE":
                        os.environ["X1300_DEVICE"] = v
                    elif k == "WIDTH":
                        os.environ["X1300_WIDTH"] = v
                    elif k == "HEIGHT":
                        os.environ["X1300_HEIGHT"] = v
                    elif k == "PIXELFORMAT":
                        os.environ["X1300_PIXFMT"] = v
    except Exception as e:
        _log(f"ENV load warn: {e}")

DEFAULT_ROIS = {
    "roi1": (1600, 780, 26, 26),
    "roi2": (1748, 780, 26, 26),
    "roi3": (1600, 928, 26, 26),
    "roi4": (1748, 928, 26, 26),
}

class Monitoring_v4l2:
    """
    HDMI/CSI-2 Grabber über V4L2 mit Background-Thread, niedriger Latenz.
    - Puffert Full-HD nach Farbkonvertierung
    - Liefert Center-Crop + beliebige ROIs aus dem Full-HD-Frame
    """

    @staticmethod
    def get_monitors_info():
        dev = os.environ.get("X1300_DEVICE", "/dev/video0")
        w = int(os.environ.get("X1300_WIDTH", "1920"))
        h = int(os.environ.get("X1300_HEIGHT", "1080"))
        label = f"HDMI-IN {dev} ({w}x{h})"
        return [(label, dev)]

    @staticmethod
    def get_monitors_info_raw():
        dev = os.environ.get("X1300_DEVICE", "/dev/video0")
        w = int(os.environ.get("X1300_WIDTH", "1920"))
        h = int(os.environ.get("X1300_HEIGHT", "1080"))
        return [{"id": dev, "name": f"HDMI-IN {dev}", "left": 0, "top": 0, "width": w, "height": h}]

    def __init__(self, device=None, width=None, height=None, pixfmt=None,
                 crop_size: Union[int, Tuple[int,int]]=224, y_bias=0,
                 swap_rb=None, out_fmt=None, monitor_id=None, **_ignored):
        env = os.environ
        self.device = device or monitor_id or env.get("X1300_DEVICE", "/dev/video0")
        self.width = int(width or env.get("X1300_WIDTH", "1920"))
        self.height = int(height or env.get("X1300_HEIGHT", "1080"))
        self.requested_pixfmt = (pixfmt or env.get("X1300_PIXFMT", "BGR3")).upper()
        self.pixfmt = self.requested_pixfmt
        self.out_fmt = (out_fmt or env.get("X1300_OUTPUT", "BGR")).upper()
        self.smart_swap = (env.get("X1300_SMART_SWAP", "1") == "1")
        if swap_rb is not None:
            self.swap_rb = bool(swap_rb)
        else:
            env_swap = env.get("X1300_SWAP_RB", "-1").strip()
            self.swap_rb = None if env_swap == "-1" else (env_swap == "1")
        if isinstance(crop_size, (tuple, list)) and len(crop_size) == 2:
            self.crop_w, self.crop_h = int(crop_size[0]), int(crop_size[1])
            if self.crop_w <= 0 or self.crop_h <= 0:
                raise ValueError("crop_size tuple must be positive (W,H).")
            self.crop_size = None
        else:
            self.crop_size = int(crop_size)
            if self.crop_size <= 0:
                raise ValueError("crop_size must be > 0.")
            self.crop_w = self.crop_h = self.crop_size
        self.y_bias = int(y_bias or 0)
        self.strict_pixfmt = (env.get("X1300_STRICT_PIXFMT", "0") == "1")
        fb = env.get("X1300_FALLBACKS", "BGR3,RGB3,YUYV,UYVY")
        self.fallback_pixfmts = [self.requested_pixfmt] + [p.strip().upper() for p in fb.split(",") if p.strip()]
        seen = set()
        self.fallback_pixfmts = [p for p in self.fallback_pixfmts if not (p in seen or seen.add(p))]
        self.warmup_ms = int(env.get("X1300_WARMUP_MS", "1500"))
        self.cap = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._last_frame = None
        self._last_full_frame = None
        self._last_ts = 0.0
        self._frame_event = threading.Event()

    def _try_configure(self, cap, pf):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimale Buffer-Größe für geringere Latenz
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
        fourcc = cv2.VideoWriter_fourcc(*pf)
        ok = cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        _log(f"configure fourcc={pf} -> {ok}")
        return ok

    def _warmup_read(self, cap, timeout_ms):
        t0 = time.time()
        reads = 0
        while (time.time() - t0) * 1000.0 < timeout_ms:
            ok, raw = cap.read()
            reads += 1
            if ok and raw is not None and getattr(raw, "size", 0) > 0:
                shape = getattr(raw, "shape", None)
                return True, raw, f"Frame {shape} after {reads} reads"
            time.sleep(0.005)
        return False, None, f"no frame after {reads} reads / {timeout_ms} ms"

    def _decide_swap_if_auto(self):
        if self.swap_rb is None:
            if (self.pixfmt == "RGB3" and self.out_fmt == "RGB") or (self.pixfmt == "BGR3" and self.out_fmt == "BGR"):
                self.swap_rb = True
            else:
                self.swap_rb = False

    def _open(self):
        if self.cap is not None and self.cap.isOpened():
            return
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {self.device}")
        tried = []
        for pf in (self.fallback_pixfmts if not self.strict_pixfmt else [self.requested_pixfmt]):
            tried.append(pf)
            self._try_configure(cap, pf)
            _log(f"probing pf={pf} {self.width}x{self.height}")
            ok, raw, info = self._warmup_read(cap, self.warmup_ms)
            _log(f"probe result pf={pf}: ok={ok} ({info})")
            if ok:
                self.pixfmt = pf
                self.cap = cap
                self._decide_swap_if_auto()
                swap_disp = "a" if self.swap_rb is None else str(int(self.swap_rb))
                _log(f"USING pf={pf}, out_fmt={self.out_fmt}, swap_rb={swap_disp}")
                break
        if self.cap is None:
            raise RuntimeError(f"No frame from HDMI input (tried: {', '.join(tried)})")

    def _close(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None

    @staticmethod
    def _center_crop_rect(img, tw, th, y_bias=0):
        H, W = img.shape[:2]
        if H < th or W < tw:
            scale = max(th / max(H, 1), tw / max(W,1))
            newW, newH = int(round(W*scale)), int(round(H*scale))
            img = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_NEAREST)
        H, W = img.shape[:2]
        cx = W // 2; cy = H // 2 + int(y_bias)
        x0 = int(cx - tw // 2); y0 = int(cy - th // 2)
        x0 = max(0, min(x0, W - tw)); y0 = max(0, min(y0, H - th))
        return img[y0:y0+th, x0:x0+tw]

    @staticmethod
    def _center_crop(img, size):
        h, w = img.shape[:2]
        if h < size or w < size:
            img = cv2.resize(img, (max(size, w), max(size, h)), interpolation=cv2.INTER_NEAREST)
            h, w = img.shape[:2]
        y0 = (h - size) // 2
        x0 = (w - size) // 2
        return img[y0:y0+size, x0:x0+size]

    def _convert(self, raw):
        pf = self.pixfmt
        out = self.out_fmt
        if pf == "YUYV":
            code = cv2.COLOR_YUV2BGR_YUYV if out == "BGR" else cv2.COLOR_YUV2RGB_YUYV
            img = cv2.cvtColor(raw, code)
        elif pf == "UYVY":
            code = cv2.COLOR_YUV2BGR_UYVY if out == "BGR" else cv2.COLOR_YUV2RGB_UYVY
            img = cv2.cvtColor(raw, code)
        elif pf == "BGR3":
            img = raw if out == "BGR" else raw[:, :, ::-1]
        elif pf == "RGB3":
            img = raw if out == "RGB" else raw[:, :, ::-1]
        else:
            img = raw if out == "BGR" else raw[:, :, ::-1]
        if self.swap_rb:
            img = img[:, :, ::-1]
        return img

    def _loop(self):
        cap = self.cap
        while self._running:
            ok, raw = cap.read()
            if not ok or raw is None:
                time.sleep(0.001)
                continue
            try:
                full = self._convert(raw)
                center = (self._center_crop_rect(full, self.crop_w, self.crop_h, self.y_bias)
                          if getattr(self, 'crop_size', None) is None
                          else self._center_crop(full, self.crop_size))
                with self._lock:
                    self._last_full_frame = full
                    self._last_frame = center
                    self._last_ts = time.time()
                    self._frame_event.set()
            except Exception as e:
                _log(f"convert/crop error: {e}")
            time.sleep(0.001)

    def start(self, monitor_id=None):
        if monitor_id:
            was_running = self._running
            self.stop()
            self.device = monitor_id
            if was_running:
                self._open()
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            return
        if self._running:
            return
        self._open()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        t = self._thread
        self._thread = None
        if t is not None:
            t.join(timeout=0.5)
        self._close()

    def wait_for_new_frame(self, timeout_ms=1000):
        """
        Wartet auf das Eintreffen eines neuen Frames.
        Gibt das Frame als NumPy-Array zurück oder None bei Timeout.
        """
        if not self._frame_event.wait(timeout_ms / 1000.0):
            return None
        with self._lock:
            frame = self._last_frame.copy() if self._last_frame is not None else None
        self._frame_event.clear()
        return frame

    def get_full_frame(self):
        """
        Gibt das zuletzt empfangene Full-HD-Frame zurück (NumPy Array).
        """
        with self._lock:
            frame = self._last_full_frame.copy() if self._last_full_frame is not None else None
        return frame
