# dbd/utils/monitoring_hdmi.py
import os
import time
import threading
import numpy as np
import cv2
import sys

def _log(msg):
    print(f"[HDMI] {msg}", file=sys.stderr, flush=True)

class Monitoring_v4l2:
    """
    HDMI/CSI-2 Grabber über V4L2 mit Background-Thread, niedriger Latenz.
    Liefert Frames im gewünschten Farbformat (BGR oder RGB), 224x224 Center-Crop.

    Defaults (BGR-first):
      X1300_PIXFMT       = "BGR3"
      X1300_OUTPUT       = "BGR"
      X1300_SWAP_RB      = "0"
      X1300_FALLBACKS    = "BGR3,RGB3,YUYV,UYVY"
      X1300_WARMUP_MS    = "1500"   (1.5 s)
      X1300_STRICT_PIXFMT= "0"      (1 = kein Fallback)

    Weitere Env:
      X1300_DEVICE   "/dev/video0"
      X1300_WIDTH    "1920"
      X1300_HEIGHT   "1080"
    """

    # ---------- Statische API für UI ----------
    @staticmethod
    def get_monitors_info():
        dev = os.environ.get("X1300_DEVICE", "/dev/video0")
        w   = int(os.environ.get("X1300_WIDTH",  "1920"))
        h   = int(os.environ.get("X1300_HEIGHT", "1080"))
        label = f"HDMI-IN {dev} ({w}x{h})"
        return [(label, dev)]

    @staticmethod
    def get_monitors_info_raw():
        dev = os.environ.get("X1300_DEVICE", "/dev/video0")
        w   = int(os.environ.get("X1300_WIDTH",  "1920"))
        h   = int(os.environ.get("X1300_HEIGHT", "1080"))
        return [{"id": dev, "name": f"HDMI-IN {dev}", "left": 0, "top": 0, "width": w, "height": h}]

    # -----------------------------------------------------------

    def __init__(self, device=None, width=None, height=None, pixfmt=None,
                 crop_size=224, swap_rb=None, out_fmt=None, monitor_id=None,
                 **_ignored):
        env = os.environ
        self.device     = device or monitor_id or env.get("X1300_DEVICE", "/dev/video0")
        self.width      = int(width  or env.get("X1300_WIDTH",  "1920"))
        self.height     = int(height or env.get("X1300_HEIGHT", "1080"))

        # User-Wunsch zuerst:
        self.requested_pixfmt = (pixfmt or env.get("X1300_PIXFMT", "BGR3")).upper()
        # Tatsächlich gewähltes Pixelformat nach erfolgreichem Open:
        self.pixfmt     = self.requested_pixfmt

        self.out_fmt    = (out_fmt or env.get("X1300_OUTPUT", "BGR")).upper()
        self.swap_rb    = (str(swap_rb if swap_rb is not None else env.get("X1300_SWAP_RB", "0")) == "1")
        self.crop_size  = int(crop_size)

        self.strict_pixfmt = (env.get("X1300_STRICT_PIXFMT", "0") == "1")
        fb = env.get("X1300_FALLBACKS", "BGR3,RGB3,YUYV,UYVY")
        # Fallback-Reihenfolge: gewünschtes Format zuerst, dann Liste (ohne Duplikate)
        self.fallback_pixfmts = [self.requested_pixfmt] + [p.strip().upper() for p in fb.split(",") if p.strip()]
        # Duplikate entfernen, Reihenfolge bewahren
        seen = set()
        self.fallback_pixfmts = [p for p in self.fallback_pixfmts if not (p in seen or seen.add(p))]

        self.warmup_ms  = int(env.get("X1300_WARMUP_MS", "1500"))

        self.cap = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._last_frame = None  # out_fmt, 224x224
        self._last_ts = 0.0
        self._frame_event = threading.Event()

    # --- Öffnen/Schließen ------------------------------------------------------
    def _try_configure(self, cap, pf):
        """Setzt Backend-Flags für gewünschtes Pixelformat."""
        # Minimale Latenz:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Keine implizite BGR-Konvertierung durch OpenCV:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)

        fourcc = cv2.VideoWriter_fourcc(*pf)
        ok = cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        _log(f"configure fourcc={pf} -> {ok}")
        return ok

    def _warmup_read(self, cap, timeout_ms):
        """Versucht bis timeout einen Frame zu lesen. Gibt (ok, raw, info_str) zurück."""
        t0 = time.time()
        last_err = None
        reads = 0
        while (time.time() - t0) * 1000.0 < timeout_ms:
            ok, raw = cap.read()
            reads += 1
            if ok and raw is not None and raw.size > 0:
                shape = getattr(raw, "shape", None)
                return True, raw, f"Frame {shape} after {reads} reads"
            time.sleep(0.005)
        return False, None, f"no frame after {reads} reads / {timeout_ms} ms"

    def _open(self):
        if self.cap is not None and self.cap.isOpened():
            return

        # Device öffnen
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
                _log(f"USING pf={pf}, out_fmt={self.out_fmt}, swap_rb={int(self.swap_rb)}")
                break
            # Reset Stream (einige Treiber mögen Neu-Setzen)
            cap.release()
            cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot re-open {self.device} after failed probe pf={pf}")

        if self.cap is None:
            raise RuntimeError(f"No frame from HDMI input (tried: {', '.join(tried)})")

    def _close(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    # --- Hintergrund-Loop ------------------------------------------------------
    def start(self, monitor_id=None):
        """
        monitor_id (optional) wird als Device-Pfad interpretiert.
        """
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

    def is_alive(self):
        return self._running and self._thread is not None and self._thread.is_alive()

    def stop(self):
        self._running = False
        t = self._thread
        self._thread = None
        if t is not None:
            t.join(timeout=0.5)
        self._close()

    # --- Frame-Pipeline --------------------------------------------------------
    def _convert(self, raw):
        """Konvertiert vom Video-Node-Pixelformat (self.pixfmt) nach out_fmt."""
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
            # Fallback: so behandeln als käme BGR rein
            img = raw if out == "BGR" else raw[:, :, ::-1]

        if self.swap_rb:
            img = img[:, :, ::-1]
        return img

    @staticmethod
    def _center_crop(img, size):
        h, w = img.shape[:2]
        if h < size or w < size:
            img = cv2.resize(img, (max(size, w), max(size, h)), interpolation=cv2.INTER_NEAREST)
            h, w = img.shape[:2]
        y0 = (h - size) // 2
        x0 = (w - size) // 2
        return img[y0:y0+size, x0:x0+size]

    def _loop(self):
        cap = self.cap
        while self._running:
            ok, raw = cap.read()
            if not ok or raw is None:
                # kurz retry, keine harten Exceptions im Loop
                time.sleep(0.001)
                continue
            try:
                img = self._convert(raw)
                if img.shape[0] != self.crop_size or img.shape[1] != self.crop_size:
                    img = self._center_crop(img, self.crop_size)
                with self._lock:
                    self._last_frame = img
                    self._last_ts = time.time()
                    self._frame_event.set()
            except Exception as e:
                _log(f"convert/crop error: {e}")
                time.sleep(0.001)

    # --- Öffentliche API -------------------------------------------------------
    def grab_screenshot(self):
        """
        Rückgabe: uint8 (224,224,3) im `out_fmt` ("BGR" oder "RGB").
        Falls Thread nicht läuft: synchron lesen (inkl. Fallback-Probing).
        """
        if not self.is_alive():
            self._open()
            ok, raw = self.cap.read()
            if not ok or raw is None:
                raise RuntimeError("HDMI read failed")
            img = self._convert(raw)
            if img.shape[0] != self.crop_size or img.shape[1] != self.crop_size:
                img = self._center_crop(img, self.crop_size)
            return img

        with self._lock:
            frame = self._last_frame
        if frame is None:
            # Erster Frame noch nicht da -> kurz warten
            if not self._frame_event.wait(timeout=0.3):
                raise RuntimeError("No HDMI frame yet")
            with self._lock:
                frame = self._last_frame
        return frame

    def wait_for_new_frame(self):
        """Blockiert bis neuer Frame vorliegt und liefert ihn zurück."""
        if not self.is_alive():
            return self.grab_screenshot()
        self._frame_event.wait()
        with self._lock:
            frame = self._last_frame
        self._frame_event.clear()
        if frame is None:
            raise RuntimeError("No HDMI frame yet")
        return frame

    # Aliasse für bestehende Aufrufer
    def get_frame_np(self):
        return self.grab_screenshot()

    def set_monitor(self, monitor_id):
        """Rückwärtskompatibel: auf neues Device umschalten."""
        self.start(monitor_id=monitor_id)

# Alias, falls alter Code 'Monitoring_mss' importiert:
Monitoring_mss = Monitoring_v4l2
