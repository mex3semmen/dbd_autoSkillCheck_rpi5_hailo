# /home/leroy/dbd/dbd/utils/datacollector.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, time, threading, hashlib, queue, atexit
from typing import Optional
import numpy as np
from PIL import Image  # speichert RGB oder L

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _timestamp_dir() -> str:
    t = time.localtime()
    return time.strftime("%Y%m%d_%H%M%S", t) + f"_{int((time.time()%1)*1_000_000):06d}"

def _ts_ms() -> int:
    return int(time.time() * 1000)

# ───────────────────────── Skillcheck ─────────────────────────
class SkillcheckCollector:
    """
    Ziel: **jedes Kamera-Frame** während eines Skillchecks speichern.
    Start, sobald class != 0. Ende erst, wenn 0 wieder dominiert (Hysterese/Grace).
    Speicherung erfolgt **asynchron** (Writer-Thread), damit der Grabber nie blockiert.

    Public API:
      - update_pred(class_id: int): wird im Inferenz-Loop aufgerufen.
      - ingest_frame(frame_rgb: np.ndarray): wird vom Grabber-Thread auf JEDES Frame aufgerufen.
    """
    def __init__(
        self,
        enabled: bool,
        base_dir: str = "Skillcheck Data",
        file_format: str = "png",           # "png" oder "jpg"
        jpg_quality: int = 92,              # falls file_format="jpg"
        png_compress_level: int = 1,        # 0..9, 0/1 = schnell
        close_zero_consec: int = 2,         # Hysterese: wie viele 0-Frames zum Schließen
        save_zero_while_active: bool = True,
        linger_ms_after_nonzero: int = 180, # nach letztem !=0 mind. so lange NICHT schließen
        dont_close_before_ms: int = 120     # Session mindestens so lange offen halten
    ):
        self.enabled = bool(enabled)
        self.base_dir = base_dir
        self.file_format = (file_format or "png").lower()
        self.jpg_quality = int(jpg_quality)
        self.png_compress_level = max(0, min(9, int(png_compress_level)))
        self.close_zero_consec = max(1, int(close_zero_consec))
        self.save_zero_while_active = bool(save_zero_while_active)
        self.linger_ms_after_nonzero = max(0, int(linger_ms_after_nonzero))
        self.dont_close_before_ms = max(0, int(dont_close_before_ms))

        self._lock = threading.Lock()
        self._active: bool = False
        self._session_dir: Optional[str] = None
        self._idx: int = 0
        self._zero_streak: int = 0
        self._last_nonzero_ts: float = 0.0
        self._session_start_ts: float = 0.0

        # Async Writer
        self._q: "queue.Queue[tuple[str, np.ndarray]]" = queue.Queue(maxsize=4000)
        self._writer_stop = threading.Event()
        self._writer = threading.Thread(target=self._writer_loop, daemon=True)
        if self.enabled:
            _ensure_dir(self.base_dir)
            self._writer.start()
            atexit.register(self._graceful_stop)

    def _graceful_stop(self):
        try:
            self._writer_stop.set()
            self._q.put_nowait(("__STOP__", np.empty((0, 0), dtype=np.uint8)))
        except Exception:
            pass

    def _writer_loop(self):
        while not self._writer_stop.is_set():
            try:
                path, arr = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if path == "__STOP__":
                break
            try:
                img = Image.fromarray(arr)
                if self.file_format in ("jpg", "jpeg"):
                    img.save(path, format="JPEG", quality=self.jpg_quality,
                             subsampling=0, optimize=False)
                else:
                    img.save(path, format="PNG", compress_level=self.png_compress_level)
            except Exception:
                # lieber EIN Bild verlieren als den Echtzeit-Loop zu blockieren
                pass

    # ── Runtime Toggle ────────────────────────────────────────────────────────
    def set_enabled(self, enabled: bool):
        enabled = bool(enabled)
        if enabled == self.enabled:
            return
        if enabled:
            # aktivieren
            self.enabled = True
            _ensure_dir(self.base_dir)
            if not (self._writer and self._writer.is_alive()):
                self._writer_stop.clear()
                self._writer = threading.Thread(target=self._writer_loop, daemon=True)
                self._writer.start()
        else:
            # deaktivieren
            self.enabled = False
            with self._lock:
                if self._active:
                    self._stop_session()
            self._graceful_stop()

    # ── Session-State ────────────────────────────────────────────────────────
    def _start_session(self):
        self._session_dir = os.path.join(self.base_dir, _timestamp_dir())
        _ensure_dir(self._session_dir)
        self._idx = 0
        self._active = True
        self._zero_streak = 0
        self._session_start_ts = time.perf_counter()
        self._last_nonzero_ts = self._session_start_ts

    def _stop_session(self):
        self._active = False
        self._session_dir = None
        self._idx = 0
        self._zero_streak = 0
        self._last_nonzero_ts = 0.0
        self._session_start_ts = 0.0

    def _can_close_now(self, now: float) -> bool:
        # Zeitfenster: nicht vor Mindestdauer und nicht direkt nach !=0 schließen
        if (now - self._session_start_ts) * 1000.0 < self.dont_close_before_ms:
            return False
        if (now - self._last_nonzero_ts) * 1000.0 < self.linger_ms_after_nonzero:
            return False
        return self._zero_streak >= self.close_zero_consec

    # 1) Inferenz-Loop ruft diese Funktion mit der **aktuellen Klasse** auf:
    def update_pred(self, current_class: int):
        if not self.enabled:
            return
        now = time.perf_counter()
        c = int(current_class)
        with self._lock:
            if not self._active:
                if c != 0:
                    self._start_session()
                else:
                    return
            # Session aktiv:
            if c == 0:
                self._zero_streak += 1
                if self._can_close_now(now):
                    self._stop_session()
            else:
                self._zero_streak = 0
                self._last_nonzero_ts = now

    # 2) Grabber-Thread ruft diese Funktion **auf jedes Frame** auf:
    def ingest_frame(self, frame_rgb: np.ndarray):
        if not self.enabled or frame_rgb is None:
            return
        with self._lock:
            if not self._active:
                return
            if (not self.save_zero_while_active) and self._zero_streak > 0:
                # Optionalfilter: 0-Frames während aktiver Session droppen
                return
            ext = "jpg" if self.file_format in ("jpg", "jpeg") else "png"
            fn = f"{self._idx:05d}_{_ts_ms()}.{ext}"
            path = os.path.join(self._session_dir, fn)
            self._idx += 1
        # außerhalb des Locks enqueuen (nicht blockieren)
        arr = np.ascontiguousarray(frame_rgb)
        try:
            self._q.put_nowait((path, arr))
        except queue.Full:
            pass

# ───────────────────────── Hyperfocus ────────────────────────
class HyperfocusCollector:
    """
    Speichert das **Original-ROI** (farbig) in "Hyperfocus Data/<token>/timestamp.png".
    Dedup via Hash, damit identische Frames nicht gespammt werden.
    """
    def __init__(self, enabled: bool, base_dir: str = "Hyperfocus Data"):
        self.enabled = bool(enabled)
        self.base_dir = base_dir
        self._lock = threading.Lock()
        self._last_hash: Optional[str] = None
        if self.enabled:
            for d in range(0, 7):
                _ensure_dir(os.path.join(self.base_dir, str(d)))

    # ── Runtime Toggle ────────────────────────────────────────────────────────
    def set_enabled(self, enabled: bool):
        enabled = bool(enabled)
        if enabled == self.enabled:
            return
        self.enabled = enabled
        if self.enabled:
            for d in range(0, 7):
                _ensure_dir(os.path.join(self.base_dir, str(d)))

    @staticmethod
    def _hash_img(arr: np.ndarray) -> str:
        h = hashlib.blake2b(digest_size=16)
        h.update(str(arr.shape).encode("ascii"))
        h.update(arr.tobytes())
        return h.hexdigest()

    def save_if_updated(self, roi_orig: np.ndarray, token: int):
        if not self.enabled or roi_orig is None:
            return
        try:
            token = int(token)
            if token < 0 or token > 6:
                token = 0
        except Exception:
            token = 0
        key = self._hash_img(roi_orig)
        with self._lock:
            if key == self._last_hash:
                return
            self._last_hash = key
            sub = os.path.join(self.base_dir, str(token))
            _ensure_dir(sub)
            fn = f"{_ts_ms()}.png"
            path = os.path.join(sub, fn)
        try:
            arr = np.ascontiguousarray(roi_orig)
            img = Image.fromarray(arr)  # behält Modus (RGB/L)
            img.save(path, format="PNG", compress_level=1)
        except Exception:
            pass  # Echtzeit geht vor
