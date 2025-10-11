import os
import json
import numpy as np
import onnxruntime as ort
from time import perf_counter_ns
from dbd.utils.monitoring_hdmi import Monitoring_v4l2  # V4L2-Grabber (Center-Crop 620x420)

# Optional (nur falls du später TRT willst)
try:
    import tensorrt as trt  # noqa: F401
    import pycuda.driver as cuda  # noqa: F401
    trt_ok = True
except Exception:
    trt_ok = False

# Hailo-Backend (erwartet Hailo8Session mit infer_rgb(img_uint8)->logits)
try:
    from dbd.utils.hailo_backend import Hailo8Session
    hailo_ok = True
except Exception as _hailo_err:
    hailo_ok = False
    _hailo_import_err = _hailo_err  # noqa: F401


class AI_model:
    """
    Einheitliche Inferenz für CPU(ONNX) / Hailo8(HEF).
    Gibt (pred_id, pred_desc, probs_dict, should_hit) zurück und hält Timings für das UI bereit.
    """

    # Imagenet-Normalisierung für ONNX-Modelle (falls benötigt)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # ── Klassenmapping (AKTUALISIERT: 11 Klassen; Wiggle-"frontier/ante" entfernt) ─────────
    # 0..10:
    #   0  None
    #   1  repair-heal (great)                [hit]
    #   2  repair-heal (ante-frontier)        [hit]
    #   3  repair-heal (out)
    #   4  full white (ante-frontier)         [hit]
    #   5  full white (great)                 [hit]
    #   6  full white (out)
    #   7  full black (great)                 [hit]
    #   8  full black (out)
    #   9  wiggle (great)                     [hit]
    #   10 wiggle (out)
    pred_dict = {
        0:  {"desc": "None",                         "hit": False},
        1:  {"desc": "repair-heal (great)",          "hit": True },
        2:  {"desc": "repair-heal (ante-frontier)",  "hit": True },
        3:  {"desc": "repair-heal (out)",            "hit": False},
        4:  {"desc": "full white (ante-frontier)",   "hit": True },
        5:  {"desc": "full white (great)",           "hit": True },
        6:  {"desc": "full white (out)",             "hit": False},
        7:  {"desc": "full black (great)",           "hit": True },
        8:  {"desc": "full black (out)",             "hit": False},
        9:  {"desc": "wiggle (great)",               "hit": True },
        10: {"desc": "wiggle (out)",                 "hit": False},
    }

    # Nur diese Klassen gelten als „Ante-Frontier“ (Timing-Delay):
    ANTE_FRONTIER_IDS = {2, 4}

    def __init__(self, model_path="models/model.onnx", provider="CPU",
                 nb_cpu_threads=None, monitor_id=1, use_bettercam=False):
        self.model_path = model_path
        self.provider = (provider or "CPU").upper()
        self.nb_cpu_threads = nb_cpu_threads

        # HDMI/CSI-Grabber (separater Thread). Center-Crop = 620x420, RGB ausgeben.
        self.monitor = Monitoring_v4l2(monitor_id=monitor_id, crop_size=(620, 420), y_bias=-16, out_fmt="RGB")
        self.monitor.start()

        # Backend-Handles
        self.ort_session = None
        self.input_name = None
        self.hailo_session = None

        # Timings
        self._last_timings = {"pre_ms": None, "infer_ms": None, "post_ms": None}

        # Temperatur/Schwellen (aus class_thresholds.json neben dem Modell oder in models/)
        self.temperature, self.class_thresholds, self.none_bias = self._load_class_thresholds(self.model_path)

        # Backend laden
        if self.provider == "HAILO":
            assert self.model_path.endswith(".hef"), "Für HAILO muss ein .hef angegeben werden."
            assert hailo_ok, f"Hailo Backend nicht verfügbar: {_hailo_import_err}"
            self._load_hailo()
        else:
            self._load_onnx()

    # ── Utilities ─────────────────────────────────────────────────────────────
    @staticmethod
    def softmax(x, T: float = 1.0):
        x = np.asarray(x, dtype=np.float32) / max(T, 1e-6)
        x = x - np.max(x)
        exp_x = np.exp(x)
        den = np.sum(exp_x)
        return exp_x / den if den > 0 else np.zeros_like(exp_x, dtype=np.float32)

    def _load_class_thresholds(self, model_path):
        temperature = 1.0
        thresholds = {}
        none_bias = None
        cand = []

        if model_path:
            cand.append(os.path.join(os.path.dirname(os.path.abspath(model_path)), "class_thresholds.json"))
            cand.append(os.path.join("models", "class_thresholds.json"))

        for p in cand:
            try:
                if os.path.exists(p):
                    with open(p, "r") as f:
                        d = json.load(f)
                        if isinstance(d, dict):
                            if "temperature" in d:
                                temperature = float(d.get("temperature") or 1.0)
                            if "none_bias" in d:
                                none_bias = float(d.get("none_bias"))
                            thr = d.get("thresholds")
                            if isinstance(thr, list):
                                thresholds = {i: float(v) for i, v in enumerate(thr)}
                                break
                            if isinstance(thr, dict):
                                for k, v in thr.items():
                                    try:
                                        thresholds[int(k)] = float(v)
                                    except Exception:
                                        pass
                                break
            except Exception:
                # still use defaults
                pass
        return temperature, thresholds, none_bias

    def _load_onnx(self):
        so = ort.SessionOptions()
        if isinstance(self.nb_cpu_threads, int) and self.nb_cpu_threads > 0:
            so.intra_op_num_threads = self.nb_cpu_threads
            so.inter_op_num_threads = self.nb_cpu_threads
        providers = ["CPUExecutionProvider"]
        self.ort_session = ort.InferenceSession(self.model_path, providers=providers, sess_options=so)
        self.input_name = self.ort_session.get_inputs()[0].name

    def _preprocess_image_for_onnx(self, img_np):
        img = np.asarray(img_np, dtype=np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = (img - self.MEAN[:, None, None]) / self.STD[:, None, None]
        img = np.expand_dims(img, axis=0)
        return np.ascontiguousarray(img)

    def _load_hailo(self):
        self.hailo_session = Hailo8Session(self.model_path)

    def is_ante_frontier(self, class_index: int) -> bool:
        try:
            return int(class_index) in self.ANTE_FRONTIER_IDS
        except Exception:
            return False

    def check_provider(self):
        return "HAILO" if self.hailo_session is not None else "CPU"

    def get_timings(self):
        return dict(self._last_timings)

    # ── Inferenz ──────────────────────────────────────────────────────────────
    def predict(self, img_np):
        """Gibt (pred_id, pred_desc, probs_dict, should_hit) zurück."""
        pre_ms = infer_ms = post_ms = None

        if self.hailo_session is not None:
            t0 = perf_counter_ns()
            # Hailo erwartet uint8 RGB contiguous
            img_uint8 = img_np if (img_np.dtype is np.uint8 and img_np.flags.c_contiguous) else np.ascontiguousarray(
                img_np.astype(np.uint8))
            pre_ms = (perf_counter_ns() - t0) / 1e6

            t1 = perf_counter_ns()
            logits = self.hailo_session.infer_rgb(img_uint8)  # (C,)
            infer_ms = (perf_counter_ns() - t1) / 1e6

        elif self.ort_session is not None:
            t0 = perf_counter_ns()
            inp = self._preprocess_image_for_onnx(img_np)
            pre_ms = (perf_counter_ns() - t0) / 1e6

            t1 = perf_counter_ns()
            output = self.ort_session.run(None, {self.input_name: inp})
            infer_ms = (perf_counter_ns() - t1) / 1e6
            logits = np.squeeze(output)

        else:
            raise RuntimeError("Kein Backend initialisiert.")

        # Softmax mit Temperature
        probs = self.softmax(logits, T=float(self.temperature or 1.0))

        pred = int(np.argmax(probs))
        desc = self.pred_dict.get(pred, {"desc": f"class_{pred}"}).get("desc", f"class_{pred}")
        is_hit_class = bool(self.pred_dict.get(pred, {}).get("hit", False))

        p_pred = float(probs[pred])
        thr = float(self.class_thresholds.get(pred, 0.50))
        should_hit = is_hit_class and (p_pred >= thr)

        # Für UI: dict "label -> prob"
        probs_dict = {}
        for i, p in enumerate(probs):
            info = self.pred_dict.get(i, {"desc": f"class_{i}"})
            probs_dict[info["desc"]] = float(p)

        self._last_timings = {"pre_ms": pre_ms, "infer_ms": infer_ms, "post_ms": post_ms}
        return pred, desc, probs_dict, should_hit

    # ── Ressourcenpflege ──────────────────────────────────────────────────────
    def cleanup(self):
        if getattr(self, "monitor", None):
            try:
                self.monitor.stop()
            except Exception:
                pass
            self.monitor = None
        if self.hailo_session is not None:
            try:
                self.hailo_session.close()
            except Exception:
                pass
            self.hailo_session = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __del__(self):
        self.cleanup()
