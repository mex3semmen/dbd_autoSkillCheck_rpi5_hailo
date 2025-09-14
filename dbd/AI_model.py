# dbd/AI_model.py
import numpy as np
import onnxruntime as ort
import cv2

from dbd.utils.monitoring_hdmi import Monitoring_mss

try:
    import torch
    torch_ok = True
    print("Info: torch library found.")
except ImportError:
    torch_ok = False

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    trt_ok = True
    print("Info: tensorRT and pycuda library found.")
except ImportError:
    trt_ok = False

# Hailo-Backend
try:
    from dbd.utils.hailo_backend import Hailo8Session
    hailo_ok = True
except Exception as _hailo_err:
    hailo_ok = False
    _hailo_import_err = _hailo_err

try:
    import bettercam
    from dbd.utils.monitoring_bettercam import Monitoring_bettercam
    bettercam_ok = True
    print("Info: Bettercam feature available.")
except ImportError:
    bettercam_ok = False


class AI_model:
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    pred_dict = {
        0: {"desc": "None", "hit": False},
        1: {"desc": "repair-heal (great)", "hit": True},
        2: {"desc": "repair-heal (ante-frontier)", "hit": True},
        3: {"desc": "repair-heal (out)", "hit": False},
        4: {"desc": "full white (great)", "hit": True},
        5: {"desc": "full white (out)", "hit": False},
        6: {"desc": "full black (great)", "hit": True},
        7: {"desc": "full black (out)", "hit": False},
        8: {"desc": "wiggle (great)", "hit": True},
        9: {"desc": "wiggle (frontier)", "hit": False},
        10: {"desc": "wiggle (out)", "hit": False}
    }

    def __init__(self, model_path="models/model.onnx", provider="CPU",
                 nb_cpu_threads=None, monitor_id=1, use_bettercam=False):
        """
        provider: "CPU" (ONNX) oder "HAILO"
        model_path: *.onnx (CPU) oder *.hef (HAILO)
        """
        self.model_path = model_path
        self.provider = (provider or "CPU").upper()
        self.nb_cpu_threads = nb_cpu_threads

        # Monitoring (HDMI/BetterCam)
        if use_bettercam and bettercam_ok:
            self.monitor = Monitoring_bettercam(monitor_id=monitor_id, crop_size=224, target_fps=240)
        else:
            self.monitor = Monitoring_mss(monitor_id=monitor_id, crop_size=224)
        self.monitor.start()

        # Backends
        self.ort_session = None
        self.input_name = None

        self.cuda_context = None
        self.engine = None
        self.context = None
        self.stream = None
        self.tensor_shapes = None
        self.bindings = None

        self.hailo_session = None

        # Provider laden
        if self.provider == "HAILO":
            assert self.model_path.endswith(".hef"), "Für HAILO muss ein .hef angegeben werden."
            assert hailo_ok, f"Hailo Backend nicht verfügbar: {_hailo_import_err}"
            self._load_hailo()
        elif self.model_path.endswith(".trt"):
            self._load_tensorrt()
        else:
            self._load_onnx()

    # ---------- Monitoring ----------
    def grab_screenshot(self) -> np.ndarray:
        """RGB uint8 (224,224,3)"""
        return self.monitor.get_frame_np()

    def wait_for_new_frame(self) -> np.ndarray:
        """RGB uint8 (224,224,3) – blockiert bis neuer Frame verfügbar."""
        return self.monitor.wait_for_new_frame()

    # ---------- Utils ----------
    @staticmethod
    def softmax(x):
        x = np.asarray(x, dtype=np.float32)
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    # ---------- ONNX ----------
    def _load_onnx(self):
        sess_options = ort.SessionOptions()
        if self.nb_cpu_threads is not None:
            sess_options.intra_op_num_threads = self.nb_cpu_threads
            sess_options.inter_op_num_threads = self.nb_cpu_threads
        execution_providers = ["CPUExecutionProvider"]
        self.ort_session = ort.InferenceSession(self.model_path, providers=execution_providers, sess_options=sess_options)
        self.input_name = self.ort_session.get_inputs()[0].name

    def _preprocess_image_for_onnx(self, img_np: np.ndarray):
        img = np.asarray(img_np, dtype=np.float32) / 255.0            # NHWC float
        img = np.transpose(img, (2, 0, 1))                            # (C,H,W)
        img = (img - self.MEAN[:, None, None]) / self.STD[:, None, None]
        img = np.expand_dims(img, axis=0)                             # (1,C,H,W)
        img = np.ascontiguousarray(img)
        return img

    # ---------- TensorRT (legacy optional) ----------
    def _load_tensorrt(self):
        assert torch_ok, "TensorRT engine model requires torch lib. Aborting."
        assert trt_ok, "TensorRT engine model requires tensorrt lib. Aborting."
        import pycuda.driver as cuda  # ensure local
        cuda.init()
        device = cuda.Device(0)
        self.cuda_context = device.make_context()

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(self.model_path, "rb") as f:
            engine_data = f.read()
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        assert len(tensor_names) == 2
        self.tensor_shapes = [self.engine.get_tensor_shape(n) for n in tensor_names]
        tensor_in = np.empty(self.tensor_shapes[0], dtype=np.float32)
        tensor_out = np.empty(self.tensor_shapes[1], dtype=np.float32)

        self.p_input = cuda.mem_alloc(1 * tensor_in.nbytes)
        self.p_output = cuda.mem_alloc(1 * tensor_out.nbytes)
        self.context.set_tensor_address(tensor_names[0], int(self.p_input))
        self.context.set_tensor_address(tensor_names[1], int(self.p_output))
        self.stream = cuda.Stream()

    # ---------- Hailo ----------
    def _load_hailo(self):
        self.hailo_session = Hailo8Session(self.model_path)

    # ---------- Inferenz ----------
    def predict(self, img_np: np.ndarray):
        if self.hailo_session is not None:
            # Hailo erwartet NHWC uint8 → Monitoring liefert genau das
            logits = self.hailo_session.infer_rgb224(img_np)
        elif self.engine is not None:
            import pycuda.driver as cuda
            inp = self._preprocess_image_for_onnx(img_np)
            output = np.empty(self.tensor_shapes[1], dtype=np.float32)
            cuda.memcpy_htod_async(self.p_input, inp, self.stream)
            self.context.execute_async_v3(self.stream.handle)
            cuda.memcpy_dtoh_async(output, self.p_output, self.stream)
            self.stream.synchronize()
            logits = np.squeeze(output)
        else:
            inp = self._preprocess_image_for_onnx(img_np)
            output = self.ort_session.run(None, {self.input_name: inp})
            logits = np.squeeze(output)

        probs = self.softmax(logits)
        pred = int(np.argmax(probs))
        desc = self.pred_dict[pred]["desc"]
        hit = self.pred_dict[pred]["hit"]
        probs_dict = {self.pred_dict[i]["desc"]: float(probs[i]) for i in range(len(probs))}
        return pred, desc, probs_dict, hit

    def check_provider(self):
        if self.hailo_session is not None:
            return "HAILO"
        if self.engine is not None:
            return "TensorRT"
        return "CPUExecutionProvider"

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

        if self.engine is not None:
            try:
                self.stream = None
                self.context = None
                self.engine = None
                self.p_input = None
                self.p_output = None
                if self.cuda_context is not None:
                    self.cuda_context.pop()
            except Exception:
                pass
            self.cuda_context = None

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    def __del__(self):
        self.cleanup()
