# dbd/utils/hailo_backend.py
# Hailo-8 Backend – robust gegenüber API-Varianten.
# I/O:
#   Input  : RGB uint8, NHWC (224,224,3)  -> wir geben als (1,224,224,3) an
#   Output : FLOAT32-Logits (11,)         -> HailoRT dequantisiert von UINT8 auf FP32

import inspect
import importlib
import numpy as np

try:
    import hailo_platform as hp
    from hailo_platform import (
        VDevice, HEF, ConfigureParams,
        InputVStreamParams, OutputVStreamParams, InferVStreams, FormatType
    )
    hailo_ok = True
    _hailo_import_err = None
except Exception as _e:
    hailo_ok = False
    _hailo_import_err = _e


# --------- Helpers -----------------------------------------------------------

def _resolve_stream_interface():
    """
    Liefert Enum-Member für PCIe – robust über verschiedene Exporte.
    Versucht:
      - hailo_platform.HailoStreamInterface.{PCIE|PCIe|pcie}
      - hailo_platform.StreamInterface.{...}
      - hailo_platform.pyhailort[._pyhailort|.pyhailort].{StreamInterface|HailoStreamInterface}.{...}
    """
    candidates = []

    for name in ("HailoStreamInterface", "StreamInterface"):
        if hasattr(hp, name):
            candidates.append(getattr(hp, name))

    for modname in ("hailo_platform.pyhailort",
                    "hailo_platform.pyhailort._pyhailort",
                    "hailo_platform.pyhailort.pyhailort"):
        try:
            m = importlib.import_module(modname)
            for n in ("StreamInterface", "HailoStreamInterface"):
                if hasattr(m, n):
                    candidates.append(getattr(m, n))
        except Exception:
            pass

    for enum in candidates:
        for member in ("PCIE", "PCIe", "pcie"):
            if hasattr(enum, member):
                return getattr(enum, member)
    return None


def _make_params_dict_with_format(ParamsCls, configured_ng, fmt):
    """
    Erzeugt IMMER ein Dict {vstream_name: ParamsObj}.
    Bevorzugt moderne .make(configured_ng, format_type=...), fällt sonst auf
    ältere Fabriken zurück; wenn kein format_type unterstützt wird, lässt es weg.
    """
    # 1) Moderne API: .make(configured_ng, ...)
    if hasattr(ParamsCls, "make"):
        try:
            sig = inspect.signature(ParamsCls.make)
            kwargs = {}
            if "format_type" in sig.parameters and fmt is not None:
                kwargs["format_type"] = fmt
            # KEIN network_name übergeben (vermeidet "Network name ... not found")
            params = ParamsCls.make(configured_ng, **kwargs)
            if isinstance(params, dict):
                return params
        except Exception:
            pass

    # 2) Ältere API: ...make_from_network_group(...)
    for attr in ("make_from_network_group", "create_from_network_group"):
        if hasattr(ParamsCls, attr):
            fn = getattr(ParamsCls, attr)
            try:
                sig = inspect.signature(fn)
                kwargs = {}
                if "format_type" in sig.parameters and fmt is not None:
                    kwargs["format_type"] = fmt
                return fn(configured_ng, **kwargs)
            except Exception:
                pass

    raise RuntimeError(f"{ParamsCls.__name__}: keine passende Factory in dieser HailoRT-Version.")


# --------- Hauptklasse -------------------------------------------------------

class Hailo8Session:
    def __init__(self, hef_path: str):
        if not hailo_ok:
            raise RuntimeError(f"Hailo Backend nicht verfügbar: {_hailo_import_err}")

        # HEF & virtuelles Gerät
        self._hef = HEF(hef_path)
        self._vdev = VDevice()

        # PCIe-Interface (Enum) auflösen
        si = _resolve_stream_interface()
        if si is None:
            raise RuntimeError(
                "Konnte StreamInterface.PCIE nicht finden. Prüfe HailoRT/Python-Installation "
                "(Treiber/Wheels gleiche Version)."
            )

        # ConfigureParams + konfigurierte NetworkGroup
        cfg = ConfigureParams.create_from_hef(self._hef, interface=si)
        configured = self._vdev.configure(self._hef, cfg)
        self._cng = configured[0] if isinstance(configured, (list, tuple)) else configured

        # VStream-Parameter (als DICTS) – Output explizit FLOAT32 (Dequantisierung durch HailoRT)
        in_params  = _make_params_dict_with_format(InputVStreamParams,  self._cng, fmt=FormatType.UINT8)
        out_params = _make_params_dict_with_format(OutputVStreamParams, self._cng, fmt=FormatType.FLOAT32)

        if not isinstance(in_params, dict) or not isinstance(out_params, dict):
            raise RuntimeError("Input/OutputVStreamParams.make*(...) muss Dicts liefern.")

        # Erwartet genau einen In/Out-Stream laut deinem HEF
        if len(in_params) != 1 or len(out_params) != 1:
            raise RuntimeError(f"Unerwartete Streamanzahl: inputs={len(in_params)}, outputs={len(out_params)}")
        self.input_name  = next(iter(in_params.keys()))   # z.B. "model/input_layer1"
        self.output_name = next(iter(out_params.keys()))  # z.B. "model/fc20"

        # Network Group aktivieren (wenn benötigt)
        self._activation_cm = None
        if hasattr(self._cng, "activate"):
            try:
                act_sig = inspect.signature(self._cng.activate)
                if len(act_sig.parameters) >= 1 and hasattr(self._cng, "create_params"):
                    self._activation_cm = self._cng.activate(self._cng.create_params())
                else:
                    self._activation_cm = self._cng.activate()
                self._activation_cm.__enter__()
            except Exception:
                self._activation_cm = None  # manche Builds aktivieren implizit

        # Inferenz-Pipeline
        self._streams_cm = InferVStreams(self._cng, in_params, out_params)
        self._streams    = self._streams_cm.__enter__()

        self._num_classes = 11  # falls nicht auslesbar

    def infer_rgb224(self, frame_rgb224_uint8: np.ndarray) -> np.ndarray:
        """
        Input:  RGB uint8, NHWC (224,224,3)
        Output: float32-Logits (11,)
        """
        if frame_rgb224_uint8.dtype != np.uint8 or frame_rgb224_uint8.shape != (224, 224, 3):
            raise ValueError(
                f"Erwartet RGB uint8 (224,224,3), bekam {frame_rgb224_uint8.dtype} {frame_rgb224_uint8.shape}"
            )

        # Batch-Dimension N=1 hinzufügen + C-contiguous
        batched = np.expand_dims(frame_rgb224_uint8, axis=0)  # (1,224,224,3)
        batched = np.ascontiguousarray(batched, dtype=np.uint8)

        outputs = self._streams.infer({ self.input_name: batched })
        out_fp32 = np.asarray(outputs[self.output_name], dtype=np.float32).reshape(-1)
        return out_fp32

    # --------- Cleanup -------------------------------------------------------

    def close(self):
        try:
            if getattr(self, "_streams_cm", None) is not None:
                self._streams_cm.__exit__(None, None, None)
                self._streams_cm = None
            if getattr(self, "_activation_cm", None) is not None:
                self._activation_cm.__exit__(None, None, None)
                self._activation_cm = None
            if getattr(self, "_vdev", None) is not None:
                self._vdev.release()
                self._vdev = None
        except Exception:
            pass

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): self.close()
    def __del__(self):
        try: self.close()
        except Exception: pass
