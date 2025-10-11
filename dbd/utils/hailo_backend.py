# -*- coding: utf-8 -*-
"""
Hailo8-Session für HEF auf RPi OS.
Input:  UINT8, NHWC(HxWx3)
Output: FLOAT32 logits (C,)
Wichtig: Aktivierung + VStreams werden EINMAL geöffnet und wiederverwendet.
"""

import numpy as np
import hailo_platform as hpf

def _info(msg: str):
    print(f"[HAILO] {msg}", flush=True)

class Hailo8Session:
    def __init__(self, hef_path: str):
        self.hef = hpf.HEF(hef_path)

        ins = list(self.hef.get_input_vstream_infos())
        outs = list(self.hef.get_output_vstream_infos())
        if len(ins) != 1 or len(outs) != 1:
            raise RuntimeError(f"Erwarte 1 Input/Output, habe {len(ins)}/{len(outs)}")

        self.in_info = ins[0]
        self.out_info = outs[0]
        ih, iw, ic = self.in_info.shape
        oc = int(np.prod(self.out_info.shape))
        self.in_h, self.in_w = int(ih), int(iw)
        self.num_classes = oc

        _info(f"HEF Input: {self.in_info.name} shape={self.in_info.shape}")
        _info(f"HEF Output: {self.out_info.name} shape={self.out_info.shape}")

        # Device + Network konfigurieren
        self.vdevice = hpf.VDevice()
        cfg = hpf.ConfigureParams.create_from_hef(self.hef, interface=hpf.HailoStreamInterface.PCIe)
        self.network_groups = self.vdevice.configure(self.hef, cfg)
        self.network_group = self.network_groups[0]
        self.network_group_params = self.network_group.create_params()

        # VStream-Parameter (quantized in, dequantized out)
        self.input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
            self.network_group, quantized=True, format_type=hpf.FormatType.UINT8
        )
        self.output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
        )

        self.input_name = self.in_info.name
        self.output_name = self.out_info.name

        # ----- EINMAL aktivieren + VStreams öffnen (persistente Pipeline) -----
        self._activation = self.network_group.activate(self.network_group_params)
        # Manuell "enter", damit der Kontext aktiv bleibt, bis wir .close() rufen
        self._activation.__enter__()
        self._infer_pipeline = hpf.InferVStreams(
            self.network_group, self.input_vstreams_params, self.output_vstreams_params
        )
        self._infer_pipeline.__enter__()

        self._opened = True
        _info("Hailo Session bereit (persistent).")

    def infer_rgb(self, img_hwc_u8: np.ndarray) -> np.ndarray:
        """Schnelle Inferenz: erwartet uint8 (H,W,3) exakt in HEF-Eingangsgröße."""
        if not self._opened:
            raise RuntimeError("Session closed.")
        if img_hwc_u8.dtype != np.uint8 or img_hwc_u8.shape[:2] != (self.in_h, self.in_w):
            raise ValueError(f"Expect uint8 ({self.in_h},{self.in_w},3); got {img_hwc_u8.shape} {img_hwc_u8.dtype}")

        batched = np.expand_dims(img_hwc_u8, 0)  # (1,H,W,3)
        results = self._infer_pipeline.infer({self.input_name: batched})
        out = np.squeeze(results[self.output_name]).astype(np.float32)
        return out

    def close(self):
        if self._opened:
            try:
                # Reihenfolge: VStreams, dann Aktivierung, dann Gerät
                if getattr(self, "_infer_pipeline", None) is not None:
                    self._infer_pipeline.__exit__(None, None, None)
                    self._infer_pipeline = None
                if getattr(self, "_activation", None) is not None:
                    self._activation.__exit__(None, None, None)
                    self._activation = None
            finally:
                self.vdevice.release()
                self._opened = False
                _info("Hailo Session closed.")
