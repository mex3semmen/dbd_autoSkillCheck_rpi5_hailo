# app.py — präzises Press-Timing (Deadline wait) + kein Hyperfocus-Delay bei Wiggle
import os, json, time, threading
from time import perf_counter_ns, perf_counter
from collections import deque, defaultdict
import gradio as gr

from dbd.AI_model import AI_model
from dbd.utils.directkeys import PressKey, ReleaseKey, SPACE
from dbd.hyperfocus import Hyperfocus, HyperfocusConfig
from dbd.utils.datacollector import SkillcheckCollector, HyperfocusCollector

# ── /tmp/x1300_env einlesen ───────────────────────────────────────────────────
def _try_load_x1300_env():
    p = "/tmp/x1300_env"
    try:
        if os.path.exists(p):
            with open(p, "r") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln or ln.startswith("#") or "=" not in ln:
                        continue
                    k, v = ln.split("=", 1)
                    k = k.strip().upper(); v = v.strip()
                    if k == "VIDEO_NODE": os.environ["X1300_DEVICE"] = v
                    elif k == "WIDTH":   os.environ["X1300_WIDTH"]  = v
                    elif k == "HEIGHT":  os.environ["X1300_HEIGHT"] = v
    except Exception as _e:
        print(f"[ENV] Warn: {p}: {_e}", flush=True)
_try_load_x1300_env()

# ── präzises Deadline-Warten: Sleep grob, dann Spin ───────────────────────────
def wait_until_ns(deadline_ns: int, spin_threshold_ns: int = 700_000):
    """Wartet bis deadline_ns (CLOCK_MONOTONIC-Äquivalent via perf_counter_ns).
       Erst schlafen (grobe Restzeit), dann Busy-Wait für die letzten ~0.7 ms.
    """
    while True:
        now = perf_counter_ns()
        rem = deadline_ns - now
        if rem <= 0:
            return
        if rem > spin_threshold_ns + 1_000_000:  # > 1.7 ms -> grob schlafen
            time.sleep((rem - spin_threshold_ns) / 1e9)
        else:
            while perf_counter_ns() < deadline_ns:
                pass
            return

MODELS_FOLDER = "models"
SETTINGS_FILE = "settings.json"
DEFAULT_SETTINGS = {
    "hitdelay": 80,
    "delays": { "wiggle_9": 10, "fullblack_7": 10 },  # NEU: klassenspezifische Delays
    "hyperfocus": {
        "enabled": True, "slot": 2, "use_full_frame_for_ocr": False,
        "rois": {
            "1": [1745, 860, 25, 25],
            "2": [1815, 920, 25, 25],
            "3": [1745, 990, 25, 25],
            "4": [1675, 920, 25, 25],
        }
    }
}

def load_settings():
    s = json.loads(json.dumps(DEFAULT_SETTINGS))  # tiefe Kopie
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                j = json.load(f)
            if "hitdelay" in j: s["hitdelay"] = int(j["hitdelay"])
            if "delays" in j and isinstance(j["delays"], dict):
                dj = j["delays"]
                if "wiggle_9" in dj:    s["delays"]["wiggle_9"] = int(dj["wiggle_9"])
                if "fullblack_7" in dj: s["delays"]["fullblack_7"] = int(dj["fullblack_7"])
            if "hyperfocus" in j and isinstance(j["hyperfocus"], dict):
                hf = s["hyperfocus"]; jhf = j["hyperfocus"]
                for k in ("enabled","slot","use_full_frame_for_ocr","rois"):
                    if k in jhf: hf[k] = jhf[k]
        except Exception as e:
            print(f"[SET] Warn: {e}")
    return s

def save_settings(hitdelay, hf_cfg: HyperfocusConfig, wiggle_delay=None, fullblack_delay=None):
    blob = {
        "hitdelay": int(hitdelay),
        "delays": {
            "wiggle_9": int(wiggle_delay if wiggle_delay is not None else DEFAULT_SETTINGS["delays"]["wiggle_9"]),
            "fullblack_7": int(fullblack_delay if fullblack_delay is not None else DEFAULT_SETTINGS["delays"]["fullblack_7"]),
        },
        "hyperfocus": {
            "enabled": bool(hf_cfg.enabled),
            "slot": int(hf_cfg.slot),
            "use_full_frame_for_ocr": bool(hf_cfg.use_full_frame_for_ocr),
            "rois": {str(k): list(v) for k, v in hf_cfg.rois.items()}
        }
    }
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(blob, f, indent=2)
    except Exception as e:
        print(f"[SET] Warn beim Speichern: {e}")

SET = load_settings()
g_hit_ante_ms  = int(SET["hitdelay"])
g_wiggle9_ms   = int(SET["delays"]["wiggle_9"])
g_fullblk7_ms  = int(SET["delays"]["fullblack_7"])

# ── On-the-fly Toggle Hooks für Collector ────────────────────────────────────
_RUNTIME_LOCK  = threading.Lock()
_RUNTIME_HOOKS = {"sc": None, "hf": None}   # werden auf lebende Objekte gesetzt
_RUNTIME_FLAGS = {"sc": None, "hf": None}   # letzter Wunschzustand (vor Objektlebenszeit möglich)

def _toggle_sc_runtime(enabled: bool):
    with _RUNTIME_LOCK:
        _RUNTIME_FLAGS["sc"] = bool(enabled)
        fn = _RUNTIME_HOOKS.get("sc")
        if callable(fn):
            fn(bool(enabled))

def _toggle_hf_runtime(enabled: bool):
    with _RUNTIME_LOCK:
        _RUNTIME_FLAGS["hf"] = bool(enabled)
        fn = _RUNTIME_HOOKS.get("hf")
        if callable(fn):
            fn(bool(enabled))

# ── Helpers ──────────────────────────────────────────────────────────────────
def _ns_to_ms(ns: int) -> float:
    return round(ns / 1_000_000.0, 3)

class RollingAverages:
    def __init__(self, window=60):
        self.window = window
        self.buffers = defaultdict(lambda: deque(maxlen=window))
    def add(self, name: str, value_ms: float):
        if value_ms is None: return
        self.buffers[name].append(float(value_ms))
    def table(self):
        avgs = {k: (sum(v) / len(v) if v else 0.0) for k, v in self.buffers.items()}
        grab = avgs.get("grab_ms", 0.0)
        pre = avgs.get("pre_ms", 0.0)
        infer = avgs.get("infer_ms", 0.0)
        post = avgs.get("post_ms", 0.0)
        loop = avgs.get("loop_ms", 0.0)
        known_sum = grab + pre + infer + post
        other = max(loop - known_sum, 0.0)
        total = grab + pre + infer + post + other
        def pct(x): return 0.0 if total <= 0 else round(100.0 * x / total, 1)
        return [
            ["HDMI/CSI grab", round(grab, 3), pct(grab)],
            ["Preprocess", round(pre, 3), pct(pre)],
            ["AI inference", round(infer, 3), pct(infer)],
            ["Postprocess", round(post, 3), pct(post)],
            ["Other/overhead", round(other, 3), pct(other)],
            ["Total (loop)", round(total, 3), 100.0],
        ]

def _model_choices_for_provider(provider_label: str):
    if not os.path.isdir(MODELS_FOLDER): return []
    if provider_label.lower().startswith("cpu"):
        exts = (".onnx",)
    else:
        exts = (".hef",)
    files = []
    for f in sorted(os.listdir(MODELS_FOLDER)):
        fp = os.path.join(MODELS_FOLDER, f)
        if os.path.isfile(fp) and f.lower().endswith(exts):
            files.append((f, fp))
    return files

def _validate_model_provider(path: str, provider_label: str):
    if not path or not os.path.exists(path):
        raise gr.Error("Kein gültiger Modelpfad.")
    ext = os.path.splitext(path)[1].lower()
    prov = "CPU" if provider_label.lower().startswith("cpu") else "HAILO"
    if prov == "HAILO":
        if ext == ".hef": return prov, path
        gr.Warning("Hailo8 gewählt, aber .onnx gefunden → schalte auf CPU/ONNX um.")
        return "CPU", path
    if ext == ".onnx": return prov, path
    if ext == ".hef":
        gr.Warning("CPU gewählt, aber .hef gefunden → schalte auf Hailo8 um.")
        return "HAILO", path
    raise gr.Error("Nicht unterstütztes Modellformat für die gewählte Ausführung.")

# ── Hyperfocus Defaults ──────────────────────────────────────────────────────
_hf_cfg = HyperfocusConfig(
    enabled=bool(SET["hyperfocus"].get("enabled", True)),
    slot=int(SET["hyperfocus"].get("slot", 2)),
    use_full_frame_for_ocr=bool(SET["hyperfocus"].get("use_full_frame_for_ocr", False)),
)
_ro = SET["hyperfocus"].get("rois", {})
def _r(key, default):
    v = _ro.get(key)
    return tuple(map(int, v)) if (isinstance(v, (list, tuple)) and len(v) == 4) else default
_hf_cfg.rois[1] = _r("1", _hf_cfg.rois[1]); _hf_cfg.rois[2] = _r("2", _hf_cfg.rois[2])
_hf_cfg.rois[3] = _r("3", _hf_cfg.rois[3]); _hf_cfg.rois[4] = _r("4", _hf_cfg.rois[4])

# ── Haupt-Loop ───────────────────────────────────────────────────────────────
def monitor(ai_model_path, provider_label, monitor_id,
            sc_data_collect, hf_data_collect,
            hit_ante, wiggle_delay, fullblack_delay, cpu_threads,   # NEU: zwei Delays
            hf_enabled, hf_slot, hf_roi1, hf_roi2, hf_roi3, hf_roi4,
            hf_async, hf_hz, live_roi):

    provider, ai_model_path = _validate_model_provider(ai_model_path, provider_label)

    global g_hit_ante_ms, _hf_cfg
    g_hit_ante_ms = int(hit_ante)

    _hf_cfg.enabled = bool(hf_enabled)
    _hf_cfg.slot = int(hf_slot or 2)
    def _parse_roi(txt, fallback):
        try:
            parts = [int(x.strip()) for x in str(txt).replace(";",",").split(",")]
            return tuple(parts) if len(parts)==4 else fallback
        except Exception:
            return fallback
    _hf_cfg.rois[1] = _parse_roi(hf_roi1, _hf_cfg.rois[1])
    _hf_cfg.rois[2] = _parse_roi(hf_roi2, _hf_cfg.rois[2])
    _hf_cfg.rois[3] = _parse_roi(hf_roi3, _hf_cfg.rois[3])
    _hf_cfg.rois[4] = _parse_roi(hf_roi4, _hf_cfg.rois[4])

    # Settings inkl. klassenspezifischer Delays persistieren
    save_settings(g_hit_ante_ms, _hf_cfg, wiggle_delay=wiggle_delay, fullblack_delay=fullblack_delay)

    ai_model = AI_model(model_path=ai_model_path, provider=provider, nb_cpu_threads=cpu_threads, monitor_id=monitor_id)
    gr.Info(f"Running on {ai_model.check_provider()}")

    hf = Hyperfocus(_hf_cfg)
    if bool(hf_enabled) and bool(hf_async):
        hf.start_worker(float(hf_hz))
    else:
        hf.stop_worker()

    # ── Data Collection init ─────────────────────────────────────────────────
    sc_collector = SkillcheckCollector(
        enabled=bool(sc_data_collect),
        base_dir="Skillcheck Data",
        close_zero_consec=2,
        linger_ms_after_nonzero=180,
        dont_close_before_ms=120,
        save_zero_while_active=True
    )
    hf_collector = HyperfocusCollector(enabled=bool(hf_data_collect), base_dir="Hyperfocus Data")

    # Hooks für On-the-fly Toggle registrieren (lebende Objekte)
    def _set_sc(en: bool): sc_collector.set_enabled(bool(en))
    def _set_hf(en: bool): hf_collector.set_enabled(bool(en))
    with _RUNTIME_LOCK:
        _RUNTIME_HOOKS["sc"] = _set_sc
        _RUNTIME_HOOKS["hf"] = _set_hf
        # ggf. zuletzt gewünschte Flags sofort übernehmen
        if _RUNTIME_FLAGS["sc"] is not None and _RUNTIME_FLAGS["sc"] != sc_collector.enabled:
            sc_collector.set_enabled(_RUNTIME_FLAGS["sc"])
        if _RUNTIME_FLAGS["hf"] is not None and _RUNTIME_FLAGS["hf"] != hf_collector.enabled:
            hf_collector.set_enabled(_RUNTIME_FLAGS["hf"])

    # Frame-Tap-Thread: liest JEDES Kamera-Frame und liefert es an Collector + Inferenz
    stop_evt = threading.Event()
    newframe_evt = threading.Event()
    latest = {"frame": None}
    def _tap():
        while not stop_evt.is_set():
            f = ai_model.monitor.wait_for_new_frame()
            if f is None:
                continue
            sc_collector.ingest_frame(f)   # jedes Kamera-Frame landet hier
            latest["frame"] = f
            newframe_evt.set()
    tap_th = threading.Thread(target=_tap, name="FrameTap", daemon=True)
    tap_th.start()

    last_hit_time_ns = 0
    min_hit_interval_ms = 500

    stats = RollingAverages(window=60)
    fps_t0_ns = perf_counter_ns()
    nb_frames = 0

    ui_roi_period = 0.25
    next_roi_ts = perf_counter()

    try:
        while True:
            loop_start_ns = perf_counter_ns()

            # Frame aus Tap beziehen (keine Event-Kollision mit Inferenz)
            t0_ns = perf_counter_ns()
            newframe_evt.wait(timeout=1.0)
            frame_np = latest["frame"]
            newframe_evt.clear()
            grab_ms = _ns_to_ms(perf_counter_ns() - t0_ns)

            # HF push
            if hf.cfg.enabled:
                if perf_counter() >= next_roi_ts or (not hf_async):
                    full = ai_model.monitor.get_full_frame()
                    hf.push_full_frame(full)
                    next_roi_ts = perf_counter() + ui_roi_period
                if not bool(hf_async):
                    hf.maybe_update_tokens()

            # Vorhersage
            t1_ns = perf_counter_ns()
            pred, desc, probs, _ = ai_model.predict(frame_np)
            predict_total_ms = _ns_to_ms(perf_counter_ns() - t1_ns)

            # Schwellwert: ausschließlich class_thresholds.json
            p_pred = float(probs.get(desc, 0.0))
            is_hit_class = bool(getattr(ai_model, "pred_dict", {}).get(pred, {}).get("hit", False))
            thr_model = float(getattr(ai_model, "class_thresholds", {}).get(pred, 0.50))
            should_hit = is_hit_class and (p_pred >= thr_model)

            # Skillcheck-State basierend auf Vorhersage updaten (steuert Session offen/zu)
            sc_collector.update_pred(int(pred))

            # Latenzen
            t = ai_model.get_timings() or {}
            pre_ms = t.get("pre_ms"); infer_ms = t.get("infer_ms"); post_ms = t.get("post_ms")
            stats.add("grab_ms", grab_ms)
            if pre_ms is not None:   stats.add("pre_ms", pre_ms)
            if infer_ms is not None: stats.add("infer_ms", infer_ms)
            if post_ms is not None:  stats.add("post_ms", post_ms)
            if pre_ms is None and infer_ms is None and post_ms is None:
                stats.add("infer_ms", predict_total_ms)

            # >>> präzises Press-Timing
            detect_ms = float(grab_ms) + float(pre_ms or 0.0) + float((infer_ms if infer_ms is not None else predict_total_ms)) + float(post_ms or 0.0)
            t_detect_ns = loop_start_ns + int(detect_ms * 1e6)

            # SPACE (mit individuellen Delays & Entprellung)
            now_ns = perf_counter_ns()
            if should_hit:
                dt_ms = _ns_to_ms(now_ns - last_hit_time_ns)
                if dt_ms > min_hit_interval_ms:
                    pred_int = int(pred)
                    is_ante = pred_int in ai_model.ANTE_FRONTIER_IDS
                    is_wiggle_any = pred_int in (9, 10)  # für HF-Anpassung ausnehmen
                    is_wiggle9 = (pred_int == 9)
                    is_fullblack7 = (pred_int == 7)

                    # Prioritäten: klassenspezifische Delays zuerst, dann Ante
                    if is_wiggle9:
                        target_ms = int(wiggle_delay or 0)
                    elif is_fullblack7:
                        target_ms = int(fullblack_delay or 0)
                    elif is_ante and g_hit_ante_ms > 0:
                        target_ms = int(g_hit_ante_ms)
                        # HF-Delay-Reduktion NICHT bei Wiggle-Klassen
                        if hf.cfg.enabled and (not is_wiggle_any) and target_ms > 0:
                            target_ms = int(round(hf.adjust_delay_ms(target_ms)))
                    else:
                        target_ms = 0

                    deadline_ns = t_detect_ns + int(target_ms * 1e6)
                    if deadline_ns > perf_counter_ns():
                        wait_until_ns(deadline_ns)

                    PressKey(SPACE)
                    last_hit_time_ns = perf_counter_ns()

                    # Sofort-UI
                    loop_ms = _ns_to_ms(perf_counter_ns() - loop_start_ns)
                    stats.add("loop_ms", loop_ms)
                    latency_rows = stats.table()

                    hf_tokens = int(hf.tokens_live) if hf.cfg.enabled else 0
                    ocr_status = f"Detected: {hf_tokens}" if hf.cfg.enabled else "OCR disabled"
                    ocr_engine = "tesseract" if hf.cfg.enabled else "-"

                    # Hyperfocus Data collection: Original-ROI speichern (0..6)
                    if hf_collector.enabled and hf.cfg.enabled and getattr(hf, "_dbg_roi_orig", None) is not None:
                        hf_collector.save_if_updated(hf._dbg_roi_orig, token=int(hf_tokens))

                    if hf.cfg.enabled and perf_counter() >= next_roi_ts:
                        roi_prev = hf.preview_roi()
                        full = hf._last_full_from_app
                        def cut(f, r):
                            if f is None: return None
                            x,y,w,h = r; x=max(0,x); y=max(0,y); return f[y:y+h, x:x+w] if w>0 and h>0 else None
                        r1,r2,r3,r4 = hf.cfg.rois[1],hf.cfg.rois[2],hf.cfg.rois[3],hf.cfg.rois[4]
                        roi_imgs = [cut(full,r1),cut(full,r2),cut(full,r3),cut(full,r4)]
                        next_roi_ts = perf_counter() + ui_roi_period
                    else:
                        roi_prev = gr.update()
                        roi_imgs = [gr.update(), gr.update(), gr.update(), gr.update()]
                    dbg_pre = hf._dbg_pre if hf.cfg.enabled else gr.update()

                    yield gr.skip(), frame_np, probs, latency_rows, \
                          hf_tokens, ocr_status, ocr_engine, dbg_pre, roi_prev, \
                          roi_imgs[0], roi_imgs[1], roi_imgs[2], roi_imgs[3]
                    # kein künstlicher Delay – sonst droppen wir Post-Hit-Frames
                    fps_t0_ns = perf_counter_ns(); nb_frames = 0
                    continue

            # regelmäßiges UI-Update
            loop_ms = _ns_to_ms(perf_counter_ns() - loop_start_ns)
            stats.add("loop_ms", loop_ms)
            nb_frames += 1
            t_diff_ms = _ns_to_ms(perf_counter_ns() - fps_t0_ns)
            if t_diff_ms > 1000.0:
                fps = round(nb_frames / (t_diff_ms / 1000.0), 1)
                latency_rows = stats.table()

                hf_tokens = int(hf.tokens_live) if hf.cfg.enabled else 0
                ocr_status = f"Detected: {hf_tokens}" if hf.cfg.enabled else "OCR disabled"
                ocr_engine = "tesseract" if hf.cfg.enabled else "-"

                # Hyperfocus Data collection: auch beim regulären Update
                if hf_collector.enabled and hf.cfg.enabled and getattr(hf, "_dbg_roi_orig", None) is not None:
                    hf_collector.save_if_updated(hf._dbg_roi_orig, token=int(hf_tokens))

                if hf.cfg.enabled and perf_counter() >= next_roi_ts:
                    roi_prev = hf.preview_roi()
                    full = hf._last_full_from_app
                    def cut(f, r):
                        if f is None: return None
                        x,y,w,h = r; x=max(0,x); y=max(0,y); return f[y:y+h, x:x+w] if w>0 and h>0 else None
                    r1,r2,r3,r4 = hf.cfg.rois[1],hf.cfg.rois[2],hf.cfg.rois[3],hf.cfg.rois[4]
                    roi_imgs = [cut(full,r1),cut(full,r2),cut(full,r3),cut(full,r4)]
                    next_roi_ts = perf_counter() + ui_roi_period
                else:
                    roi_prev = gr.update()
                    roi_imgs = [gr.update(), gr.update(), gr.update(), gr.update()]
                dbg_pre = hf._dbg_pre if hf.cfg.enabled else gr.update()

                yield fps, gr.skip(), gr.skip(), latency_rows, \
                      hf_tokens, ocr_status, ocr_engine, dbg_pre, roi_prev, \
                      roi_imgs[0], roi_imgs[1], roi_imgs[2], roi_imgs[3]
                fps_t0_ns = perf_counter_ns(); nb_frames = 0

    finally:
        try: hf.stop_worker()
        except Exception: pass
        try: ai_model.cleanup()
        except Exception: pass
        try:
            stop_evt.set()
            tap_th.join(timeout=0.2)
        except Exception:
            pass
        # Hooks wieder freigeben (verhindert Toggle auf stale Objekte)
        with _RUNTIME_LOCK:
            if _RUNTIME_HOOKS.get("sc") is _set_sc: _RUNTIME_HOOKS["sc"] = None
            if _RUNTIME_HOOKS.get("hf") is _set_hf: _RUNTIME_HOOKS["hf"] = None

# ── UI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    default_provider = "Hailo8"
    model_files = _model_choices_for_provider(default_provider)
    if not model_files:
        default_provider = "CPU"
        model_files = _model_choices_for_provider(default_provider)

    def _on_provider_change(new_provider_label):
        files = _model_choices_for_provider(new_provider_label)
        if not files:
            gr.Warning("Kein passendes Modell im models/-Ordner gefunden.")
            return gr.update(choices=[], value=None)
        return gr.update(choices=files, value=files[0][1])

    def _roi_str(t): return ", ".join(map(str, t))

    hf_default = SET["hyperfocus"]

    with gr.Blocks(title="Auto skill check") as webui:
        gr.Markdown("<h1 style='text-align: center;'>DBD Auto skill check</h1>")
        with gr.Row():
            with gr.Column(variant="panel"):
                gr.Markdown("AI inference settings")
                ai_model_path = gr.Dropdown(choices=model_files, value=(model_files[0][1] if model_files else ""),
                                            label="Model file (ONNX / Hailo HEF)")
                provider_label = gr.Radio(choices=["CPU", "Hailo8"], value=default_provider, label="Execution device")
                provider_label.change(fn=_on_provider_change, inputs=provider_label, outputs=ai_model_path)

                monitor_id = gr.Dropdown(
                    choices=[(os.environ.get("X1300_DEVICE","/dev/video0"),
                              os.environ.get("X1300_DEVICE","/dev/video0"))],
                    value=os.environ.get("X1300_DEVICE","/dev/video0"), label="Monitor")

                # Data Collection Optionen direkt unter „Monitor“
                sc_data_collect = gr.Checkbox(value=False, label="Skillcheck Data collection")
                hf_data_collect = gr.Checkbox(value=False, label="Hyperfocus Data collection")
                # On-the-fly Toggle → setzt Runtime-Flags/Hooks, kein Reload nötig
                sc_data_collect.change(fn=lambda en: _toggle_sc_runtime(en), inputs=sc_data_collect, outputs=[])
                hf_data_collect.change(fn=lambda en: _toggle_hf_runtime(en), inputs=hf_data_collect, outputs=[])

                gr.Markdown("AI Features options")
                hit_ante = gr.Slider(0, 120, step=1, value=g_hit_ante_ms, label="Ante-frontier hit delay (ms)")
                wiggle_delay = gr.Slider(0, 50, step=1, value=g_wiggle9_ms,   label="Wiggle Hit delay (Klasse 9)")
                fullblack_delay = gr.Slider(0, 50, step=1, value=g_fullblk7_ms, label="Full Black hit delay (Klasse 7)")
                cpu_threads = gr.Radio(label="CPU threads (ONNX)", choices=[1,2,4,8], value=4)

                with gr.Accordion("Hyperfocus (Perk OCR)", open=False):
                    hf_enabled = gr.Checkbox(value=bool(hf_default.get("enabled", True)),
                                             label="Enable Hyperfocus (+4%/token delay reduction)")
                    hf_slot = gr.Number(value=int(hf_default.get("slot", 2)), precision=0, label="Selected slot (1–4)")
                    ro = hf_default.get("rois", {})
                    hf_roi1 = gr.Textbox(value=_roi_str(ro.get("1", (1745,860,25,25))), label="Slot 1 ROI (Top)")
                    hf_roi2 = gr.Textbox(value=_roi_str(ro.get("2", (1815,920,25,25))), label="Slot 2 ROI (Right)")
                    hf_roi3 = gr.Textbox(value=_roi_str(ro.get("3", (1745,990,25,25))), label="Slot 3 ROI (Bottom)")
                    hf_roi4 = gr.Textbox(value=_roi_str(ro.get("4", (1675,920,25,25))), label="Slot 4 ROI (Left)")
                    hf_async = gr.Checkbox(value=True, label="Run OCR in background (non-blocking)")
                    hf_hz    = gr.Slider(5, 20, step=1, value=10, label="OCR target Hz (worker)")
                    live_roi = gr.Checkbox(value=True, label="Live ROI thumbnails")

            with gr.Column(variant="panel"):
                fps = gr.Number(label="AI model FPS", interactive=False)
                image_visu = gr.Image(label="Last hit skill check frame", height=420, interactive=False)
                probs = gr.Label(label="Skill check AI recognition (class → prob)")
                gr.Markdown("### Latency (ms) — rolling average (60 frames)")
                latency_df = gr.Dataframe(
                    headers=["Step", "ms", "% of total"],
                    value=[["HDMI/CSI grab",0,0],["Preprocess",0,0],["AI inference",0,0],
                           ["Postprocess",0,0],["Other/overhead",0,0],["Total (loop)",0,0]],
                    datatype=["str","number","number"], interactive=False, wrap=True)

                with gr.Row():
                    hf_tokens_live = gr.Number(label="Hyperfocus Tokens (live)", precision=0, interactive=False, value=0)
                    ocr_status = gr.Label(label="OCR status")
                    ocr_engine = gr.Label(label="OCR engine")

                with gr.Row():
                    pre_dbg = gr.Image(label="Preprocessed (debug)", interactive=False, height=110)
                    roi_prev = gr.Image(label="ROI preview", interactive=False, height=110)
                with gr.Row():
                    roi1_img = gr.Image(label="Perk ROI 1 (Top)", interactive=False, height=110)
                    roi2_img = gr.Image(label="Perk ROI 2 (Right)", interactive=False, height=110)
                with gr.Row():
                    roi3_img = gr.Image(label="Perk ROI 3 (Bottom)", interactive=False, height=110)
                    roi4_img = gr.Image(label="Perk ROI 4 (Left)", interactive=False, height=110)

        run_button  = gr.Button("RUN", variant="primary")
        stop_button = gr.Button("STOP", variant="stop")

        monitoring = run_button.click(
            fn=monitor,
            inputs=[ai_model_path, provider_label, monitor_id,
                    sc_data_collect, hf_data_collect,
                    hit_ante, wiggle_delay, fullblack_delay, cpu_threads,
                    hf_enabled, hf_slot, hf_roi1, hf_roi2, hf_roi3, hf_roi4,
                    hf_async, hf_hz, live_roi],
            outputs=[fps, image_visu, probs, latency_df,
                     hf_tokens_live, ocr_status, ocr_engine,
                     pre_dbg, roi_prev, roi1_img, roi2_img, roi3_img, roi4_img]
        )
        stop_button.click(fn=lambda: 0.0, inputs=None, outputs=fps, cancels=monitoring)

    webui.launch()
