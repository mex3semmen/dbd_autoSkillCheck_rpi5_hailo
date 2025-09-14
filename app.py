import os
from time import perf_counter_ns, sleep
from collections import deque, defaultdict
import gradio as gr

from dbd.AI_model import AI_model
from dbd.utils.directkeys import PressKey, ReleaseKey, SPACE
from dbd.utils.monitoring_hdmi import Monitoring_mss

ai_model = None
MODELS_FOLDER = "models"

# --- NEU: globale Runtime-Variable für Live-Update des Sliders -------------
g_hit_ante_ms = 80  # wird beim RUN initialisiert; Slider .change ändert das live


# ---------- Helpers -----------------------------------------------------------

def _ns_to_ms(ns: int) -> float:
    return round(ns / 1_000_000.0, 3)


class RollingAverages:
    """Rolling averages for named metrics over last N samples."""

    def __init__(self, window=60):
        self.window = window
        self.buffers = defaultdict(lambda: deque(maxlen=window))

    def add(self, name: str, value_ms: float):
        if value_ms is None:
            return
        self.buffers[name].append(float(value_ms))

    def get_avg(self, name: str) -> float:
        """Get average for a specific metric."""
        buf = self.buffers.get(name, [])
        return sum(buf) / len(buf) if buf else 0.0

    def table(self):
        """Return rows suitable for a Gradio Dataframe: [Step, ms, % of total]."""
        avgs = {k: (sum(v) / len(v) if v else 0.0) for k, v in self.buffers.items()}
        grab = avgs.get("grab_ms", 0.0)
        pre = avgs.get("pre_ms", 0.0)
        infer = avgs.get("infer_ms", 0.0)
        post = avgs.get("post_ms", 0.0)
        loop = avgs.get("loop_ms", 0.0)

        known_sum = grab + pre + infer + post
        other = max(loop - known_sum, 0.0)
        total = grab + pre + infer + post + other

        def pct(x):
            return 0.0 if total <= 0 else round(100.0 * x / total, 1)

        rows = [
            ["HDMI/CSI grab", round(grab, 3), pct(grab)],
            ["Preprocess", round(pre, 3), pct(pre)],
            ["AI inference", round(infer, 3), pct(infer)],
            ["Postprocess", round(post, 3), pct(post)],
            ["Other/overhead", round(other, 3), pct(other)],
            ["Total (loop)", round(total, 3), 100.0],
        ]
        return rows


# ---------- App lifecycle -----------------------------------------------------

def cleanup():
    global ai_model
    if ai_model is not None:
        del ai_model
        ai_model = None
    return 0.0


def _list_models_for_provider(provider_label: str):
    if not os.path.isdir(MODELS_FOLDER):
        return []
    if provider_label.startswith("CPU"):
        exts = (".onnx", ".trt")
    else:
        exts = (".hef",)
    items = []
    for f in sorted(os.listdir(MODELS_FOLDER)):
        if f.lower().endswith(exts):
            items.append((f, f"{MODELS_FOLDER}/{f}"))
    return items


def _validate_model_provider(path: str, provider_label: str):
    prov = "CPU" if provider_label.startswith("CPU") else "HAILO"
    if prov == "CPU":
        if path is None or not os.path.exists(path) or not path.lower().endswith((".onnx", ".trt")):
            raise gr.Error("CPU/ONNX-Modus benötigt .onnx (oder optional .trt).", duration=0)
        return prov, path
    # HAILO
    if path and path.lower().endswith(".hef") and os.path.exists(path):
        return prov, path
    default_hef = os.path.join(MODELS_FOLDER, "model.hef")
    if os.path.exists(default_hef):
        return prov, default_hef
    raise gr.Error("Hailo8-Modus benötigt .hef (z.B. models/model.hef).", duration=0)


# ---------- Main monitor loop -------------------------------------------------

def monitor(ai_model_path, provider_label, monitor_id, hit_ante, nb_cpu_threads):
    if monitor_id is None:
        raise gr.Error("Invalid monitor option")

    provider, ai_model_path = _validate_model_provider(ai_model_path, provider_label)

    try:
        global ai_model, g_hit_ante_ms
        # Initialwert aus RUN übernehmen
        g_hit_ante_ms = int(hit_ante) if hit_ante is not None else g_hit_ante_ms

        ai_model = AI_model(model_path=ai_model_path, provider=provider,
                            nb_cpu_threads=nb_cpu_threads, monitor_id=monitor_id)
        execution_provider = ai_model.check_provider()
    except Exception as e:
        raise gr.Error(f"Error when loading AI model: {e}", duration=0)

    if execution_provider == "HAILO":
        gr.Info("Running AI model on Hailo-8 (success)")
    elif execution_provider == "TensorRT":
        gr.Info("Running AI model on GPU (success, TensorRT)")
    elif execution_provider in ("CUDAExecutionProvider", "DmlExecutionProvider"):
        gr.Info(f"Running AI model on GPU (success, {execution_provider})")
    else:
        gr.Info(f"Running AI model on CPU (success, {nb_cpu_threads} threads)")

    fps_t0_ns = perf_counter_ns()
    nb_frames = 0
    stats = RollingAverages(window=60)

    # Timing & Debounce
    last_hit_time_ns = 0
    min_hit_interval_ms = 500  # block duplicate triggers

    # Pre-build an empty table
    latency_rows = [
        ["HDMI/CSI grab", 0.0, 0.0],
        ["Preprocess", 0.0, 0.0],
        ["AI inference", 0.0, 0.0],
        ["Postprocess", 0.0, 0.0],
        ["Other/overhead", 0.0, 0.0],
        ["Total (loop)", 0.0, 0.0],
    ]

    try:
        while True:
            loop_start_ns = perf_counter_ns()

            # ---- Grab from HDMI/CSI -----------------------------------------
            t0_ns = perf_counter_ns()
            frame_np = ai_model.wait_for_new_frame()
            grab_ms = _ns_to_ms(perf_counter_ns() - t0_ns)

            # ---- Predict -----------------------------------------------------
            t1_ns = perf_counter_ns()
            pred, desc, probs, should_hit = ai_model.predict(frame_np)
            predict_total_ms = _ns_to_ms(perf_counter_ns() - t1_ns)

            # Optional: fine-grained timings
            pre_ms = infer_ms = post_ms = None
            get_timings = getattr(ai_model, "get_timings", None)
            if callable(get_timings):
                t = get_timings() or {}
                pre_ms = float(t.get("pre_ms")) if t.get("pre_ms") is not None else None
                infer_ms = float(t.get("infer_ms")) if t.get("infer_ms") is not None else None
                post_ms = float(t.get("post_ms")) if t.get("post_ms") is not None else None

            # ---- Rolling stats update ----------------------------------------
            stats.add("grab_ms", grab_ms)
            if pre_ms is not None or infer_ms is not None or post_ms is not None:
                if pre_ms is not None:   stats.add("pre_ms", pre_ms)
                if infer_ms is not None: stats.add("infer_ms", infer_ms)
                if post_ms is not None:  stats.add("post_ms", post_ms)
            else:
                stats.add("infer_ms", predict_total_ms)

            # ---- Trigger logic (reads global live) ---------------------------
            current_time_ns = perf_counter_ns()

            if should_hit:
                time_since_last_hit_ms = _ns_to_ms(current_time_ns - last_hit_time_ns)

                if time_since_last_hit_ms > min_hit_interval_ms:
                    processing_elapsed_ms = _ns_to_ms(current_time_ns - loop_start_ns)

                    # --- NEU: live gelesener Wert
                    local_hit_ante = int(g_hit_ante_ms)  # atomare Int-Reads sind threadsafe

                    if pred == 2 and local_hit_ante > 0:
                        adjusted_delay = max(0, local_hit_ante - processing_elapsed_ms)
                        if adjusted_delay > 0:
                            sleep(adjusted_delay * 0.001)

                    PressKey(SPACE)
                    last_hit_time_ns = perf_counter_ns()

                    loop_ms = _ns_to_ms(perf_counter_ns() - loop_start_ns)
                    stats.add("loop_ms", loop_ms)

                    latency_rows = stats.table()
                    yield gr.skip(), frame_np, probs, latency_rows

                    sleep(0.05)
                    fps_t0_ns = perf_counter_ns()
                    nb_frames = 0
                    continue

            # Regular loop timing update
            loop_ms = _ns_to_ms(perf_counter_ns() - loop_start_ns)
            stats.add("loop_ms", loop_ms)
            nb_frames += 1

            # Once per ~1s: update FPS and latency table
            t_diff_ms = _ns_to_ms(perf_counter_ns() - fps_t0_ns)
            if t_diff_ms > 1000.0:
                fps = round(nb_frames / (t_diff_ms / 1000.0), 1)
                latency_rows = stats.table()
                yield fps, gr.skip(), gr.skip(), latency_rows
                fps_t0_ns = perf_counter_ns()
                nb_frames = 0

    except Exception:
        pass
    finally:
        print("Monitoring stopped.")


# ---------- UI ---------------------------------------------------------------

if __name__ == "__main__":
    fps_info = "Number of frames per second the AI model analyses the monitored frame."
    providers = ["CPU (default)", "Hailo8"]
    cpu_choices = [("Low", 2), ("Normal", 4), ("High", 6), ("Computer Killer Mode", 8)]

    # Initial model list
    model_files = _list_models_for_provider(providers[0])
    if len(model_files) == 0 and os.path.isdir(MODELS_FOLDER):
        for f in sorted(os.listdir(MODELS_FOLDER)):
            if f.endswith((".onnx", ".trt", ".hef")):
                model_files.append((f, f'{MODELS_FOLDER}/{f}'))
    if len(model_files) == 0:
        raise gr.Error(f"No AI model found in {MODELS_FOLDER}/", duration=0)

    monitor_choices = Monitoring_mss.get_monitors_info()

    def switch_monitor_cb(monitor_id):
        with Monitoring_mss(monitor_id, crop_size=520) as monitor:
            return monitor.get_frame_np()

    def on_provider_change(new_provider_label):
        choices = _list_models_for_provider(new_provider_label)
        if not choices:
            gr.Warning("No models for selected provider found in models/ folder.")
            return gr.update(), gr.update()
        if not new_provider_label.startswith("CPU"):
            pref = os.path.join(MODELS_FOLDER, "model.hef")
            value = next((v for (lbl, v) in choices if v == pref), choices[0][1])
            cpu_interactive = False
        else:
            value = choices[0][1]
            cpu_interactive = True
        return gr.update(choices=choices, value=value), gr.update(interactive=cpu_interactive)

    # --- NEU: Callback zum Live-Setzen des globalen Wertes -------------------
    def set_hit_ante_cb(v):
        global g_hit_ante_ms
        try:
            g_hit_ante_ms = int(v)
        except Exception:
            pass  # ignorieren
        # keine Outputs nötig

    with (gr.Blocks(title="Auto skill check") as webui):
        gr.Markdown("<h1 style='text-align: center;'>DBD Auto skill check</h1>", elem_id="title")
        gr.Markdown("https://github.com/Manuteaa/dbd_autoSkillCheck")

        with gr.Row():
            with gr.Column(variant="panel"):
                with gr.Column(variant="panel"):
                    gr.Markdown("AI inference settings")
                    ai_model_path = gr.Dropdown(
                        choices=model_files,
                        value=model_files[0][1],
                        label="Name the AI model to use (ONNX / TensorRT / Hailo HEF)"
                    )
                    provider_label = gr.Radio(
                        choices=providers, value=providers[0],
                        label="Device the AI model will use"
                    )
                    monitor_id = gr.Dropdown(
                        choices=monitor_choices,
                        value=monitor_choices[0][1],
                        label="Monitor to use"
                    )
                with gr.Column(variant="panel"):
                    gr.Markdown("AI Features options")
                    hit_ante = gr.Slider(minimum=0, maximum=100, step=5, value=80,
                                         label="Ante-frontier hit delay in ms")
                    cpu_stress = gr.Radio(
                        label="CPU workload for AI model inference (increase to improve AI model FPS or decrease to reduce CPU stress)",
                        choices=cpu_choices,
                        value=cpu_choices[1][1],
                    )
                with gr.Column():
                    run_button = gr.Button("RUN", variant="primary")
                    stop_button = gr.Button("STOP", variant="stop")

            with gr.Column(variant="panel"):
                fps = gr.Number(label="AI model FPS", info=fps_info, interactive=False)
                image_visu = gr.Image(label="Last hit skill check frame", height=224, interactive=False)
                probs = gr.Label(label="Skill check AI recognition")

                gr.Markdown("### Latency (ms) — rolling average of last 60 frames")
                latency_df = gr.Dataframe(
                    headers=["Step", "ms", "% of total"],
                    value=[
                        ["HDMI/CSI grab", 0.0, 0.0],
                        ["Preprocess", 0.0, 0.0],
                        ["AI inference", 0.0, 0.0],
                        ["Postprocess", 0.0, 0.0],
                        ["Other/overhead", 0.0, 0.0],
                        ["Total (loop)", 0.0, 0.0],
                    ],
                    datatype=["str", "number", "number"],
                    interactive=False,
                    wrap=True,
                )

        provider_label.change(fn=on_provider_change, inputs=provider_label, outputs=[ai_model_path, cpu_stress])

        monitoring = run_button.click(
            fn=monitor,
            inputs=[ai_model_path, provider_label, monitor_id, hit_ante, cpu_stress],
            outputs=[fps, image_visu, probs, latency_df]
        )

        stop_button.click(fn=cleanup, inputs=None, outputs=fps)
        monitor_id.blur(fn=switch_monitor_cb, inputs=monitor_id, outputs=image_visu)

        # --- NEU: Live-Update des Delays auch während der Monitor-Loop läuft
        hit_ante.change(fn=set_hit_ante_cb, inputs=hit_ante, outputs=None)

    try:
        webui.launch()
    except:
        print("User stopped the web UI. Please wait to cleanup resources...")
    finally:
        cleanup()
