# DBD AutoSkillCheck — RPi 5 + CSI-HDMI (+ optional Hailo-8)

Fork of **dbd_autoSkillCheck**, adapted for **Raspberry Pi 5** with **CSI-HDMI capture** (TC358743 / Geekworm X1300). Optional inference via **Hailo-8**.

**Upstream/Credit:**  
➡️ https://github.com/Manuteaa/dbd_autoSkillCheck

> ⚠️ Automation may violate the game’s Terms of Service. Use at your own risk.

---

## What this fork adds

- Plug-and-play **CSI-HDMI controller** setup for 1080p60 via `opencv_setup.sh` (media graph, EDID, pixel format).
- OpenCV/V4L2 capture from `/dev/video0` with correct RGB/BGR mapping.
- App runner `app.py` with live preview and timing controls.
- Fixed center **ROI (224×224)** around the skill-check ring.
- Space-bar trigger with configurable **lead/lag (ms)**.
- Optional **Hailo-8** inference path.

---

## Hardware

- **Raspberry Pi 5** (64-bit, Bookworm recommended)
- **HDMI splitter** (PC → Monitor + Pi)
- **CSI-HDMI adapter** (e.g., Geekworm X1300, TC358743, 4-lane)
- Correct **FFC cable** for Pi 5 (22-pin Type-A)
- (Optional) **Hailo-8 M.2** + HailoRT

---

## Install & Run (quick)

```bash
# 1) Clone
git clone https://github.com/mex3semmen/dbd_autoskillcheck_rpi5_halo.git
cd dbd_autoskillcheck_rpi5_halo

# 2) Python env
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt  # opencv, gradio, etc.

# 3) Enable CSI-HDMI controller (builds /dev/video0)
chmod +x opencv_setup.sh
sudo ./opencv_setup.sh
# If your script lives in ./scripts/opencv_setup.sh, adjust the path.

# 4) Launch the app
python3 app.py
