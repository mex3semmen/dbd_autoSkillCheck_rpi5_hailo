#!/usr/bin/env bash
# X1300 / TC358743 Setup – BGR-Ausgabe am Video-Node (BGR3, 24 bpp)
# Pi 5, Raspberry Pi OS/X11, ohne GStreamer. OpenCV liest direkt aus /dev/videoX.
# - EDID setzen (pad=0), DV-Timings abfragen
# - Media-Graph verlinken (csi2:4 -> rp1-cfe-csi2_ch0:0)
# - CSI-BUS-Format: RGB888_1X24
# - Video-Node-Format: BGR3  (OpenCV erwartet BGR → kein Farb-Swap nötig)
# Hinweis: Kernel 6.12.x hat teils R/B-Swap/Streaming-Issues mit tc358743. 6.6 LTS ist robuster.

set -Eeuo pipefail
export LC_ALL=C LANG=C

# ---------- Styling ----------
if [ -t 1 ]; then
  BOLD=$'\e[1m'; DIM=$'\e[2m'; RED=$'\e[31m'; YEL=$'\e[33m'; GRN=$'\e[32m'; CLR=$'\e[0m'
else
  BOLD=""; DIM=""; RED=""; YEL=""; GRN=""; CLR=""
fi
sec() { printf "\n${BOLD}%s${CLR}\n" "$*"; }
ok()  { printf "${GRN}[OK]${CLR} %s\n" "$*"; }
warn(){ printf "${YEL}[WARN]${CLR} %s\n" "$*"; }
err() { printf "${RED}[FATAL]${CLR} %s\n" "$*"; }

# ---------- Optionen ----------
EDID_FILE="${EDID_FILE:-/tmp/X1300_EDID_1080P60.txt}"
RETRIES="${RETRIES:-4}"
SLEEP_S="${SLEEP_S:-0.3}"
ENV_OUT="${ENV_OUT:-/tmp/x1300_env}"
MD_LIMIT="${MD_LIMIT:-160}"

# ---------- Kernelinfo ----------
KREL="$(uname -r || true)"
sec "[X1300] Kernel: $KREL"
[[ "$KREL" =~ ^6\.12\. ]] && warn "Kernel 6.12.x auf Pi 5 ist für TC358743 anfällig. 6.6.x läuft stabiler."

# ---------- Geräte finden ----------
SUBDEV=""
for p in /sys/class/video4linux/v4l-subdev*; do
  [[ -r "$p/name" ]] || continue
  if grep -qi '^tc358743' "$p/name"; then SUBDEV="/dev/$(basename "$p")"; break; fi
done
[[ -n "$SUBDEV" ]] || { err "tc358743-Subdev nicht gefunden (Overlay/Verkabelung)."; exit 1; }

MD=""
for m in /dev/media*; do
  [[ -e "$m" ]] || continue
  if media-ctl -d "$m" -p 2>/dev/null | grep -Fq "device node name $SUBDEV"; then MD="$m"; break; fi
done
[[ -n "$MD" ]] || { err "Kein /dev/mediaN referenziert $SUBDEV."; exit 1; }

VID=""
for v in /dev/video*; do
  n="/sys/class/video4linux/$(basename "$v")/name"
  [[ -r "$n" ]] || continue
  if grep -qi 'rp1-cfe-csi2_ch0' "$n"; then VID="$v"; break; fi
done
VID="${VID:-/dev/video0}"

printf "%s\n" "------------------------------------------------------------"
echo "[X1300] Media: $MD"
echo "[X1300] Subdev: $SUBDEV"
echo "[X1300] Video:  $VID"
printf "%s\n" "------------------------------------------------------------"

# ---------- EDID 1080p60 ----------
if [ ! -s "$EDID_FILE" ]; then
  cat > "$EDID_FILE" <<'EOF'
00ffffffffffff005262888800888888
1c150103800000780aEE91A3544C9926
0F505400000001010101010101010101
010101010101011d007251d01e206e28
5500c48e2100001e8c0ad08a20e02d10
103e9600138e2100001e000000fc0054
6f73686962612d4832430a20000000FD
003b3d0f2e0f1e0a202020202020014f
020322444f841303021211012021223c
3d3e101f2309070766030c00300080E3
007F8c0ad08a20e02d10103e9600c48e
210000188c0ad08a20e02d10103e9600
138e210000188c0aa01451f01600267c
4300138e210000980000000000000000
00000000000000000000000000000000
00000000000000000000000000000015
EOF
  ok "EDID geschrieben: $EDID_FILE"
fi

# ---------- EDID setzen (pad=0) ----------
sec "[X1300] >> v4l2-ctl --set-edid"
v4l2-ctl -d "$SUBDEV" --clear-edid 0 >/dev/null 2>&1 || true
if ! v4l2-ctl -d "$SUBDEV" --set-edid=pad=0,file="$EDID_FILE",format=hex --fix-edid-checksums >/dev/null 2>&1; then
  v4l2-ctl -d "$SUBDEV" --set-edid="file=$EDID_FILE" --fix-edid-checksums >/dev/null 2>&1 || true
fi
v4l2-ctl -d "$SUBDEV" --info-edid 0 2>/dev/null | sed -n '1,60p' || true

# ---------- DV-Timings mehrfach abfragen ----------
W=0; H=0
for i in $(seq 1 "$RETRIES"); do
  sleep "$SLEEP_S"
  if v4l2-ctl -d "$SUBDEV" --query-dv-timings >/tmp/_dv 2>&1; then
    W=$(awk -F': *' '/Active width/ {print $2}'  /tmp/_dv | head -n1)
    H=$(awk -F': *' '/Active height/ {print $2}' /tmp/_dv | head -n1)
    W=${W:-0}; H=${H:-0}
  fi
  [[ "$W" -gt 0 && "$H" -gt 0 ]] && break
  echo "[X1300] DV retry $i/$RETRIES … (Quelle auf 1080p60 stellen/umschalten)"
done

if [[ "$W" -eq 0 || "$H" -eq 0 ]]; then
  printf "%s\n" "------------------------------------------------------------"
  echo "[X1300] DV current: 0x0"
  cat /tmp/_dv || true
  printf "%s\n" "------------------------------------------------------------"
  echo "[X1300] Subdev --all (HDMI-Presence):"
  v4l2-ctl -d "$SUBDEV" --all 2>/dev/null | sed -n '/Digital Video Controls/,+12p' || true
  err "Kein gültiges HDMI-Signal (DV=0x0). EDID/HPD/Quelle/Kabel prüfen."
  exit 2
else
  echo "[X1300] DV current: ${W}x${H}"
  sed -n '1,30p' /tmp/_dv
fi

# Bestätigen (üblich in den Guides)
v4l2-ctl -d "$SUBDEV" --set-dv-bt-timings query >/dev/null 2>&1 || true

# ---------- Media-Graph & Formate ----------
BUS_CODE="RGB888_1X24"
ENTITY_NAME="$(media-ctl -d "$MD" -p | awk '/- entity .*tc358743/ {sub(/.*: /,""); print; exit}')"
ENTITY_NAME="${ENTITY_NAME:-tc358743 11-000f}"

printf "%s\n" "------------------------------------------------------------"
echo "[X1300] Entity: ${ENTITY_NAME}"
echo "[X1300] BUS-Code (tc358743:0) = ${BUS_CODE}"
printf "%s\n" "------------------------------------------------------------"
echo "[X1300] Setze Link & csi2-Pad-Formate (BUS=${BUS_CODE}, Size=${W}x${H})…"

media-ctl -d "$MD" -r >/dev/null 2>&1 || true
media-ctl -d "$MD" -l "'csi2':4 -> 'rp1-cfe-csi2_ch0':0 [1]" || true
media-ctl -d "$MD" -V "'csi2':0 [fmt:${BUS_CODE}/${W}x${H} field:none colorspace:srgb]" || true
media-ctl -d "$MD" -V "'csi2':4 [fmt:${BUS_CODE}/${W}x${H} field:none colorspace:srgb]" || true

sec "Topologie (relevante Zeilen inkl. Links & Formaten):"
media-ctl -d "$MD" -p | awk '
/- entity .*tc358743/ || /- entity .*csi2/ || /rp1-cfe-csi2_ch0/ {print; show=1; next}
show && /pad[0-9]:/ {print}
show && /->/ {print}
' | sed -n "1,${MD_LIMIT}p" || true

if ! media-ctl -d "$MD" -p | grep -Fq -- '-> "rp1-cfe-csi2_ch0":0 [ENABLED]'; then
  err "Link csi2:4 -> rp1-cfe-csi2_ch0:0 ist NICHT enabled."
  exit 3
fi

# ---------- Video-Node auf BGR3 ----------
printf "%s\n" "------------------------------------------------------------"
echo "[X1300] Video-Node auf BGR3 ${W}x${H} setzen…"
set_bgr_ok=0
for attempt in 1 2; do
  v4l2-ctl -d "$VID" -v width="$W",height="$H",pixelformat=BGR3 >/dev/null 2>&1 || true
  fmt="$(v4l2-ctl -d "$VID" --get-fmt-video 2>/dev/null | awk -F"'" '/Pixel Format/ {print $2}')"
  echo "[X1300] --get-fmt-video:"; v4l2-ctl -d "$VID" --get-fmt-video || true
  if [[ "$fmt" == "BGR3" ]]; then set_bgr_ok=1; break; fi
  sleep 0.1
done
if [[ "$set_bgr_ok" -ne 1 ]]; then
  err "Konnte BGR3 nicht setzen (aktuell: '$fmt'). Treiber/Kernel prüfen."
  exit 4
fi

echo "[X1300] --list-formats-ext (Auszug):"
v4l2-ctl -d "$VID" --list-formats-ext | sed -n '1,120p' || true

# ---------- Kurzstream-Test ----------
printf "%s\n" "------------------------------------------------------------"
echo "[X1300] 2-Frame-Test (BGR3, verbose)…"
if v4l2-ctl --verbose -d "$VID" --stream-mmap=4 --stream-count=2 --stream-to=/dev/null --stream-poll >/dev/null 2>&1; then
  ok "Streamtest OK."
else
  warn "STREAMON fehlgeschlagen – meist Pad/BUS-Mismatch oder Kernel 6.12-Bug."
  sec "[X1300] dmesg (tc358743|csi|rp1-cfe, letzte 80 Zeilen):"
  dmesg | grep -Ei 'tc358743|csi|rp1-cfe' | tail -n 80 || true
  err "Streaming-Start gescheitert."
  exit 5
fi

# ---------- ENV für Python/OpenCV ----------
cat > "$ENV_OUT" <<EOF
VIDEO_NODE=$VID
WIDTH=$W
HEIGHT=$H
PIXELFORMAT=BGR3
EOF

sec "[X1300] Setup OK."
echo "[X1300]   Media     : $MD"
echo "[X1300]   Subdev    : $SUBDEV"
echo "[X1300]   VideoNode : $VID"
echo "[X1300]   Size/Pix  : ${W}x${H} BGR3"
echo "[X1300]   Env       : $ENV_OUT"
exit 0
