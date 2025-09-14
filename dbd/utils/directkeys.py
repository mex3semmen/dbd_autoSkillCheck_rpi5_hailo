# directkeys.py (STM32 GPIO-only Version)
# Sendet Leertaste-Trigger per GPIO-Puls zu STM32

import lgpio
import time
import random

PIN = 17
h = lgpio.gpiochip_open(0)   # Öffnet das Standard-GPIO-Device
lgpio.gpio_write(h, PIN, 0)  # Initial LOW

MITTELWERT = 0.12   # 120ms Durchschnitt
VARIANZ = 0.02      # ±20ms Varianz (typisch menschlich)

SPACE = "SPACE"

def PressKey(key):
    """Sendet einen Leertaste-Trigger via GPIO-Puls zu STM32."""
    if key == SPACE:
        druckdauer = random.gauss(MITTELWERT, VARIANZ)
        druckdauer = max(0.08, min(druckdauer, 0.16))  # 80-160ms Begrenzung (120ms ± 40ms)
        lgpio.gpio_write(h, PIN, 1)     # HIGH (Taste gedrückt)
        time.sleep(druckdauer)          # Menschliche Varianz
        lgpio.gpio_write(h, PIN, 0)     # LOW (Taste losgelassen)
    else:
        raise ValueError(f"Unbekannte Taste: {key}")

def ReleaseKey(key):
    """GPIO benötigt kein explizites Loslassen, bleibt für API-Kompatibilität leer."""
    pass

if __name__ == "__main__":
    print("GPIO-Test: SPACE-Taste")
    PressKey(SPACE)
    print("Done!")