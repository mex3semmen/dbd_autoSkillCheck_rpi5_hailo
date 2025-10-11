# directkeys.py — Pi→STM32 (PULSE, Rising-Edge). STM32 hält die Taste.
# STM32: TRIGGER_PIN=PA0, INPUT_PULLDOWN, attachInterrupt(..., RISING)

import os, atexit
from time import perf_counter_ns
import lgpio

PIN  = int(os.getenv("STM32_GPIO", "17"))   # BCM-Pin zum STM32 (an PA0)
CHIP = int(os.getenv("GPIOCHIP", "0"))

# Pulsparameter: starte großzügig, dann runtertesten
PULSE_US      = int(os.getenv("PULSE_US", "2000"))   # 2 ms
PULSE_RETRIES = int(os.getenv("PULSE_RETRIES", "1")) # i. d. R. 1 reicht
PULSE_GAP_US  = int(os.getenv("PULSE_GAP_US", "3000"))

ACTIVE = 1  # HIGH = aktiv (INPUT_PULLDOWN + RISING)
IDLE   = 0  # LOW  = idle

# GPIO-Init
H = lgpio.gpiochip_open(CHIP)
lgpio.gpio_claim_output(H, PIN, IDLE)

def _cleanup():
    try: lgpio.gpio_write(H, PIN, IDLE)
    except: pass
    try: lgpio.gpio_free(H, PIN)
    except: pass
    try: lgpio.gpiochip_close(H)
    except: pass
atexit.register(_cleanup)

def _busy_wait_us(us: int):
    deadline = perf_counter_ns() + us * 1_000
    while perf_counter_ns() < deadline:
        pass

def _prime_low():
    # Stelle sicher, dass vor der Flanke sicher LOW anliegt (echte Rising-Edge)
    lgpio.gpio_write(H, PIN, IDLE)
    _busy_wait_us(200)  # 0,2 ms

SPACE = "SPACE"

def PressKey(key) -> bool:
    if key != SPACE:
        raise ValueError("Unbekannte Taste")
    _prime_low()
    for i in range(max(1, PULSE_RETRIES)):
        lgpio.gpio_write(H, PIN, ACTIVE)   # Rising → ISR am STM32
        _busy_wait_us(PULSE_US)
        lgpio.gpio_write(H, PIN, IDLE)
        if i + 1 < PULSE_RETRIES:
            _busy_wait_us(PULSE_GAP_US)
    return True

def ReleaseKey(key):
    try: lgpio.gpio_write(H, PIN, IDLE)
    except: pass

if __name__ == "__main__":
    print(f"[directkeys] PULSE_US={PULSE_US}  RETRIES={PULSE_RETRIES}  GAP_US={PULSE_GAP_US}")
    PressKey(SPACE)
    print("OK")
