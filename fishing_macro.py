"""Roblox fishing macro with screen-based meter tracking.

Hotkeys:
  - or F6: start / pause
  = or F7: quit

Requirements:
  pip install opencv-python mss numpy pyautogui pydirectinput pynput
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import mss
import numpy as np
import pyautogui
import pydirectinput
from pynput import keyboard


@dataclass(frozen=True)
class Settings:
    # Fractions of the primary monitor: (left, top, right, bottom).
    roi: tuple[float, float, float, float] = (0.73, 0.28, 0.80, 0.75)

    scan_interval: float = 0.02
    min_zone_area: float = 12
    min_indicator_area: float = 5
    missing_frames_to_end: int = 12
    meter_timeout: float = 40.0

    input_latency: float = 0.06
    velocity_smoothing: float = 0.20
    target_offset: float = 0.0
    target_hysteresis: float = 1.0
    max_prediction: float = 0.10
    min_velocity_samples: int = 3
    max_abs_velocity: float = 2500.0

    collect_delay: float = 1.0
    collect_hold_time: float = 3.0
    recast_delay: float = 0.6

    white_v_min: int = 145
    white_s_max: int = 100


CFG = Settings()

pyautogui.FAILSAFE = True
pydirectinput.PAUSE = 0

running = threading.Event()
quit_program = threading.Event()
input_lock = threading.Lock()
mouse_down = False
key_down = False


def release_controls() -> None:
    """Release any inputs this script may have held."""
    global mouse_down, key_down

    with input_lock:
        try:
            pyautogui.mouseUp()
        except Exception:
            pass
        mouse_down = False

        try:
            pydirectinput.keyUp("e")
        except Exception:
            pass
        key_down = False


def set_mouse(should_hold: bool, position: tuple[int, int]) -> None:
    """Change mouse state only when necessary."""
    global mouse_down

    with input_lock:
        if should_hold and not mouse_down:
            pyautogui.moveTo(*position)
            pyautogui.mouseDown()
            mouse_down = True
        elif not should_hold and mouse_down:
            pyautogui.mouseUp()
            mouse_down = False


def toggle_macro() -> None:
    if quit_program.is_set():
        return
    if running.is_set():
        running.clear()
        release_controls()
        print("PAUSED — controls released")
    else:
        running.set()
        print("STARTING")


def quit_macro() -> None:
    running.clear()
    quit_program.set()
    release_controls()
    print("EXITING")


def wait_while_running(seconds: float) -> bool:
    deadline = time.monotonic() + max(0.0, seconds)
    while time.monotonic() < deadline:
        if not running.is_set() or quit_program.is_set():
            return False
        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
    return running.is_set() and not quit_program.is_set()


def capture_roi(sct: mss.mss) -> np.ndarray:
    """Capture only the configured portion of the primary monitor."""
    monitor = sct.monitors[1]
    left, top, right, bottom = CFG.roi
    x1 = monitor["left"] + int(monitor["width"] * left)
    y1 = monitor["top"] + int(monitor["height"] * top)
    x2 = monitor["left"] + int(monitor["width"] * right)
    y2 = monitor["top"] + int(monitor["height"] * bottom)

    shot = sct.grab({
        "left": x1,
        "top": y1,
        "width": max(1, x2 - x1),
        "height": max(1, y2 - y1),
    })
    return cv2.cvtColor(np.asarray(shot), cv2.COLOR_BGRA2BGR)


def largest_zone(hsv: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the most plausible colored target zone as x, y, width, height."""
    yellow = cv2.inRange(
        hsv, np.array([12, 45, 60]), np.array([42, 255, 255])
    )
    green = cv2.inRange(
        hsv, np.array([35, 40, 50]), np.array([100, 255, 255])
    )
    zone_mask = cv2.bitwise_or(yellow, green)
    kernel = np.ones((3, 3), np.uint8)
    zone_mask = cv2.morphologyEx(zone_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if area >= CFG.min_zone_area and h >= 4:
            # Prefer substantial vertical target shapes without overfavoring height.
            score = area * min(h, 20)
            candidates.append((score, x, y, w, h))

    if not candidates:
        return None

    _, x, y, w, h = max(candidates, key=lambda item: item[0])
    return x, y, w, h


def get_meter_data(sct: mss.mss) -> tuple[float, float, float] | None:
    """Return indicator center-y and target top/bottom in ROI coordinates."""
    bgr = capture_roi(sct)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    zone = largest_zone(hsv)
    if zone is None:
        return None

    zone_x, zone_top, zone_w, zone_h = zone
    zone_center_x = zone_x + zone_w / 2

    white = cv2.inRange(
        hsv,
        np.array([0, 0, CFG.white_v_min]),
        np.array([180, CFG.white_s_max, 255]),
    )
    white = cv2.morphologyEx(
        white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )

    contours, _ = cv2.findContours(
        white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    indicators = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if area < CFG.min_indicator_area or not (2 <= h <= 35) or w > 35:
            continue
        center_x = x + w / 2
        center_y = y + h / 2
        # The horizontal distance is a useful discriminator when UI clutter is present.
        indicators.append((abs(center_x - zone_center_x), -area, center_y))

    if not indicators:
        return None

    indicator_y = min(indicators, key=lambda item: (item[0], item[1]))[2]
    return float(indicator_y), float(zone_top), float(zone_top + zone_h)


def collect_item() -> None:
    global key_down

    if not wait_while_running(CFG.collect_delay):
        return

    print("Attempting collection with E...")
    release_controls()
    if not wait_while_running(0.15):
        return

    with input_lock:
        if not running.is_set() or quit_program.is_set():
            return
        pydirectinput.keyDown("e")
        key_down = True

    try:
        wait_while_running(CFG.collect_hold_time)
    finally:
        with input_lock:
            try:
                pydirectinput.keyUp("e")
            finally:
                key_down = False

    print("Collection key released.")


def control_meter(
    sct: mss.mss, cast_position: tuple[int, int]
) -> bool:
    """Track the indicator; return True when the meter disappears."""
    velocity = 0.0
    last_time = None
    last_y = None
    missing_frames = 0
    detected_frames = 0
    holding = False

    print("Tracking indicator...")
    while running.is_set() and not quit_program.is_set():
        now = time.monotonic()
        data = get_meter_data(sct)

        if data is None:
            missing_frames += 1
            if missing_frames >= CFG.missing_frames_to_end:
                set_mouse(False, cast_position)
                print("Meter disappeared.")
                return True
            time.sleep(CFG.scan_interval)
            continue

        missing_frames = 0
        indicator_y, zone_top, zone_bottom = data
        detected_frames += 1

        if last_time is not None and last_y is not None:
            dt = now - last_time
            if dt > 0:
                raw_velocity = (indicator_y - last_y) / dt
                if abs(raw_velocity) <= CFG.max_abs_velocity:
                    alpha = CFG.velocity_smoothing
                    velocity = alpha * raw_velocity + (1.0 - alpha) * velocity

        last_time = now
        last_y = indicator_y

        if detected_frames < CFG.min_velocity_samples:
            time.sleep(CFG.scan_interval)
            continue

        zone_height = max(1.0, zone_bottom - zone_top)
        target_y = (zone_top + zone_bottom) / 2 + zone_height * CFG.target_offset
        prediction_time = min(CFG.input_latency, CFG.max_prediction)
        predicted_y = indicator_y + velocity * prediction_time
        margin = max(2.0, zone_height * 0.03)
        hysteresis = max(0.0, CFG.target_hysteresis)

        # Image y increases downward. Hold while below the target, release above it.
        if holding:
            if predicted_y <= target_y + margin - hysteresis:
                holding = False
        elif predicted_y >= target_y + margin + hysteresis:
            holding = True

        set_mouse(holding, cast_position)
        time.sleep(CFG.scan_interval)

    release_controls()
    return False


def main() -> None:
    listener = keyboard.GlobalHotKeys({
        "-": toggle_macro,
        "=": quit_macro,
        "<f6>": toggle_macro,
        "<f7>": quit_macro,
    })
    listener.start()

    print("Roblox Fishing Macro ready")
    print("- / F6 = start or pause")
    print("= / F7 = exit")
    print("Move your mouse over the casting spot before starting.")

    try:
        with mss.mss() as sct:
            while not quit_program.is_set():
                if not running.is_set():
                    release_controls()
                    time.sleep(0.05)
                    continue

                pos = pyautogui.position()
                cast_position = (int(pos.x), int(pos.y))
                print("Casting...")
                pyautogui.click(*cast_position)

                start_time = time.monotonic()
                consecutive_hits = 0
                found_meter = False

                while running.is_set() and not quit_program.is_set():
                    if get_meter_data(sct) is not None:
                        consecutive_hits += 1
                    else:
                        consecutive_hits = 0

                    if consecutive_hits >= 3:
                        found_meter = True
                        break
                    if time.monotonic() - start_time >= CFG.meter_timeout:
                        break
                    time.sleep(CFG.scan_interval)

                if not running.is_set() or quit_program.is_set():
                    release_controls()
                    continue

                if not found_meter:
                    print("Meter not detected; retrying.")
                    wait_while_running(CFG.recast_delay)
                    continue

                print("Meter detected — precise control enabled")
                finished = control_meter(sct, cast_position)
                release_controls()

                if not running.is_set() or quit_program.is_set():
                    continue
                if finished:
                    collect_item()
                wait_while_running(CFG.recast_delay)

    except pyautogui.FailSafeException:
        print("Failsafe triggered.")
    except KeyboardInterrupt:
        print("Stopped.")
    except Exception as exc:
        print(f"Unexpected error: {exc!r}")
        raise
    finally:
        running.clear()
        quit_program.set()
        release_controls()
        try:
            listener.stop()
        except Exception:
            pass
        print("Macro closed.")


if __name__ == "__main__":
    main()
