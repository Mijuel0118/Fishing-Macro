import time
import threading
from collections import deque

import cv2
import mss
import numpy as np
import pyautogui
import pydirectinput
from pynput import keyboard


# =========================
# SETTINGS
# =========================

# Screen region to scan, as proportions of screen width/height.
# Adjust these if your fishing meter is in a different location.
ROI = (0.73, 0.28, 0.80, 0.75)

SCAN_INTERVAL = 0.02

MIN_ZONE_AREA = 12
MIN_INDICATOR_AREA = 5

MISSING_FRAMES_TO_END = 12
METER_TIMEOUT = 40

# Control tuning
INPUT_LATENCY = 0.10
VELOCITY_SMOOTHING = 0.25
TARGET_OFFSET = 0.0
TARGET_HYSTERESIS = 1
MAX_PREDICTION = 0.12
MIN_VELOCITY_SAMPLES = 3

# Collection and recasting
COLLECT_DELAY = 1.0
COLLECT_HOLD_TIME = 3.0
RECAST_DELAY = 0.6

# Broad color detection for different lighting conditions.
# These HSV ranges may need adjusting for your game's meter.
ZONE_LOWER = np.array([15, 35, 100], dtype=np.uint8)
ZONE_UPPER = np.array([45, 255, 255], dtype=np.uint8)

INDICATOR_LOWER = np.array([0, 0, 180], dtype=np.uint8)
INDICATOR_UPPER = np.array([180, 100, 255], dtype=np.uint8)


# =========================
# STATE
# =========================

running = False
exiting = False

state_lock = threading.Lock()
worker_thread = None


# =========================
# INPUT
# =========================

def set_mouse(holding, cast_position=False):
    """Hold or release the left mouse button."""
    try:
        if holding:
            if cast_position:
                pyautogui.mouseDown()
            else:
                pyautogui.mouseDown()
        else:
            pyautogui.mouseUp()
    except Exception as exc:
        print(f"Mouse input error: {exc}")


def release_all():
    """Release mouse and keyboard inputs safely."""
    try:
        pyautogui.mouseUp()
    except Exception:
        pass

    try:
        pydirectinput.keyUp("e")
    except Exception:
        pass


# =========================
# SCREEN DETECTION
# =========================

def get_screen_region(sct):
    monitor = sct.monitors[1]
    screen_width = monitor["width"]
    screen_height = monitor["height"]

    left = int(screen_width * ROI[0])
    top = int(screen_height * ROI[1])
    right = int(screen_width * ROI[2])
    bottom = int(screen_height * ROI[3])

    return {
        "left": left,
        "top": top,
        "width": max(1, right - left),
        "height": max(1, bottom - top),
    }


def largest_component(mask, min_area):
    """Return the bounding box of the largest sufficiently large component."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    best = None
    best_area = min_area

    for i in range(1, count):
        x, y, width, height, area = stats[i]

        if area >= best_area:
            best_area = area
            best = (x, y, width, height, area)

    return best


def get_meter_data(frame):
    """
    Detect the target zone and white indicator.

    Returns:
        (zone_center_y, zone_height, indicator_center_y)
    or None if the meter cannot be detected.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Target zone
    zone_mask = cv2.inRange(hsv, ZONE_LOWER, ZONE_UPPER)

    # Indicator: bright, low-saturation pixels
    indicator_mask = cv2.inRange(
        hsv, INDICATOR_LOWER, INDICATOR_UPPER
    )

    # Remove tiny isolated noise.
    kernel = np.ones((3, 3), np.uint8)
    zone_mask = cv2.morphologyEx(
        zone_mask, cv2.MORPH_OPEN, kernel
    )
    indicator_mask = cv2.morphologyEx(
        indicator_mask, cv2.MORPH_OPEN, kernel
    )

    zone = largest_component(zone_mask, MIN_ZONE_AREA)
    indicator = largest_component(
        indicator_mask, MIN_INDICATOR_AREA
    )

    if zone is None or indicator is None:
        return None

    zx, zy, zw, zh, _ = zone
    ix, iy, iw, ih, _ = indicator

    zone_center_y = zy + zh / 2
    indicator_center_y = iy + ih / 2

    return zone_center_y, zh, indicator_center_y


# =========================
# FISHING CONTROL
# =========================

def control_meter(sct, region):
    """
    Control the mouse based on the indicator's position and velocity.

    The image coordinate system has y increasing downward:
    - Indicator moving down: positive velocity.
    - Indicator moving up: negative velocity.
    """
    global running

    start_time = time.time()
    missing_frames = 0
    holding = False

    previous_y = None
    previous_time = None
    smoothed_velocity = 0.0
    velocity_samples = deque(maxlen=5)

    while running and not exiting:
        if time.time() - start_time > METER_TIMEOUT:
            print("Meter timeout; moving to collection.")
            break

        frame_start = time.time()

        try:
            screenshot = np.array(sct.grab(region))
            frame = cv2.cvtColor(
                screenshot, cv2.COLOR_BGRA2BGR
            )
        except Exception as exc:
            print(f"Screen capture error: {exc}")
            time.sleep(SCAN_INTERVAL)
            continue

        data = get_meter_data(frame)

        if data is None:
            missing_frames += 1

            if missing_frames >= MISSING_FRAMES_TO_END:
                break

            time.sleep(SCAN_INTERVAL)
            continue

        missing_frames = 0

        zone_center_y, zone_height, indicator_y = data
        now = time.time()

        # Estimate indicator velocity.
        if previous_y is not None and previous_time is not None:
            dt = max(now - previous_time, 0.001)
            velocity = (indicator_y - previous_y) / dt

            velocity_samples.append(velocity)

            if len(velocity_samples) >= MIN_VELOCITY_SAMPLES:
                average_velocity = sum(velocity_samples) / len(
                    velocity_samples
                )

                smoothed_velocity = (
                    VELOCITY_SMOOTHING * average_velocity
                    + (1 - VELOCITY_SMOOTHING) * smoothed_velocity
                )

        previous_y = indicator_y
        previous_time = now

        # Predict where the indicator will be after input delay.
        predicted_y = indicator_y + (
            smoothed_velocity * INPUT_LATENCY
        )

        max_prediction_distance = max(
            1.0, zone_height * MAX_PREDICTION
        )

        predicted_y = max(
            indicator_y - max_prediction_distance,
            min(
                indicator_y + max_prediction_distance,
                predicted_y,
            ),
        )

        # Aim slightly above the center of the target zone.
        target_y = (
            zone_center_y
            + zone_height * TARGET_OFFSET
        )

        # Start holding before the indicator falls too low.
        margin = max(2.0, zone_height * 0.03)
        early_hold_distance = zone_height * 0.10

        # Since screen y increases downward, a larger y is lower.
        start_holding_y = target_y - early_hold_distance
        release_holding_y = start_holding_y - margin

        if holding:
            # Release after the indicator has moved sufficiently upward.
            if predicted_y < release_holding_y:
                holding = False
        else:
            # Begin holding early as the indicator falls toward the zone.
            if predicted_y > start_holding_y:
                holding = True

        set_mouse(holding)

        # Keep the loop responsive without intentionally adding extra delay.
        elapsed = time.time() - frame_start
        time.sleep(max(0, SCAN_INTERVAL - elapsed))

    set_mouse(False)


# =========================
# CAST AND COLLECT
# =========================

def cast_line():
    """Cast the fishing line with a left click."""
    print("Casting...")

    pyautogui.mouseUp()
    time.sleep(0.1)

    pyautogui.mouseDown()
    time.sleep(0.12)
    pyautogui.mouseUp()

    time.sleep(0.4)


def wait_for_meter(sct, region):
    """Wait until the fishing meter appears."""
    print("Waiting for fishing meter...")

    start_time = time.time()
    consecutive_detections = 0

    while running and not exiting:
        if time.time() - start_time > METER_TIMEOUT:
            print("No meter detected before timeout.")
            return False

        try:
            screenshot = np.array(sct.grab(region))
            frame = cv2.cvtColor(
                screenshot, cv2.COLOR_BGRA2BGR
            )
            data = get_meter_data(frame)
        except Exception as exc:
            print(f"Detection error: {exc}")
            data = None

        if data is not None:
            consecutive_detections += 1

            if consecutive_detections >= 2:
                print("Fishing meter detected.")
                return True
        else:
            consecutive_detections = 0

        time.sleep(SCAN_INTERVAL)

    return False


def collect_item():
    """Wait for the catch and hold E to collect it."""
    global running

    if not running or exiting:
        return

    print("Waiting before collection...")
    time.sleep(COLLECT_DELAY)

    if not running or exiting:
        return

    print("Holding E to collect...")
    try:
        pydirectinput.keyDown("e")
        time.sleep(COLLECT_HOLD_TIME)
    finally:
        pydirectinput.keyUp("e")

    if running and not exiting:
        print("Collection complete.")
        time.sleep(RECAST_DELAY)


# =========================
# MAIN LOOP
# =========================

def fishing_loop():
    global running

    print("Fishing macro started.")

    try:
        with mss.mss() as sct:
            region = get_screen_region(sct)

            while running and not exiting:
                cast_line()

                if not running or exiting:
                    break

                detected = wait_for_meter(sct, region)

                if not running or exiting:
                    break

                if detected:
                    print("Controlling fishing meter...")
                    control_meter(sct, region)
                else:
                    print("Meter not found; attempting collection.")

                if not running or exiting:
                    break

                collect_item()

    except Exception as exc:
        print(f"Macro error: {exc}")

    finally:
        release_all()
        print("Fishing macro stopped.")


# =========================
# START / PAUSE / EXIT
# =========================

def toggle_macro():
    global running, worker_thread

    if exiting:
        return

    if running:
        running = False
        release_all()
        print("Macro paused.")
    else:
        running = True

        if worker_thread is None or not worker_thread.is_alive():
            worker_thread = threading.Thread(
                target=fishing_loop,
                daemon=True,
            )
            worker_thread.start()
        else:
            print("Macro resumed.")


def exit_macro():
    global running, exiting

    running = False
    exiting = True
    release_all()

    print("Exiting macro...")


def on_press(key):
    try:
        if key == keyboard.Key.f6:
            toggle_macro()
            return

        if key == keyboard.Key.f7:
            exit_macro()
            return

        if hasattr(key, "char") and key.char is not None:
            if key.char == "-":
                toggle_macro()
            elif key.char == "=":
                exit_macro()

    except Exception as exc:
        print(f"Keyboard listener error: {exc}")


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    pyautogui.FAILSAFE = False
    pydirectinput.PAUSE = 0

    print("--------------------------------")
    print("Roblox Fishing Macro")
    print("--------------------------------")
    print("- or F6 = Start / Pause")
    print("= or F7 = Exit")
    print("Keep Roblox visible while running.")
    print("--------------------------------")

    try:
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
    except KeyboardInterrupt:
        exit_macro()
    finally:
        release_all()
