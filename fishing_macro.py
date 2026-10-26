import time
import threading
from collections import deque

import cv2
import mss
import numpy as np
import pyautogui
import pydirectinput
from pynput import keyboard


# ============== SETTINGS ==============

# Screen area containing the fishing meter
ROI = (0.73, 0.28, 0.80, 0.75)

# Detection
SCAN_INTERVAL = 0.025
MIN_ZONE_AREA = 12
MIN_INDICATOR_AREA = 5
MISSING_FRAMES_TO_END = 12
METER_TIMEOUT = 40

# Movement and lag compensation
INPUT_LATENCY = 0.15
VELOCITY_SMOOTHING = 0.35
TARGET_OFFSET = 0.05
TARGET_HYSTERESIS = 3
MAX_PREDICTION = 0.30
MIN_VELOCITY_SAMPLES = 3

# Fishing cycle
COLLECT_DELAY = 1.0
COLLECT_HOLD_TIME = 3.0
RECAST_DELAY = 0.6

# ======================================

pyautogui.FAILSAFE = True
pydirectinput.PAUSE = 0

running = threading.Event()
quit_program = threading.Event()

mouse_down = False


def release_controls():
    global mouse_down

    try:
        pyautogui.mouseUp()
    except Exception:
        pass

    try:
        pydirectinput.keyUp("e")
    except Exception:
        pass

    mouse_down = False


def toggle_macro():
    if running.is_set():
        running.clear()
        release_controls()
        print("PAUSED - controls released")
    else:
        running.set()
        print("STARTING")


def quit_macro():
    running.clear()
    quit_program.set()
    release_controls()
    print("EXITING")


listener = keyboard.GlobalHotKeys({
    "-": toggle_macro,
    "=": quit_macro,
    "<f6>": toggle_macro,
    "<f7>": quit_macro,
})
listener.start()


def get_meter_data(sct):
    """
    Return indicator y, target top, target bottom.
    Coordinates are relative to the captured ROI.
    """

    screen = sct.monitors[1]

    left = screen["left"]
    top = screen["top"]
    width = screen["width"]
    height = screen["height"]

    x1 = left + int(width * ROI[0])
    y1 = top + int(height * ROI[1])
    x2 = left + int(width * ROI[2])
    y2 = top + int(height * ROI[3])

    frame = np.asarray(sct.grab({
        "left": x1,
        "top": y1,
        "width": x2 - x1,
        "height": y2 - y1,
    }))

    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Broad color ranges for day/night lighting.
    white = cv2.inRange(
        hsv,
        np.array([0, 0, 145]),
        np.array([180, 100, 255]),
    )

    yellow = cv2.inRange(
        hsv,
        np.array([12, 45, 60]),
        np.array([42, 255, 255]),
    )

    green = cv2.inRange(
        hsv,
        np.array([35, 40, 50]),
        np.array([100, 255, 255]),
    )

    kernel = np.ones((3, 3), np.uint8)

    white = cv2.morphologyEx(
        white, cv2.MORPH_OPEN, kernel
    )
    yellow = cv2.morphologyEx(
        yellow, cv2.MORPH_OPEN, kernel
    )
    green = cv2.morphologyEx(
        green, cv2.MORPH_OPEN, kernel
    )

    zone_mask = cv2.bitwise_or(yellow, green)

    zone_contours, _ = cv2.findContours(
        zone_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    zones = []

    for contour in zone_contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        if area >= MIN_ZONE_AREA and h >= 4:
            zones.append((area, x, y, w, h))

    if not zones:
        return None

    # Prefer a tall target component over tiny colored objects.
    zones.sort(
        key=lambda z: z[0] * min(z[4], 20),
        reverse=True,
    )

    _, zone_x, zone_top, zone_w, zone_h = zones[0]

    indicator_contours, _ = cv2.findContours(
        white,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    indicators = []

    for contour in indicator_contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        if (
            area >= MIN_INDICATOR_AREA
            and 2 <= h <= 35
            and w <= 35
        ):
            # Prefer white components near the target's horizontal
            # position, where possible.
            zone_center_x = zone_x + zone_w / 2
            indicator_center_x = x + w / 2
            horizontal_distance = abs(
                indicator_center_x - zone_center_x
            )

            indicators.append((
                horizontal_distance,
                area,
                y + h / 2,
            ))

    if not indicators:
        return None

    indicators.sort(
        key=lambda item: (item[0], -item[1])
    )

    indicator_y = indicators[0][2]

    return (
        float(indicator_y),
        float(zone_top),
        float(zone_top + zone_h),
    )


def set_mouse(should_hold, cast_position):
    global mouse_down

    if should_hold and not mouse_down:
        pyautogui.moveTo(cast_position.x, cast_position.y)
        pyautogui.mouseDown()
        mouse_down = True

    elif not should_hold and mouse_down:
        pyautogui.mouseUp()
        mouse_down = False


def wait_while_running(seconds):
    """Wait in short intervals so pausing remains responsive."""
    end_time = time.monotonic() + seconds

    while time.monotonic() < end_time:
        if not running.is_set() or quit_program.is_set():
            return False
        time.sleep(0.02)

    return running.is_set() and not quit_program.is_set()


def collect_item():
    if not wait_while_running(COLLECT_DELAY):
        return

    print("Attempting collection with E...")

    pyautogui.mouseUp()
    time.sleep(0.15)

    if not running.is_set() or quit_program.is_set():
        return

    pydirectinput.keyDown("e")

    try:
        wait_while_running(COLLECT_HOLD_TIME)
    finally:
        pydirectinput.keyUp("e")

    print("Collection key released.")


def control_meter(sct, cast_position):
    """
    Track the indicator, smooth its measured velocity, and predict
    where it will be after the configured input latency.
    """

    global mouse_down

    samples = deque(maxlen=5)
    velocity = 0.0
    last_time = None
    last_y = None
    missing_frames = 0
    detected_frames = 0
    holding = False

    print("Waiting for a stable meter reading...")

    while running.is_set() and not quit_program.is_set():

        now = time.monotonic()
        data = get_meter_data(sct)

        if data is None:
            missing_frames += 1

            if missing_frames >= MISSING_FRAMES_TO_END:
                print("Meter disappeared.")
                return True

            time.sleep(SCAN_INTERVAL)
            continue

        missing_frames = 0

        indicator_y, zone_top, zone_bottom = data

        detected_frames += 1
        samples.append((now, indicator_y))

        if last_time is not None and last_y is not None:
            dt = now - last_time

            if dt > 0:
                raw_velocity = (indicator_y - last_y) / dt

                # Ignore implausible one-frame jumps.
                if abs(raw_velocity) < 2500:
                    velocity = (
                        VELOCITY_SMOOTHING * raw_velocity
                        + (1 - VELOCITY_SMOOTHING) * velocity
                    )

        last_time = now
        last_y = indicator_y

        if detected_frames < MIN_VELOCITY_SAMPLES:
            time.sleep(SCAN_INTERVAL)
            continue

        zone_height = max(1.0, zone_bottom - zone_top)
        zone_center = (zone_top + zone_bottom) / 2

        # Positive screen velocity means moving downward.
        target_y = zone_center + zone_height * TARGET_OFFSET

        # Predict the position when the next input is likely to
        # take effect. Cap prediction to avoid extreme extrapolation.
        prediction_time = min(
            INPUT_LATENCY,
            MAX_PREDICTION,
        )

        predicted_y = indicator_y + velocity * prediction_time

        # Hysteresis prevents rapid mouse down/up flickering.
        margin = max(
            TARGET_HYSTERESIS,
            zone_height * 0.08,
        )

        if holding:
            # Keep holding until the predicted position gets
            # safely above the target threshold.
            if predicted_y < target_y - margin:
                holding = False
        else:
            # Start holding when prediction says the indicator
            # will be too low.
            if predicted_y > target_y + margin:
                holding = True

        set_mouse(holding, cast_position)

        time.sleep(SCAN_INTERVAL)

    release_controls()
    return False


print("Roblox Fishing Macro ready")
print("- = start / pause")
print("= = exit")
print("F6 = start / pause")
print("F7 = exit")
print("Position the mouse over your casting spot before starting.")

try:
    with mss.mss() as sct:

        while not quit_program.is_set():

            if not running.is_set():
                release_controls()
                time.sleep(0.05)
                continue

            cast_position = pyautogui.position()

            print("Casting...")
            pyautogui.click(
                cast_position.x,
                cast_position.y,
            )

            start_time = time.monotonic()
            consecutive_hits = 0
            found_meter = False

            while running.is_set() and not quit_program.is_set():

                data = get_meter_data(sct)

                if data is not None:
                    consecutive_hits += 1
                else:
                    consecutive_hits = 0

                if consecutive_hits >= 3:
                    found_meter = True
                    break

                if time.monotonic() - start_time >= METER_TIMEOUT:
                    break

                time.sleep(SCAN_INTERVAL)

            if not running.is_set() or quit_program.is_set():
                release_controls()
                continue

            if not found_meter:
                print("Meter not detected; retrying.")
                wait_while_running(RECAST_DELAY)
                continue

            print("Meter detected - predictive control enabled")

            finished = control_meter(sct, cast_position)
            release_controls()

            if not running.is_set() or quit_program.is_set():
                continue

            if finished:
                collect_item()

            wait_while_running(RECAST_DELAY)

except pyautogui.FailSafeException:
    print("Failsafe triggered.")

except KeyboardInterrupt:
    print("Stopped.")

finally:
    release_controls()
    running.clear()
    quit_program.set()

    try:
        listener.stop()
    except Exception:
        pass

    print("Macro closed.")
