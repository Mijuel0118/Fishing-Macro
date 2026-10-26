import time
import threading

import cv2
import mss
import numpy as np
import pyautogui
import pydirectinput
from pynput import keyboard


# ------------ SETTINGS ------------

SCAN_INTERVAL = 0.01
BAR_TIMEOUT = 30
BAR_END_DELAY = 0.5
RECAST_DELAY = 0.5

COLLECT_DELAY = 1.0
COLLECT_HOLD_TIME = 3.0

# Approximate fishing meter region (screen fractions)
ROI = (0.73, 0.28, 0.80, 0.75)

MIN_ZONE_AREA = 12
MIN_INDICATOR_AREA = 5

# ---------------------------------

pyautogui.FAILSAFE = True
pydirectinput.PAUSE = 0


running = threading.Event()
quit_program = threading.Event()


def release_controls():
    pyautogui.mouseUp()
    pydirectinput.keyUp("e")


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
    "<f6>": toggle_macro,
    "<f7>": quit_macro,
})
listener.start()


def get_meter_data(sct):
    """Detect the colored target zone and white indicator."""

    screen = sct.monitors[1]

    left = screen["left"]
    top = screen["top"]
    width = screen["width"]
    height = screen["height"]

    x1 = left + int(width * ROI[0])
    y1 = top + int(height * ROI[1])
    x2 = left + int(width * ROI[2])
    y2 = top + int(height * ROI[3])

    frame = np.array(sct.grab({
        "left": x1,
        "top": y1,
        "width": x2 - x1,
        "height": y2 - y1,
    }))

    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # White indicator
    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 180]),
        np.array([180, 75, 255]),
    )

    # Yellow target zone
    yellow_mask = cv2.inRange(
        hsv,
        np.array([15, 65, 80]),
        np.array([38, 255, 255]),
    )

    # Green target zone
    green_mask = cv2.inRange(
        hsv,
        np.array([39, 65, 70]),
        np.array([100, 255, 255]),
    )

    kernel = np.ones((3, 3), np.uint8)

    white_mask = cv2.morphologyEx(
        white_mask, cv2.MORPH_OPEN, kernel
    )
    yellow_mask = cv2.morphologyEx(
        yellow_mask, cv2.MORPH_OPEN, kernel
    )
    green_mask = cv2.morphologyEx(
        green_mask, cv2.MORPH_OPEN, kernel
    )

    zone_mask = cv2.bitwise_or(yellow_mask, green_mask)

    contours, _ = cv2.findContours(
        zone_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    zones = []

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        if area >= MIN_ZONE_AREA and h >= 4:
            zones.append((area, y, y + h))

    if not zones:
        return None

    zones.sort(key=lambda item: item[0], reverse=True)
    _, zone_top, zone_bottom = zones[0]

    contours, _ = cv2.findContours(
        white_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    indicators = []

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        if (
            area >= MIN_INDICATOR_AREA
            and 2 <= h <= 35
            and w <= 35
        ):
            indicators.append((area, y + h // 2))

    if not indicators:
        return None

    indicators.sort(key=lambda item: item[0], reverse=True)
    _, indicator_y = indicators[0]

    return indicator_y, zone_top, zone_bottom


print("Roblox Fishing Macro ready")
print("F6 = start / pause")
print("F7 = exit")
print("Position mouse over the casting spot before starting.")

mouse_down = False

try:
    with mss.mss() as sct:
        while not quit_program.is_set():

            if not running.is_set():
                if mouse_down:
                    pyautogui.mouseUp()
                    mouse_down = False

                time.sleep(0.05)
                continue

            cast_position = pyautogui.position()

            # Cast once
            pyautogui.click(
                cast_position.x,
                cast_position.y,
            )
            print("Cast! Waiting for meter...")

            start_time = time.time()
            last_seen = start_time
            found_meter = False
            consecutive_hits = 0

            # Wait for the meter to appear
            while running.is_set() and not quit_program.is_set():

                data = get_meter_data(sct)

                if data is not None:
                    consecutive_hits += 1
                else:
                    consecutive_hits = 0

                if consecutive_hits >= 3:
                    found_meter = True
                    last_seen = time.time()
                    break

                if time.time() - start_time >= BAR_TIMEOUT:
                    break

                time.sleep(SCAN_INTERVAL)

            if not running.is_set() or quit_program.is_set():
                continue

            if not found_meter:
                print("Meter not detected; retrying.")
                time.sleep(RECAST_DELAY)
                continue

            print("Meter detected - controlling indicator")

            # Control indicator until the meter disappears
            while running.is_set() and not quit_program.is_set():

                data = get_meter_data(sct)

                if data is None:
                    if time.time() - last_seen >= BAR_END_DELAY:
                        print("Meter disappeared.")
                        break

                    time.sleep(SCAN_INTERVAL)
                    continue

                last_seen = time.time()

                indicator_y, zone_top, zone_bottom = data

                zone_center = (zone_top + zone_bottom) / 2
                zone_height = zone_bottom - zone_top

                # Aim slightly below center
                target_y = zone_center + zone_height * 0.10

                if indicator_y > target_y:
                    # Indicator too low: hold mouse to move up
                    if not mouse_down:
                        pyautogui.mouseDown()
                        mouse_down = True
                else:
                    # Indicator reached target: release mouse
                    if mouse_down:
                        pyautogui.mouseUp()
                        mouse_down = False

                time.sleep(SCAN_INTERVAL)

            # Always release mouse after the minigame
            if mouse_down:
                pyautogui.mouseUp()
                mouse_down = False

            # Collect the item
            if running.is_set() and not quit_program.is_set():

                print("Waiting for collection prompt...")
                time.sleep(COLLECT_DELAY)

                if running.is_set() and not quit_program.is_set():
                    print("Holding E to collect item...")

                    pyautogui.mouseUp()
                    pydirectinput.keyUp("e")
                    time.sleep(0.2)

                    pydirectinput.keyDown("e")
                    try:
                        time.sleep(COLLECT_HOLD_TIME)
                    finally:
                        pydirectinput.keyUp("e")

                    print("Collection key released.")

                if running.is_set() and not quit_program.is_set():
                    time.sleep(RECAST_DELAY)

except pyautogui.FailSafeException:
    print("Failsafe triggered.")

except KeyboardInterrupt:
    print("Stopped.")

finally:
    release_controls()
    running.clear()
    quit_program.set()
    listener.stop()
    print("Macro closed.")
