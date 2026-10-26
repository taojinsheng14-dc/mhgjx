"""Locate a task by its green heading, never by its list index."""
import re
import time

import numpy as np

from .humanizer import RelativeRegion


def task_body_box(image, result, title, min_confidence=0.70):
    height, width = image.shape[:2]
    lines = []
    for polygon, text, confidence in result or []:
        if float(confidence) < min_confidence:
            continue
        points = np.asarray(polygon)
        left, top = np.floor(points.min(axis=0)).astype(int)
        right, bottom = np.ceil(points.max(axis=0)).astype(int)
        left, top = max(0, left), max(0, top)
        right, bottom = min(width, right), min(height, bottom)
        if right <= left or bottom <= top:
            continue
        patch = image[top:bottom, left:right].astype(float)
        blue, green, red = patch[:, :, 0], patch[:, :, 1], patch[:, :, 2]
        is_heading = np.mean((green > 100) & (green > red * 1.35) & (green > blue * 1.35)) > 0.08
        lines.append((left, top, right, bottom, re.sub(r"\s+", "", str(text)), is_heading))
    matches = [line for line in lines if line[4] == title and line[5]]
    if len(matches) != 1:
        return None
    heading = matches[0]
    boundary = min((line[1] for line in lines if line[5] and line[1] >= heading[3]), default=height)
    bodies = [line for line in lines if not line[5] and line[1] >= heading[3]
              and line[3] < boundary and line[1] - heading[3] <= 100
              and abs(line[0] - heading[0]) <= 80]
    if not bodies:
        return None
    return min(bodies, key=lambda line: line[1])[:4]


def click_named_task(runner, step):
    title = str(step["task_title"])
    roi = RelativeRegion.from_ltrb(*step.get("task_search_region", [0.785, 0.24, 0.95, 0.70]))
    deadline = time.monotonic() + float(step.get("task_timeout", 6.0))
    while not runner.stop_event.is_set() and time.monotonic() < deadline:
        image = runner._capture_color_region(None, use_adb_snapshot=True)
        height, width = image.shape[:2]
        if list(step.get("screen_size", [width, height])) != [width, height]:
            raise RuntimeError("Task screenshot size differs from configured input size")
        crop = runner._crop_image_region(image, roi)
        with runner._ocr_lock:
            if runner._ocr_engine is None:
                from rapidocr_onnxruntime import RapidOCR
                runner._ocr_engine = RapidOCR()
            result, _ = runner._ocr_engine(crop)
        box = task_body_box(crop, result, title)
        scores = [(str(item[1]), round(float(item[2]), 3)) for item in result or []]
        runner.log(f"task target: title={title}, box={box}, OCR_scores={scores!r}")
        if box is None:
            runner._interruptible_wait(0.3)
            continue
        left, top, right, bottom = box
        x0, y0 = int(roi.left * width), int(roi.top * height)
        click = dict(step)
        click.pop("task_title", None)
        click.pop("verify_after", None)
        click.pop("verify_retries", None)
        click["use_click_region"] = True
        click["region"] = [(x0 + left) / width, (y0 + top) / height,
                           (x0 + right) / width, (y0 + bottom) / height]
        runner._region_click(click)
        return
    if not runner.stop_event.is_set():
        runner._handle_timeout(step, f"未确认绿色任务标题及任务内容: {title}；不点击其他任务")
