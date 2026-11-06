"""Bounded, read-only retries for a ghost task's coordinate line."""
import re
import time

import cv2


def retry_coordinate_line(crop, result, detail, engine, render, valid, confidence, stopped, log):
    if valid(detail) or "鬼" not in detail:
        return detail
    candidates = [item for item in (result or []) if len(item) >= 3
                  and float(item[2]) >= confidence
                  and re.search(r"去.*\d.*抓", str(item[1]))]
    if len(candidates) != 1:
        return detail
    item = candidates[0]
    h, w = crop.shape[:2]
    points = item[0]
    left = max(0, int(min(p[0] for p in points)) - 8)
    right = min(w, int(max(p[0] for p in points)) + 9)
    top = max(0, int(min(p[1] for p in points)) - 4)
    bottom = min(h, int(max(p[1] for p in points)) + 5)
    if right <= left or bottom <= top:
        return detail
    line = crop[top:bottom, left:right]
    for attempt, scale in enumerate((2, 3), 1):
        if stopped():
            break
        started = time.perf_counter()
        enlarged = cv2.resize(line, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        retry, _ = engine(enlarged)
        text = render(retry, confidence)
        # Replace, never append: old unreadable digits must not remain in the evidence.
        candidate = detail.replace(str(item[1]).strip(), text, 1)
        success = bool(re.search(r"去.*\d{1,3}[,，.．、]\d{1,3}.*抓", text)) and valid(candidate)
        log(f"coordinate OCR retry {attempt}/2: box=({left},{top},{right},{bottom}) "
            f"scale={scale} elapsed={(time.perf_counter()-started)*1000:.0f}ms "
            f"success={success} text={text!r}")
        if success:
            return candidate
    return detail
