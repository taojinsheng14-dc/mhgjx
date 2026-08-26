from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .device import AdbDevice
from .frame_source import AdbFrameSource, FrameSource
from .humanizer import RelativeRegion


@dataclass(frozen=True)
class TemplateMatch:
    confidence: float
    region: RelativeRegion
    pixel_box: tuple[int, int, int, int]


class TemplateMatcher:
    def __init__(self, device: AdbDevice, frame_source: FrameSource | None = None):
        self.device = device
        self.frame_source = frame_source or AdbFrameSource(device)

    def screenshot(self) -> np.ndarray:
        return self.frame_source.latest().image

    def find_template(
        self,
        template_path: Path,
        threshold: float = 0.88,
        screenshot: np.ndarray | None = None,
        mode: str = "color",
        search_region: RelativeRegion | None = None,
    ) -> TemplateMatch | None:
        haystack = screenshot if screenshot is not None else self.screenshot()
        try:
            template_bytes = template_path.read_bytes()
        except OSError as exc:
            raise FileNotFoundError(f"Template not found or unreadable: {template_path}") from exc
        try:
            template = cv2.imdecode(np.frombuffer(template_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        except cv2.error as exc:
            raise FileNotFoundError(f"Template not found or unreadable: {template_path}") from exc
        if template is None:
            raise FileNotFoundError(f"Template not found or unreadable: {template_path}")

        screen_h, screen_w = haystack.shape[:2]
        offset_x = 0
        offset_y = 0
        if search_region is not None:
            region = search_region.clamp()
            left = max(0, min(screen_w - 1, int(region.left * screen_w)))
            top = max(0, min(screen_h - 1, int(region.top * screen_h)))
            right = max(left + 1, min(screen_w, int(region.right * screen_w)))
            bottom = max(top + 1, min(screen_h, int(region.bottom * screen_h)))
            haystack = haystack[top:bottom, left:right]
            offset_x = left
            offset_y = top

        match_haystack = haystack
        match_template = template
        if mode == "edges":
            match_haystack = cv2.Canny(cv2.cvtColor(haystack, cv2.COLOR_BGR2GRAY), 80, 160)
            match_template = cv2.Canny(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), 80, 160)
        elif mode != "color":
            raise ValueError(f"Unsupported template matching mode: {mode}")

        result = cv2.matchTemplate(match_haystack, match_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < threshold:
            return None

        x, y = max_loc
        x += offset_x
        y += offset_y
        h, w = template.shape[:2]
        region = RelativeRegion.from_ltrb(
            x / screen_w,
            y / screen_h,
            (x + w) / screen_w,
            (y + h) / screen_h,
        )
        return TemplateMatch(confidence=float(max_val), region=region, pixel_box=(x, y, x + w, y + h))
