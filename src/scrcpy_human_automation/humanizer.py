from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from .device import AdbDevice


@dataclass(frozen=True)
class RelativePoint:
    x: float
    y: float

    def clamp(self) -> "RelativePoint":
        return RelativePoint(x=max(0.0, min(1.0, self.x)), y=max(0.0, min(1.0, self.y)))


@dataclass(frozen=True)
class RelativeRegion:
    left: float
    top: float
    right: float
    bottom: float

    @classmethod
    def from_ltrb(cls, left: float, top: float, right: float, bottom: float) -> "RelativeRegion":
        return cls(left=left, top=top, right=right, bottom=bottom)

    @property
    def width(self) -> float:
        return max(0.0, self.right - self.left)

    @property
    def height(self) -> float:
        return max(0.0, self.bottom - self.top)

    @property
    def center(self) -> RelativePoint:
        return RelativePoint((self.left + self.right) / 2.0, (self.top + self.bottom) / 2.0)

    def clamp(self) -> "RelativeRegion":
        return RelativeRegion(
            left=max(0.0, min(1.0, self.left)),
            top=max(0.0, min(1.0, self.top)),
            right=max(0.0, min(1.0, self.right)),
            bottom=max(0.0, min(1.0, self.bottom)),
        )

    def expand(self, margin_ratio: float) -> "RelativeRegion":
        margin_x = self.width * margin_ratio
        margin_y = self.height * margin_ratio
        return RelativeRegion(
            self.left - margin_x,
            self.top - margin_y,
            self.right + margin_x,
            self.bottom + margin_y,
        ).clamp()


class HumanizedController:
    def __init__(self, device: AdbDevice, seed: int | None = None):
        self.device = device
        self.random = random.Random(seed)

    def sleep_random(self, delay_range: tuple[float, float]) -> float:
        delay = self.random.uniform(*delay_range)
        time.sleep(delay)
        return delay

    def to_absolute(self, point: RelativePoint) -> tuple[int, int]:
        size = self.device.screen_size_for_input()
        safe_point = point.clamp()
        return int(safe_point.x * size.width), int(safe_point.y * size.height)

    def random_point_in_region(
        self,
        region: RelativeRegion,
        padding_ratio: float = 0.12,
        center_bias: float = 0.6,
    ) -> RelativePoint:
        region = region.clamp()
        inner_left = region.left + region.width * padding_ratio
        inner_top = region.top + region.height * padding_ratio
        inner_right = region.right - region.width * padding_ratio
        inner_bottom = region.bottom - region.height * padding_ratio

        if inner_right <= inner_left or inner_bottom <= inner_top:
            return region.center.clamp()

        cx = (inner_left + inner_right) / 2.0
        cy = (inner_top + inner_bottom) / 2.0
        spread_x = (inner_right - inner_left) / 2.0
        spread_y = (inner_bottom - inner_top) / 2.0

        gaussian_x = self.random.gauss(0.0, 0.35) * spread_x * center_bias
        gaussian_y = self.random.gauss(0.0, 0.35) * spread_y * center_bias
        uniform_x = self.random.uniform(-spread_x, spread_x) * (1.0 - center_bias)
        uniform_y = self.random.uniform(-spread_y, spread_y) * (1.0 - center_bias)

        x = min(inner_right, max(inner_left, cx + gaussian_x + uniform_x))
        y = min(inner_bottom, max(inner_top, cy + gaussian_y + uniform_y))
        return RelativePoint(x=x, y=y)

    def random_tap(
        self,
        region: RelativeRegion,
        pre_delay: tuple[float, float] = (0.15, 0.45),
        post_delay: tuple[float, float] = (0.18, 0.55),
        padding_ratio: float = 0.12,
        tap_mode: str = "tap",
    ) -> tuple[int, int]:
        self.sleep_random(pre_delay)
        point = self.random_point_in_region(region, padding_ratio=padding_ratio)
        x, y = self.to_absolute(point)
        if tap_mode == "swipe":
            self.device.short_swipe_tap(x, y)
        else:
            self.device.tap(x, y)
        self.sleep_random(post_delay)
        return x, y

    def natural_swipe(
        self,
        start: RelativePoint,
        end: RelativePoint,
        duration_ms: tuple[int, int] = (650, 1100),
        steps_range: tuple[int, int] = (7, 12),
        jitter_ratio: float = 0.006,
        pre_delay: tuple[float, float] = (0.2, 0.55),
        post_delay: tuple[float, float] = (0.25, 0.7),
    ) -> list[tuple[int, int]]:
        self.sleep_random(pre_delay)

        total_ms = self.random.randint(*duration_ms)
        steps = self.random.randint(*steps_range)
        points = self._build_swipe_path(start, end, steps=steps, jitter_ratio=jitter_ratio)
        segment_weights = self._build_segment_weights(len(points) - 1)
        segment_durations = [max(25, int(total_ms * weight)) for weight in segment_weights]

        for index in range(len(points) - 1):
            x1, y1 = self.to_absolute(points[index])
            x2, y2 = self.to_absolute(points[index + 1])
            self.device.swipe(x1, y1, x2, y2, segment_durations[index])
            time.sleep(self.random.uniform(0.008, 0.03))

        self.sleep_random(post_delay)
        return [self.to_absolute(point) for point in points]

    def _build_swipe_path(
        self,
        start: RelativePoint,
        end: RelativePoint,
        steps: int,
        jitter_ratio: float,
    ) -> list[RelativePoint]:
        path: list[RelativePoint] = []
        control_x = (start.x + end.x) / 2.0 + self.random.uniform(-0.04, 0.04)
        control_y = (start.y + end.y) / 2.0 + self.random.uniform(-0.04, 0.04)

        for i in range(steps + 1):
            t = i / steps
            eased = self._ease_in_out_cubic(t)
            curve_x = ((1 - eased) ** 2) * start.x + 2 * (1 - eased) * eased * control_x + (eased**2) * end.x
            curve_y = ((1 - eased) ** 2) * start.y + 2 * (1 - eased) * eased * control_y + (eased**2) * end.y

            wave = math.sin(t * math.pi * self.random.uniform(1.1, 1.8))
            jitter_x = self.random.uniform(-jitter_ratio, jitter_ratio) + wave * jitter_ratio * 0.45
            jitter_y = self.random.uniform(-jitter_ratio, jitter_ratio) - wave * jitter_ratio * 0.45

            point = RelativePoint(curve_x + jitter_x, curve_y + jitter_y).clamp()
            if i == 0:
                point = start.clamp()
            elif i == steps:
                point = end.clamp()
            path.append(point)
        return path

    def _build_segment_weights(self, segments: int) -> list[float]:
        values: list[float] = []
        for index in range(segments):
            t = index / max(1, segments - 1)
            base = 0.6 + abs(t - 0.5) * 1.3
            values.append(base * self.random.uniform(0.9, 1.15))
        total = sum(values)
        return [value / total for value in values]

    @staticmethod
    def _ease_in_out_cubic(t: float) -> float:
        if t < 0.5:
            return 4 * t * t * t
        return 1 - pow(-2 * t + 2, 3) / 2
