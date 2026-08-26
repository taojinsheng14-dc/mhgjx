from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .frame_source import FrameSource
from .humanizer import RelativeRegion
from .vision import TemplateMatcher


@dataclass(frozen=True)
class StateSignal:
    name: str
    confidence: float = 0.0
    detail: str = ""


@dataclass(frozen=True)
class GameState:
    name: str
    confidence: float
    signals: list[StateSignal] = field(default_factory=list)


class GameStateMachine:
    def __init__(self, matcher: TemplateMatcher, templates_dir: Path, frame_source: FrameSource):
        self.matcher = matcher
        self.templates_dir = templates_dir
        self.frame_source = frame_source

    def observe(self) -> GameState:
        frame = self.frame_source.latest(max_age=0.2)
        image = frame.image
        signals: list[StateSignal] = []

        battle = self._match("战斗回合标志.png", image, 0.8, [0.35, 0.0, 0.65, 0.13])
        if battle is not None:
            signals.append(StateSignal("battle", battle))
            return GameState("battle", battle, signals)

        fight_button = self._match("抓鬼战斗按钮.png", image, 0.84, [0.66, 0.45, 0.95, 0.72])
        if fight_button is not None:
            signals.append(StateSignal("ghost_fight_dialog", fight_button))
            return GameState("ghost_fight_dialog", fight_button, signals)

        accept = self._match("好的我帮你.png", image, 0.86, [0.62, 0.48, 0.98, 0.78])
        if accept is not None:
            signals.append(StateSignal("zhongkui_accept_dialog", accept))
            return GameState("zhongkui_accept_dialog", accept, signals)

        activity = self._match("钟馗抓鬼识别文字.png", image, 0.84, [0.30, 0.12, 0.84, 0.78])
        if activity is not None:
            signals.append(StateSignal("activity_ghost_card", activity))
            return GameState("activity_ghost_card", activity, signals)

        task_panel = self._task_panel_activity(image)
        if task_panel > 0.02:
            signals.append(StateSignal("right_task_panel", task_panel, "right panel has bright task text"))
            return GameState("right_task_panel", min(0.75, task_panel * 10), signals)

        return GameState("unknown", 0.0, signals)

    def _match(self, template: str, image: np.ndarray, threshold: float, region_values: list[float]) -> float | None:
        path = self.templates_dir / template
        if not path.exists():
            return None
        match = self.matcher.find_template(
            path,
            threshold=threshold,
            screenshot=image,
            search_region=RelativeRegion.from_ltrb(*region_values),
        )
        if match is None:
            return None
        return match.confidence

    @staticmethod
    def _task_panel_activity(image: np.ndarray) -> float:
        height, width = image.shape[:2]
        left = int(width * 0.78)
        top = int(height * 0.2)
        right = int(width * 0.97)
        bottom = int(height * 0.65)
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            return 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Green/yellow/white task text in the fixed right task panel.
        green = cv2.inRange(hsv, (35, 45, 80), (90, 255, 255))
        yellow = cv2.inRange(hsv, (18, 50, 100), (38, 255, 255))
        white = cv2.inRange(hsv, (0, 0, 160), (179, 70, 255))
        mask = cv2.bitwise_or(cv2.bitwise_or(green, yellow), white)
        return float(cv2.countNonZero(mask)) / float(mask.size)
