from __future__ import annotations

import threading
import time
import json
import re
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .device import AdbDevice
from .frame_source import AdbFrameSource, FrameSource
from .humanizer import HumanizedController, RelativePoint, RelativeRegion
from .state_machine import GameStateMachine
from .vision import TemplateMatcher


LogCallback = Callable[[str], None]


class RestartWorkflow(Exception):
    """Signal that the current round should start from the first step again."""


class WorkflowRunner:
    """Execute toolbox workflow dictionaries against an adb device."""

    def __init__(
        self,
        device: AdbDevice,
        templates_dir: Path,
        log: LogCallback = print,
        stop_event: threading.Event | None = None,
        frame_source: FrameSource | None = None,
    ):
        self.device = device
        self.templates_dir = templates_dir
        self.failures_dir = templates_dir.parent / "failures"
        self.log = log
        self.stop_event = stop_event or threading.Event()
        self.human = HumanizedController(device)
        self.frame_source = frame_source or AdbFrameSource(device)
        self.matcher = TemplateMatcher(device, frame_source=self.frame_source)
        self.state_machine = GameStateMachine(self.matcher, templates_dir, self.frame_source)
        self.column_cache: dict[str, str] = {}
        self.state_dir = templates_dir.parent / "state"
        self.flags: dict[str, bool] = {}
        self.flag_events: dict[str, threading.Event] = {}
        self.flag_details: dict[str, str] = {}
        self._ocr_engine = None
        self._ocr_lock = threading.Lock()
        self._debug_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="debug-shot")
        self._debug_futures: list[Future] = []
        self._debug_lock = threading.Lock()

    def run(self, steps: list[dict], repeat: int = 1, start_index: int = 0) -> None:
        self.device.ensure_connected()
        infinite = repeat <= 0
        rounds = None if infinite else max(1, repeat)
        start_index = max(0, min(start_index, len(steps)))
        max_round_restarts = 2
        try:
            round_index = 0
            while not self.stop_event.is_set() and (infinite or round_index < rounds):
                if self.stop_event.is_set():
                    break
                self.flags.clear()
                self.flag_events.clear()
                self.flag_details.clear()
                restart_count = 0
                while not self.stop_event.is_set():
                    try:
                        round_label = f"{round_index + 1}/∞" if infinite else f"{round_index + 1}/{rounds}"
                        if start_index:
                            self.log(f"??? {round_label} ???? {start_index + 1} ???")
                        else:
                            self.log(f"??? {round_label} ?")
                        for index, step in enumerate(steps[start_index:], start=start_index + 1):
                            if self.stop_event.is_set():
                                break
                            self.log(f"?? {index}: {self.describe_step(step)}")
                            self._run_step(step)
                        break
                    except RestartWorkflow:
                        restart_count += 1
                        if restart_count > max_round_restarts:
                            raise TimeoutError("流程从头重试次数过多，已停止以便人工检查。")
                        self.log(f"当前轮从头重试 {restart_count}/{max_round_restarts}")
                round_index += 1
            self.log("流程已停止" if self.stop_event.is_set() else "流程执行完成")
        finally:
            self.frame_source.close()
            self._debug_executor.shutdown(wait=False, cancel_futures=True)

    def _run_step(self, step: dict) -> None:
        if step.get("disabled", False):
            self.log(f"跳过未配置步骤: {step.get('label') or step.get('template') or step.get('type')}")
            return
        action = step.get("type")
        if action == "template_click":
            self._wait_template(step, click=True)
        elif action == "region_click":
            self._region_click(step)
        elif action == "region_click_until_template":
            self._region_click_until_template(step)
        elif action == "delay":
            self._interruptible_delay(float(step["min"]), float(step["max"]))
        elif action == "wait_template":
            self._wait_template(step, click=False)
        elif action == "wait_template_click":
            self._wait_template(step, click=True)
        elif action == "template_scroll_click":
            self._template_scroll_click(step)
        elif action == "template_column_region_click":
            self._template_column_region_click(step)
        elif action == "template_grid_slot_click":
            self._template_grid_slot_click(step)
        elif action == "detect_template_flag":
            self._detect_template_flag(step)
        elif action == "detect_text_flag":
            self._detect_text_flag(step)
        elif action == "detect_text_flags":
            self._detect_text_flags(step)
        elif action == "ensure_text_after_click":
            self._ensure_text_after_click(step)
        elif action == "wait_text":
            self._wait_text(step)
        elif action == "conditional_region_click":
            self._conditional_region_click(step)
        elif action == "wait_template_gone":
            self._wait_template_gone(step)
        elif action == "wait_region_still":
            self._wait_region_still(step)
        elif action == "wait_battle_complete":
            self._wait_battle_complete(step)
        elif action == "swimming_task_loop":
            self._swimming_task_loop(step)
        elif action == "swipe":
            self._swipe(step)
        elif action == "observe_state":
            self._observe_state(step)
        else:
            raise ValueError(f"不支持的动作类型: {action}")

    def _observe_state(self, step: dict) -> None:
        state = self.state_machine.observe()
        signals = ", ".join(f"{signal.name}:{signal.confidence:.3f}" for signal in state.signals) or "none"
        self.log(f"state={state.name} confidence={state.confidence:.3f} signals={signals}")
        expected = step.get("expected")
        if expected and state.name != str(expected):
            self.log(f"state mismatch: expected {expected}, got {state.name}")

    def _region_click(self, step: dict) -> None:
        if self.stop_event.is_set():
            return
        label = str(step.get("label") or step.get("template") or step.get("type") or "")
        if step.get("use_click_region", False):
            values = step.get("region")
            if not isinstance(values, list) or len(values) != 4:
                raise ValueError("use_click_region requires region [left, top, right, bottom]")
            region = RelativeRegion.from_ltrb(*(float(value) for value in values)).clamp()
            pre_delay = tuple(step.get("pre_delay", [0.15, 0.4]))
            post_delay = tuple(step.get("post_delay", [0.25, 0.65]))
            padding_ratio = float(step.get("padding_ratio", 0.12))
            tap_mode = str(step.get("tap_mode", "tap"))
            x, y = self._tap_region(
                region,
                pre_delay=pre_delay,
                post_delay=post_delay,
                padding_ratio=padding_ratio,
                tap_mode=tap_mode,
            )
            self._log_click_detail(
                label=label,
                source="click_region",
                x=x,
                y=y,
                region=region,
                pre_delay=pre_delay,
                post_delay=post_delay,
                padding_ratio=padding_ratio,
                tap_mode=tap_mode,
            )
            self._maybe_save_debug_screenshot(step, "after_click")
            self._verify_after_step(step)
            return

        pixel_point = step.get("pixel_point")
        drift_px = int(step.get("drift_px", 20))
        if isinstance(pixel_point, list) and len(pixel_point) == 2:
            base_x = int(pixel_point[0])
            base_y = int(pixel_point[1])
            x = base_x + self.human.random.randint(-drift_px, drift_px)
            y = base_y + self.human.random.randint(-drift_px, drift_px)
            pre_delay = tuple(step.get("pre_delay", [0.15, 0.4]))
            post_delay = tuple(step.get("post_delay", [0.25, 0.65]))
            tap_mode = str(step.get("tap_mode", "tap"))
            self._interruptible_delay(*pre_delay)
            if self.stop_event.is_set():
                return
            if tap_mode == "swipe":
                self.device.short_swipe_tap(x, y, int(step.get("tap_duration_ms", 65)))
            else:
                self.device.tap(x, y)
            self._interruptible_delay(*post_delay)
            self._log_click_detail(
                label=label,
                source="pixel_point",
                x=x,
                y=y,
                base=(base_x, base_y),
                drift_px=drift_px,
                pre_delay=pre_delay,
                post_delay=post_delay,
                tap_mode=tap_mode,
            )
            self._maybe_save_debug_screenshot(step, "after_click")
            self._verify_after_step(step)
            return

        values = step.get("region")
        if not isinstance(values, list) or len(values) != 4:
            raise ValueError("region_click requires [left, top, right, bottom]")
        region = RelativeRegion.from_ltrb(*(float(value) for value in values)).clamp()
        pre_delay = tuple(step.get("pre_delay", [0.15, 0.4]))
        post_delay = tuple(step.get("post_delay", [0.25, 0.65]))
        padding_ratio = float(step.get("padding_ratio", 0.12))
        tap_mode = str(step.get("tap_mode", "tap"))
        x, y = self._tap_region(
            region,
            pre_delay=pre_delay,
            post_delay=post_delay,
            padding_ratio=padding_ratio,
            tap_mode=tap_mode,
        )
        self._log_click_detail(
            label=label,
            source="relative_region",
            x=x,
            y=y,
            region=region,
            pre_delay=pre_delay,
            post_delay=post_delay,
            padding_ratio=padding_ratio,
            tap_mode=tap_mode,
        )
        self._maybe_save_debug_screenshot(step, "after_click")
        self._verify_after_step(step)


    def _verify_after_step(self, step: dict) -> None:
        verify = step.get("verify_after")
        if not isinstance(verify, dict):
            return
        if self._verify_condition(verify):
            return
        retries = max(0, int(step.get("verify_retries", verify.get("retries", 0))))
        for attempt in range(retries):
            if self.stop_event.is_set():
                return
            self.log(f"?????????? {attempt + 1}/{retries}: {step.get('label', '')}")
            retry_step = dict(step)
            retry_step.pop("verify_after", None)
            self._region_click(retry_step)
            if self._verify_condition(verify):
                self.log(f"?????????: {step.get('label', '')}")
                return
        message = str(verify.get("message") or f"???????: {step.get('label', step.get('type'))}")
        self._handle_timeout(step, message)

    def _verify_condition(self, verify: dict) -> bool:
        timeout = float(verify.get("timeout", 1.2))
        interval = float(verify.get("interval", 0.2))
        mode = str(verify.get("mode", "color"))
        threshold = float(verify.get("threshold", 0.84))
        search_region = self._relative_region_from_values(verify.get("search_region"))
        any_templates = verify.get("any_template")
        if isinstance(any_templates, str):
            any_templates = [any_templates]
        if isinstance(any_templates, list) and any_templates:
            deadline = time.monotonic() + timeout
            while not self.stop_event.is_set() and time.monotonic() < deadline:
                for template in any_templates:
                    path = self.templates_dir / str(template)
                    if path.exists():
                        match = self.matcher.find_template(
                            path,
                            threshold=threshold,
                            mode=mode,
                            search_region=search_region,
                        )
                        if match:
                            self.log(f"???????: {path.name} {match.confidence:.3f}")
                            return True
                self._interruptible_wait(interval)
            return False
        gone_template = verify.get("gone_template")
        if isinstance(gone_template, str) and gone_template:
            path = self.templates_dir / gone_template
            deadline = time.monotonic() + timeout
            while not self.stop_event.is_set() and time.monotonic() < deadline:
                match = None
                if path.exists():
                    match = self.matcher.find_template(
                        path,
                        threshold=threshold,
                        mode=mode,
                        search_region=search_region,
                    )
                if match is None:
                    self.log(f"???????: {gone_template} ???")
                    return True
                self._interruptible_wait(interval)
            return False
        expected_state = verify.get("state")
        if isinstance(expected_state, str) and expected_state:
            deadline = time.monotonic() + timeout
            while not self.stop_event.is_set() and time.monotonic() < deadline:
                state = self.state_machine.observe()
                if state.name == expected_state:
                    self.log(f"?????????: {state.name}")
                    return True
                self._interruptible_wait(interval)
            return False
        return True

    def _tap_region(
        self,
        region: RelativeRegion,
        pre_delay: tuple[float, float],
        post_delay: tuple[float, float],
        padding_ratio: float = 0.12,
        tap_mode: str = "tap",
    ) -> tuple[int, int]:
        self._interruptible_delay(*pre_delay)
        point = self.human.random_point_in_region(region, padding_ratio=padding_ratio)
        x, y = self.human.to_absolute(point)
        if not self.stop_event.is_set():
            if tap_mode == "swipe":
                self.device.short_swipe_tap(x, y)
            else:
                self.device.tap(x, y)
        self._interruptible_delay(*post_delay)
        return x, y

    def _log_click_detail(
        self,
        *,
        label: str,
        source: str,
        x: int,
        y: int,
        region: RelativeRegion | None = None,
        base: tuple[int, int] | None = None,
        drift_px: int | None = None,
        pre_delay: tuple[float, float] | None = None,
        post_delay: tuple[float, float] | None = None,
        padding_ratio: float | None = None,
        tap_mode: str = "tap",
        match_box: tuple[int, int, int, int] | None = None,
    ) -> None:
        parts = [f"click detail: label={label}", f"source={source}", f"point=({x},{y})", f"mode={tap_mode}"]
        if base is not None:
            parts.append(f"base=({base[0]},{base[1]})")
        if drift_px is not None:
            parts.append(f"drift=±{drift_px}px")
        if region is not None:
            parts.append(f"click_box={self._region_to_pixel_box(region)}")
            parts.append(f"region=({region.left:.4f},{region.top:.4f},{region.right:.4f},{region.bottom:.4f})")
        if match_box is not None:
            parts.append(f"match_box={match_box}")
        if padding_ratio is not None:
            parts.append(f"padding={padding_ratio:.2f}")
        if pre_delay is not None:
            parts.append(f"pre={float(pre_delay[0]):.2f}-{float(pre_delay[1]):.2f}s")
        if post_delay is not None:
            parts.append(f"post={float(post_delay[0]):.2f}-{float(post_delay[1]):.2f}s")
        self.log(" | ".join(parts))

    def _region_click_until_template(self, step: dict) -> None:
        template_path = self.templates_dir / str(step["template"])
        attempts = max(1, int(step.get("attempts", 3)))
        threshold = float(step.get("threshold", 0.84))
        verify_timeout = float(step.get("verify_timeout", 1.2))
        interval = float(step.get("interval", 0.25))
        search_region = self._optional_search_region(step)
        for attempt in range(attempts):
            self.log(f"范围点击并验证 {step.get('label', '')}: {attempt + 1}/{attempts}")
            self._region_click(step)
            match = self._find_template_until(
                template_path,
                threshold=threshold,
                timeout=verify_timeout,
                interval=interval,
                mode=str(step.get("match_mode", "color")),
                search_region=search_region,
            )
            if match:
                self.log(f"点击后验证成功 {template_path.name} {match.confidence:.3f}")
                return
        self._handle_timeout(step, f"点击后未出现目标模板: {template_path.name}")

    def _wait_template(self, step: dict, click: bool) -> None:
        template_path = self.templates_dir / step["template"]
        threshold = float(step.get("threshold", 0.88))
        timeout = float(step.get("timeout", 10.0))
        interval = float(step.get("interval", 0.5))
        started_at = time.monotonic()
        initial_delay = step.get("initial_delay")
        if isinstance(initial_delay, list) and len(initial_delay) == 2:
            self._interruptible_delay(float(initial_delay[0]), float(initial_delay[1]))
        scan_started_at = time.monotonic()
        deadline = time.monotonic() + timeout

        while not self.stop_event.is_set() and time.monotonic() < deadline:
            match_started_at = time.monotonic()
            match = self.matcher.find_template(
                template_path,
                threshold=threshold,
                mode=str(step.get("match_mode", "color")),
                search_region=self._optional_search_region(step),
            )
            match_elapsed_ms = (time.monotonic() - match_started_at) * 1000.0
            if match:
                total_elapsed = time.monotonic() - started_at
                scan_elapsed = time.monotonic() - scan_started_at
                if click:
                    x, y = self._click_match(match, step)
                    self.log(
                        f"识别成功 {match.confidence:.3f}，点击 ({x}, {y})，"
                        f"总等待{total_elapsed:.2f}s，扫描{scan_elapsed:.2f}s，单次{match_elapsed_ms:.0f}ms"
                    )
                else:
                    self.log(
                        f"识别成功 {match.confidence:.3f}: {step['template']}，"
                        f"总等待{total_elapsed:.2f}s，扫描{scan_elapsed:.2f}s，单次{match_elapsed_ms:.0f}ms"
                    )
                return
            self._interruptible_wait(interval)
        if not self.stop_event.is_set():
            self._handle_timeout(step, f"超时未识别到模板: {step['template']}")

    def _detect_template_flag(self, step: dict) -> None:
        flag = str(step.get("flag") or "")
        if not flag:
            raise ValueError("detect_template_flag requires flag")
        template = str(step.get("template") or "")
        template_path = self.templates_dir / template
        self.flags[flag] = False
        if not template or not template_path.exists():
            self.log(f"flag {flag}=false, template missing: {template}")
            return
        initial_delay = step.get("initial_delay")
        if isinstance(initial_delay, list) and len(initial_delay) == 2:
            self._interruptible_delay(float(initial_delay[0]), float(initial_delay[1]))
        match = self._find_template_until(
            template_path,
            threshold=float(step.get("threshold", 0.84)),
            timeout=float(step.get("timeout", 1.5)),
            interval=float(step.get("interval", 0.25)),
            mode=str(step.get("match_mode", "color")),
            search_region=self._optional_search_region(step),
        )
        self.flags[flag] = bool(match)
        if match:
            self.log(f"flag {flag}=true, matched {template} {match.confidence:.3f}")
        else:
            self.log(f"flag {flag}=false, not matched {template}")

    def _detect_text_flag(self, step: dict) -> None:
        flag = str(step.get("flag") or "")
        if not flag:
            raise ValueError("detect_text_flag requires flag")
        targets = step.get("texts")
        if not isinstance(targets, list) or not targets:
            targets = [step.get("text", "")]
        target_texts = [str(value).strip() for value in targets if str(value).strip()]
        if not target_texts:
            raise ValueError("detect_text_flag requires text or texts")

        self.flags[flag] = False
        self.flag_details[flag] = ""
        event = threading.Event()
        self.flag_events[flag] = event
        crop = self._capture_color_region(self._optional_search_region(step))

        def worker() -> None:
            try:
                found, detail = self._ocr_contains_text(
                    crop,
                    target_texts,
                    min_confidence=float(step.get("min_confidence", 0.45)),
                )
                self.flags[flag] = found
                self.flag_details[flag] = detail
                self.log(f"flag {flag}={str(found).lower()}, ocr='{detail[:80]}'")
            except Exception as exc:
                self.flags[flag] = False
                self.flag_details[flag] = str(exc)
                self.log(f"flag {flag}=false, ocr error: {exc}")
            finally:
                event.set()

        if bool(step.get("async", True)):
            thread = threading.Thread(target=worker, name=f"ocr-{flag}", daemon=True)
            thread.start()
            self.log(f"flag {flag}: async OCR started for {target_texts}")
        else:
            worker()

    def _detect_text_flags(self, step: dict) -> None:
        definitions = step.get("flags")
        if not isinstance(definitions, list) or not definitions:
            raise ValueError("detect_text_flags requires flags")

        normalized_definitions: list[tuple[str, list[str], bool]] = []
        for item in definitions:
            if not isinstance(item, dict):
                continue
            flag = str(item.get("flag") or "").strip()
            targets = item.get("texts")
            if not isinstance(targets, list) or not targets:
                targets = [item.get("text", "")]
            target_texts = [str(value).strip() for value in targets if str(value).strip()]
            if flag and target_texts:
                normalized_definitions.append((flag, target_texts, bool(item.get("require_all", False))))
                self.flags[flag] = False
                self.flag_details[flag] = ""
                self.flag_events[flag] = threading.Event()

        if not normalized_definitions:
            raise ValueError("detect_text_flags has no valid flags")

        crop = self._capture_color_region(self._optional_search_region(step))

        def worker() -> None:
            try:
                with self._ocr_lock:
                    if self._ocr_engine is None:
                        from rapidocr_onnxruntime import RapidOCR

                        self._ocr_engine = RapidOCR()
                    result, _elapsed = self._ocr_engine(crop)
                detail = self._ocr_detail(result, float(step.get("min_confidence", 0.45)))
                normalized = re.sub(r"\s+", "", detail)
                for flag, target_texts, require_all in normalized_definitions:
                    normalized_targets = [re.sub(r"\s+", "", target) for target in target_texts]
                    found = all(target in normalized for target in normalized_targets) if require_all else any(
                        target in normalized for target in normalized_targets
                    )
                    self.flags[flag] = found
                    self.flag_details[flag] = detail
                    self.log(f"flag {flag}={str(found).lower()}, ocr='{detail[:80]}'")
            except Exception as exc:
                for flag, _target_texts, _require_all in normalized_definitions:
                    self.flags[flag] = False
                    self.flag_details[flag] = str(exc)
                    self.log(f"flag {flag}=false, ocr error: {exc}")
            finally:
                for flag, _target_texts, _require_all in normalized_definitions:
                    event = self.flag_events.get(flag)
                    if event is not None:
                        event.set()

        if bool(step.get("async", True)):
            thread = threading.Thread(target=worker, name="ocr-flags", daemon=True)
            thread.start()
            self.log(f"flags OCR started: {[flag for flag, _targets, _require_all in normalized_definitions]}")
        else:
            worker()

    def _wait_text(self, step: dict) -> None:
        targets = step.get("texts")
        if not isinstance(targets, list) or not targets:
            targets = [step.get("text", "")]
        target_texts = [str(value).strip() for value in targets if str(value).strip()]
        if not target_texts:
            raise ValueError("wait_text requires text or texts")

        require_all = bool(step.get("require_all", False))
        timeout = float(step.get("timeout", 8.0))
        interval = step.get("interval", [0.35, 0.75])
        min_confidence = float(step.get("min_confidence", 0.45))
        search_region = self._optional_search_region(step)
        deadline = time.monotonic() + timeout
        last_detail = ""
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            crop = self._capture_color_region(search_region)
            found, detail = self._ocr_contains_text(
                crop,
                target_texts,
                min_confidence=min_confidence,
                require_all=require_all,
            )
            last_detail = detail
            if found:
                self.log(f"等待文本成功: {target_texts}，ocr='{detail[:80]}'")
                return
            self.log(f"等待文本未命中: {target_texts}，ocr='{detail[:60]}'")
            if isinstance(interval, list) and len(interval) == 2:
                self._interruptible_wait(self.human.random.uniform(float(interval[0]), float(interval[1])))
            else:
                self._interruptible_wait(float(interval))
        self._handle_timeout(step, f"等待文本超时: {target_texts}，最后OCR='{last_detail[:80]}'")

    def _ensure_text_after_click(self, step: dict) -> None:
        targets = step.get("texts")
        if not isinstance(targets, list) or not targets:
            targets = [step.get("text", "")]
        target_texts = [str(value).strip() for value in targets if str(value).strip()]
        if not target_texts:
            raise ValueError("ensure_text_after_click requires text or texts")

        click_region = self._optional_fixed_click_region(step) or self._optional_relative_region(step, "click_region")
        if click_region is None:
            region_values = step.get("region")
            if isinstance(region_values, list) and len(region_values) == 4:
                click_region = RelativeRegion.from_ltrb(*(float(value) for value in region_values)).clamp()
        if click_region is None:
            raise ValueError("ensure_text_after_click requires click_region, fixed_click_region or region")

        search_region = self._optional_search_region(step)
        min_confidence = float(step.get("min_confidence", 0.45))
        require_all = bool(step.get("require_all", False))
        attempts = max(1, int(step.get("attempts", 2)))
        verify_timeout = float(step.get("verify_timeout", 2.0))
        verify_interval = float(step.get("verify_interval", 0.35))
        last_detail = ""

        if bool(step.get("coordinate_must_change", False)):
            self._ensure_coordinate_changes_after_click(
                step=step,
                click_region=click_region,
                search_region=search_region,
                min_confidence=min_confidence,
                attempts=attempts,
                verify_timeout=verify_timeout,
                verify_interval=verify_interval,
            )
            return

        if bool(step.get("click_first", False)):
            self.log("ensure text click first")
            pre_delay = tuple(step.get("pre_delay", [0.08, 0.22]))
            post_delay = tuple(step.get("post_delay", [0.08, 0.18]))
            padding_ratio = float(step.get("padding_ratio", 0.0))
            tap_mode = str(step.get("tap_mode", "tap"))
            x, y = self._tap_region(
                click_region,
                pre_delay=pre_delay,
                post_delay=post_delay,
                padding_ratio=padding_ratio,
                tap_mode=tap_mode,
            )
            self._log_click_detail(
                label=str(step.get("label") or step.get("type") or ""),
                source="ensure_text_click_first",
                x=x,
                y=y,
                region=click_region,
                pre_delay=pre_delay,
                post_delay=post_delay,
                padding_ratio=padding_ratio,
                tap_mode=tap_mode,
            )

        for attempt in range(attempts + 1):
            if self.stop_event.is_set():
                return
            found, detail = self._ocr_region_contains(
                search_region,
                target_texts,
                min_confidence=min_confidence,
                require_all=require_all,
            )
            if bool(step.get("ghost_coordinate_pattern", False)):
                found = self._looks_like_ghost_coordinate(detail)
            last_detail = detail
            if found:
                self.log(f"ensure text ok before/after click: {target_texts}, ocr='{detail[:80]}'")
                return
            if attempt >= attempts:
                break
            self.log(f"ensure text missing, click retry {attempt + 1}/{attempts}: ocr='{detail[:80]}'")
            pre_delay = tuple(step.get("pre_delay", [0.08, 0.22]))
            post_delay = tuple(step.get("post_delay", [0.08, 0.18]))
            padding_ratio = float(step.get("padding_ratio", 0.0))
            tap_mode = str(step.get("tap_mode", "tap"))
            x, y = self._tap_region(
                click_region,
                pre_delay=pre_delay,
                post_delay=post_delay,
                padding_ratio=padding_ratio,
                tap_mode=tap_mode,
            )
            self._log_click_detail(
                label=str(step.get("label") or step.get("type") or ""),
                source=f"ensure_text_retry_{attempt + 1}",
                x=x,
                y=y,
                region=click_region,
                pre_delay=pre_delay,
                post_delay=post_delay,
                padding_ratio=padding_ratio,
                tap_mode=tap_mode,
            )
            deadline = time.monotonic() + verify_timeout
            while not self.stop_event.is_set() and time.monotonic() < deadline:
                found, detail = self._ocr_region_contains(
                    search_region,
                    target_texts,
                    min_confidence=min_confidence,
                    require_all=require_all,
                )
                if bool(step.get("ghost_coordinate_pattern", False)):
                    found = self._looks_like_ghost_coordinate(detail)
                last_detail = detail
                if found:
                    self.log(f"ensure text ok after click retry {attempt + 1}: ocr='{detail[:80]}'")
                    self._maybe_save_debug_screenshot(step, "after_ensure_click")
                    return
                self._interruptible_wait(verify_interval)

        self._maybe_save_debug_screenshot(step, "ensure_text_failed")
        self._handle_timeout(step, f"ensure text failed: {target_texts}, last OCR='{last_detail[:100]}'")

    def _ocr_region_contains(
        self,
        region: RelativeRegion | None,
        target_texts: list[str],
        min_confidence: float,
        require_all: bool = False,
    ) -> tuple[bool, str]:
        crop = self._capture_color_region(region)
        return self._ocr_contains_text(
            crop,
            target_texts,
            min_confidence=min_confidence,
            require_all=require_all,
        )

    def _ensure_coordinate_changes_after_click(
        self,
        step: dict,
        click_region: RelativeRegion,
        search_region: RelativeRegion | None,
        min_confidence: float,
        attempts: int,
        verify_timeout: float,
        verify_interval: float,
    ) -> None:
        before_detail = self._ocr_region_detail(search_region, min_confidence)
        before_coord = self._extract_ghost_coordinate(before_detail)
        self.log(f"eye check before: coord={before_coord}, ocr='{before_detail[:80]}'")

        last_detail = before_detail
        last_coord = before_coord
        for attempt in range(attempts + 1):
            if self.stop_event.is_set():
                return
            self.log(f"eye click {attempt + 1}/{attempts + 1}")
            pre_delay = tuple(step.get("pre_delay", [0.08, 0.22]))
            post_delay = tuple(step.get("post_delay", [0.08, 0.18]))
            padding_ratio = float(step.get("padding_ratio", 0.0))
            tap_mode = str(step.get("tap_mode", "tap"))
            x, y = self._tap_region(
                click_region,
                pre_delay=pre_delay,
                post_delay=post_delay,
                padding_ratio=padding_ratio,
                tap_mode=tap_mode,
            )
            self._log_click_detail(
                label=str(step.get("label") or step.get("type") or ""),
                source=f"eye_click_{attempt + 1}",
                x=x,
                y=y,
                region=click_region,
                pre_delay=pre_delay,
                post_delay=post_delay,
                padding_ratio=padding_ratio,
                tap_mode=tap_mode,
            )

            deadline = time.monotonic() + verify_timeout
            while not self.stop_event.is_set() and time.monotonic() < deadline:
                detail = self._ocr_region_detail(search_region, min_confidence)
                coord = self._extract_ghost_coordinate(detail)
                last_detail = detail
                last_coord = coord
                if coord is not None and before_coord is not None and coord != before_coord:
                    self.log(f"eye check ok: {before_coord} -> {coord}, ocr='{detail[:80]}'")
                    self._maybe_save_debug_screenshot(step, "after_eye_coordinate_changed")
                    return
                if coord is not None and before_coord is None:
                    self.log(f"eye check ok: new coord={coord}, ocr='{detail[:80]}'")
                    self._maybe_save_debug_screenshot(step, "after_eye_coordinate_found")
                    return
                self._interruptible_wait(verify_interval)

        self._maybe_save_debug_screenshot(step, "eye_coordinate_failed")
        self._handle_timeout(
            step,
            f"eye coordinate did not change: before={before_coord}, after={last_coord}, last OCR='{last_detail[:100]}'",
        )

    def _ocr_region_detail(self, region: RelativeRegion | None, min_confidence: float) -> str:
        crop = self._capture_color_region(region)
        with self._ocr_lock:
            if self._ocr_engine is None:
                from rapidocr_onnxruntime import RapidOCR

                self._ocr_engine = RapidOCR()
            result, _elapsed = self._ocr_engine(crop)
        return self._ocr_detail(result, min_confidence)

    @staticmethod
    def _extract_ghost_coordinate(detail: str) -> tuple[int, int] | None:
        normalized = re.sub(r"\s+", "", detail)
        if "鬼" not in normalized:
            return None
        match = re.search(r"去.{0,12}?([0-9]{1,3})[,，.．、]([0-9]{1,3}).{0,12}?抓", normalized)
        if not match:
            match = re.search(r"([0-9]{1,3})[,，.．、]([0-9]{1,3})", normalized)
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    @staticmethod
    def _looks_like_ghost_coordinate(detail: str) -> bool:
        normalized = re.sub(r"\s+", "", detail)
        if not normalized:
            return False
        has_task = ("捉鬼" in normalized) or ("抓鬼" in normalized) or ("钟道抓鬼" in normalized) or ("钟馗抓鬼" in normalized)
        has_route = "去" in normalized and "抓" in normalized
        has_coordinate = re.search(r"\d{1,3}[,，.．、]\d{1,3}", normalized) is not None
        return has_task and has_route and has_coordinate

    def _conditional_region_click(self, step: dict) -> None:
        conditions = step.get("conditions")
        if isinstance(conditions, dict) and conditions:
            matched, mismatches = self._conditions_match(conditions, timeout=float(step.get("flag_wait_timeout", 8.0)))
            if not matched:
                self.log(f"skip conditional click {step.get('label', '')}: " + "; ".join(mismatches))
                return
            self.log(f"run conditional click {step.get('label', '')}: conditions matched")
            self._region_click(step)
            return

        flag = str(step.get("flag") or "")
        event = self.flag_events.get(flag)
        if event is not None and not event.is_set():
            timeout = float(step.get("flag_wait_timeout", 8.0))
            self.log(f"waiting flag {flag} up to {timeout:.1f}s")
            event.wait(timeout)
        expected = bool(step.get("expected", True))
        actual = bool(self.flags.get(flag, False))
        if actual != expected:
            self.log(f"skip conditional click {step.get('label', '')}: flag {flag}={actual}")
            return
        self.log(f"run conditional click {step.get('label', '')}: flag {flag}={actual}")
        self._region_click(step)

    def _conditions_match(self, conditions: dict, timeout: float = 0.0) -> tuple[bool, list[str]]:
        mismatches: list[str] = []
        for flag_name, expected_value in conditions.items():
            flag = str(flag_name)
            event = self.flag_events.get(flag)
            if timeout > 0 and event is not None and not event.is_set():
                self.log(f"waiting flag {flag} up to {timeout:.1f}s")
                event.wait(timeout)
            expected = bool(expected_value)
            actual = bool(self.flags.get(flag, False))
            if actual != expected:
                mismatches.append(f"{flag}={actual}, expected {expected}")
        return not mismatches, mismatches

    def _template_scroll_click(self, step: dict) -> None:
        template_path = self.templates_dir / step["template"]
        threshold = float(step.get("threshold", 0.86))
        attempts = max(1, int(step.get("attempts", 2)))
        timeout = float(step.get("timeout_per_attempt", step.get("timeout", 3.0)))
        interval = float(step.get("interval", 0.5))
        pre_swipes = self._pre_swipes_for_step(step)
        for swipe_index in range(pre_swipes):
            self.log(f"识别前先滑动 {swipe_index + 1}/{pre_swipes}")
            self._swipe(
                {
                    "direction": step.get("swipe_direction", "up"),
                    "duration_min": int(step.get("swipe_duration_min", 550)),
                    "duration_max": int(step.get("swipe_duration_max", 850)),
                    "jitter": float(step.get("swipe_jitter", 0.004)),
                }
            )
            self._interruptible_delay(*tuple(step.get("after_swipe_delay", [0.7, 1.2])))
        for attempt in range(attempts):
            match = self._find_template_until(
                template_path,
                threshold=threshold,
                timeout=timeout,
                interval=interval,
                mode=str(step.get("match_mode", "color")),
                search_region=self._optional_search_region(step),
            )
            if match:
                x, y = self._click_match(match, step)
                self.log(f"滑动重试识别成功 {match.confidence:.3f}，点击 ({x}, {y})")
                return
            if attempt < attempts - 1 and self._allow_retry_swipe(step):
                self.log(f"未找到 {step['template']}，执行滑动后重试 {attempt + 2}/{attempts}")
                self._swipe(
                    {
                        "direction": step.get("swipe_direction", "up"),
                        "duration_min": int(step.get("swipe_duration_min", 550)),
                        "duration_max": int(step.get("swipe_duration_max", 850)),
                        "jitter": float(step.get("swipe_jitter", 0.004)),
                    }
                )
                self._interruptible_delay(*tuple(step.get("after_swipe_delay", [0.7, 1.2])))
        if not self.stop_event.is_set():
            self._handle_timeout(step, f"滑动重试后仍未识别到模板: {step['template']}")

    def _template_column_region_click(self, step: dict) -> None:
        template_path = self.templates_dir / step["template"]
        match = self._find_template_with_optional_scroll(step, template_path)
        if not match:
            if not self.stop_event.is_set():
                self._handle_timeout(step, f"未识别到目标卡片: {step['template']}")
            return

        left, _top, right, _bottom = match.pixel_box
        center_x = (left + right) * 0.5
        screen_w = self.device.screen_size_for_input().width
        detected_column = "left" if center_x < screen_w * float(step.get("column_split", 0.5)) else "right"

        lock_key = str(step.get("lock_key") or step.get("template") or "column_region")
        if step.get("lock_column", False):
            column = self.column_cache.setdefault(lock_key, detected_column)
            if column != detected_column:
                self.log(f"识别到 {detected_column} 列，但已锁定使用 {column} 列点击范围")
        else:
            column = detected_column

        row_offsets = step.get("row_click_region_offset")
        if isinstance(row_offsets, list) and len(row_offsets) == 4:
            x, y = self._tap_match_offset_region(match, row_offsets, step)
            self.log(f"识别到 {detected_column} 列，按识别行动态点击参加 ({x}, {y})")
            return

        regions = step.get("column_click_regions")
        if isinstance(regions, dict) and column in regions:
            values = regions[column]
            if not isinstance(values, list) or len(values) != 4:
                raise ValueError(f"{column} column click region requires [left, top, right, bottom]")
            click_region = RelativeRegion.from_ltrb(*(float(value) for value in values)).clamp()
            pre_delay = tuple(step.get("pre_delay", [0.12, 0.35]))
            post_delay = tuple(step.get("post_delay", [0.25, 0.65]))
            padding_ratio = float(step.get("padding_ratio", 0.18))
            x, y = self.human.random_tap(
                click_region,
                pre_delay=pre_delay,
                post_delay=post_delay,
                padding_ratio=padding_ratio,
            )
            self._maybe_save_debug_screenshot(step, f"after_{column}_column_click")
            self._log_click_detail(
                label=str(step.get("label") or step.get("template") or step.get("type") or ""),
                source=f"template_column_{column}",
                x=x,
                y=y,
                region=click_region,
                pre_delay=pre_delay,
                post_delay=post_delay,
                padding_ratio=padding_ratio,
                match_box=match.pixel_box,
            )
            self.log(f"识别到 {detected_column} 列，使用 {column} 列固定范围随机点击 ({x}, {y})")
            return

        dynamic_offsets = step.get("match_click_region_offset")
        if isinstance(dynamic_offsets, list) and len(dynamic_offsets) == 4:
            x, y = self._tap_match_offset_region(match, dynamic_offsets, step)
            self.log(f"识别到 {detected_column} 列，未配置列范围，按模板位置动态点击 ({x}, {y})")
            return

        x, y = self._click_match(match, step)
        self.log(f"未配置 {column} 列点击范围，回退卡片内点击 ({x}, {y})")
        return


    def _template_grid_slot_click(self, step: dict) -> None:
        template_path = self.templates_dir / step["template"]
        match = self._find_template_with_optional_scroll(step, template_path)
        if not match:
            if not self.stop_event.is_set():
                self._handle_timeout(step, f"????????: {step['template']}")
            return

        left, top, right, bottom = match.pixel_box
        center_x = (left + right) * 0.5
        center_y = (top + bottom) * 0.5
        slots = step.get("slots", {})
        matched_slot = None
        for slot_name, slot in slots.items():
            bounds = slot.get("bounds") if isinstance(slot, dict) else None
            if not isinstance(bounds, list) or len(bounds) != 4:
                continue
            x1, y1, x2, y2 = (float(value) for value in bounds)
            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                matched_slot = (slot_name, slot)
                break
        if matched_slot is None:
            self._handle_timeout(step, f"????????????: center=({center_x:.0f},{center_y:.0f})")
            return

        slot_name, slot = matched_slot
        click_region_values = slot.get("click_region")
        if not isinstance(click_region_values, list) or len(click_region_values) != 4:
            self._handle_timeout(step, f"?? {slot_name} ?????????")
            return
        click_region = RelativeRegion.from_ltrb(*(float(value) for value in click_region_values)).clamp()
        x, y = self._tap_region(
            click_region,
            pre_delay=tuple(step.get("pre_delay", [0.05, 0.16])),
            post_delay=tuple(step.get("post_delay", [0.02, 0.06])),
            padding_ratio=float(step.get("padding_ratio", 0.03)),
        )
        self._maybe_save_annotated_debug_screenshot(
            step,
            f"after_slot_{slot_name}_click",
            match_box=match.pixel_box,
            click_box=self._region_to_pixel_box(click_region),
            point=(x, y),
        )
        self.log(
            f"??? {step['template']} ???? {slot_name}: "
            f"match_box={match.pixel_box}, ???? ({x}, {y})"
        )

    def _find_template_with_optional_scroll(self, step: dict, template_path: Path):
        threshold = float(step.get("threshold", 0.86))
        attempts = max(1, int(step.get("attempts", 2)))
        timeout = float(step.get("timeout_per_attempt", step.get("timeout", 3.0)))
        interval = float(step.get("interval", 0.5))
        initial_delay = step.get("initial_delay")
        if isinstance(initial_delay, list) and len(initial_delay) == 2:
            self._interruptible_delay(float(initial_delay[0]), float(initial_delay[1]))
        if self.stop_event.is_set():
            return None
        pre_swipes = self._pre_swipes_for_step(step)
        for swipe_index in range(pre_swipes):
            if self.stop_event.is_set():
                return None
            self.log(f"识别前先滑动 {swipe_index + 1}/{pre_swipes}")
            self._fast_swipe(
                {
                    "direction": step.get("swipe_direction", "up"),
                    "duration_min": int(step.get("swipe_duration_min", 550)),
                    "duration_max": int(step.get("swipe_duration_max", 850)),
                }
            )
            self._interruptible_delay(*tuple(step.get("after_swipe_delay", [0.7, 1.2])))
        for attempt in range(attempts):
            if self.stop_event.is_set():
                return None
            match = self._find_template_until(
                template_path,
                threshold=threshold,
                timeout=timeout,
                interval=interval,
                mode=str(step.get("match_mode", "color")),
                search_region=self._optional_search_region(step),
            )
            if match:
                self.log(f"卡片识别成功 {match.confidence:.3f}: {step['template']}")
                return match
            if self.stop_event.is_set():
                return None
            if attempt < attempts - 1 and self._allow_retry_swipe(step):
                self.log(f"未找到 {step['template']}，执行滑动后重试 {attempt + 2}/{attempts}")
                self._fast_swipe(
                    {
                        "direction": step.get("swipe_direction", "up"),
                        "duration_min": int(step.get("swipe_duration_min", 550)),
                        "duration_max": int(step.get("swipe_duration_max", 850)),
                    }
                )
                self._interruptible_delay(*tuple(step.get("after_swipe_delay", [0.7, 1.2])))
        return None

    def _find_template_until(
        self,
        template_path: Path,
        threshold: float,
        timeout: float,
        interval: float,
        mode: str,
        search_region: RelativeRegion | None,
    ):
        deadline = time.monotonic() + timeout
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            match = self.matcher.find_template(
                template_path,
                threshold=threshold,
                mode=mode,
                search_region=search_region,
            )
            if match:
                return match
            self._interruptible_wait(interval)
        return None

    def _wait_template_gone(self, step: dict) -> None:
        template_path = self.templates_dir / step["template"]
        threshold = float(step.get("threshold", 0.88))
        timeout = float(step.get("timeout", 300.0))
        interval = float(step.get("interval", 5.0))
        required_misses = int(step.get("confirm_count", 2))
        misses = 0
        deadline = time.monotonic() + timeout

        while not self.stop_event.is_set() and time.monotonic() < deadline:
            match = self.matcher.find_template(
                template_path,
                threshold=threshold,
                mode=str(step.get("match_mode", "color")),
                search_region=self._optional_search_region(step),
            )
            if match:
                misses = 0
                self.log(f"模板仍存在 {match.confidence:.3f}: {step['template']}")
            else:
                misses += 1
                self.log(f"模板未出现，确认 {misses}/{required_misses}")
                if misses >= required_misses:
                    return
            self._interruptible_wait(interval)
        if not self.stop_event.is_set():
            self._handle_timeout(step, f"等待模板消失超时: {step['template']}")

    def _wait_region_still(self, step: dict) -> None:
        region_values = step.get("region", [0.0, 0.0, 1.0, 1.0])
        if not isinstance(region_values, list) or len(region_values) != 4:
            raise ValueError("wait_region_still requires [left, top, right, bottom]")
        region = RelativeRegion.from_ltrb(*(float(value) for value in region_values)).clamp()
        timeout = float(step.get("timeout", 480.0))
        interval = float(step.get("interval", 5.0))
        stable_count = int(step.get("stable_count", 3))
        diff_threshold = float(step.get("diff_threshold", 4.0))
        initial_delay = step.get("initial_delay", [3.0, 5.0])
        if isinstance(initial_delay, list) and len(initial_delay) == 2:
            self._interruptible_delay(float(initial_delay[0]), float(initial_delay[1]))

        previous = self._capture_region_gray(region)
        stable_seen = 0
        deadline = time.monotonic() + timeout
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            self._interruptible_wait(interval)
            current = self._capture_region_gray(region)
            diff = float(np.mean(cv2.absdiff(previous, current)))
            previous = current
            if diff <= diff_threshold:
                stable_seen += 1
                self.log(f"区域趋于稳定 diff={diff:.2f}，确认 {stable_seen}/{stable_count}")
                if stable_seen >= stable_count:
                    return
            else:
                stable_seen = 0
                self.log(f"区域仍在变化 diff={diff:.2f}")
        if not self.stop_event.is_set():
            self._handle_timeout(step, "等待区域稳定超时")

    def _wait_battle_complete(self, step: dict) -> None:
        template_paths = self._battle_template_paths(step)
        missing = [path.name for path in template_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"缺少战斗标志模板: {', '.join(missing)}。请在战斗中框选稳定的战斗标志并保存为对应模板名。")
        threshold = float(step.get("threshold", 0.86))
        interval = float(step.get("interval", 5.0))
        end_interval = step.get("end_interval", [interval, interval])
        enter_timeout = float(step.get("enter_timeout", 240.0))
        battle_timeout = float(step.get("battle_timeout", 180.0))
        confirm_gone = int(step.get("confirm_gone", 3))
        search_region = self._optional_search_region(step)

        self.log(f"等待进入战斗，最多 {enter_timeout:.0f}s")
        mode = str(step.get("match_mode", "color"))
        entered = self._wait_battle_enter_with_optional_clicks(
            step=step,
            template_paths=template_paths,
            threshold=threshold,
            interval=interval,
            timeout=enter_timeout,
            search_region=search_region,
            mode=mode,
        )
        if not entered:
            if self.stop_event.is_set():
                return
            self._handle_timeout(step, f"等待进入战斗超时: {self._battle_template_names(step)}")
            return

        self.log(f"已识别到战斗标志，等待战斗结束，最多 {battle_timeout:.0f}s")
        misses = 0
        battle_started_at = time.monotonic()
        deadline = time.monotonic() + battle_timeout
        round_region = self._optional_relative_region(step, "round_ocr_region")
        round_ocr_interval = float(step.get("round_ocr_interval", max(interval, 3.0)))
        round_stall_seconds = float(step.get("round_stall_alert_seconds", 0.0))
        max_round_alert = int(step.get("max_round_alert", 0))
        battle_alert_seconds = float(step.get("battle_alert_seconds", 0.0))
        last_round: int | None = None
        last_round_changed_at = battle_started_at
        last_round_ocr_at = 0.0
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            now = time.monotonic()
            match = self._find_any_template(
                template_paths,
                threshold=threshold,
                mode=mode,
                search_region=search_region,
            )
            if match:
                misses = 0
                self.log(f"战斗中，标志匹配 {match.confidence:.3f}")
                if battle_alert_seconds > 0 and now - battle_started_at >= battle_alert_seconds:
                    self._handle_timeout(step, f"疑似验证/战斗异常: 战斗持续 {now - battle_started_at:.0f}s")
                    return
                if round_region is not None and now - last_round_ocr_at >= round_ocr_interval:
                    last_round_ocr_at = now
                    round_number, detail = self._ocr_battle_round(round_region, float(step.get("round_min_confidence", 0.45)))
                    if round_number is not None:
                        if last_round != round_number:
                            last_round = round_number
                            last_round_changed_at = now
                            self.log(f"识别到战斗回合: 第{round_number}回合")
                        else:
                            stalled = now - last_round_changed_at
                            self.log(f"战斗回合未变化: 第{round_number}回合 {stalled:.0f}s")
                            if round_stall_seconds > 0 and stalled >= round_stall_seconds:
                                self._handle_timeout(step, f"疑似验证: 第{round_number}回合停留 {stalled:.0f}s")
                                return
                        if max_round_alert > 0 and round_number >= max_round_alert:
                            self._handle_timeout(step, f"疑似验证/战斗异常: 已到第{round_number}回合")
                            return
                    elif detail:
                        self.log(f"战斗回合 OCR 未解析: {detail[:60]}")
            else:
                misses += 1
                self.log(f"战斗标志未出现，结束确认 {misses}/{confirm_gone}")
                if misses >= confirm_gone:
                    self._interruptible_delay(*tuple(step.get("post_delay", [2.0, 4.0])))
                    self._increment_progress_if_configured(step)
                    return
            if isinstance(end_interval, list) and len(end_interval) == 2:
                self._interruptible_wait(self.human.random.uniform(float(end_interval[0]), float(end_interval[1])))
            else:
                self._interruptible_wait(interval)
        if not self.stop_event.is_set():
            self._handle_timeout(step, f"等待战斗结束超时: {self._battle_template_names(step)}")

    def _wait_battle_enter_with_optional_clicks(
        self,
        step: dict,
        template_paths: list[Path],
        threshold: float,
        interval: float,
        timeout: float,
        search_region: RelativeRegion | None,
        mode: str,
    ) -> bool:
        retry_clicks = step.get("enter_retry_clicks")
        if not isinstance(retry_clicks, list) or not retry_clicks:
            return self._wait_any_template_state(
                template_paths,
                threshold=threshold,
                interval=interval,
                timeout=timeout,
                present=True,
                search_region=search_region,
                mode=mode,
            )

        retry_delay = step.get("enter_retry_delay", [3.0, 4.5])
        if not isinstance(retry_delay, list) or len(retry_delay) != 2:
            retry_delay = [3.0, 4.5]
        next_retry_at = time.monotonic() + self.human.random.uniform(float(retry_delay[0]), float(retry_delay[1]))
        deadline = time.monotonic() + timeout
        retry_count = 0

        while not self.stop_event.is_set() and time.monotonic() < deadline:
            match = self._find_any_template(
                template_paths,
                threshold=threshold,
                mode=mode,
                search_region=search_region,
            )
            if match:
                self.log(f"模板出现 {match.confidence:.3f}")
                return True

            now = time.monotonic()
            if now >= next_retry_at:
                for retry_step in retry_clicks:
                    if not isinstance(retry_step, dict):
                        continue
                    conditions = retry_step.get("conditions")
                    if isinstance(conditions, dict) and conditions:
                        matched, mismatches = self._conditions_match(conditions)
                        if not matched:
                            self.log(
                                f"skip enter retry click {retry_step.get('label', '')}: "
                                + "; ".join(mismatches)
                            )
                            continue
                    self.log(f"enter retry click {retry_count + 1}: {retry_step.get('label', '')}")
                    self._region_click(retry_step)
                    retry_count += 1
                    break
                next_retry_at = time.monotonic() + self.human.random.uniform(float(retry_delay[0]), float(retry_delay[1]))

            self._interruptible_wait(interval)
        return False

    def _swimming_task_loop(self, step: dict) -> None:
        template_paths = self._battle_template_paths(step)
        missing = [path.name for path in template_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing battle marker template: {', '.join(missing)}")

        task_region = self._optional_relative_region(step, "task_region")
        if task_region is None:
            task_region = RelativeRegion.from_ltrb(0.82, 0.30, 0.96, 0.48).clamp()

        threshold = float(step.get("threshold", 0.8))
        search_region = self._optional_search_region(step)
        mode = str(step.get("match_mode", "color"))
        check_interval = float(step.get("check_interval", 0.65))
        battle_timeout = float(step.get("battle_timeout", 240.0))
        confirm_gone = max(1, int(step.get("confirm_gone", 1)))
        max_duration = float(step.get("max_duration", 1800.0))
        max_task_clicks = max(1, int(step.get("max_task_clicks", 300)))
        idle_delay = step.get("idle_delay", [2.0, 4.0])
        if not isinstance(idle_delay, list) or len(idle_delay) != 2:
            idle_delay = [2.0, 4.0]

        started_at = time.monotonic()
        task_clicks = 0
        self.log(
            f"swimming loop started: max_duration={max_duration:.0f}s, "
            f"max_task_clicks={max_task_clicks}, idle={float(idle_delay[0]):.1f}-{float(idle_delay[1]):.1f}s"
        )

        while not self.stop_event.is_set():
            if time.monotonic() - started_at >= max_duration:
                self.log("swimming loop finished: max duration reached")
                return
            if task_clicks >= max_task_clicks:
                self.log("swimming loop finished: max task clicks reached")
                return

            match = self._find_any_template(
                template_paths,
                threshold=threshold,
                mode=mode,
                search_region=search_region,
            )
            if match:
                self.log(f"swimming battle detected {match.confidence:.3f}, wait until battle ends")
                self._wait_current_battle_gone(
                    template_paths=template_paths,
                    threshold=threshold,
                    mode=mode,
                    search_region=search_region,
                    interval=check_interval,
                    battle_timeout=battle_timeout,
                    confirm_gone=confirm_gone,
                    step=step,
                )
                continue

            pre_delay = tuple(step.get("pre_delay", [0.08, 0.22]))
            post_delay = tuple(step.get("post_delay", [0.02, 0.08]))
            padding_ratio = float(step.get("padding_ratio", 0.05))
            tap_mode = str(step.get("tap_mode", "tap"))
            x, y = self._tap_region(
                task_region,
                pre_delay=pre_delay,
                post_delay=post_delay,
                padding_ratio=padding_ratio,
                tap_mode=tap_mode,
            )
            task_clicks += 1
            self._log_click_detail(
                label=str(step.get("label") or "swimming_task_loop"),
                source=f"swimming_task_click_{task_clicks}",
                x=x,
                y=y,
                region=task_region,
                pre_delay=pre_delay,
                post_delay=post_delay,
                padding_ratio=padding_ratio,
                tap_mode=tap_mode,
            )
            self.log(f"swimming task click {task_clicks}/{max_task_clicks}: ({x}, {y})")
            self._interruptible_wait(self.human.random.uniform(float(idle_delay[0]), float(idle_delay[1])))

    def _wait_current_battle_gone(
        self,
        template_paths: list[Path],
        threshold: float,
        mode: str,
        search_region: RelativeRegion | None,
        interval: float,
        battle_timeout: float,
        confirm_gone: int,
        step: dict,
    ) -> None:
        misses = 0
        deadline = time.monotonic() + battle_timeout
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            match = self._find_any_template(
                template_paths,
                threshold=threshold,
                mode=mode,
                search_region=search_region,
            )
            if match:
                misses = 0
            else:
                misses += 1
                self.log(f"swimming battle gone confirm {misses}/{confirm_gone}")
                if misses >= confirm_gone:
                    self._interruptible_delay(*tuple(step.get("battle_end_delay", [0.3, 0.8])))
                    return
            self._interruptible_wait(interval)
        if not self.stop_event.is_set():
            self._handle_timeout(step, "swimming battle wait timeout")

    def _ocr_battle_round(self, region: RelativeRegion, min_confidence: float) -> tuple[int | None, str]:
        image = self._capture_color_region(region)
        with self._ocr_lock:
            if self._ocr_engine is None:
                from rapidocr_onnxruntime import RapidOCR

                self._ocr_engine = RapidOCR()
            result, _elapsed = self._ocr_engine(image)

        parts: list[str] = []
        if result:
            for item in result:
                if len(item) < 3:
                    continue
                text = str(item[1]).strip()
                try:
                    confidence = float(item[2])
                except (TypeError, ValueError):
                    confidence = 0.0
                if text and confidence >= min_confidence:
                    parts.append(text)
        detail = " ".join(parts)
        normalized = re.sub(r"\s+", "", detail)
        match = re.search(r"第?([0-9]{1,2})回合", normalized)
        if match:
            return int(match.group(1)), detail
        chinese_digits = {
            "一": 1,
            "二": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
            "十": 10,
        }
        match = re.search(r"第?([一二三四五六七八九十])回合", normalized)
        if match:
            return chinese_digits.get(match.group(1)), detail
        return None, detail

    def _wait_any_template_state(
        self,
        template_paths: list[Path],
        threshold: float,
        interval: float,
        timeout: float,
        present: bool,
        search_region: RelativeRegion | None,
        mode: str,
    ) -> bool:
        deadline = time.monotonic() + timeout
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            match = self._find_any_template(
                template_paths,
                threshold=threshold,
                mode=mode,
                search_region=search_region,
            )
            if bool(match) is present:
                if match:
                    self.log(f"模板出现 {match.confidence:.3f}")
                return True
            self._interruptible_wait(interval)
        return False

    def _find_any_template(
        self,
        template_paths: list[Path],
        threshold: float,
        mode: str,
        search_region: RelativeRegion | None,
    ):
        best_match = None
        for template_path in template_paths:
            match = self.matcher.find_template(
                template_path,
                threshold=threshold,
                mode=mode,
                search_region=search_region,
            )
            if match and (best_match is None or match.confidence > best_match.confidence):
                best_match = match
        return best_match

    def _battle_template_paths(self, step: dict) -> list[Path]:
        names = step.get("templates")
        if not isinstance(names, list) or not names:
            names = [step["template"]]
        return [self.templates_dir / str(name) for name in names]

    @staticmethod
    def _battle_template_names(step: dict) -> str:
        names = step.get("templates")
        if isinstance(names, list) and names:
            return ", ".join(str(name) for name in names)
        return str(step.get("template", ""))

    def _capture_region_gray(self, region: RelativeRegion) -> np.ndarray:
        image = self.matcher.screenshot()
        height, width = image.shape[:2]
        left = max(0, min(width - 1, int(region.left * width)))
        top = max(0, min(height - 1, int(region.top * height)))
        right = max(left + 1, min(width, int(region.right * width)))
        bottom = max(top + 1, min(height, int(region.bottom * height)))
        crop = image[top:bottom, left:right]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)

    def _capture_color_region(self, region: RelativeRegion | None) -> np.ndarray:
        image = self.matcher.screenshot()
        if region is None:
            return image
        height, width = image.shape[:2]
        region = region.clamp()
        left = max(0, min(width - 1, int(region.left * width)))
        top = max(0, min(height - 1, int(region.top * height)))
        right = max(left + 1, min(width, int(region.right * width)))
        bottom = max(top + 1, min(height, int(region.bottom * height)))
        return image[top:bottom, left:right].copy()

    def _ocr_contains_text(
        self,
        image: np.ndarray,
        targets: list[str],
        min_confidence: float,
        require_all: bool = False,
    ) -> tuple[bool, str]:
        with self._ocr_lock:
            if self._ocr_engine is None:
                from rapidocr_onnxruntime import RapidOCR

                self._ocr_engine = RapidOCR()
            result, _elapsed = self._ocr_engine(image)

        detail = self._ocr_detail(result, min_confidence)
        normalized = re.sub(r"\s+", "", detail)
        normalized_targets = [re.sub(r"\s+", "", target) for target in targets]
        if require_all:
            found = all(target in normalized for target in normalized_targets)
        else:
            found = any(target in normalized for target in normalized_targets)
        return found, detail

    @staticmethod
    def _ocr_detail(result, min_confidence: float) -> str:
        parts: list[str] = []
        if result:
            for item in result:
                if len(item) < 3:
                    continue
                text = str(item[1]).strip()
                try:
                    confidence = float(item[2])
                except (TypeError, ValueError):
                    confidence = 0.0
                if text and confidence >= min_confidence:
                    parts.append(text)
        return " ".join(parts)

    def _pre_swipes_for_step(self, step: dict) -> int:
        base = max(0, int(step.get("pre_swipes", 0)))
        key = step.get("progress_state")
        if not key:
            return base
        completed = self._load_progress(str(key))
        threshold = int(step.get("progress_threshold", 20))
        if completed >= threshold:
            value = max(0, int(step.get("pre_swipes_after_progress", base)))
        else:
            value = max(0, int(step.get("pre_swipes_before_progress", base)))
        self.log(f"progress {key}: completed={completed}, threshold={threshold}, pre_swipes={value}")
        return value

    def _allow_retry_swipe(self, step: dict) -> bool:
        key = step.get("progress_state")
        if not key:
            return True
        completed = self._load_progress(str(key))
        threshold = int(step.get("progress_threshold", 20))
        if completed >= threshold:
            return bool(step.get("retry_swipe_after_progress", True))
        return bool(step.get("retry_swipe_before_progress", False))

    def _increment_progress_if_configured(self, step: dict) -> None:
        key = step.get("progress_increment_state") or step.get("progress_state")
        if not key:
            return
        completed = self._load_progress(str(key)) + int(step.get("progress_increment", 1))
        completed = max(0, completed)
        self._save_progress(str(key), completed)
        self.log(f"progress {key}: completed={completed}")

    def _progress_path(self, key: str) -> Path:
        safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in key)
        return self.state_dir / f"{safe}.json"

    def _load_progress(self, key: str) -> int:
        path = self._progress_path(key)
        if not path.exists():
            return 0
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            self.log(f"progress {key}: state unreadable, assume 0 ({exc})")
            return 0
        saved_date = str(data.get("date", ""))
        today = datetime.now().strftime("%Y-%m-%d")
        if saved_date and saved_date != today:
            self.log(f"progress {key}: saved date {saved_date} != {today}, assume 0")
            return 0
        try:
            return max(0, int(data.get("completed", 0)))
        except (TypeError, ValueError):
            return 0

    def _save_progress(self, key: str, completed: int) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "task": key,
            "completed": max(0, int(completed)),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._progress_path(key).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _optional_search_region(step: dict) -> RelativeRegion | None:
        return WorkflowRunner._optional_relative_region(step, "search_region")

    @staticmethod
    def _optional_fixed_click_region(step: dict) -> RelativeRegion | None:
        return WorkflowRunner._optional_relative_region(step, "fixed_click_region")

    @staticmethod
    def _relative_region_from_values(values: object) -> RelativeRegion | None:
        if not isinstance(values, list) or len(values) != 4:
            return None
        return RelativeRegion.from_ltrb(*(float(value) for value in values)).clamp()

    @staticmethod
    def _optional_relative_region(step: dict, key: str) -> RelativeRegion | None:
        values = step.get(key)
        return WorkflowRunner._relative_region_from_values(values)

    @staticmethod
    def _click_region(match_region: RelativeRegion, click_area: object) -> RelativeRegion:
        """Map an optional [left, top, right, bottom] sub-area into a match."""
        if not isinstance(click_area, list) or len(click_area) != 4:
            return match_region
        left, top, right, bottom = (float(value) for value in click_area)
        return RelativeRegion.from_ltrb(
            match_region.left + match_region.width * left,
            match_region.top + match_region.height * top,
            match_region.left + match_region.width * right,
            match_region.top + match_region.height * bottom,
        ).clamp()

    def _click_match(self, match, step: dict) -> tuple[int, int]:
        label = str(step.get("label") or step.get("template") or step.get("type") or "")
        fixed_region = self._optional_fixed_click_region(step)
        if fixed_region is not None:
            pre_delay = tuple(step.get("pre_delay", [0.12, 0.35]))
            post_delay = tuple(step.get("post_delay", [0.25, 0.65]))
            padding_ratio = float(step.get("padding_ratio", 0.18))
            x, y = self.human.random_tap(
                fixed_region,
                pre_delay=pre_delay,
                post_delay=post_delay,
                padding_ratio=padding_ratio,
            )
            self._log_click_detail(
                label=label,
                source="template_fixed_region",
                x=x,
                y=y,
                region=fixed_region,
                pre_delay=pre_delay,
                post_delay=post_delay,
                padding_ratio=padding_ratio,
                match_box=match.pixel_box,
            )
            self._maybe_save_debug_screenshot(step, "after_fixed_region_click")
            return x, y
        offset = step.get("click_offset")
        if isinstance(offset, list) and len(offset) == 2:
            left, top, right, bottom = match.pixel_box
            center_x = int((left + right) * 0.5 + float(offset[0]))
            center_y = int((top + bottom) * 0.5 + float(offset[1]))
            drift = int(step.get("drift_px", 16))
            x = center_x + self.human.random.randint(-drift, drift)
            y = center_y + self.human.random.randint(-drift, drift)
            pre_delay = tuple(step.get("pre_delay", [0.12, 0.35]))
            post_delay = tuple(step.get("post_delay", [0.25, 0.65]))
            self.human.sleep_random(pre_delay)
            self.device.tap(x, y)
            self.human.sleep_random(post_delay)
            self._log_click_detail(
                label=label,
                source="template_offset",
                x=x,
                y=y,
                base=(center_x, center_y),
                drift_px=drift,
                pre_delay=pre_delay,
                post_delay=post_delay,
                match_box=match.pixel_box,
            )
            self._maybe_save_debug_screenshot(step, "after_template_click")
            return x, y
        click_region = self._click_region(match.region, step.get("click_area"))
        pre_delay = tuple(step.get("pre_delay", [0.12, 0.35]))
        post_delay = tuple(step.get("post_delay", [0.25, 0.65]))
        x, y = self.human.random_tap(
            click_region,
            pre_delay=pre_delay,
            post_delay=post_delay,
        )
        self._log_click_detail(
            label=label,
            source="template_match_area",
            x=x,
            y=y,
            region=click_region,
            pre_delay=pre_delay,
            post_delay=post_delay,
            match_box=match.pixel_box,
        )
        self._maybe_save_debug_screenshot(step, "after_template_click")
        return x, y

    def _tap_match_offset_region(self, match, offsets: list, step: dict) -> tuple[int, int]:
        left, top, _right, _bottom = match.pixel_box
        dx1, dy1, dx2, dy2 = (int(float(value)) for value in offsets)
        screen = self.device.screen_size_for_input()
        box_left = max(0, min(screen.width - 1, left + min(dx1, dx2)))
        box_top = max(0, min(screen.height - 1, top + min(dy1, dy2)))
        box_right = max(box_left + 1, min(screen.width, left + max(dx1, dx2)))
        box_bottom = max(box_top + 1, min(screen.height, top + max(dy1, dy2)))
        region = RelativeRegion.from_ltrb(
            box_left / screen.width,
            box_top / screen.height,
            box_right / screen.width,
            box_bottom / screen.height,
        ).clamp()
        x, y = self._tap_region(
            region,
            pre_delay=tuple(step.get("pre_delay", [0.12, 0.35])),
            post_delay=tuple(step.get("post_delay", [0.25, 0.65])),
            padding_ratio=float(step.get("padding_ratio", 0.18)),
        )
        self._log_click_detail(
            label=str(step.get("label") or step.get("template") or step.get("type") or ""),
            source="template_dynamic_offset_region",
            x=x,
            y=y,
            region=region,
            pre_delay=tuple(step.get("pre_delay", [0.12, 0.35])),
            post_delay=tuple(step.get("post_delay", [0.25, 0.65])),
            padding_ratio=float(step.get("padding_ratio", 0.18)),
            match_box=match.pixel_box,
        )
        self._maybe_save_annotated_debug_screenshot(
            step,
            "after_dynamic_match_click",
            match_box=match.pixel_box,
            click_box=(box_left, box_top, box_right, box_bottom),
            point=(x, y),
        )
        return x, y

    def _swipe(self, step: dict) -> None:
        directions = {
            "up": (RelativePoint(0.52, 0.80), RelativePoint(0.48, 0.25)),
            "down": (RelativePoint(0.48, 0.25), RelativePoint(0.52, 0.80)),
            "left": (RelativePoint(0.82, 0.52), RelativePoint(0.20, 0.48)),
            "right": (RelativePoint(0.20, 0.48), RelativePoint(0.82, 0.52)),
        }
        direction = step.get("direction", "up")
        if direction not in directions:
            raise ValueError(f"不支持的滑动方向: {direction}")
        start, end = directions[direction]
        start_x, start_y = self.human.to_absolute(start)
        end_x, end_y = self.human.to_absolute(end)
        duration_min = int(step.get("duration_min", 650))
        duration_max = int(step.get("duration_max", 1100))
        jitter = float(step.get("jitter", 0.006))
        self.log(
            f"swipe detail: label={step.get('label', step.get('type', ''))} | "
            f"direction={direction} | start=({start_x},{start_y}) | end=({end_x},{end_y}) | "
            f"duration={duration_min}-{duration_max}ms | jitter={jitter:.4f}"
        )
        self.human.natural_swipe(
            start,
            end,
            duration_ms=(duration_min, duration_max),
            jitter_ratio=jitter,
        )

    def _fast_swipe(self, step: dict) -> None:
        directions = {
            "up": (RelativePoint(0.52, 0.78), RelativePoint(0.50, 0.36)),
            "down": (RelativePoint(0.50, 0.36), RelativePoint(0.52, 0.78)),
            "left": (RelativePoint(0.78, 0.52), RelativePoint(0.30, 0.50)),
            "right": (RelativePoint(0.30, 0.50), RelativePoint(0.78, 0.52)),
        }
        direction = step.get("direction", "up")
        start, end = directions.get(direction, directions["up"])
        x1, y1 = self.human.to_absolute(start)
        x2, y2 = self.human.to_absolute(end)
        duration = self.human.random.randint(int(step.get("duration_min", 260)), int(step.get("duration_max", 420)))
        if not self.stop_event.is_set():
            self.log(
                f"fast swipe detail: label={step.get('label', step.get('type', ''))} | "
                f"direction={direction} | start=({x1},{y1}) | end=({x2},{y2}) | duration={duration}ms"
            )
            self.device.swipe(x1, y1, x2, y2, duration)

    def _handle_timeout(self, step: dict, message: str) -> None:
        if step.get("screenshot_on_timeout", True):
            path = self._save_failure_screenshot()
            self.log(f"{message}??????: {path}")
            self._save_failure_context(path, step, message)
        strategy = str(step.get("on_timeout", "error"))
        if strategy == "continue":
            self.log(f"{message}????????")
            return
        if strategy == "restart_flow":
            self.log(f"{message}???????????")
            raise RestartWorkflow(message)
        if strategy == "pause":
            self.stop_event.set()
            raise TimeoutError(f"{message}??????????")
        raise TimeoutError(message)

    def _save_failure_screenshot(self) -> Path:
        self.failures_dir.mkdir(exist_ok=True)
        path = self.failures_dir / f"failure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path.write_bytes(self.device.screenshot_png())
        return path

    def _save_failure_context(self, screenshot_path: Path, step: dict, message: str) -> None:
        context_path = screenshot_path.with_suffix(".txt")
        try:
            label = step.get("label") or step.get("template") or step.get("type")
            lines = [
                f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"message: {message}",
                f"step: {label}",
                f"step_json: {json.dumps(step, ensure_ascii=False)}",
                "",
                "recent_log:",
            ]
            log_dir = self.templates_dir.parent / "logs"
            log_files = sorted(log_dir.glob("toolbox_*.log"), key=lambda path: path.stat().st_mtime)
            if log_files:
                recent = log_files[-1].read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
                lines.extend(recent)
            context_path.write_text("\n".join(lines), encoding="utf-8")
            self.log(f"failure context saved: {context_path}")
        except Exception as exc:
            self.log(f"failure context save failed: {exc}")



    def _maybe_save_debug_screenshot(self, step: dict, suffix: str) -> None:
        if not step.get("debug_screenshot_after", False):
            return
        self.failures_dir.mkdir(exist_ok=True)
        label = str(step.get("label") or step.get("template") or step.get("type") or "step")
        safe_label = "".join(char if char.isalnum() or char in "-_" else "_" for char in label)[:32] or "step"
        path = self.failures_dir / f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_label}_{suffix}.png"
        if self._queue_debug_screenshot(path):
            self.log(f"debug screenshot queued: {path}")
        else:
            self.log("debug screenshot skipped: queue busy")

    def _maybe_save_annotated_debug_screenshot(
        self,
        step: dict,
        suffix: str,
        match_box: tuple[int, int, int, int],
        click_box: tuple[int, int, int, int],
        point: tuple[int, int],
    ) -> None:
        if not step.get("debug_screenshot_after", False):
            return
        self.failures_dir.mkdir(exist_ok=True)
        label = str(step.get("label") or step.get("template") or step.get("type") or "step")
        safe_label = "".join(char if char.isalnum() or char in "-_" else "_" for char in label)[:32] or "step"
        path = self.failures_dir / f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_label}_{suffix}_annotated.png"
        try:
            frame = self.frame_source.latest().image.copy()
            cv2.rectangle(frame, (match_box[0], match_box[1]), (match_box[2], match_box[3]), (0, 255, 255), 3)
            cv2.rectangle(frame, (click_box[0], click_box[1]), (click_box[2], click_box[3]), (0, 0, 255), 4)
            cv2.circle(frame, point, 14, (0, 0, 255), -1)
            ok, encoded = cv2.imencode(".png", frame)
            if ok:
                path.write_bytes(encoded.tobytes())
                self.log(f"annotated debug screenshot saved: {path}")
        except Exception as exc:
            self.log(f"annotated debug screenshot failed: {exc}")

    def _region_to_pixel_box(self, region: RelativeRegion) -> tuple[int, int, int, int]:
        screen = self.device.screen_size_for_input()
        return (
            int(region.left * screen.width),
            int(region.top * screen.height),
            int(region.right * screen.width),
            int(region.bottom * screen.height),
        )

    def _queue_debug_screenshot(self, path: Path) -> bool:
        with self._debug_lock:
            self._debug_futures = [future for future in self._debug_futures if not future.done()]
            if len(self._debug_futures) >= 2:
                return False
            future = self._debug_executor.submit(self._write_debug_screenshot, path)
            self._debug_futures.append(future)
            return True

    def _write_debug_screenshot(self, path: Path) -> None:
        try:
            path.write_bytes(self.device.screenshot_png())
        except Exception:
            # Debug screenshots are diagnostic only; never break the main workflow.
            return

    def _interruptible_delay(self, minimum: float, maximum: float) -> None:
        seconds = self.human.random.uniform(minimum, maximum)
        self.log(f"随机等待 {seconds:.2f} 秒")
        self._interruptible_wait(seconds)

    def _interruptible_wait(self, seconds: float) -> None:
        self.stop_event.wait(max(0.0, seconds))

    @staticmethod
    def describe_step(step: dict) -> str:
        prefix = "未配置/跳过 - " if step.get("disabled", False) else ""
        action = step.get("type")
        if action == "template_click":
            return prefix + f"识别并点击 {step.get('template', '')} (阈值 {step.get('threshold', 0.88)})"
        if action == "wait_template":
            return prefix + f"等待模板出现 {step.get('template', '')}，最多 {step.get('timeout', 10)}s"
        if action == "wait_template_click":
            return prefix + f"等待模板出现并点击 {step.get('template', '')}，最多 {step.get('timeout', 10)}s"
        if action == "template_scroll_click":
            offset = step.get("click_offset")
            offset_text = f" 偏移{offset}" if isinstance(offset, list) else ""
            pre_swipes = int(step.get("pre_swipes", 0))
            pre_text = f"，先滑{pre_swipes}次" if pre_swipes > 0 else ""
            return prefix + f"找模板点击/滑动重试 {step.get('template', '')}{offset_text}{pre_text}，最多{step.get('attempts', 2)}次"
        if action == "template_column_region_click":
            locked = "锁定列" if step.get("lock_column", False) else "每次判列"
            pre_swipes = int(step.get("pre_swipes", 0))
            pre_text = f"，先滑{pre_swipes}次" if pre_swipes > 0 else ""
            return prefix + f"识别卡片按列点范围 {step.get('template', '')}，{locked}{pre_text}"
        if action == "detect_template_flag":
            return prefix + f"detect flag {step.get('flag', '')} by {step.get('template', '')}"
        if action == "detect_text_flag":
            return prefix + f"detect flag {step.get('flag', '')} by text {step.get('text', '')}"
        if action == "detect_text_flags":
            flags = step.get("flags")
            if isinstance(flags, list):
                names = ", ".join(str(item.get("flag", "")) for item in flags if isinstance(item, dict))
            else:
                names = ""
            return prefix + f"detect flags by OCR: {names}"
        if action == "wait_text":
            texts = step.get("texts")
            if isinstance(texts, list):
                text = " + ".join(str(value) for value in texts)
            else:
                text = str(step.get("text", ""))
            return prefix + f"等待区域文字 {text}，最多 {float(step.get('timeout', 8.0)):.0f}s"
        if action == "conditional_region_click":
            return prefix + f"conditional click {step.get('label', '')} if {step.get('flag', '')}"
        if action == "wait_template_gone":
            return prefix + f"等待模板消失 {step.get('template', '')}，最多 {step.get('timeout', 300)}s"
        if action == "wait_region_still":
            label = step.get("label", "区域")
            initial = step.get("initial_delay", [0, 0])
            interval = float(step.get("interval", 5.0))
            stable_count = int(step.get("stable_count", 3))
            timeout = float(step.get("timeout", 480.0))
            if isinstance(initial, list) and len(initial) == 2:
                initial_text = f"先等{float(initial[0]):.0f}-{float(initial[1]):.0f}s，"
            else:
                initial_text = ""
            return prefix + f"{label}: {initial_text}每{interval:.0f}s比对截图，连续{stable_count}次不动则完成，最多{timeout:.0f}s"
        if action == "wait_battle_complete":
            templates = step.get("templates")
            template_text = ", ".join(str(name) for name in templates) if isinstance(templates, list) and templates else step.get("template", "")
            return prefix + (
                f"等待战斗完成 {template_text}: "
                f"寻路最多{float(step.get('enter_timeout', 240)):.0f}s，"
                f"战斗最多{float(step.get('battle_timeout', 180)):.0f}s，"
                f"每{float(step.get('interval', 5)):.0f}s识别"
            )
        if action == "swimming_task_loop":
            return prefix + (
                f"游泳循环: 非战斗每 {step.get('idle_delay', [2, 4])} 秒点任务栏，"
                f"最多 {float(step.get('max_duration', 1800)):.0f}s"
            )
        if action == "region_click":
            label = step.get("label", "")
            pixel_point = step.get("pixel_point")
            point = step.get("point")
            drift = step.get("drift_px")
            wait = step.get("post_delay", [0.25, 0.65])
            wait_text = f" 等待{float(wait[0]):.2f}-{float(wait[1]):.2f}s" if len(wait) == 2 else ""
            if isinstance(pixel_point, list) and len(pixel_point) == 2:
                return prefix + f"坐标点击 {label} 像素({int(pixel_point[0])}, {int(pixel_point[1])}) ±{drift}px{wait_text}"
            if isinstance(point, list) and len(point) == 2:
                return prefix + f"坐标点击 {label} ({point[0]:.3f}, {point[1]:.3f}) ±{drift}px{wait_text}"
            return (prefix + f"固定相对区域随机点击 {label}{wait_text}").strip()
        if action == "region_click_until_template":
            return prefix + (
                f"范围点击 {step.get('label', '')} 并验证 {step.get('template', '')}，"
                f"最多{int(step.get('attempts', 3))}次"
            )
        if action == "delay":
            return prefix + f"随机等待 {step.get('min', 0.5)}-{step.get('max', 1.2)} 秒"
        if action == "swipe":
            names = {"up": "向上", "down": "向下", "left": "向左", "right": "向右"}
            return prefix + f"{names.get(step.get('direction'), step.get('direction'))}自然滑动"
        if action == "observe_state":
            return prefix + f"观察当前状态 expected={step.get('expected', '')}"
        return prefix + str(action)
