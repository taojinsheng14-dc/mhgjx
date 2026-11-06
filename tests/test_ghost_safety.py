import json
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scrcpy_human_automation.device import AdbDevice
from scrcpy_human_automation.ghost_safety import GhostRecoveryRequired, GhostSafety
from scrcpy_human_automation.humanizer import RelativeRegion
from scrcpy_human_automation.runner import WorkflowRunner


class Clock:
    def __init__(self):
        self.now = 100.0

    def wait(self, seconds):
        self.now += max(0.01, seconds)


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.steps = json.loads((ROOT / "flows/钟馗抓鬼_视频帧测试.json").read_text(encoding="utf-8"))["steps"]
        self.r = WorkflowRunner(AdbDevice(), ROOT / "templates", log=Mock())
        self.addCleanup(lambda: self.r._debug_executor.shutdown(wait=False))
        self.s = GhostSafety(self.r, self.steps, {"max_recoveries": 2})
        self.r.ghost_safety = self.s
        self.clock = Clock()
        self.patcher = patch("scrcpy_human_automation.ghost_safety.time.monotonic", lambda: self.clock.now)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.r._interruptible_wait = self.clock.wait
        self.r._interruptible_delay = lambda lo, hi: self.clock.wait(lo)
        self.image = {"hud": True}
        self.s.fresh = Mock(side_effect=lambda: self.image)
        self.r.frame_source = SimpleNamespace(latest=lambda **kw: SimpleNamespace(image=self.image), close=lambda: None)
        self.s.battle = lambda im: im.get("battle", False)
        self.s.activity = lambda im: im.get("activity", False)
        self.s.normal_hud = lambda im: im.get("hud", False)
        self.s.nonbattle_item = lambda im: im.get("item", False)
        self.s.flag_window = lambda im: im.get("flag", False)
        self.s.role_match = lambda im, role: SimpleNamespace(confidence=1) if im.get(role) else None
        self.s.task_text = lambda im: im.get("text", "")
        self.box = RelativeRegion(0.52, 0.80, 0.53, 0.82)
        self.s.eye_icon = lambda im, step: self.box if im.get("eye") else None
        self.r._region_click = Mock()
        self.r._click_match = Mock(return_value=(1, 2))
        self.r._handle_timeout = Mock(side_effect=lambda step, message: (_ for _ in ()).throw(GhostRecoveryRequired(message)))
        self.r.device.ensure_connected = Mock()

    def test_battle_miss_requires_fresh_nonbattle_hud(self):
        self.image = {"battle": True, "hud": False}
        self.assertFalse(self.s.confirm_battle_end())
        self.image = {"hud": False}
        self.assertFalse(self.s.confirm_battle_end())
        self.image = {"hud": True}
        self.assertTrue(self.s.confirm_battle_end())
        self.assertTrue(self.s.battle_finished)
        self.r._region_click.assert_not_called()

    def test_battle_end_accepts_item_without_activity(self):
        self.image = {"hud": False, "item": True}
        self.assertTrue(self.s.confirm_battle_end())
        self.s.fresh.assert_called_once()
        self.r._region_click.assert_not_called()

    def test_round_marker_vetoes_visible_item(self):
        self.image = {"battle": True, "hud": True, "item": True}
        self.assertFalse(self.s.confirm_battle_end())
        self.assertFalse(self.s.battle_finished)

    def test_end_checks_run_concurrently_on_same_frame(self):
        barrier = threading.Barrier(3)
        frames = []
        def check(image):
            frames.append(image)
            barrier.wait(timeout=2)
            return False
        self.s.battle = check
        self.s.normal_hud = check
        self.s.nonbattle_item = check
        self.assertFalse(self.s.confirm_battle_end())
        self.assertEqual(len(frames), 3)
        self.assertTrue(all(image is self.image for image in frames))
        self.s.fresh.assert_called_once()

    def test_popup_close_is_deferred_and_uses_original_protection(self):
        start = self.clock.now
        self.assertFalse(self.s.before(self.s.step("close_activity")))
        self.assertEqual(self.clock.now, start)
        self.assertGreaterEqual(self.s.close_at - start, 9.5)
        self.assertLessEqual(self.s.close_at - start, 11.5)
        self.image = {"activity": True}
        close_times = []

        def click(step):
            close_times.append(self.clock.now)
            self.image = {"accept": True}

        self.r._region_click.side_effect = click
        self.s.wait_accept(self.s.step("accept"))
        self.assertEqual(len(close_times), 1)
        self.assertGreaterEqual(close_times[0] - start, 9.5)
        self.assertLess(self.clock.now - start, 13)
        self.r._click_match.assert_called_once()

    def test_old_popup_frame_cannot_authorize_close(self):
        self.image = {"activity": True}
        self.s.fresh.side_effect = lambda: {"accept": True}
        self.s.wait_accept(self.s.step("accept"))
        self.r._region_click.assert_not_called()
        self.r._click_match.assert_called_once()

    def test_persistent_popup_has_bounded_clicks(self):
        self.image = {"activity": True}
        with self.assertRaises(GhostRecoveryRequired):
            self.s.wait_accept(self.s.step("accept"))
        self.assertEqual(self.r._region_click.call_count, 3)
        self.assertEqual(self.s.fresh.call_count, 3)

    def test_stop_prevents_accept_and_close(self):
        self.r.stop_event.set()
        self.s.wait_accept(self.s.step("accept"))
        self.r._region_click.assert_not_called()
        self.r._click_match.assert_not_called()

    def test_existing_precise_task_does_not_consume_eye(self):
        self.image = {"hud": True, "text": "捉鬼 去大唐境外123,57处抓野鬼"}
        self.s.open_eye(self.s.step("eye"))
        self.r._region_click.assert_not_called()
        self.assertTrue(self.r.flags["need_fly_mount_for_ghost"])

    def test_missing_eye_never_clicks(self):
        self.image = {"hud": True, "text": "捉鬼 去建邺城123,57附近抓野鬼"}
        with self.assertRaises(GhostRecoveryRequired):
            self.s.open_eye(self.s.step("eye"))
        self.r._region_click.assert_not_called()

    def test_eye_success_updates_map_flags(self):
        self.image = {"hud": True, "eye": True, "text": "捉鬼 去五庄观123,57附近抓野鬼"}
        self.r._region_click.side_effect = lambda step: self.image.update(text="捉鬼 去五庄观91,154处抓野鬼")
        self.s.open_eye(self.s.step("eye"))
        self.r._region_click.assert_called_once()
        self.assertTrue(self.r.flags["need_fly_mount_for_ghost"])
        self.assertFalse(self.r.flags["fly_station_after_battle"])

    def test_eye_retry_requires_unchanged_task_and_visible_icon(self):
        self.image = {"hud": True, "eye": True, "text": "捉鬼 去建邺城123,57附近抓野鬼"}
        with self.assertRaises(GhostRecoveryRequired):
            self.s.open_eye(self.s.step("eye"))
        self.assertEqual(self.r._region_click.call_count, 2)

    def test_unknown_after_eye_never_retries(self):
        self.image = {"hud": True, "eye": True, "text": "捉鬼 去建邺城123,57附近抓野鬼"}
        self.r._region_click.side_effect = lambda step: self.image.update(text="无法识别", eye=False)
        with self.assertRaises(GhostRecoveryRequired):
            self.s.open_eye(self.s.step("eye"))
        self.r._region_click.assert_called_once()

    def test_station_blocked_when_flag_window_missing(self):
        self.r.flags["fly_station_after_battle"] = True
        with self.assertRaises(GhostRecoveryRequired):
            self.s.before(self.s.step("station"))
        self.r._region_click.assert_not_called()

    def test_station_accepts_item_when_activity_missing(self):
        self.image = {"hud": False, "item": True}
        self.s.flag_open_requested = True
        self.s.verify_station(self.s.step("station"))
        self.assertFalse(self.s.flag_open_requested)
        self.r._region_click.assert_not_called()

    def test_station_rejects_activity_popup_even_with_item(self):
        self.image = {"activity": True, "item": True}
        with self.assertRaises(GhostRecoveryRequired):
            self.s.verify_station(self.s.step("station"))

    def test_station_window_remains_requests_recovery(self):
        self.image = {"flag": True}
        with self.assertRaises(GhostRecoveryRequired):
            self.s.verify_station(self.s.step("station"))

    def test_recovery_to_battle_has_no_ground_click(self):
        self.image = {"battle": True}
        self.assertEqual(self.s.recover(self.s.roles["activity"], "test"), self.s.roles["battle"])
        self.r._region_click.assert_not_called()

    def test_recovery_precise_task_skips_eye_and_flying_toggle(self):
        self.image = {"hud": True, "text": "捉鬼 去大唐境外91,154处抓野鬼"}
        self.assertEqual(self.s.recover(self.s.roles["fight"], "test"), self.s.roles["route"])
        self.r._region_click.assert_not_called()

    def test_unowned_flag_window_does_not_trigger_station(self):
        self.image = {"flag": True}
        with self.assertRaisesRegex(TimeoutError, "无法确认"):
            self.s.recover(self.s.roles["activity"], "test")
        self.r._region_click.assert_not_called()

    def test_recovery_missing_task_is_not_assumed_empty(self):
        self.image = {"hud": True, "text": "任务"}
        with self.assertRaises(TimeoutError):
            self.s.recover(self.s.roles["accept"], "test")
        self.r._region_click.assert_not_called()

    def test_recovery_is_limited_per_round(self):
        self.image = {"battle": True}
        for _ in range(2):
            self.s.recover(self.s.roles["activity"], "test")
        with self.assertRaisesRegex(TimeoutError, "已达2次"):
            self.s.recover(self.s.roles["activity"], "test")

    def test_progress_only_once_until_next_accept(self):
        self.r._load_progress = Mock(return_value=10)
        self.r._save_progress = Mock()
        self.r._increment_progress_if_configured(self.s.step("battle"))
        self.r._increment_progress_if_configured(self.s.step("battle"))
        self.r._save_progress.assert_called_once_with("zhongkui_ghost", 11)
        self.s.after(self.s.step("accept"))
        self.r._increment_progress_if_configured(self.s.step("battle"))
        self.assertEqual(self.r._save_progress.call_count, 2)

    def test_recovery_only_enabled_in_video_ghost_flow(self):
        for name in ("钟馗抓鬼.json", "厢房.json", "皇宫飞贼.json"):
            data = json.loads((ROOT / "flows" / name).read_text(encoding="utf-8-sig"))
            self.assertFalse(any(s.get("ghost_safety") for s in data["steps"]))


class RunnerIntegrationTests(unittest.TestCase):
    def test_recovery_resumes_selected_step_and_continues(self):
        r = WorkflowRunner(AdbDevice(), ROOT / "templates", log=Mock())
        r.device.ensure_connected = Mock()
        r.device.screen_size_for_input = Mock(return_value=SimpleNamespace(width=2608, height=1200))
        r.frame_source = SimpleNamespace(close=Mock())
        steps = [{"ghost_safety": {"max_recoveries": 2}}, {}, {}]
        history = []

        def execute(step):
            history.append(next(i for i, candidate in enumerate(steps) if candidate is step))
            if len(history) == 1:
                raise GhostRecoveryRequired("test")

        r._run_step = execute
        safety = Mock()
        safety.before.return_value = True
        safety.recover.return_value = 1
        with patch("scrcpy_human_automation.runner.GhostSafety", return_value=safety):
            r.run(steps)
        self.assertEqual(history, [0, 1, 2])
        safety.recover.assert_called_once_with(0, "test")
        r.frame_source.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
