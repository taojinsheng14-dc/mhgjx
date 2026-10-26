import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scrcpy_human_automation.task_target import task_body_box, click_named_task


class TaskTargetTests(unittest.TestCase):
    def scene(self, titles):
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        result = []
        for index, title in enumerate(titles):
            y = index * 150 + 10
            image[y:y+20, 10:150] = (0, 230, 0)
            result.append(([[10,y],[150,y],[150,y+20],[10,y+20]], title, .99))
            result.append(([[10,y+30],[300,y+30],[300,y+50],[10,y+50]], "前去任务地点", .99))
        return image, result

    def test_first_and_second_row(self):
        for titles, expected in [(["皇宫飞贼", "师门任务"], 40),
                                 (["师门任务", "皇宫飞贼"], 190)]:
            image, result = self.scene(titles)
            self.assertEqual(task_body_box(image, result, "皇宫飞贼")[1], expected)

    def test_missing_ambiguous_or_non_green_title(self):
        for titles in [["师门任务"], ["皇宫飞贼", "皇宫飞贼"]]:
            image, result = self.scene(titles)
            self.assertIsNone(task_body_box(image, result, "皇宫飞贼"))
        image, result = self.scene(["皇宫飞贼"])
        image[:] = 255
        self.assertIsNone(task_body_box(image, result, "皇宫飞贼"))

    def test_never_uses_next_task_body(self):
        image, result = self.scene(["皇宫飞贼", "师门任务"])
        result.pop(1)
        self.assertIsNone(task_body_box(image, result, "皇宫飞贼"))

    def test_timeout_or_stop_never_clicks(self):
        runner = SimpleNamespace(stop_event=threading.Event(), _region_click=Mock(), _handle_timeout=Mock())
        click_named_task(runner, {"task_title": "皇宫飞贼", "task_timeout": 0})
        runner._handle_timeout.assert_called_once()
        runner._region_click.assert_not_called()
        runner.stop_event.set()
        runner._handle_timeout.reset_mock()
        click_named_task(runner, {"task_title": "皇宫飞贼"})
        runner._handle_timeout.assert_not_called()
        runner._region_click.assert_not_called()


if __name__ == "__main__":
    unittest.main()
