import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scrcpy_human_automation.coordinate_ocr import retry_coordinate_line
from scrcpy_human_automation.runner import WorkflowRunner


class CoordinateRetryTests(unittest.TestCase):
    def run_retry(self, texts, stopped=False, original="去建城1706附近抓"):
        box = [[5, 5], [150, 5], [150, 25], [5, 25]]
        engine = Mock(side_effect=[([[box, text, .99]], None) for text in texts])
        detail = "捉鬼 " + original + "野鬼 师门任务"
        result = retry_coordinate_line(
            np.zeros((80, 200, 3), dtype=np.uint8), [[box, original, .99]],
            detail, engine, WorkflowRunner._ocr_detail,
            WorkflowRunner._looks_like_ghost_coordinate, .4, lambda: stopped, Mock())
        return result, engine

    def test_retry_recovers_separator(self):
        result, engine = self.run_retry(["去建邺城170,6附近抓"])
        self.assertEqual(WorkflowRunner._extract_ghost_coordinate(result), (170, 6))
        self.assertNotIn("1706", result)
        engine.assert_called_once()

    def test_never_guesses_joined_digits(self):
        result, engine = self.run_retry(["去建城1706附近抓"] * 2)
        self.assertIsNone(WorkflowRunner._extract_ghost_coordinate(result))
        self.assertEqual(engine.call_count, 2)

    def test_successful_normal_path_has_no_extra_ocr(self):
        _, engine = self.run_retry([], original="去建邺城170,6附近抓")
        engine.assert_not_called()

    def test_stop_prevents_retry(self):
        _, engine = self.run_retry([], stopped=True)
        engine.assert_not_called()


if __name__ == "__main__":
    unittest.main()
