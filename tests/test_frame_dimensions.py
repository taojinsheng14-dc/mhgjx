import unittest,tempfile
from pathlib import Path
from unittest.mock import Mock
import cv2,numpy as np
from scrcpy_human_automation.vision import TemplateMatcher
from scrcpy_human_automation.device import AdbDevice,ScreenSize

class FrameDimensionsTests(unittest.TestCase):
    def test_portrait_cache_refreshes_after_rotation(self):
        d=AdbDevice();d._screen_size=ScreenSize(1200,2608)
        d.current_capture_size=Mock(return_value=(2608,1200))
        self.assertEqual(d.screen_size_for_input(refresh=True),ScreenSize(2608,1200))
    def test_template_bigger_on_one_axis_returns_no_match(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'template.png';cv2.imwrite(str(p),np.zeros((60,330,3),dtype=np.uint8))
            matcher=TemplateMatcher(Mock())
            for size in [(70,100,3),(30,400,3),(30,100,3)]:
                self.assertIsNone(matcher.find_template(p,screenshot=np.zeros(size,dtype=np.uint8)))
if __name__=='__main__':unittest.main()
