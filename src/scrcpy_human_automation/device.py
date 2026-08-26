from __future__ import annotations

import re
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ScreenSize:
    width: int
    height: int


class AdbDevice:
    def __init__(self, serial: str | None = None, adb_path: str = "adb"):
        self.serial = serial
        self.adb_path = adb_path
        self._screen_size: ScreenSize | None = None
        self._screenshot_lock = threading.Lock()
        self._input_lock = threading.Lock()
        self._input_shell: subprocess.Popen[str] | None = None

    def adb_args(self) -> list[str]:
        args = [self.adb_path]
        if self.serial:
            args.extend(["-s", self.serial])
        return args

    @staticmethod
    def _subprocess_options() -> dict:
        if sys.platform != "win32":
            return {}
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        return {
            "creationflags": subprocess.CREATE_NO_WINDOW,
            "startupinfo": startupinfo,
        }

    def run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*self.adb_args(), *args],
            check=check,
            capture_output=True,
            text=True,
            **self._subprocess_options(),
        )

    def run_bytes(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [*self.adb_args(), *args],
            check=check,
            capture_output=True,
            text=False,
            **self._subprocess_options(),
        )

    def ensure_connected(self) -> None:
        result = self.run("get-state")
        if "device" not in result.stdout:
            raise RuntimeError("No Android device is connected to adb.")

    def screen_size(self, refresh: bool = False) -> ScreenSize:
        if self._screen_size is not None and not refresh:
            return self._screen_size

        result = self.run("shell", "wm", "size")
        match = re.search(r"Physical size:\s*(\d+)x(\d+)", result.stdout)
        if not match:
            raise RuntimeError(f"Unable to parse device size from output: {result.stdout!r}")
        physical_width = int(match.group(1))
        physical_height = int(match.group(2))

        capture_size = self.current_capture_size()
        if capture_size is not None:
            capture_width, capture_height = capture_size
            physical_is_landscape = physical_width > physical_height
            capture_is_landscape = capture_width > capture_height
            if physical_is_landscape != capture_is_landscape:
                physical_width, physical_height = physical_height, physical_width

        self._screen_size = ScreenSize(width=physical_width, height=physical_height)
        return self._screen_size

    def current_capture_size(self) -> tuple[int, int] | None:
        screenshot = self.screenshot_png()
        if len(screenshot) >= 24 and screenshot[:8] == b"\x89PNG\r\n\x1a\n":
            return struct.unpack(">II", screenshot[16:24])
        return None

    def screenshot_raw(self) -> tuple[int, int, bytes]:
        with self._screenshot_lock:
            last_error = ""
            for attempt in range(3):
                result = self.run_bytes("exec-out", "screencap", check=False)
                data = result.stdout or b""
                if len(data) >= 12:
                    width, height, _pixel_format = struct.unpack_from("<III", data, 0)
                    pixel_bytes = width * height * 4
                    if width > 0 and height > 0 and len(data) >= 12 + pixel_bytes:
                        return width, height, data[12 : 12 + pixel_bytes]
                stderr = (result.stderr or b"").decode("utf-8", "ignore").strip()
                last_error = stderr or f"empty/truncated raw screenshot ({len(data)} bytes)"
                time.sleep(0.08 * (attempt + 1))
            raise RuntimeError(f"ADB raw screenshot failed after retries: {last_error}")

    def screen_size_for_input(self, refresh: bool = False) -> ScreenSize:
        if self._screen_size is not None and not refresh:
            return self._screen_size

        capture_size = self.current_capture_size()
        if capture_size is None:
            return self.screen_size(refresh=refresh)
        width, height = capture_size
        size = ScreenSize(width=width, height=height)
        if self._screen_size != size:
            self._screen_size = size
        return size

    def tap(self, x: int, y: int) -> None:
        if not self._send_input_command(f"input tap {x} {y}"):
            self.run("shell", "input", "tap", str(x), str(y))

    def short_swipe_tap(self, x: int, y: int, duration_ms: int = 65) -> None:
        # Some games occasionally treat `input tap` as a highlight-only press.
        # A tiny swipe produces an explicit down/up gesture while staying inside
        # the same button.
        self.swipe(x, y, x + 1, y + 1, duration_ms)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int) -> None:
        command = f"input swipe {x1} {y1} {x2} {y2} {duration_ms}"
        if not self._send_input_command(command):
            self.run(
                "shell",
                "input",
                "swipe",
                str(x1),
                str(y1),
                str(x2),
                str(y2),
                str(duration_ms),
            )

    def shell(self, command: str) -> None:
        self.run("shell", command)

    def _send_input_command(self, command: str) -> bool:
        """Reuse one adb shell for input commands to avoid per-click adb startup."""
        with self._input_lock:
            process = self._input_shell
            if process is None or process.poll() is not None or process.stdin is None:
                try:
                    process = subprocess.Popen(
                        [*self.adb_args(), "shell"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        text=True,
                        encoding="utf-8",
                        **self._subprocess_options(),
                    )
                except Exception:
                    self._input_shell = None
                    return False
                self._input_shell = process

            try:
                process.stdin.write(command + "\n")
                process.stdin.flush()
                return True
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
                self._input_shell = None
                return False

    def screenshot_png(self) -> bytes:
        # `adb exec-out screencap` can occasionally return an empty/truncated
        # buffer, especially if another thread is also taking a screenshot.
        # Serialize screenshots and retry briefly so OpenCV never receives an
        # empty buffer.
        with self._screenshot_lock:
            last_error = ""
            for attempt in range(3):
                result = self.run_bytes("exec-out", "screencap", "-p", check=False)
                data = result.stdout or b""
                if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 32:
                    return data
                stderr = (result.stderr or b"").decode("utf-8", "ignore").strip()
                last_error = stderr or f"empty/truncated screenshot ({len(data)} bytes)"
                time.sleep(0.12 * (attempt + 1))
            raise RuntimeError(f"ADB screenshot failed after retries: {last_error}")
