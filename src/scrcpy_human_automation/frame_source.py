from __future__ import annotations

import threading
import time
from dataclasses import dataclass
import io
import subprocess
import sys

import cv2
import numpy as np

from .device import AdbDevice


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    captured_at: float
    source: str


class FrameSource:
    def latest(self, max_age: float = 0.0) -> Frame:
        raise NotImplementedError

    def close(self) -> None:
        return


class NonSeekableReader(io.RawIOBase):
    """Wrap a pipe so PyAV does not try unsupported seeks on Windows pipes."""

    def __init__(self, raw):
        self.raw = raw

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def read(self, size: int = -1) -> bytes:
        return self.raw.read(size)

    def readinto(self, buffer) -> int:
        data = self.raw.read(len(buffer))
        if not data:
            return 0
        buffer[: len(data)] = data
        return len(data)

    def close(self) -> None:
        try:
            self.raw.close()
        finally:
            super().close()


class AdbFrameSource(FrameSource):
    def __init__(self, device: AdbDevice, cache_ttl: float = 0.15):
        self.device = device
        self.cache_ttl = max(0.0, cache_ttl)
        self._lock = threading.Lock()
        self._cached: Frame | None = None

    def latest(self, max_age: float = 0.0) -> Frame:
        now = time.monotonic()
        ttl = max(max_age, self.cache_ttl)
        with self._lock:
            if self._cached is not None and now - self._cached.captured_at <= ttl:
                return self._cached
            frame = self._capture()
            self._cached = frame
            return frame

    def _capture(self) -> Frame:
        try:
            width, height, pixels = self.device.screenshot_raw()
            rgba = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 4))
            image = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            return Frame(image=image, captured_at=time.monotonic(), source="adb-raw")
        except Exception:
            png_bytes = self.device.screenshot_png()
            image = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("Unable to decode screenshot from device.")
            return Frame(image=image, captured_at=time.monotonic(), source="adb-png")


class ScrcpyVideoFrameSource(FrameSource):
    """Experimental live video frame source.

    On Windows scrcpy does not expose a simple Python frame pipe, so this uses
    Android's H264 screenrecord stream as a low-risk video-source test. It is
    only enabled for flows that explicitly request frame_source="video".
    """

    def __init__(
        self,
        device: AdbDevice,
        max_size: int = 4096,
        bit_rate: str = "4M",
        max_fps: int = 15,
        startup_timeout: float = 8.0,
    ):
        self.device = device
        self.max_size = max_size
        self.bit_rate = bit_rate
        self.max_fps = max_fps
        self.startup_timeout = startup_timeout
        self._lock = threading.Lock()
        self._cached: Frame | None = None
        self._fallback = AdbFrameSource(device)
        self._closed = threading.Event()
        self._ready = threading.Event()
        self._error: Exception | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._thread = threading.Thread(target=self._reader, name="video-frame-source", daemon=True)
        self._thread.start()

    def latest(self, max_age: float = 0.0) -> Frame:
        if not self._ready.wait(self.startup_timeout):
            return self._fallback.latest(max_age=max_age)
        if self._error is not None:
            return self._fallback.latest(max_age=max_age)
        with self._lock:
            frame = self._cached
        if frame is None:
            return self._fallback.latest(max_age=max_age)
        allowed_age = max_age if max_age > 0 else 2.0
        if time.monotonic() - frame.captured_at > allowed_age:
            return self._fallback.latest(max_age=max_age)
        return frame

    def close(self) -> None:
        self._closed.set()
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass
            try:
                process.wait(timeout=2)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass

    def _reader(self) -> None:
        try:
            import av

            width, height = self._scaled_size()
            command = [
                *self.device.adb_args(),
                "exec-out",
                "screenrecord",
                "--output-format=h264",
                "--size",
                f"{width}x{height}",
                "--bit-rate",
                str(self.bit_rate),
                "-",
            ]
            options = self.device._subprocess_options()
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                **options,
            )
            self._process = process
            if process.stdout is None:
                raise RuntimeError("screenrecord stdout is not available")
            stream = io.BufferedReader(NonSeekableReader(process.stdout), buffer_size=1024 * 1024)
            container = av.open(
                stream,
                format="h264",
                mode="r",
                options={"probesize": "32768", "analyzeduration": "0"},
            )
            last_frame_at = 0.0
            min_interval = 1.0 / max(1, self.max_fps)
            for packet in container.demux(video=0):
                if self._closed.is_set():
                    break
                for video_frame in packet.decode():
                    if self._closed.is_set():
                        break
                    now = time.monotonic()
                    if now - last_frame_at < min_interval:
                        continue
                    image = video_frame.to_ndarray(format="bgr24")
                    with self._lock:
                        self._cached = Frame(image=image, captured_at=now, source="video-screenrecord")
                    last_frame_at = now
                    self._ready.set()
            if process.poll() not in (None, 0) and not self._closed.is_set():
                raise RuntimeError(f"screenrecord exited with code {process.returncode}")
        except Exception as exc:
            self._error = exc
            self._ready.set()
        finally:
            self.close()

    def _scaled_size(self) -> tuple[int, int]:
        size = self.device.screen_size_for_input()
        width = size.width
        height = size.height
        largest = max(width, height)
        if largest <= self.max_size:
            return width, height
        scale = self.max_size / largest
        # H264 encoders generally prefer even dimensions.
        scaled_width = max(2, int(width * scale) // 2 * 2)
        scaled_height = max(2, int(height * scale) // 2 * 2)
        return scaled_width, scaled_height
