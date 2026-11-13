"""Camera input: threaded live capture with reconnect, or synchronous video-file reading."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import NamedTuple, Optional, Union

import cv2
import numpy as np


class Frame(NamedTuple):
    image: np.ndarray
    timestamp: float   # time.perf_counter() at capture
    index: int


class Camera:
    """Live sources (webcam index / stream URL) are read in a background thread and only the
    newest frame is kept (frame-drop policy: never build up latency). Video files are read
    synchronously so no frame is skipped. Live capture reconnects after read failures.
    """

    def __init__(self, source: Union[int, str] = 0, width: int = 1280, height: int = 720,
                 fps: int = 30, reconnect_attempts: int = 10, reconnect_delay_s: float = 1.0,
                 stale_frame_timeout_s: float = 2.0):
        self.source = source
        self.width, self.height, self.fps = width, height, fps
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_s = reconnect_delay_s
        self.stale_timeout = stale_frame_timeout_s
        self.is_file = isinstance(source, str) and Path(source).is_file()
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._latest: Optional[Frame] = None
        self._last_returned = 0
        self._index = 0
        self._state = "closed"        # closed | ok | reconnecting | lost | ended
        self._last_frame_time = 0.0

    # ------------------------------------------------------------------ lifecycle
    def _open(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.release()
            return None
        if not self.is_file:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def start(self) -> bool:
        self._cap = self._open()
        if self._cap is None:
            self._state = "lost"
            return False
        self._state = "ok"
        self._last_frame_time = time.perf_counter()
        if not self.is_file:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True, name="camera")
            self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        self._cap, self._thread, self._state = None, None, "closed"

    # ------------------------------------------------------------------ live thread
    def _loop(self) -> None:
        fails = 0
        while not self._stop.is_set():
            ok, img = self._cap.read() if self._cap is not None else (False, None)
            if ok and img is not None:
                fails = 0
                self._state = "ok"
                now = time.perf_counter()
                with self._cond:
                    self._index += 1
                    self._latest = Frame(img, now, self._index)
                    self._last_frame_time = now
                    self._cond.notify_all()
            else:
                fails += 1
                if fails >= 3:
                    if not self._reconnect():
                        return
                    fails = 0
                else:
                    time.sleep(0.01)

    def _reconnect(self) -> bool:
        self._state = "reconnecting"
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        for _ in range(self.reconnect_attempts):
            if self._stop.is_set():
                return False
            time.sleep(self.reconnect_delay_s)
            cap = self._open()
            if cap is not None:
                self._cap = cap
                self._state = "ok"
                return True
        self._state = "lost"
        with self._cond:
            self._cond.notify_all()
        return False

    # ------------------------------------------------------------------ reading
    def read(self, timeout: float = 1.0) -> Optional[Frame]:
        """Return the next frame, or None on timeout / end of file / lost camera."""
        if self.is_file:
            if self._cap is None:
                return None
            ok, img = self._cap.read()
            if not ok:
                self._state = "ended"
                return None
            self._index += 1
            now = time.perf_counter()
            self._last_frame_time = now
            return Frame(img, now, self._index)
        with self._cond:
            self._cond.wait_for(
                lambda: (self._latest is not None and self._latest.index != self._last_returned)
                or self._state == "lost",
                timeout=timeout)
            if self._latest is None or self._latest.index == self._last_returned:
                return None
            self._last_returned = self._latest.index
            return self._latest

    # ------------------------------------------------------------------ status
    @property
    def state(self) -> str:
        if self._state == "ok" and not self.is_file and \
                time.perf_counter() - self._last_frame_time > self.stale_timeout:
            return "stale"
        return self._state

    @property
    def ended(self) -> bool:
        return self._state in ("ended", "lost")

    @property
    def native_fps(self) -> float:
        if self._cap is not None:
            v = self._cap.get(cv2.CAP_PROP_FPS)
            if v and v > 0:
                return float(v)
        return float(self.fps)
