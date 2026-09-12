"""Unlabelled full-window capture; independent of CNN and calibration."""
import hashlib
import json
import threading
import time
import shutil
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
import win32_screen


class FrameFilter:
    def __init__(self):
        self.hashes = set()
        self.recent = deque(maxlen=64)

    def accept(self, image):
        h, w = image.shape[:2]
        # Only the central board zone drives deduplication, not avatar clocks.
        # This is a layout heuristic, NOT a training crop; originals stay full size.
        roi = image[int(h*.08):int(h*.94), int(w*.25):int(w*.75)]
        thumb = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (256, 320))
        key = hashlib.sha256((thumb//8).tobytes()+str((h,w)).encode()).hexdigest()
        if key in self.hashes:
            return False, key
        for shape, previous in self.recent:
            if shape != (h,w):
                continue
            delta = cv2.absdiff(thumb, previous)
            if np.mean(delta > 16) < .001 and float(delta.mean()) < .8:
                return False, key
        self.hashes.add(key)
        self.recent.append(((h,w),thumb))
        return True, key


class DatasetCapture(QThread):
    progress = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stop_event = threading.Event()
        self.directory = Path(__file__).resolve().parent/'datasets'/'raw'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    def stop(self):
        self.stop_event.set()

    def run(self):
        try:
            self._collect()
        except Exception as exc:
            self.progress.emit(f'采集已停止：{type(exc).__name__}: {exc}')

    def _collect(self):
        self.directory.mkdir(parents=True, exist_ok=False)
        gate = FrameFilter()
        saved = seen = total_bytes = 0
        previous_geom = None
        announced = ''
        started = time.monotonic()
        with (self.directory/'frames.jsonl').open('a', encoding='utf-8') as manifest:
            while not self.stop_event.is_set():
                tick = time.monotonic()
                found = win32_screen.find_xiangqi_window()
                reason = None
                if not found:
                    reason = '等待游戏窗口'
                else:
                    hwnd, title, _ = found
                    geom = win32_screen.get_client_rect(hwnd)
                    if (win32_screen.is_iconic(hwnd) or
                            not win32_screen.is_window_unobscured(hwnd, geom)):
                        reason = '窗口被遮挡或最小化，暂停采集'
                    elif (hwnd, geom) != previous_geom:
                        previous_geom = (hwnd, geom)
                        reason = '窗口位置变化，等待下一帧稳定'
                if reason:
                    if reason != announced:
                        self.progress.emit(reason)
                        announced = reason
                    self.stop_event.wait(.125)
                    continue
                # Desktop region capture does not activate/restore the target.
                image, method = win32_screen.capture_window(
                    hwnd, *geom, allow_fallback=True, try_print=False,
                    activate_fallback=False, return_method=True)
                if (image is not None and win32_screen.get_client_rect(hwnd) == geom
                        and win32_screen.is_window_unobscured(hwnd, geom)):
                    seen += 1
                    accepted, signature = gate.accept(image)
                    if accepted:
                        if total_bytes >= 10*1024**3 or shutil.disk_usage(self.directory).free < 1024**3:
                            raise RuntimeError('达到本次10GB上限或磁盘可用空间不足1GB')
                        name = f'{saved:06d}.png'
                        ok, encoded = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                        if not ok:
                            raise RuntimeError('PNG编码失败')
                        (self.directory/name).write_bytes(encoded.tobytes())
                        record = {'file': name, 'timestamp': datetime.now().isoformat(),
                                  'elapsed_seconds': tick-started, 'geometry': geom,
                                  'window_title': title, 'capture': method,
                                  'width': image.shape[1], 'height': image.shape[0],
                                  'signature': signature, 'label': None,
                                  'dedup_roi_normalized': [.25,.08,.75,.94]}
                        manifest.write(json.dumps(record, ensure_ascii=False)+'\n')
                        manifest.flush()
                        saved += 1
                        total_bytes += encoded.nbytes
                        announced = ''
                        self.progress.emit(f'采集中：已保存 {saved} 帧 / 检查 {seen} 帧，{total_bytes/1024**2:.1f}MB；{self.directory}')
                self.stop_event.wait(max(0, .125-(time.monotonic()-tick)))
        self.progress.emit(f'采集结束：保存 {saved} 帧；{self.directory}')
