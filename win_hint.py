"""
Windows 天天象棋 (微信客户端) 视觉提示 bot。

路线: 窗口截图 (mss) -> CNN 读盘 -> 稳定/合法局面确认 -> Pikafish 主变化
      -> 点击穿透悬浮窗画箭头 (当前最佳走法=黄色, 随后应对=红色)

用法:
  python win_hint.py --list-windows   # 诊断: 列出所有窗口
  python win_hint.py --calibrate      # 检测棋盘区域, 写 calib.json, 存 debug/win_calib.png
  python win_hint.py --show           # 单帧: 读盘 + 引擎提示 + 存 debug/win_show.png (带箭头)
  python win_hint.py                  # 实时提示循环 (Ctrl+C 结束)
  python win_hint.py --opp-arrows 1   # 保留兼容参数；当前显示一条红线和一条金线
  python win_hint.py --movetime 2000  # 引擎每步搜索毫秒数 (默认 1200)
  python win_hint.py --turn b         # 指定当前轮次；中途接入建议显式指定
  python win_hint.py --no-overlay     # 不画悬浮箭头, 仅控制台输出

说明: 当前版本只提示，不自动点击。箭头画在“点击穿透”的置顶悬浮层上。目标窗口最小化、隐藏
或棋盘被其它窗口遮挡时，脚本暂停识别/引擎并隐藏箭头；后台但无遮挡时仍允许继续绘图。
"""
import argparse
import io
import json
import math
import os
import sys
import time
from collections import deque

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import win32_screen
from xiangqi_bot import Bot, CALIB_PATH, _SCRIPT_DIR
from engine_session import EngineSession, fen_prevalidate
from board_geometry import detect_grid

DEBUG_DIR = os.path.join(_SCRIPT_DIR, 'debug')

# Arrow colors (BGR)
GOLD = (40, 200, 255)
RED = (60, 80, 255)

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"


class FramePending(RuntimeError):
    """Transient window movement, not an application failure."""


def refine_grid(img, x1, y1, x2, y2, halfwin=40):
    """Locate the actual 9x10 grid lines inside a rough board rect.

    The grid lines are thin dark lines on the wood. Per-column/per-row LOW
    percentiles of darkness are piece-robust (pieces block only a minority of
    any line), so each of the 9+10 lines is found by a local peak search
    around the rough estimate. Returns (cols, rows) in image-px.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
    board = gray[int(y1):int(y2), int(x1):int(x2)]
    dark = 255 - board
    H, W = board.shape
    # vertical lines: p10 of darkness down each column (skip 5% margins)
    colprof = np.percentile(dark[int(H * 0.05):int(H * 0.95), :], 10, axis=0)
    colprof = cv2.GaussianBlur(colprof.astype(np.float32), (9, 1), 0)
    rowprof = np.percentile(dark[:, int(W * 0.05):int(W * 0.95)], 10, axis=1)
    rowprof = cv2.GaussianBlur(rowprof.astype(np.float32), (1, 9), 0)

    cols = []
    for i in range(9):
        cx = int((x2 - x1) * i / 8)
        lo, hi = max(0, cx - halfwin), min(W, cx + halfwin)
        seg = colprof[lo:hi]
        cols.append(x1 + lo + int(seg.argmax()))
    rows = []
    for j in range(10):
        cy = int((y2 - y1) * j / 9)
        lo, hi = max(0, cy - halfwin), min(H, cy + halfwin)
        seg = rowprof[lo:hi]
        rows.append(y1 + lo + int(seg.argmax()))
    return cols, rows


def refine_grid_pieces(img, x1, y1, x2, y2):
    """Fit the exact 9x10 lattice from piece centers (HoughCircles).

    Uses the two most populous rows (the back ranks when the game is fresh:
    9 pieces each). Works even when the wood edge and grid extent differ
    more than a fixed margin can express. Returns (cols, rows) in image-px
    or None if the fit looks wrong.
    """
    pad = 30
    x0, y0 = max(0, int(x1) - pad), max(0, int(y1) - pad)
    x1_, y1_ = min(img.shape[1], int(x2) + pad), min(img.shape[0], int(y2) + pad)
    sub = cv2.cvtColor(img[y0:y1_, x0:x1_], cv2.COLOR_BGR2GRAY)
    sub = cv2.GaussianBlur(sub, (5, 5), 1)
    circles = cv2.HoughCircles(sub, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                               param1=120, param2=22, minRadius=22, maxRadius=45)
    if circles is None:
        return None
    pts = sorted((x + x0, y + y0) for x, y, _r in circles[0])
    # cluster into rows
    rows = []
    for x, y in pts:
        for row in rows:
            if abs(np.mean([p[1] for p in row]) - y) < 25:
                row.append((x, y))
                break
        else:
            rows.append([(x, y)])
    rows = [r for r in rows if len(r) >= 7]
    rows.sort(key=lambda r: np.mean([p[1] for p in r]))
    if len(rows) < 2:
        return None
    full = [r for r in rows if len(r) == 9]
    if not full:
        return None
    def colfit(row):
        xs = np.array(sorted(p[0] for p in row), dtype=float)
        ranks = np.arange(len(xs), dtype=float)
        return np.polyfit(ranks, xs, 1)
    fits = [colfit(r) for r in full]
    a = float(np.mean([f[0] for f in fits]))
    b = float(np.mean([f[1] for f in fits]))
    if not (50 <= a <= 90):
        return None
    y_top = float(np.mean([p[1] for p in rows[0]]))
    y_bot = float(np.mean([p[1] for p in rows[-1]]))
    if not (400 <= y_bot - y_top <= 900):
        return None
    cols = [b + a * i for i in range(9)]
    rows_ = [y_top + j * (y_bot - y_top) / 9 for j in range(10)]
    return list(cols), list(rows_)


# ----------------------------------------------------------------------
# image helpers
# ----------------------------------------------------------------------

def composite(base, sub, ox, oy):
    """Overlay a BGRA `sub` onto a BGR `base` at (ox, oy), with clipping."""
    sh, sw = sub.shape[:2]
    bh, bw = base.shape[:2]
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(bw, ox + sw), min(bh, oy + sh)
    if x1 <= x0 or y1 <= y0:
        return
    region = base[y0:y1, x0:x1]
    a = sub[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    for ch in range(3):
        region[..., ch] = np.where(a[..., 3] > 0, a[..., ch], region[..., ch])
    base[y0:y1, x0:x1] = region


# ----------------------------------------------------------------------
# CV board-rectangle detection (wood-color segmentation)
# ----------------------------------------------------------------------

def detect_board_rect(img, hsv_lo=(10, 30, 165), hsv_hi=(30, 160, 255)):
    """Find the wooden board rectangle in a window capture (BGR).
    Returns (x1, y1, x2, y2) in image coords, or None."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lo, hsv_hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 20000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / max(1, h)
        # 9:10 grid -> aspect ~0.9, accept 0.78..1.05
        if 0.78 <= aspect <= 1.05:
            if best is None or area > best[4]:
                best = (x, y, x + w, y + h, area)
    if best:
        return best[:4]
    return None


# ----------------------------------------------------------------------
# Windows Bot: overrides the macOS-only methods of the repo's Bot
# ----------------------------------------------------------------------

class WinHintBot(Bot):
    def __init__(self, hwnd=None, title=None, rect=None,
                 opp_arrows=1, movetime_ms=1200, start_turn='w',
                 use_overlay=True, require_foreground=False):
        super().__init__()
        self.win_hwnd = hwnd
        self.win_title = title
        self.win_rect = rect or (0, 0, 0, 0)
        self.opp_arrows = opp_arrows
        self.movetime_ms = movetime_ms
        self.start_turn = start_turn
        self.use_overlay = use_overlay
        self.require_foreground = require_foreground
        self.engine = None
        self._legal_cache = {}
        self.overlay = None
        # Kept for compatibility with older callers. The overlay no longer
        # draws historical opponent moves; it draws the current PV instead.
        self.opp_moves = deque(maxlen=max(1, opp_arrows))
        # Actual opponent moves are kept separately from the engine's
        # predicted reply to the currently displayed user hint.
        self.reply_hint = None
        self._calib_cols_norm = None
        self._calib_rows_norm = None
        self._geometry_generation = 0
        self._geometry_changed_at = time.monotonic()
        self._allow_empty_repairs = True
        self._recovery_key = None
        self._recovery_count = 0
        self._recovery_since = 0.0
        self._recognition_failed_since = None
        self._diagnostic_at = 0.0
        self._manual_turn_used = False
        self._grid_present = False
        self._turn_ui_key = None
        self._turn_ui_count = 0
        self._turn_ui_announced = None
        self._last_transition_side = None
        self._capture_method = None
        self._capture_fallbacks = 0
        self.retina_scale = 1.0  # physical pixels everywhere

    # ---------------- window / screenshot ----------------

    def find_window(self):
        r = win32_screen.find_xiangqi_window()
        if not r:
            raise RuntimeError("天天象棋 window not found (see --list-windows)")
        self.win_hwnd, self.win_title, self.win_rect = r
        if win32_screen.is_iconic(self.win_hwnd):
            print("  Window is minimized — waiting for user to restore it.")
        # Use the CLIENT rect: PrintWindow emits the client area, so all
        # downstream pixel math (calib ratios, overlay) stays aligned.
        self._set_client_geometry(
            win32_screen.get_client_rect(self.win_hwnd), apply_calibration=False)
        print(f"  Window: '{self.win_title}' hwnd={self.win_hwnd:#x} "
              f"client=({self.win_x},{self.win_y}) {self.win_w}x{self.win_h} "
              f"(full window {self.win_rect})")
        return r

    def _set_client_geometry(self, geom, apply_calibration=True):
        """Store the current client rect and rebuild the absolute grid.

        Calibration is stored as client-relative normalized coordinates, so a
        move or resize must update both the screenshot origin and every board
        intersection used by CNN parsing and the overlay.
        """
        x, y, w, h = (int(v) for v in geom)
        old = tuple(getattr(self, k, None)
                    for k in ('win_x', 'win_y', 'win_w', 'win_h'))
        new = (x, y, w, h)
        changed = old != new
        self.win_x, self.win_y, self.win_w, self.win_h = new
        if apply_calibration and self._calib_cols_norm is not None:
            self.cols_logical = [self.win_x + c * self.win_w
                                 for c in self._calib_cols_norm]
            self.rows_logical = [self.win_y + r * self.win_h
                                 for r in self._calib_rows_norm]
            self.cell_w = ((self.cols_logical[-1] - self.cols_logical[0]) / 8)
            self.cell_h = ((self.rows_logical[-1] - self.rows_logical[0]) / 9)
        if changed:
            self._geometry_generation += 1
            self._geometry_changed_at = time.monotonic()
        return changed

    def _refresh_client_geometry(self):
        """Refresh geometry every frame; return True if it moved/resized."""
        return self._set_client_geometry(
            win32_screen.get_client_rect(self.win_hwnd), apply_calibration=True)

    def window_ready(self):
        """Return (ok, reason) without changing focus or mouse state.

        A background window is valid when its client area is fully visible.
        Only hidden/minimized/occluded targets are paused.
        """
        if not win32_screen.is_window_visible(self.win_hwnd):
            return False, 'window is hidden'
        if win32_screen.is_iconic(self.win_hwnd):
            return False, 'window is minimized'
        if self.require_foreground and not win32_screen.is_foreground(self.win_hwnd):
            return False, 'window is not foreground (--require-foreground)'
        if time.monotonic() - self._geometry_changed_at < 0.6:
            return False, 'waiting for window geometry to settle'
        ignore = []
        if self.overlay is not None and self.overlay.hwnd:
            ignore.append(self.overlay.hwnd)
        if not win32_screen.is_window_unobscured(
                self.win_hwnd,
                (self.win_x, self.win_y, self.win_w, self.win_h),
                ignore_hwnds=ignore):
            return False, 'window is occluded'
        return True, None

    def wait_for_window_ready(self, interval=0.35):
        """Wait until the target is visible and unobscured.

        This is used before the first frame so a background/covered window is
        never parsed from a stale or mixed screen capture.
        """
        last_reason = None
        while True:
            self._refresh_client_geometry()
            ok, reason = self.window_ready()
            if ok:
                return True
            if reason != last_reason:
                print(f"  paused: {reason}; make the target visible and unobscured to resume")
                last_reason = reason
            if self.overlay is not None:
                self.overlay.set_visible(False)
                self.overlay.pump()
            time.sleep(interval)

    def screenshot_for_processing(self):
        # All callers (including startup/calibration and the parser's second
        # shot) use the same wait path. Dragging must never escape main().
        while True:
            self.wait_for_window_ready(interval=0.1)
            generation = self._geometry_generation
            try:
                img = self._capture_settled_frame()
                self._locate_grid(img)
                return img
            except FramePending:
                continue
            except RuntimeError:
                self._refresh_client_geometry()
                ready, _reason = self.window_ready()
                if not ready or generation != self._geometry_generation:
                    continue
                raise

    def read_initial_board(self):
        """Do not exit startup because a drag/animation invalidates a frame."""
        announced = False
        while True:
            img = self.screenshot_for_processing()
            generation = self._geometry_generation
            board = self.parse_board_cnn(img)
            self._refresh_client_geometry()
            quality = getattr(self, '_last_parse_quality', None)
            if (board is not None and quality is not None and
                    not quality.get('corrections') and
                    generation == self._geometry_generation):
                return img, board
            if not announced:
                print('  waiting for a clear initial board; startup will retry')
                announced = True
            time.sleep(0.2)

    def _locate_grid(self, img):
        # Once a grid is known, search a small client-relative neighborhood
        # first. This is the normal per-frame path and stays fast while the
        # whole OS window is dragged or resized. If a different game screen
        # moved the board inside the client, fall back to the global detector
        # (which includes the Hough-border recovery path).
        fit = None
        old_cols = [c - self.win_x for c in getattr(self, 'cols_logical', [])]
        old_rows = [r - self.win_y for r in getattr(self, 'rows_logical', [])]
        if len(old_cols) == 9 and len(old_rows) == 10:
            pad = max(80, int(max(getattr(self, 'cell_w', 0),
                                 getattr(self, 'cell_h', 0)) * 1.5))
            x1 = max(0, int(min(old_cols)) - pad)
            y1 = max(0, int(min(old_rows)) - pad)
            x2 = min(img.shape[1], int(max(old_cols)) + pad)
            y2 = min(img.shape[0], int(max(old_rows)) + pad)
            if x2 - x1 >= 300 and y2 - y1 >= 350:
                local = detect_grid(img[y1:y2, x1:x2], _allow_hough=False)
                if local is not None:
                    fit = ([float(v) + x1 for v in local[0]],
                           [float(v) + y1 for v in local[1]])
        if fit is None:
            fit = detect_grid(img)
        if fit is None:
            # A calibrated client-relative grid is the authoritative fallback
            # once the target window has moved or the skin's thin lines are
            # temporarily hidden by pieces/selection markers.  The line
            # detector is only a refinement signal; making its failure disable
            # the already-valid calibration turns an otherwise readable board
            # into WAITING_FOR_GAME after every window drag.
            calibrated = (
                len(old_cols) == 9 and len(old_rows) == 10 and
                all(0 <= c < img.shape[1] for c in old_cols) and
                all(0 <= r < img.shape[0] for r in old_rows) and
                old_cols[-1] > old_cols[0] and old_rows[-1] > old_rows[0] and
                (old_cols[-1] - old_cols[0]) / 8 >= 20 and
                (old_rows[-1] - old_rows[0]) / 9 >= 20
            )
            self._grid_present = calibrated
            if calibrated:
                if getattr(self, '_grid_detector_miss_announced', False) is False:
                    print('  board line detector missed; keeping calibrated grid')
                    self._grid_detector_miss_announced = True
                return True
            return False
        self._grid_present = True
        self._grid_detector_miss_announced = False
        cols, rows = fit
        tolerance = max(2., (cols[-1]-cols[0])/8*.035)
        if (len(old_cols) == 9 and len(old_rows) == 10 and
                max(abs(a-b) for a, b in zip(cols+rows, old_cols+old_rows)) <= tolerance):
            return True
        self._calib_cols_norm = [float(c)/self.win_w for c in cols]
        self._calib_rows_norm = [float(r)/self.win_h for r in rows]
        self._set_client_geometry((self.win_x, self.win_y, self.win_w, self.win_h))
        # Scene changes can move the board without moving the OS window.
        self._geometry_generation += 1
        print(f'  board grid updated from lines: cell={self.cell_w:.1f}x{self.cell_h:.1f}')
        return True

    def parse_board_cnn(self, img):
        if not self._grid_present:
            self._last_parse_quality = None
            return None
        return super().parse_board_cnn(img)

    def _capture_settled_frame(self):
        if win32_screen.is_iconic(self.win_hwnd):
            raise FramePending('window is minimized')

        # A drag/resize can change the client rect between PrintWindow and the
        # fallback screen grab. Never parse an image with a grid from another
        # geometry generation. The outer wrapper waits before retrying.
        self._refresh_client_geometry()
        geom = (self.win_x, self.win_y, self.win_w, self.win_h)
        generation = self._geometry_generation
        img, method = win32_screen.capture_window(
            self.win_hwnd, *geom, allow_fallback=False,
            return_method=True)
        if img is None:
            # Modern Windows excludes our overlay from capture without hiding
            # it on the monitor. Only unsupported systems need hide/restore.
            was_visible = (bool(getattr(self.overlay, '_visible', False)) and
                           not getattr(self.overlay, 'capture_excluded', False))
            try:
                if was_visible:
                    self.overlay.set_visible(False)
                    self.overlay.pump()
                    time.sleep(0.04)
                img, method = win32_screen.capture_window(
                    self.win_hwnd, *geom, allow_fallback=True,
                    try_print=False, activate_fallback=False,
                    return_method=True)
            finally:
                self._refresh_client_geometry()
                ready, _reason = self.window_ready()
                if was_visible and ready and generation == self._geometry_generation:
                    self.overlay.set_visible(True)
                    self.overlay.pump()
            self._capture_fallbacks += 1
        self._note_capture_method(method)
        self._refresh_client_geometry()
        if generation != self._geometry_generation:
            raise FramePending('window geometry changed during capture')
        return img

    def _note_capture_method(self, method):
        if method != self._capture_method:
            self._capture_method = method
            label = 'isolated PrintWindow' if method == 'printwindow' else method
            print(f"  capture: {label}")

    def _get_window_width(self):
        return self.win_w

    def _get_window_height(self):
        return self.win_h

    def activate_window(self):
        win32_screen.activate(self.win_hwnd)
        time.sleep(0.2)

    def click(self, lx, ly):
        win32_screen.click(int(lx), int(ly))

    def is_my_turn(self):
        # autopilot compatibility; hint mode doesn't rely on this
        try:
            return getattr(self, '_tracked_turn', self.start_turn) in ('w', 'b')
        except Exception:
            return True

    # ---------------- engine (persistent session) ----------------

    def pikafish(self, fen, move_history=None, excluded=None):
        if self.engine is None:
            self.engine = EngineSession(verbose=False)
        if move_history:
            fen += " moves " + ' '.join(move_history)
        best, info, _ = self._analyse_screen(fen, self.movetime_ms, excluded)
        return best, info

    def get_legal_moves(self, fen):
        if self.engine is None:
            self.engine = EngineSession(verbose=False)
        if fen not in self._legal_cache:
            moves = tuple(self._flip_rank(m) for m in
                          self.engine.legal_moves(self._engine_fen(fen)))
            if len(self._legal_cache) >= 64:
                self._legal_cache.pop(next(iter(self._legal_cache)))
            self._legal_cache[fen] = moves
        return list(self._legal_cache[fen])

    @staticmethod
    def _flip_rank(move):
        """Convert screen notation <-> UCI; both directions are identical."""
        if (not isinstance(move, str) or len(move) != 4 or
                move[0] not in 'abcdefghi' or move[2] not in 'abcdefghi' or
                move[1] not in '0123456789' or move[3] not in '0123456789'):
            raise ValueError(f"invalid move: {move!r}")
        return f"{move[0]}{9-int(move[1])}{move[2]}{9-int(move[3])}"

    def _engine_fen(self, fen):
        """Translate legacy CLI side tokens at the engine boundary only.

        Internal b=bottom/player, w=top/opponent. The normalized FEN has
        uppercase bottom pieces, so actual UCI w=bottom and b=top.
        """
        fields = fen.split()
        if len(fields) < 2 or fields[1] not in ('w', 'b'):
            raise ValueError('position requires an explicit side')
        fields[1] = 'w' if fields[1] == 'b' else 'b'
        if 'moves' in fields:
            idx = fields.index('moves') + 1
            fields[idx:] = [self._flip_rank(m) for m in fields[idx:]]
        return ' '.join(fields)

    def _analyse_screen(self, fen, movetime_ms, excluded=None, pv_len=2):
        best, info, pv = self.engine.analyse(
            self._engine_fen(fen), movetime_ms=movetime_ms,
            excluded=[self._flip_rank(m) for m in excluded] if excluded else None,
            pv_len=pv_len)
        best = self._flip_rank(best) if best and best not in ('(none)', '0000') else None
        return best, info, [self._flip_rank(m) for m in pv]

    # ---------------- orientation ----------------

    def detect_orientation_from_board(self, board):
        """The player is bottom. Require two opposite kings in their palaces."""
        kings = [(r, c, p) for r, row in enumerate(board)
                 for c, p in enumerate(row) if p in ('K', 'k')]
        top = [p for r, c, p in kings if r <= 2 and 3 <= c <= 5]
        bottom = [p for r, c, p in kings if r >= 7 and 3 <= c <= 5]
        if len(kings) != 2 or len(top) != 1 or len(bottom) != 1 or top == bottom:
            return False
        playing_red = bottom[0] == 'K'
        if getattr(self, '_announced_orientation', None) != playing_red:
            print(f"  Orientation: bottom/player={'RED' if playing_red else 'BLACK'}")
            self._announced_orientation = playing_red
        self.playing_red = playing_red
        return True

    def _observe_ui_turn(self, img, position_key=None):
        """Detect the active green countdown frame around an avatar.

        The board itself contains green legal-move dots, so scanning all green
        contours is unreliable.  The only useful evidence is a roughly square
        green *ring* in the two avatar zones: left/top means opponent (``w``),
        right/bottom means player (``b``).  Missing/ambiguous evidence is
        deliberately returned as ``None``; callers then use the player side as
        the safe default instead of guessing the opponent.
        """
        cw = float(min(self.cell_w, self.cell_h))
        if cw <= 1 or len(self.cols_logical) != 9 or len(self.rows_logical) != 10:
            return None
        left = self.cols_logical[0] - self.win_x
        right = self.cols_logical[-1] - self.win_x
        top = self.rows_logical[0] - self.win_y
        bottom = self.rows_logical[-1] - self.win_y
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Timer outline is vivid green; muted green clock labels and avatar
        # backgrounds are not turn indicators.
        green = cv2.inRange(hsv, (38, 130, 100), (90, 255, 255))

        def ring_score(x1, y1, x2, y2):
            """Return the strongest square green-ring score in one avatar ROI."""
            h, w = green.shape[:2]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 <= x1 or y2 <= y1:
                return 0.0
            roi = green[y1:y2, x1:x2]
            # Join the four sides of a thin rounded countdown frame without
            # joining unrelated board dots (which are outside these ROIs).
            roi = cv2.morphologyEx(
                roi, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(
                roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best = 0.0
            for contour in contours:
                xx, yy, ww, hh = cv2.boundingRect(contour)
                if not (0.35*cw <= ww <= 2.0*cw and
                        0.35*cw <= hh <= 2.0*cw and
                        0.65 <= ww/max(hh, 1) <= 1.5):
                    continue
                patch = roi[yy:yy+hh, xx:xx+ww] > 0
                rim = max(2, int(min(ww, hh) * 0.16))
                if patch.shape[0] <= 2*rim or patch.shape[1] <= 2*rim:
                    continue
                edge = np.concatenate((
                    patch[:rim, :].ravel(), patch[-rim:, :].ravel(),
                    patch[:, :rim].ravel(), patch[:, -rim:].ravel()))
                inner = patch[rim:-rim, rim:-rim].mean()
                edge_mean = edge.mean()
                # A timer frame has green on several sides and is not a solid
                # green avatar/photo region.
                # Countdown is a shrinking perimeter, not a permanently
                # closed rectangle. Accept two strong sides; requiring all
                # four causes loss of turn detection as the timer shrinks.
                sides = sorted((patch[:rim, :].mean(),
                                patch[-rim:, :].mean(),
                                patch[:, :rim].mean(),
                                patch[:, -rim:].mean()))
                side_cov = sides[-2]
                if side_cov < 0.04 or edge_mean < inner * 0.9:
                    continue
                score = side_cov * 2.0 + edge_mean - inner * 0.35
                best = max(best, float(score))
            return best

        # These boxes track the fixed avatar layout relative to the board and
        # remain valid when the whole client window is dragged or resized.
        top_score = ring_score(left - 3.5*cw, top - 1.8*cw,
                               left - 0.35*cw, top + 1.6*cw)
        bottom_score = ring_score(right + 0.35*cw, bottom - 3.0*cw,
                                  right + 3.5*cw, bottom + 1.6*cw)
        margin = 0.08
        if max(top_score, bottom_score) < 0.16:
            side = None
        elif abs(top_score - bottom_score) <= margin:
            side = None
        else:
            side = 'w' if top_score > bottom_score else 'b'

        key = (side, position_key)
        if side is not None and key == self._turn_ui_key:
            self._turn_ui_count += 1
        else:
            self._turn_ui_key, self._turn_ui_count = key, 1 if side else 0
        if side is not None and self._turn_ui_count >= 2:
            if side != self._turn_ui_announced:
                print(f"  turn marker: {'OPPONENT' if side == 'w' else 'PLAYER'}")
                self._turn_ui_announced = side
            return side
        return None

    def _fresh_turn(self, fen, ui_turn, allow_manual=False):
        if ui_turn:
            return ui_turn
        if allow_manual and not self._manual_turn_used and self.start_turn in ('b', 'w'):
            return self.start_turn
        # Missing/ambiguous countdown evidence must never invent an opponent
        # turn.  The player is always the bottom side and is the safe default,
        # including black-side and long-thinking positions.
        fallback = 'b'
        print(f'  turn unknown; defaulting to PLAYER ({fallback})')
        return fallback

    def _snapshot_valid(self, fen):
        """Engine must accept the entire position, not just its piece count."""
        for side in ('b', 'w'):
            try:
                fen_prevalidate(self._engine_fen(f'{fen} {side} - - 0 1'))
                self.get_legal_moves(f'{fen} {side} - - 0 1')
                return True
            except Exception:
                pass
        return False

    @staticmethod
    def _initial_repairs_allowed(quality):
        """Allow only stable, bounded non-king classifier repairs at attach."""
        repairs = tuple(quality.get('repairs', ()))
        if not repairs or len(repairs) > 2:
            return False
        if not quality.get('repair_stable', False):
            return False
        return all(old not in ('K', 'k') and new not in ('K', 'k')
                   for _r, _c, old, new in repairs)

    def _recognition_problem(self, img, reason):
        now = time.monotonic()
        if self._recognition_failed_since is None:
            self._recognition_failed_since = now
        if now - self._recognition_failed_since < 10 or now - self._diagnostic_at < 30:
            return
        if not self._grid_present:
            return  # Menus are waiting-for-game, not CNN failures.
        self._diagnostic_at = now
        os.makedirs(DEBUG_DIR, exist_ok=True)
        # Bounded rotating evidence; only the target client image is stored.
        slot = int(getattr(self, '_diagnostic_slot', 0)) % 3
        self._diagnostic_slot = slot + 1
        path = os.path.join(DEBUG_DIR, f'recognition_failure_{slot}')
        try:
            cv2.imwrite(path + '.png', img)
            with open(path + '.json', 'w', encoding='utf-8') as out:
                json.dump({'reason': reason, 'geometry': [self.win_x, self.win_y, self.win_w, self.win_h],
                           'cols': self.cols_logical, 'rows': self.rows_logical,
                           'quality': getattr(self, '_last_parse_quality', None),
                           'repairs': getattr(self.cnn, '_last_validation_cells', [])}, out, ensure_ascii=False)
        except Exception as exc:
            print(f'  diagnostic save failed: {exc}')
        print(f'  recognition unstable >10s: {reason}; evidence: {path}.png; '
              'see RECOVERY_GUIDE.md for calibration / decoration / CNN optimization')

    # ---------------- FEN / move conventions ----------------
    # FEN rows run from rank 9 to rank 0; UCI rank 0 is the bottom.
    # Internal moves remain screen-row based for overlay/log compatibility.
    # Convert ranks and legacy CLI side tokens ONLY at the engine boundary.

    def _side_tokens(self):
        """(user_token, opp_token) for engine queries."""
        return 'b', 'w'

    def board_to_fen(self, board):
        """Normalize the player's color to uppercase without changing ownership.

        When playing black the screen is rotated, so swap color labels for
        the entire board. NEVER recolor a piece according to its current row.
        """
        parts = []
        for i, row in enumerate(board):
            s, e = "", 0
            for p in row:
                if p is None:
                    e += 1
                    continue
                if not self.playing_red:
                    p = p.swapcase()
                if e:
                    s += str(e)
                    e = 0
                s += p
            if e:
                s += str(e)
            parts.append(s)
        return "/".join(parts)

    def uci_to_screen_cells(self, move):
        """Internal screen notation -> cells (engine UCI is converted on receipt)."""
        fc, fr = ord(move[0]) - ord('a'), int(move[1])
        tc, tr = ord(move[2]) - ord('a'), int(move[3])
        return (fr, fc), (tr, tc)

    # ---------------- calibration ----------------

    def _grid_score(self, img, x1, y1, x2, y2):
        """CNN score of a candidate grid (image-px coords)."""
        cw = (x2 - x1) / 8.0
        ch = (y2 - y1) / 9.0
        cols = [x1 + i * cw for i in range(9)]
        rows = [y1 + j * ch for j in range(10)]
        return self._grid_score_lists(img, cols, rows)

    def _grid_score_lists(self, img, cols, rows):
        """CNN score for explicit per-line col/row positions (image-px)."""
        cw = (max(cols) - min(cols)) / 8.0
        ch = (max(rows) - min(rows)) / 9.0
        old = sys.stdout
        sys.stdout = io.StringIO()
        try:
            board = self.cnn.parse_board(img, cols, rows, 1.0, 0, 0, cw, ch)
        finally:
            sys.stdout = old
        total, n = 0.0, 0
        for r in range(10):
            for c in range(9):
                if board[r][c] is not None and self.cnn._cell_probs[r][c] is not None:
                    total += float(self.cnn._cell_probs[r][c].max())
                    n += 1
        return total, n, board

    def load_calibration(self):
        """Load calibration (client-rect based; supports saved per-line grids)."""
        import json
        if not os.path.exists(CALIB_PATH):
            return False
        try:
            with open(CALIB_PATH) as f:
                d = json.load(f)
            if 'cols' in d and 'rows' in d:
                self._calib_cols_norm = [float(c) for c in d['cols']]
                self._calib_rows_norm = [float(r) for r in d['rows']]
            else:
                x1, y1 = float(d['rx1']), float(d['ry1'])
                x2, y2 = float(d['rx2']), float(d['ry2'])
                self._calib_cols_norm = [x1 + i * (x2 - x1) / 8
                                         for i in range(9)]
                self._calib_rows_norm = [y1 + j * (y2 - y1) / 9
                                         for j in range(10)]
            self._set_client_geometry(
                (self.win_x, self.win_y, self.win_w, self.win_h),
                apply_calibration=True)
            self.cell_w = (self.cols_logical[-1] - self.cols_logical[0]) / 8
            self.cell_h = (self.rows_logical[-1] - self.rows_logical[0]) / 9
            print(f"  Loaded calibration: cell={self.cell_w:.1f}x{self.cell_h:.1f}")
            return True
        except Exception:
            return False

    def calibrate(self, img=None):
        """Calibrate from lines, including boards with only a few pieces."""
        if img is None:
            img = self.screenshot_for_processing()
        if not self._locate_grid(img):
            print('  waiting for visible 9x10 board lines')
            return False
        with open(CALIB_PATH, 'w', encoding='utf-8') as out:
            json.dump({'cols': self._calib_cols_norm, 'rows': self._calib_rows_norm}, out)
        print(f'  saved line calibration: {CALIB_PATH}')
        return True

    def _legacy_piece_calibration(self, img=None):
        """Unused legacy routine; live/manual calibration uses lines above."""
        """Detect board rect (CV), fine-tune with CNN, save calib.json."""
        if img is None:
            img = self.screenshot_for_processing()
        if not self.load_cnn():
            print("  CNN unavailable, cannot calibrate")
            return False

        det = detect_board_rect(img)
        if det:
            x1, y1, x2, y2 = det
            print(f"  CV board rect: ({x1},{y1})-({x2},{y2})  "
                  f"{x2-x1}x{y2-y1} (window {img.shape[1]}x{img.shape[0]})")
        else:
            # Fallback: try default relative ratios (mac layout)
            print("  CV detection failed, trying default ratios...")
            x1 = self.win_w * 0.2985
            y1 = self.win_h * 0.1344
            x2 = self.win_w * 0.7027
            y2 = self.win_h * 0.9052

        # Fine-tune: margin around the wood rect is skin-dependent (this
        # skin needs ~7%/side, not the original 2-4%) — search a wider
        # range on the rough lattice, then snap to the exact lattice with
        # two methods and keep the one that parses best.
        bw, bh = x2 - x1, y2 - y1
        best = None
        for m in (0.03, 0.05, 0.07, 0.09, 0.11, 0.13):
            cand = (x1 + m * bw, y1 + m * bh,
                    x2 - m * bw, y2 - m * bh)
            score, n, _ = self._grid_score(img, *cand)
            if best is None or (score, n) > (best[0], best[1]):
                best = (score, n, cand)
        score, n, (bx1, by1, bx2, by2) = best
        print(f"  Rough grid: ({bx1:.0f},{by1:.0f})-({bx2:.0f},{by2:.0f}) "
              f"pieces={n} conf={score:.1f}")

        # Exact-lattice candidates: previously saved calibration (best
        # prior knowledge), piece-circle fit (HoughCircles per rank), and
        # dark-line projection refinement. Keep the one that parses best.
        candidates = []
        if os.path.exists(CALIB_PATH):
            try:
                import json
                with open(CALIB_PATH) as f:
                    d = json.load(f)
                if 'cols' in d and 'rows' in d:
                    candidates.append(('saved', (
                        [c * self.win_w for c in d['cols']],
                        [r * self.win_h for r in d['rows']])))
            except Exception:
                pass
        pf = refine_grid_pieces(img, bx1, by1, bx2, by2)
        if pf is not None:
            candidates.append(('pieces', pf))
        candidates.append(('lines', refine_grid(img, bx1, by1, bx2, by2,
                                               halfwin=40)))
        scored = []
        for kind, fit in candidates:
            f_score, f_n, _ = self._grid_score_lists(img, *fit)
            scored.append((f_score, f_n, kind, fit))
        f_score, f_n, kind, (cols, rows) = max(
            scored, key=lambda t: (t[0], t[1]))
        use_lists = (f_score, f_n) >= (score, n)
        if use_lists:
            print(f"  Exact grid ({kind}): conf={f_score:.1f} "
                  f"pieces={f_n} (adopted)")
            score, n = f_score, f_n
        else:
            print(f"  Exact grid worse ({f_score:.1f}/{f_n}), keeping rough")

        # Commit as absolute screen coords (client-rect relative px)
        if use_lists:
            self.cols_logical = [self.win_x + c for c in cols]
            self.rows_logical = [self.win_y + r for r in rows]
            save_cols, save_rows = cols, rows
        else:
            cw0, ch0 = (bx2 - bx1) / 8.0, (by2 - by1) / 9.0
            self.cols_logical = [self.win_x + bx1 + i * cw0 for i in range(9)]
            self.rows_logical = [self.win_y + by1 + j * ch0 for j in range(10)]
            save_cols = [c - self.win_x for c in self.cols_logical]
            save_rows = [r - self.win_y for r in self.rows_logical]
        self.cell_w = (self.cols_logical[-1] - self.cols_logical[0]) / 8
        self.cell_h = (self.rows_logical[-1] - self.rows_logical[0]) / 9
        self._calib_cols_norm = [c / self.win_w for c in save_cols]
        self._calib_rows_norm = [r / self.win_h for r in save_rows]
        with open(CALIB_PATH, 'w') as f:
            import json
            json.dump({
                'rx1': bx1 / self.win_w, 'ry1': by1 / self.win_h,
                'rx2': bx2 / self.win_w, 'ry2': by2 / self.win_h,
                'cols': [c / self.win_w for c in save_cols],
                'rows': [r / self.win_h for r in save_rows],
            }, f)
        print(f"  Saved {CALIB_PATH}")

        # Visual check image (final committed lattice)
        os.makedirs(DEBUG_DIR, exist_ok=True)
        vis = img.copy()
        for c in self.cols_logical:
            px = int(c - self.win_x)
            py1 = int(min(self.rows_logical) - self.win_y) - 8
            py2 = int(max(self.rows_logical) - self.win_y) + 8
            cv2.line(vis, (px, py1), (px, py2), (0, 0, 255), 1)
        for r in self.rows_logical:
            py = int(r - self.win_y)
            px1 = int(min(self.cols_logical) - self.win_x) - 8
            px2 = int(max(self.cols_logical) - self.win_x) + 8
            cv2.line(vis, (px1, py), (px2, py), (0, 0, 255), 1)
        if use_lists:
            _, _, board = self._grid_score_lists(img, cols, rows)
        else:
            _, _, board = self._grid_score(img, bx1, by1, bx2, by2)
        cv2.imwrite(os.path.join(DEBUG_DIR, 'win_calib.png'), vis)
        print(f"  Parsed board ({n} pieces):")
        self._print_board(board)
        print(f"  FEN: {self.board_to_fen(board)}")
        print(f"  Check: {os.path.join(DEBUG_DIR, 'win_calib.png')}")
        return (sum(p == 'K' for row in board for p in row) == 1 and
                sum(p == 'k' for row in board for p in row) == 1)

    def _print_board(self, board):
        for r in range(10):
            line = " "
            for c in range(9):
                p = board[r][c]
                line += f" {p}" if p else " ."
            print(line)

    # ---------------- trusted state and change explanation ----------------

    def _apply_uci(self, board, move):
        if not isinstance(move, str) or len(move) != 4:
            return None
        try:
            fc, fr = ord(move[0]) - ord('a'), int(move[1])
            tc, tr = ord(move[2]) - ord('a'), int(move[3])
        except (TypeError, ValueError):
            return None
        nb = [row[:] for row in board]
        if not (0 <= fr < 10 and 0 <= tr < 10 and
                0 <= fc < 9 and 0 <= tc < 9):
            return None
        if nb[fr][fc] is None or (fr, fc) == (tr, tc):
            return None
        if (nb[tr][tc] is not None and
                nb[fr][fc].isupper() == nb[tr][tc].isupper()):
            return None
        nb[tr][tc] = nb[fr][fc]
        nb[fr][fc] = None
        return nb

    def _grid_to_uci(self, sr, sc, dr, dc):
        return f"{chr(97 + sc)}{sr}{chr(97 + dc)}{dr}"

    def _diff_cells(self, old_board, new_board):
        return [(r, c, old_board[r][c], new_board[r][c])
                for r in range(10) for c in range(9)
                if old_board[r][c] != new_board[r][c]]

    def _boards_equal(self, a, b):
        return a is not None and b is not None and all(
            a[r][c] == b[r][c] for r in range(10) for c in range(9))

    def _move_touched_cells(self, moves):
        touched = set()
        for move in moves:
            (sr, sc), (dr, dc) = self.uci_to_screen_cells(move)
            touched.add((sr, sc))
            touched.add((dr, dc))
        return touched

    def _exact_moves(self, old_board, new_board, legal, ignore_cells=()):
        """Require a full board match; persistent UI errors are not evidence."""
        out = []
        for move in legal:
            result = self._apply_uci(old_board, move)
            # Engine perft output should be valid, but a stale/restarted
            # session can still produce a move that no longer applies to the
            # CNN-trusted board. Treat it as a non-match, never as a crash.
            if result is None:
                continue
            matches = True
            for r in range(10):
                for c in range(9):
                    if result[r][c] != new_board[r][c]:
                        matches = False
                        break
                if not matches:
                    break
            if matches:
                out.append(move)
        return out

    def _find_transition_for_turn(self, old_board, new_board, old_fen,
                                   turn, ignore_cells=(),
                                   allow_two_ply=True):
        """Match one or two complete legal moves from a trusted position.

        The previous implementation scored only source/destination cells and
        then committed the raw CNN board anyway. This function only returns a
        transition when the engine's legal move, applied to all 90 cells,
        exactly produces the candidate board.
        """
        if turn not in ('w', 'b'):
            return None
        legal = self.get_legal_moves(f"{old_fen} {turn} - - 0 1")

        direct = self._exact_moves(old_board, new_board, legal, ignore_cells)
        if len(direct) == 1:
            return (direct[0],)
        if len(direct) > 1:
            return None

        # A wrong/stale turn marker must not fall into the expensive two-ply
        # search before the other side gets a chance to explain a normal move.
        if not allow_two_ply:
            return None

        diffs = self._diff_cells(old_board, new_board)
        if len(diffs) > 4:
            return None

        # Enumerate exact two-ply explanations locally first. In a recapture,
        # the first destination may end with the second player's piece, or a
        # square may return to its old value. Do not require both endpoints
        # of the first move to appear in the final diff.
        matches = []
        deadline = time.monotonic() + 1.5
        other = 'b' if turn == 'w' else 'w'
        for m1 in legal:
            b1 = self._apply_uci(old_board, m1)
            if b1 is None:
                continue
            second_diffs = self._diff_cells(b1, new_board)
            if len(second_diffs) != 2:
                continue
            proposed = []
            for sr, sc, piece, remaining in second_diffs:
                if piece is None or remaining is not None:
                    continue
                for dr, dc, _old, arrived in second_diffs:
                    if arrived == piece and (sr, sc) != (dr, dc):
                        proposed.append(self._grid_to_uci(sr, sc, dr, dc))
            proposed = self._exact_moves(b1, new_board, proposed)
            if not proposed:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError('two-ply validation budget exhausted')
            legal2 = self.get_legal_moves(
                f"{self.board_to_fen(b1)} {other} - - 0 1")
            matches.extend((m1, m2) for m2 in proposed if m2 in legal2)
            if len(matches) > 1:
                return None
        return matches[0] if len(matches) == 1 else None

    def _find_transition(self, old_board, new_board, old_fen, turn,
                         ignore_cells=()):
        """Return (moves, next_turn), or None when the change is untrusted."""
        turns = [turn] if turn in ('w', 'b') else ['w', 'b']

        # First test a direct move for both sides.  This is cheap and fixes the
        # common case where a missed green countdown frame left ``turn`` stale
        # (especially when the player is black).  Without this pass the old
        # code entered the two-ply engine loop and could take minutes.
        direct_matches = []
        for candidate_turn in ('w', 'b'):
            try:
                result = self._find_transition_for_turn(
                    old_board, new_board, old_fen, candidate_turn,
                    ignore_cells, allow_two_ply=False)
            except Exception as exc:
                print(f"  direct transition unavailable: {exc}")
                continue
            if result:
                direct_matches.append((candidate_turn, result))
        if len(direct_matches) == 1:
            side, moves = direct_matches[0]
            self._last_transition_side = side
            next_turn = side
            for _ in moves:
                next_turn = 'b' if next_turn == 'w' else 'w'
            return moves, next_turn
        if len(direct_matches) > 1:
            self._last_transition_side = None
            return None

        matches = []
        for candidate_turn in turns:
            try:
                result = self._find_transition_for_turn(
                    old_board, new_board, old_fen, candidate_turn,
                    ignore_cells, allow_two_ply=True)
            except Exception as exc:
                print(f"  transition unavailable: {exc}")
                self._last_transition_side = None
                return None  # Unknown legality must not count as a non-match.
            if result:
                matches.append((candidate_turn, result))
        # If the starting side was unknown, accept only an unambiguous side.
        if len(matches) != 1:
            self._last_transition_side = None
            return None
        side, moves = matches[0]
        self._last_transition_side = side
        next_turn = side
        for _ in moves:
            next_turn = 'b' if next_turn == 'w' else 'w'
        return moves, next_turn

    def _hint_for_position(self, board, fen, turn):
        """Analyse one trusted position and return (gold_move, red_move, info)."""
        if turn not in ('w', 'b'):
            return None, None, ""
        search_ms = min(max(int(self.movetime_ms), 300), 2200)
        best, info, pv = self._analyse_screen(
            f"{fen} {turn} - - 0 1", movetime_ms=search_ms, pv_len=2)
        if not best or best == '(none)' or len(best) != 4:
            return None, None, info
        if not pv or pv[0] != best:
            pv = [best]

        after = self._apply_uci(board, best)
        if after is None:
            raise ValueError('engine bestmove does not apply to the board')
        if len(pv) >= 2 and self._apply_uci(after, pv[1]) is None:
            pv = [best]

        # Some Pikafish builds omit pv from their final info line. Compute the
        # reply from the exact post-move position in that case.
        if len(pv) < 2:
            after = self._apply_uci(board, best)
            if after is not None:
                other = 'b' if turn == 'w' else 'w'
                reply, _reply_info, _reply_pv = self._analyse_screen(
                    f"{self.board_to_fen(after)} {other} - - 0 1",
                    movetime_ms=min(800, search_ms), pv_len=1)
                if reply and reply != '(none)' and len(reply) == 4:
                    pv.append(reply)

        primary = pv[0] if pv else None
        response = pv[1] if len(pv) > 1 else None
        # Colors encode PV order, not ownership: yellow is the side to move,
        # red is the reply (including when the opponent moves first).
        return primary, response, info

    # ---------------- hint loop ----------------

    def _overlay_rect(self):
        pad = int(min(self.cell_w, self.cell_h) * 0.8)
        ox = int(min(self.cols_logical)) - pad
        oy = int(min(self.rows_logical)) - pad
        w = int(max(self.cols_logical) - min(self.cols_logical)) + 2 * pad
        h = int(max(self.rows_logical) - min(self.rows_logical)) + 2 * pad
        return ox, oy, w, h

    def render_arrows(self, gold_hint, red_hint, ox, oy, w, h):
        """Draw the current principal variation on a transparent canvas.

        Gold is the current side's best move; red is the other side's reply.
        """
        canvas = np.zeros((h, w, 4), np.uint8)
        cw = min(self.cell_w, self.cell_h)

        def cell_xy(r, c):
            return int(self.cols_logical[c] - ox), int(self.rows_logical[r] - oy)

        def arrow(r1, c1, r2, c2, color, thick, alpha):
            x1, y1 = cell_xy(r1, c1)
            x2, y2 = cell_xy(r2, c2)
            dx, dy = x2 - x1, y2 - y1
            L = math.hypot(dx, dy)
            if L < 1:
                return
            ang = math.atan2(dy, dx)
            head = min(cw * 0.45, L * 0.5)
            ex = x2 - head * 0.6 * math.cos(ang)
            ey = y2 - head * 0.6 * math.sin(ang)
            cv2.line(canvas, (x1, y1), (int(ex), int(ey)),
                     color + (alpha,), thick, cv2.LINE_AA)
            a1, a2 = ang + math.pi * 0.85, ang - math.pi * 0.85
            p1 = (int(x2 + head * math.cos(a1)), int(y2 + head * math.sin(a1)))
            p2 = (int(x2 + head * math.cos(a2)), int(y2 + head * math.sin(a2)))
            # OpenCV 4.13 requires (N,1,2) for fillPoly contours
            cv2.fillPoly(canvas, [np.array([(x2, y2), p1, p2],
                                          np.int32).reshape(3, 1, 2)],
                         color + (alpha,))

        if gold_hint:
            (r1, c1), (r2, c2) = self.uci_to_screen_cells(gold_hint)
            arrow(r1, c1, r2, c2, GOLD, max(6, int(cw * 0.16)), 235)
        if red_hint:
            (r1, c1), (r2, c2) = self.uci_to_screen_cells(red_hint)
            arrow(r1, c1, r2, c2, RED, max(4, int(cw * 0.11)), 190)
        return canvas

    def hint_loop(self):
        self.engine = EngineSession(verbose=True)
        if self.use_overlay:
            try:
                from win32_overlay import ClickThroughOverlay
                self.overlay = ClickThroughOverlay()
            except Exception as e:
                print(f"  [overlay unavailable: {e} — console hints only]")

        user_token, opp_token = self._side_tokens()
        confirmed, confirmed_fen = None, None
        confirmed_generation = None
        confirmed_color = None
        candidate, candidate_fen = None, None
        candidate_generation = None
        candidate_count = 0
        turn = self.start_turn if self.start_turn in ('w', 'b') else None
        gold_hint = red_hint = None
        analysis_key = None
        conditional_key = None
        state = 'WAITING_FOR_STABLE_FRAME'
        last_draw = None
        interval = 0.35
        last_rejected_signature = None

        print(f"=== Hint loop: player is always BOTTOM (color auto-detected), "
              f"opp arrows={self.opp_arrows}, movetime={self.movetime_ms}ms ===")
        print("    Ctrl+C to stop.\n")

        try:
            while not self.stop_flag:
                # Keep the overlay window procedure responsive.
                if self.overlay is not None:
                    self.overlay.pump()
                t0 = time.time()
                try:
                    self._refresh_client_geometry()
                    ready, reason = self.window_ready()
                except Exception as e:
                    ready, reason = False, f'window geometry unavailable ({e})'
                if not ready:
                    paused_state = 'PAUSED_NOT_READY'
                    if state != paused_state:
                        print(f"  paused: {reason}; hints hidden")
                    state = paused_state
                    gold_hint = red_hint = None
                    analysis_key = None
                    candidate = candidate_fen = None
                    candidate_count = 0
                    self._recovery_key = None
                    self._turn_ui_key = None
                    self._turn_ui_count = 0
                    if self.overlay is not None and last_draw is not None:
                        self.overlay.set_visible(False)
                        last_draw = None
                    time.sleep(interval)
                    continue
                try:
                    capture_generation = getattr(self, '_geometry_generation', 0)
                    img = self.screenshot_for_processing()
                    capture_generation = getattr(self, '_geometry_generation', 0)
                    board = self.parse_board_cnn(img)
                    self._refresh_client_geometry()
                except Exception as e:
                    print(f"  frame err: {e}")
                    state = 'WINDOW_MOVING_OR_CAPTURE_FAILED'
                    gold_hint = red_hint = None
                    analysis_key = None
                    candidate = candidate_fen = None
                    candidate_count = 0
                    if self.overlay is not None:
                        self.overlay.set_visible(False)
                        last_draw = None
                    time.sleep(interval)
                    continue
                generation = getattr(self, '_geometry_generation', 0)
                if generation != capture_generation:
                    state = 'WINDOW_MOVED'
                    gold_hint = red_hint = None
                    analysis_key = None
                    last_rejected_signature = None
                    candidate = candidate_fen = None
                    candidate_generation = None
                    candidate_count = 0
                    if self.overlay is not None:
                        self.overlay.set_visible(False)
                        last_draw = None
                    time.sleep(interval)
                    continue
                quality = getattr(self, '_last_parse_quality', None)
                if not board or quality is None or not self.detect_orientation_from_board(board):
                    if confirmed is None:
                        if state != 'WAITING_FOR_GAME':
                            print('  WAITING_FOR_GAME: no readable board; waiting for game')
                        state = 'WAITING_FOR_GAME'
                    else:
                        # A transient CNN/capture miss after a move is not a
                        # new game.  Keep the trusted board and side-to-move;
                        # throwing them away forces a slow full re-attach and
                        # is particularly painful while the opponent thinks.
                        if state != 'WAITING_FOR_TRACKED_BOARD':
                            print('  waiting for the tracked board to reappear; turn retained')
                        state = 'WAITING_FOR_TRACKED_BOARD'
                    self._recognition_problem(img, 'no complete confident board')
                    self._recovery_key = None
                    self._turn_ui_key = None
                    self._turn_ui_count = 0
                    if (confirmed is None and
                            time.monotonic() - self._recognition_failed_since >= 1.5):
                        confirmed = confirmed_fen = None
                        confirmed_color = None
                        turn = None
                        conditional_key = None
                    candidate = candidate_fen = None
                    candidate_generation = None
                    candidate_count = 0
                    gold_hint = red_hint = None
                    analysis_key = None
                    if self.overlay is not None:
                        self.overlay.set_visible(False)
                        last_draw = None
                    time.sleep(interval)
                    continue
                fen = self.board_to_fen(board)
                ui_turn = self._observe_ui_turn(img, (fen, generation))
                if confirmed_color is not None and confirmed_color != self.playing_red:
                    confirmed = confirmed_fen = None
                    confirmed_color = None
                    turn = None
                    candidate = candidate_fen = None
                    candidate_count = 0
                    gold_hint = red_hint = None
                    analysis_key = conditional_key = None
                    if self.overlay is not None:
                        self.overlay.set_visible(False)
                        last_draw = None

                recovery_key = (fen, self.playing_red, generation)
                if quality.get('corrections'):
                    self._recovery_key = None
                elif self._recovery_key == recovery_key:
                    self._recovery_count += 1
                else:
                    self._recovery_key = recovery_key
                    self._recovery_count = 1
                    self._recovery_since = time.monotonic()
                recoverable = (self._recovery_key == recovery_key and
                               self._recovery_count >= 3 and
                               time.monotonic() - self._recovery_since >= 0.7)

                if (confirmed is None and quality.get('corrections') and
                        not self._initial_repairs_allowed(quality)):
                    self._recognition_problem(img, 'initial board requires CNN repairs')
                    # No history exists yet to justify any repaired square.
                    candidate = candidate_fen = None
                    candidate_count = 0
                    if state != 'WAITING_FOR_CLEAR_INITIAL_BOARD':
                        print('  waiting for an unrepaired initial board')
                    state = 'WAITING_FOR_CLEAR_INITIAL_BOARD'
                    time.sleep(interval)
                    continue
                elif confirmed is None and quality.get('corrections'):
                    if state != 'WAITING_FOR_STABLE_FRAME_WITH_REPAIR':
                        print('  accepting bounded stable non-king repair candidate')
                    state = 'WAITING_FOR_STABLE_FRAME_WITH_REPAIR'

                if confirmed_generation is not None and generation != confirmed_generation:
                    state = 'WINDOW_MOVED'
                    last_rejected_signature = None
                    candidate = candidate_fen = None
                    candidate_generation = generation
                    candidate_count = 0
                    gold_hint = red_hint = None
                    analysis_key = None
                    confirmed_generation = generation
                    if self.overlay is not None:
                        self.overlay.set_visible(False)
                        last_draw = None

                same_candidate = (candidate is not None and
                                  candidate_generation == generation and
                                  candidate_fen == fen)
                if same_candidate:
                    candidate_count += 1
                else:
                    candidate, candidate_fen = board, fen
                    candidate_generation = generation
                    candidate_count = 1

                if confirmed is not None and not self._boards_equal(board, confirmed):
                    # Hide immediately, even before the second stable frame.
                    state = 'WAITING_FOR_STABLE_FRAME'
                    gold_hint = red_hint = None
                    analysis_key = None
                    if self.overlay is not None and last_draw is not None:
                        self.overlay.set_visible(False)
                        last_draw = None
                elif confirmed is not None and (gold_hint or red_hint):
                    state = 'HINT_READY'

                transition = None
                if confirmed is None:
                    state = 'WAITING_FOR_STABLE_FRAME'
                    if candidate_count >= 2 and self._snapshot_valid(candidate_fen):
                        try:
                            fen_prevalidate(self._engine_fen(
                                f"{candidate_fen} {turn or 'b'} - - 0 1"))
                        except ValueError as exc:
                            if candidate_fen != last_rejected_signature:
                                print(f"  initial board rejected: {exc}")
                            last_rejected_signature = candidate_fen
                            time.sleep(interval)
                            continue
                        confirmed, confirmed_fen = candidate, candidate_fen
                        confirmed_color = self.playing_red
                        confirmed_generation = generation
                        candidate_count = 2
                        turn = self._fresh_turn(confirmed_fen, ui_turn, allow_manual=True)
                        self._manual_turn_used = True
                        self._recognition_failed_since = None
                        self._tracked_turn = turn
                        state = 'READY' if turn else 'WAITING_FOR_TURN'
                        print(f"  state: {confirmed_fen}  (turn={turn or '?'}; {state})")
                        if turn is None:
                            print('  turn unknown: computing both conditional plans; waiting for UI marker or move')
                    elif candidate_count >= 2:
                        self._recognition_problem(img, 'engine rejected the recognized initial position')
                        state = 'WAITING_FOR_GAME'
                elif candidate_count >= 2 and not self._boards_equal(candidate, confirmed):
                    if candidate_fen == START_FEN:
                        # Opening positions have an unambiguous red-first
                        # turn; arbitrary snapshots use the recovery path.
                        confirmed, confirmed_fen = candidate, candidate_fen
                        confirmed_color = self.playing_red
                        confirmed_generation = generation
                        candidate = candidate_fen = None
                        candidate_count = 0
                        # The active countdown marker wins.  If it is absent,
                        # follow the same safe default as every other
                        # arbitrary position: the bottom/player side moves.
                        turn = ui_turn or user_token
                        self._tracked_turn = turn
                        gold_hint = red_hint = None
                        self.reply_hint = None
                        analysis_key = None
                        last_rejected_signature = None
                        state = 'NEW_GAME'
                        print(f"  [new game] state reset (turn={turn})")
                    else:
                        state = 'VALIDATING_MOVE'
                        transition = self._find_transition(
                            confirmed, candidate, confirmed_fen, turn)

                if (confirmed is not None and candidate_count >= 2 and
                        not self._boards_equal(candidate, confirmed)
                        and candidate_fen != START_FEN):
                    if transition is None:
                        # Never turn repeated classifier errors into trusted
                        # evidence or silently skip a real move on that cell.
                        diffs = self._diff_cells(confirmed, candidate)
                        signature = tuple(diffs)
                        if signature != last_rejected_signature:
                            print(f"  [resync] rejected {len(diffs)} "
                                  f"cell(s); trusted state retained; "
                                  f"diff={diffs[:6]}")
                        last_rejected_signature = signature
                        state = 'RESYNC_REQUIRED'
                        self._recognition_problem(img, 'candidate differs from tracked position')
                        gold_hint = red_hint = None
                        analysis_key = None
                        self.reply_hint = None
                        candidate = candidate_fen = None
                        candidate_generation = generation
                        candidate_count = 0
                    else:
                        moves, next_turn = transition
                        old_turn = turn
                        if (self._last_transition_side in ('w', 'b') and
                                old_turn in ('w', 'b') and
                                self._last_transition_side != old_turn):
                            print(f"  [turn] legal move corrected stale side "
                                  f"{old_turn} -> {self._last_transition_side}")
                        # Commit the canonical engine result, not raw CNN
                        # output. This prevents decoration cells or a
                        # one-frame classifier error from entering trusted
                        # state.
                        committed = [row[:] for row in confirmed]
                        for move in moves:
                            committed = self._apply_uci(committed, move)
                        confirmed = committed
                        confirmed_color = self.playing_red
                        self._recognition_failed_since = None
                        confirmed_fen = self.board_to_fen(confirmed)
                        confirmed_generation = generation
                        candidate = [row[:] for row in confirmed]
                        candidate_fen = confirmed_fen
                        candidate_count = 2
                        last_rejected_signature = None
                        turn = next_turn
                        self._tracked_turn = turn
                        gold_hint = red_hint = None
                        self.reply_hint = None
                        analysis_key = None
                        state = 'READY'
                        label = ' + '.join(moves)
                        print(f"  [move] {label} ({old_turn or '?'} -> {turn})")

                # Independently recover a stable, unrepaired full-board
                # snapshot. No move-history or two-ply distance requirement.
                if (recoverable and not self._boards_equal(board, confirmed) and
                        self._snapshot_valid(fen)):
                    confirmed = [row[:] for row in board]
                    confirmed_fen = fen
                    confirmed_color = self.playing_red
                    confirmed_generation = generation
                    candidate = [row[:] for row in board]
                    candidate_fen, candidate_generation, candidate_count = fen, generation, 2
                    turn = self._fresh_turn(fen, ui_turn)
                    self._tracked_turn = turn
                    self._recognition_failed_since = None
                    gold_hint = red_hint = None
                    analysis_key = conditional_key = None
                    state = 'READY' if turn else 'WAITING_FOR_TURN'
                    print(f"  [recovered snapshot] {fen} (turn={turn or '?'})")

                if self._boards_equal(board, confirmed):
                    self._recognition_failed_since = None
                    if ui_turn and turn != ui_turn:
                        if turn is not None:
                            print(f"  [turn] active marker corrected {turn} -> {ui_turn}")
                        turn = ui_turn
                        self._tracked_turn = turn
                        gold_hint = red_hint = None
                        analysis_key = None

                if (confirmed is not None and turn is None and candidate_count >= 2 and
                        candidate_fen == confirmed_fen and
                        conditional_key != (confirmed_fen, generation)):
                    conditional_key = (confirmed_fen, generation)
                    print('  CONDITIONAL ONLY: turn unknown; no unconditional arrows')
                    for side in ('b', 'w'):
                        try:
                            gold, red, info = self._hint_for_position(confirmed, confirmed_fen, side)
                            print(f"  IF {'PLAYER' if side == 'b' else 'OPPONENT'} TO MOVE: "
                                  f"gold={gold or '-'} red={red or '-'} ({self.engine.score_str(info)})")
                        except Exception as exc:
                            print(f'  conditional {side} unavailable: {exc}')
                    state = 'WAITING_FOR_TURN'

                # If the frame equals the trusted board, it is stable enough
                # to analyze. A geometry change or rejected candidate always
                # clears both arrows first.
                if (confirmed is not None and candidate_count >= 2 and
                        candidate_fen == confirmed_fen and turn in ('w', 'b') and
                        analysis_key != (confirmed_fen, turn, generation)):
                    state = 'ANALYZING'
                    analysis_key = (confirmed_fen, turn, generation)
                    analysis_generation = getattr(self, '_geometry_generation', 0)
                    try:
                        gold, red, info = self._hint_for_position(
                            confirmed, confirmed_fen, turn)
                        self._refresh_client_geometry()
                        ready, _reason = self.window_ready()
                        if not ready or analysis_generation != getattr(self, '_geometry_generation', 0):
                            gold = red = None
                            state = 'WINDOW_MOVED'
                            analysis_key = None
                        else:
                            # Search blocks capture. The player/AI may have
                            # moved meanwhile; verify again before publishing.
                            fresh = self.parse_board_cnn(self.screenshot_for_processing())
                            self._refresh_client_geometry()
                            ready, _reason = self.window_ready()
                            if (not ready or
                                    analysis_generation != self._geometry_generation or
                                    not self._boards_equal(fresh, confirmed)):
                                gold = red = None
                                state = 'BOARD_CHANGED_DURING_SEARCH'
                                analysis_key = None
                                candidate = candidate_fen = None
                                candidate_count = 0
                        if gold or red:
                            gold_hint, red_hint = gold, red
                            self.reply_hint = red_hint if turn == user_token else gold_hint
                            state = 'HINT_READY'
                            print(f"  HINT: gold={gold_hint or '-'} "
                                  f"red={red_hint or '-'} "
                                  f"({self.engine.score_str(info)})")
                        else:
                            gold_hint = red_hint = None
                            self.reply_hint = None
                            if state == 'ANALYZING':
                                state = 'NO_LEGAL_MOVE_OR_NO_PV'
                    except Exception as e:
                        print(f"  engine err: {e}")
                        state = 'ENGINE_TIMEOUT_OR_ERROR'
                        gold_hint = red_hint = None
                        self.reply_hint = None
                        analysis_key = None

                # Overlay
                if self.overlay is not None:
                    try:
                        if (state not in ('HINT_READY',) or
                                win32_screen.is_iconic(self.win_hwnd) or
                                not win32_screen.is_window_visible(self.win_hwnd)):
                            if last_draw is not None:
                                self.overlay.set_visible(False)
                                last_draw = None
                        else:
                            ox, oy, w, h = self._overlay_rect()
                            key = (ox, oy, w, h, gold_hint, red_hint)
                            if key != last_draw:
                                canvas = self.render_arrows(
                                    gold_hint, red_hint, ox, oy, w, h)
                                self.overlay.update(ox, oy, w, h, canvas)
                                last_draw = key
                    except Exception as e:
                        print(f"  overlay err: {e}")

                elapsed = time.time() - t0
                if elapsed < interval:
                    time.sleep(interval - elapsed)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            if self.overlay is not None:
                try:
                    self.overlay.destroy()
                except Exception:
                    pass
            if self.engine is not None:
                self.engine.close()


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Windows 天天象棋 hint bot")
    ap.add_argument('--list-windows', action='store_true')
    ap.add_argument('--calibrate', action='store_true')
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--autopilot', action='store_true')
    ap.add_argument('--no-overlay', action='store_true')
    ap.add_argument('--require-foreground', action='store_true',
                    help='仅前台运行；默认后台无遮挡也识别并显示箭头')
    ap.add_argument('--opp-arrows', type=int, default=1)
    ap.add_argument('--movetime', type=int, default=1200)
    ap.add_argument('--turn', choices=['auto', 'w', 'b'], default='auto',
                    help='auto=开局按红先推断; w/b=手动指定轮到哪方')
    args = ap.parse_args()

    win32_screen.set_dpi_aware()

    if args.list_windows:
        print("All visible top-level windows:")
        for hwnd, title, rect in win32_screen.list_windows():
            if title:
                print(f"  {hwnd:#010x}  {title!r}  {rect}")
        r = win32_screen.find_xiangqi_window()
        print(f"\nMatcher would pick: {r}")
        return

    r = win32_screen.find_xiangqi_window()
    if not r:
        print('  WAITING_FOR_GAME: waiting for 天天象棋 window')
        while not r:
            time.sleep(0.5)
            r = win32_screen.find_xiangqi_window()
    bot = WinHintBot(*r, opp_arrows=args.opp_arrows,
                     movetime_ms=args.movetime, start_turn=args.turn,
                     use_overlay=not args.no_overlay,
                     require_foreground=args.require_foreground)
    print('  visibility: ' + ('foreground and unobscured' if args.require_foreground
                              else 'unobscured (background allowed)'))
    bot.win_x, bot.win_y, bot.win_w, bot.win_h = \
        win32_screen.get_client_rect(r[0])

    if args.calibrate:
        bot.wait_for_window_ready()
        bot.calibrate()
        return

    if args.show:
        bot.wait_for_window_ready()
        img = bot.screenshot_for_processing()
        if not bot.load_cnn():
            print("CNN not available")
            return
        if not bot.load_calibration():
            if not bot.calibrate(img):
                print("Calibration failed")
                return
        os.makedirs(DEBUG_DIR, exist_ok=True)
        img, board = bot.read_initial_board()
        bot.detect_orientation_from_board(board)
        if args.turn == 'auto':
            user_tok, opp_tok = bot._side_tokens()
            bot.start_turn = ((user_tok if bot.playing_red else opp_tok)
                              if bot.board_to_fen(board) == START_FEN else None)
        else:
            bot.start_turn = args.turn
        fen = bot.board_to_fen(board)
        bot._print_board(board)
        print(f"FEN: {fen}  (turn={bot.start_turn})")
        gold_hint = red_hint = None
        try:
            bot.engine = EngineSession(verbose=False)
            gold_hint, red_hint, info = bot._hint_for_position(
                board, fen, bot.start_turn)
            print(f"HINT: gold={gold_hint or '-'} red={red_hint or '-'} "
                  f"({bot.engine.score_str(info)})")
        except Exception as e:
            print(f"engine err: {e}")
        # draw arrows on the crop around the board
        bw, bh = bot.win_w, bot.win_h
        x1 = int(0.03 * bw); y1 = int(0.03 * bh)
        x2 = int(0.97 * bw); y2 = int(0.97 * bh)
        crop = img[y1:y2, x1:x2].copy()
        pad = int(min(bot.cell_w, bot.cell_h) * 0.8)
        oxb = int(min(bot.cols_logical))
        oyb = int(min(bot.rows_logical))
        w = int(max(bot.cols_logical) - min(bot.cols_logical)) + 2 * pad
        h = int(max(bot.rows_logical) - min(bot.rows_logical)) + 2 * pad
        if gold_hint or red_hint:
            sub = bot.render_arrows(gold_hint, red_hint,
                                    oxb - pad, oyb - pad, w, h)
            composite(crop, sub, (oxb - pad) - x1, (oyb - pad) - y1)
        out = os.path.join(DEBUG_DIR, 'win_show.png')
        cv2.imwrite(out, crop)
        print(f"Preview: {out}")
        return

    if args.autopilot:
        print("Automatic moves are disabled in hint mode; no clicks were sent.")
        return

    # Default: live hint loop
    if not bot.load_cnn():
        print("CNN not available")
        return
    if not bot.load_calibration():
        print('  WAITING_FOR_GAME: waiting for a board to calibrate')
        while True:
            img = bot.screenshot_for_processing()
            if bot._grid_present and bot.calibrate(img):
                break
            time.sleep(0.5)
    # The loop detects color and establishes any starting position itself.
    bot.start_turn = args.turn
    bot.hint_loop()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped.')
