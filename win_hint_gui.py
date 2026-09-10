"""Calibrate once, then continuously observe and render hints in Qt.

Usage:
    python -X utf8 win_hint_gui.py
"""

import json
import os
import sys
import time


def _configure_qt_environment():
    """Point Qt at this interpreter's PyQt5 plugins before Qt is loaded.

    The desktop/runtime environment can export Qt variables belonging to a
    different Qt installation (often OpenCV, another Python, or a bundled
    application).  Qt reads these variables during QApplication creation, so
    correcting them after importing Qt is too late.
    """
    import PyQt5

    pyqt_root = os.path.dirname(os.path.abspath(PyQt5.__file__))
    qt_root = os.path.join(pyqt_root, "Qt5")
    plugin_root = os.path.join(qt_root, "plugins")
    platform_root = os.path.join(plugin_root, "platforms")
    qwindows = os.path.join(platform_root, "qwindows.dll")
    if not os.path.isfile(qwindows):
        raise RuntimeError(f"PyQt5 Windows platform plugin not found: {qwindows}")

    # Remove inherited values first.  In particular, setting only
    # QT_QPA_PLATFORM_PLUGIN_PATH is insufficient when QT_PLUGIN_PATH still
    # points at another Qt distribution.
    os.environ.pop("QT_PLUGIN_PATH", None)
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    os.environ.pop("QT_QPA_PLATFORM", None)
    os.environ["QT_PLUGIN_PATH"] = plugin_root
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = platform_root
    os.environ["QT_QPA_PLATFORM"] = "windows"


_configure_qt_environment()

try:
    from PyQt5.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal, pyqtSlot
    from PyQt5.QtWidgets import (
        QApplication, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
        QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout,
        QSizePolicy, QWidget,
    )
    from PyQt5.QtGui import QImage, QPixmap
except ImportError as exc:  # pragma: no cover - only used on Windows host
    raise SystemExit(
        "PyQt5 is required for the diagnostics panel; install it with "
        "python -m pip install PyQt5"
    ) from exc

import cv2

import win32_screen
from board_geometry import detect_grid, refine_grid_centers
from engine_session import EngineSession
from gui_pipeline import (annotate_board, destroy_overlay, hide_overlay,
                           save_step, show_overlay)
from win_hint import (
    CALIB_PATH, WinHintBot, detect_board_rect, refine_grid,
    refine_grid_pieces,
)


class TaskWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    @pyqtSlot()
    def run(self):
        try:
            self.finished.emit(self.fn())
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class GuiBot(WinHintBot):
    def screenshot_for_processing(self):
        # The CNN's second shot must not enter the CLI's indefinite wait or
        # automatic grid search. Qt will retry on its next timer tick.
        self._refresh_client_geometry()
        ready, reason = self.window_ready()
        if not ready:
            raise RuntimeError(reason)
        return self._capture_settled_frame()


class DebugPanel(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("天天象棋提示调试面板")
        self.resize(1100, 900)
        self.bot = None
        self._thread = None
        self._worker = None
        self.last_result = None
        self._auto = False
        self._next_action = None
        self._candidate = None
        self._accepted = None
        self._calibrated_geometry = None
        self._closing = False
        self._pending_command = None
        self._pause_requested = False
        self._last_error = None
        self._screenshot_requested = False
        self._screenshot_active = False
        self._screenshot_resume = False
        self._timer = QTimer(self)
        self._timer.setInterval(400)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        self.window_value = QLabel("未连接")
        self.calibration_value = QLabel("未加载")
        self.player_value = QLabel("未知")
        self.turn_value = QLabel("未知")
        self.fen_value = QPlainTextEdit()
        self.fen_value.setReadOnly(True)
        self.fen_value.setMaximumHeight(60)
        self.quality_value = QLabel("—")
        self.hint_value = QPlainTextEdit()
        self.hint_value.setReadOnly(True)
        self.hint_value.setMaximumHeight(75)
        self.status_value = QLabel("就绪。请先点击“校准棋盘”或“重新识别”。")
        self.status_value.setWordWrap(True)

        self.capture_button = QPushButton("抓取截图")
        self.calibrate_button = QPushButton("校准棋盘")
        self.recognize_button = QPushButton("重新识别")
        self.hint_button = QPushButton("计算提示线")
        self.show_overlay_button = QPushButton("显示提示线")
        self.hide_overlay_button = QPushButton("隐藏提示线")
        self.visibility_button = QPushButton("显示模式：始终显示")
        self.visibility_button.setCheckable(True)
        self.visibility_button.setToolTip("点击切换仅前台显示；始终显示模式在遮挡或截图时保留已有提示，暂停识别。")
        self.visibility_button.toggled.connect(self._visibility_changed)
        self.screenshot_button = QPushButton("进入截图模式")
        self.screenshot_button.setCheckable(True)
        self.screenshot_button.toggled.connect(self._screenshot_changed)
        self.clear_button = QPushButton("清空结果")
        self.capture_button.clicked.connect(self.start_capture)
        self.calibrate_button.clicked.connect(self.start_calibration)
        self.recognize_button.clicked.connect(self.start_recognition)
        self.hint_button.clicked.connect(self.start_hint)
        self.show_overlay_button.clicked.connect(self.show_last_overlay)
        self.hide_overlay_button.clicked.connect(self.hide_current_overlay)
        self.clear_button.clicked.connect(self.clear_result)

        buttons = QHBoxLayout()
        buttons.addWidget(self.capture_button)
        buttons.addWidget(self.calibrate_button)
        buttons.addWidget(self.recognize_button)
        buttons.addWidget(self.hint_button)
        buttons.addWidget(self.show_overlay_button)
        buttons.addWidget(self.hide_overlay_button)
        buttons.addWidget(self.visibility_button)
        buttons.addWidget(self.screenshot_button)
        buttons.addWidget(self.clear_button)

        info = QFormLayout()
        info.addRow("目标窗口", self.window_value)
        info.addRow("校准状态", self.calibration_value)
        info.addRow("我方颜色", self.player_value)
        info.addRow("识别到的走子方", self.turn_value)
        info.addRow("识别 FEN", self.fen_value)
        info.addRow("识别质量", self.quality_value)
        info.addRow("提示线", self.hint_value)

        group = QGroupBox("当前状态")
        group.setLayout(info)
        self.preview = QLabel("这里显示每一步保存的诊断截图")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(760, 400)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview.setStyleSheet("QLabel { background: #202124; color: #d9d9d9; }")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(200)
        self.log.setPlaceholderText("操作日志")

        root = QVBoxLayout()
        root.addLayout(buttons)
        root.addWidget(group)
        root.addWidget(QLabel("诊断截图（截图 / 识别标注 / 提示线复合图）"))
        root.addWidget(self.preview, 1)
        root.addWidget(QLabel("诊断日志"))
        root.addWidget(self.log, 0)
        root.addWidget(self.status_value)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        self._append("面板已启动：不会移动鼠标，也不会激活天天象棋窗口。")
        self._append("开局或移动后点击校准，即自动识别并持续更新提示；隐藏提示线会暂停自动更新。")
        self._append("截图会保存到 debug/gui_pipeline/，提示线按钮只更新点击穿透悬浮层。")

    def _append(self, text):
        self.log.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {text}")

    def _set_busy(self, busy, label):
        for button in (self.capture_button, self.calibrate_button,
                       self.recognize_button, self.hint_button,
                       self.show_overlay_button, self.hide_overlay_button,
                       self.clear_button):
            button.setEnabled(not busy)
        if busy:
            self.status_value.setText(label)
        # These controls queue work even while CNN/search is running.
        for button in (self.calibrate_button, self.recognize_button,
                       self.hide_overlay_button):
            button.setEnabled(True)

    @staticmethod
    def _valid_grid(cols, rows, width, height):
        """Reject duplicate/degenerate calibration points before saving."""
        if len(cols) != 9 or len(rows) != 10 or width <= 0 or height <= 0:
            return False
        if any(not (0 <= float(x) < width) for x in cols):
            return False
        if any(not (0 <= float(y) < height) for y in rows):
            return False
        dx = [float(cols[i + 1]) - float(cols[i]) for i in range(8)]
        dy = [float(rows[i + 1]) - float(rows[i]) for i in range(9)]
        min_step = max(0.02, min(float(width), float(height)) * 0.02)
        if min(dx, default=0) < min_step or min(dy, default=0) < min_step:
            return False
        # Physical board cells are square. Normalized coordinates use
        # different x/y denominators and must be checked after conversion.
        if width > 1 and height > 1:
            ratio = (sum(dx)/8) / (sum(dy)/9)
            if not 0.95 <= ratio <= 1.05:
                return False
        return (max(dx) / min(dx) <= 1.35 and
                max(dy) / min(dy) <= 1.35)

    def _run_task(self, label, fn, done):
        if self._screenshot_requested:
            return
        if self._thread is not None:
            return
        self._set_busy(True, label)
        if label != "正在重新识别棋盘…" or not self._auto:
            self._append(label)
        thread = QThread(self)
        worker = TaskWorker(fn)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(done)
        worker.failed.connect(self._task_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._task_finished)
        self._thread = thread
        self._worker = worker
        thread.start()

    @pyqtSlot()
    def _task_finished(self):
        self._thread = None
        self._worker = None
        self._set_busy(False, "就绪")
        if self._closing:
            self.close()

    @pyqtSlot(str)
    def _task_failed(self, message):
        self._candidate = None
        self._accepted = None
        if self.bot is not None and not self._retain_obscured_hint():
            hide_overlay(self.bot)
        if self._last_error != message:
            self._append("等待恢复：" + message)
            self._last_error = message
        self.status_value.setText("操作失败：" + message)
        self._set_busy(False, self.status_value.text())

    def _ensure_bot(self, require_calibration=True):
        if self.bot is not None and not win32_screen.user32.IsWindow(self.bot.win_hwnd):
            raise RuntimeError("游戏窗口已关闭，请重新启动面板连接新窗口")
        if self.bot is None:
            found = win32_screen.find_xiangqi_window()
            if not found:
                raise RuntimeError("找不到“天天象棋”窗口")
            self.bot = GuiBot(*found, start_turn='auto', use_overlay=False)
            self.bot._set_client_geometry(
                win32_screen.get_client_rect(found[0]), apply_calibration=False)
            self.bot.load_cnn()
            # Keep the previous normalized grid available as a fallback for
            # a window move.  A line detector miss must not make the manual
            # calibration button fail when the board itself did not move.
            self.bot.load_calibration()
            if not self._valid_grid(
                    self.bot._calib_cols_norm or [],
                    self.bot._calib_rows_norm or [],
                    1.0, 1.0):
                # Normalized points must also be strictly increasing.  The
                # width/height check above uses pixel units, so use a second
                # explicit normalized spacing check here.
                self.bot._calib_cols_norm = None
                self.bot._calib_rows_norm = None
        else:
            self.bot._refresh_client_geometry()
        if require_calibration and not self.bot._calib_cols_norm:
            if not self.bot.load_calibration():
                raise RuntimeError("没有校准数据，请先点击“校准棋盘”")
        if require_calibration:
            self.bot._grid_present = True
        return self.bot

    @staticmethod
    def _capture(bot):
        bot._refresh_client_geometry()
        ready, reason = bot.window_ready()
        deadline = time.monotonic() + 1.0
        while not ready and reason == 'waiting for window geometry to settle' and time.monotonic() < deadline:
            time.sleep(0.05)
            bot._refresh_client_geometry()
            ready, reason = bot.window_ready()
        if not ready:
            raise RuntimeError(f"目标窗口不可用：{reason}")
        # Explicitly avoid PrintWindow activation/focus changes.  The normal
        # screen fallback is also safe when the target is on another monitor.
        return bot._capture_settled_frame()

    def _set_preview(self, path):
        """Show a saved diagnostic image without keeping a large array in Qt."""
        image = QImage(path)
        if image.isNull():
            self._append(f"截图无法载入：{path}")
            return
        pixmap = QPixmap.fromImage(image)
        self.preview.setPixmap(pixmap.scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.preview.setToolTip(path)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        path = self.preview.toolTip()
        if path and os.path.isfile(path):
            self._set_preview(path)

    def start_capture(self):
        def task():
            bot = self._ensure_bot(require_calibration=True)
            img = self._capture(bot)
            annotated = annotate_board(bot, img)
            path = save_step(annotated, "01_capture_grid.png")
            return {'path': path, 'shape': img.shape[:2],
                    'elapsed': 0.0}

        self._run_task("正在抓取目标窗口截图…", task, self._capture_done)

    @pyqtSlot(object)
    def _capture_done(self, result):
        self._set_preview(result['path'])
        self.window_value.setText(self._window_text())
        self.calibration_value.setText(
            f"已加载，网格 {self.bot.cell_w:.1f}x{self.bot.cell_h:.1f}")
        self.status_value.setText(
            f"截图完成：{result['shape'][1]}x{result['shape'][0]}；"
            f"文件已保存到 {result['path']}")
        self._append(f"截图完成：{result['path']}")

    def start_calibration(self):
        self._pause_requested = False
        if self._thread is not None:
            self._pending_command = 'calibrate'
            self._auto = False
            self.status_value.setText("已收到校准请求，当前操作结束后执行…")
            return
        self._auto = False
        self._next_action = None
        self._candidate = self._accepted = None
        if self.bot is not None:
            hide_overlay(self.bot)
        def task():
            bot = self._ensure_bot(require_calibration=False)
            img = self._capture(bot)
            geometry = (bot.win_x, bot.win_y, bot.win_w, bot.win_h)
            started = time.perf_counter()
            # Manual calibration must inspect the current scene, not accept
            # an old grid solely because its coordinates are well-formed.
            fit = detect_grid(img)

            # Fresh piece/line fallback for layouts whose wood joins a panel.
            rough = None
            if fit is not None:
                rough = (fit[0][0], fit[1][0], fit[0][-1], fit[1][-1])
            else:
                rough = detect_board_rect(img)

            if rough is not None:
                x1, y1, x2, y2 = rough
                candidate = refine_grid_pieces(img, x1, y1, x2, y2)
                if fit is None and candidate is not None and self._valid_grid(
                        candidate[0], candidate[1], img.shape[1], img.shape[0]):
                    fit = candidate
                elif fit is None:
                    candidate = refine_grid(img, x1, y1, x2, y2, halfwin=55)
                    if self._valid_grid(candidate[0], candidate[1],
                                        img.shape[1], img.shape[0]):
                        fit = candidate
                    else:
                        # A midgame rank need not contain nine pieces. If
                        # independent x peaks fail, use the board rectangle's
                        # center and y lattice spacing to propose square cells.
                        # The full CNN validation below must still accept it.
                        step = (candidate[1][-1] - candidate[1][0]) / 9
                        center = (x1 + x2) / 2
                        square = ([center + (i-4)*step for i in range(9)], candidate[1])
                        if self._valid_grid(*square, img.shape[1], img.shape[0]):
                            fit = square

            if fit is not None and not self._valid_grid(
                    fit[0], fit[1], img.shape[1], img.shape[0]):
                fit = None
            elapsed = (time.perf_counter() - started) * 1000
            if win32_screen.get_client_rect(bot.win_hwnd) != geometry:
                raise RuntimeError("校准期间窗口移动，请松开窗口后再次校准")
            if fit is None:
                raise RuntimeError("棋盘线检测失败：请确保棋盘完整可见且无遮挡")

            cols, rows = refine_grid_centers(img, *fit)
            if not self._valid_grid(cols, rows, bot.win_w, bot.win_h):
                raise RuntimeError("校准网格比例异常，未覆盖原有校准")
            previous_cols = bot._calib_cols_norm
            previous_rows = bot._calib_rows_norm
            previous_grid = (bot.cols_logical, bot.rows_logical, bot.cell_w, bot.cell_h,
                             bot._grid_present)
            bot._calib_cols_norm = [float(x) / bot.win_w for x in cols]
            bot._calib_rows_norm = [float(y) / bot.win_h for y in rows]
            bot._set_client_geometry(
                (bot.win_x, bot.win_y, bot.win_w, bot.win_h),
                apply_calibration=True)
            bot._grid_present = True
            # Validate the proposed crop before committing it to disk.
            board = bot.parse_board_cnn(img)
            quality = getattr(bot, '_last_parse_quality', None)
            if (board is None or quality is None or quality.get('corrections') or
                    not bot.detect_orientation_from_board(board)):
                bot._calib_cols_norm = previous_cols
                bot._calib_rows_norm = previous_rows
                bot._set_client_geometry(geometry, apply_calibration=True)
                (bot.cols_logical, bot.rows_logical, bot.cell_w, bot.cell_h,
                 bot._grid_present) = previous_grid
                raise RuntimeError("新校准未通过棋子验证，原校准未覆盖；请等待走子动画结束再校准")
            bot._grid_detector_miss_announced = False
            with open(CALIB_PATH, 'w', encoding='utf-8') as out:
                json.dump({
                    'cols': bot._calib_cols_norm,
                    'rows': bot._calib_rows_norm,
                }, out, ensure_ascii=False)
            path = save_step(annotate_board(bot, img),
                             "00_calibration.png")
            return {
                'kind': 'calibrated',
                'elapsed': elapsed,
                'message': f'新网格已保存：cell={bot.cell_w:.1f}x{bot.cell_h:.1f}',
                'path': path,
                'geometry': geometry,
            }

        self._run_task("正在执行手动棋盘校准…", task, self._calibration_done)

    @pyqtSlot(object)
    def _calibration_done(self, result):
        if self._closing or self._pending_command:
            return
        self._calibrated_geometry = result['geometry']
        self._auto = not self._pause_requested
        self._next_action = 'recognize'
        self._set_preview(result['path'])
        self.calibration_value.setText(result['message'])
        self.window_value.setText(self._window_text())
        self.status_value.setText(
            f"校准完成，耗时 {result['elapsed']:.0f}ms。已开启自动识别与提示。")
        self._append(result['message'])
        self._append(f"校准截图：{result['path']}")
        if result['kind'] == 'calibration-kept':
            self._append("提示：当前按钮保留了旧网格；如果棋盘在窗口内部换了位置，请重新校准。")

    def start_recognition(self):
        self._pause_requested = False
        if self._thread is not None:
            self._pending_command = 'recognize'
            self._auto = False
            self.status_value.setText("已收到重新识别请求，当前操作结束后执行…")
            return
        self._auto = True
        self._candidate = self._accepted = None
        self._next_action = None
        if self.bot is not None:
            hide_overlay(self.bot)
        self._recognize_once()

    def _recognize_once(self):
        def task():
            bot = self._ensure_bot(require_calibration=True)
            started = time.perf_counter()
            img = self._capture(bot)
            geometry = (bot.win_x, bot.win_y, bot.win_w, bot.win_h)
            board = bot.parse_board_cnn(img)
            elapsed = (time.perf_counter() - started) * 1000
            quality = getattr(bot, '_last_parse_quality', None)
            if board is None or quality is None:
                raise RuntimeError(
                    "CNN 没有返回可信棋盘；请确认棋盘无遮挡后再次点击“重新识别”")
            oriented = bot.detect_orientation_from_board(board)
            if not oriented or (quality.get('corrections') and
                                not bot._initial_repairs_allowed(quality)):
                raise RuntimeError("局面尚不可信，等待下一帧；持续失败请重新校准")
            fen = bot.board_to_fen(board)
            # The detector is deliberately sampled twice on the same stable
            # frame.  If no green countdown ring is present, use the requested
            # player-side fallback instead of inventing an opponent turn.
            ui_turn = bot._observe_ui_turn(img, (fen, bot._geometry_generation))
            if ui_turn is None:
                ui_turn = bot._observe_ui_turn(img, (fen, bot._geometry_generation))
            marker_detected = ui_turn in ('b', 'w')
            turn = ui_turn or 'b'
            bot._gui_last_board = board
            bot._gui_last_fen = fen
            bot._gui_last_turn = turn
            bot._gui_last_quality = quality
            bot._gui_last_image = img.copy()
            path = None
            if (fen, turn, bot.playing_red) != self._accepted:
                path = save_step(annotate_board(bot, img, board, quality), "02_recognition.png")
            return {
                'fen': fen,
                'oriented': oriented,
                'playing_red': bot.playing_red if oriented else None,
                'turn': turn,
                'marker_detected': marker_detected,
                'quality': quality,
                'elapsed': elapsed,
                'pieces': sum(p is not None for row in board for p in row),
                'path': path,
                'geometry': geometry,
            }

        self._run_task("正在重新识别棋盘…", task, self._recognition_done)

    @pyqtSlot(object)
    def _recognition_done(self, result):
        if not self._result_current(result):
            return
        key = (result['fen'], result['turn'], result['playing_red'])
        self._last_error = None
        if self._accepted == key:
            if self.bot.overlay is not None:
                self.bot.overlay.set_visible(True)
            return
        hide_overlay(self.bot)
        if self._candidate != key:
            self._candidate = key
            self.status_value.setText("局面变化，等待第二帧确认…")
            return
        self._accepted = key
        self._next_action = 'hint'
        self.last_result = result
        self._set_preview(result['path'])
        color = ('红方' if result['playing_red'] else '黑方') \
            if result['playing_red'] is not None else '无法确定'
        turn = '我方' if result['turn'] == 'b' else '对手'
        quality = result['quality']
        repairs = quality.get('repairs', ())
        overrides = quality.get('occupancy_overrides', ())
        self.window_value.setText(self._window_text())
        self.calibration_value.setText(
            f"已加载，网格 {self.bot.cell_w:.1f}x{self.bot.cell_h:.1f}")
        self.player_value.setText(color)
        marker = '绿色倒计时框' if result['marker_detected'] else '未检测到绿色，按我方默认'
        active_red = result['playing_red'] if result['turn'] == 'b' else not result['playing_red']
        self.turn_value.setText(f"{'红方' if active_red else '黑方'} / {turn}（{marker}）")
        self.fen_value.setPlainText(result['fen'])
        self.quality_value.setText(
            f"棋子 {result['pieces']}/90；min_any={quality.get('min_any', 0):.3f}；"
            f"min_piece={quality.get('min_piece', 0):.3f}；"
            f"修复={len(repairs)}；占位覆盖={len(overrides)}；"
            f"耗时={result['elapsed']:.0f}ms")
        self.hint_value.setPlainText("尚未计算。")
        self.status_value.setText("识别完成；如果颜色或走子方不对，调整窗口后再次点击“重新识别”。")
        self._append(
            f"识别完成：{color}，{turn}走，{result['pieces']}/90，"
            f"耗时 {result['elapsed']:.0f}ms；截图 {result['path']}")

    def start_hint(self):
        if self.bot is None or not getattr(self.bot, '_gui_last_board', None):
            self.status_value.setText("请先点击“重新识别”。")
            return

        def task():
            bot = self._ensure_bot(require_calibration=True)
            if bot.engine is None:
                bot.engine = EngineSession(verbose=False)
            started = time.perf_counter()
            gold, red, info = bot._hint_for_position(
                bot._gui_last_board, bot._gui_last_fen, bot._gui_last_turn)
            # Search runs asynchronously; never display a result if pieces
            # changed while the engine was thinking.
            latest = self._capture(bot)
            latest_board = bot.parse_board_cnn(latest)
            if latest_board is None or bot.board_to_fen(latest_board) != bot._gui_last_fen:
                raise RuntimeError("分析期间局面变化，正在重新识别")
            latest_turn = bot._observe_ui_turn(latest, (bot._gui_last_fen, bot._geometry_generation))
            latest_turn = bot._observe_ui_turn(latest, (bot._gui_last_fen, bot._geometry_generation)) or latest_turn or 'b'
            if latest_turn != bot._gui_last_turn:
                raise RuntimeError("分析期间走子方变化，正在重新识别")
            bot._gui_last_gold = gold
            bot._gui_last_red = red
            # The saved composite proves the render transform independently
            # of the topmost Win32 window and is also useful when the overlay
            # is excluded from screen capture by Windows.
            annotated = annotate_board(
                bot, bot._gui_last_image, bot._gui_last_board,
                bot._gui_last_quality, gold, red)
            path = save_step(annotated, "03_hint_lines.png")
            return {
                'gold': gold or '-',
                'red': red or '-',
                'score': bot.engine.score_str(info),
                'elapsed': (time.perf_counter() - started) * 1000,
                'path': path,
                'geometry': (bot.win_x, bot.win_y, bot.win_w, bot.win_h),
            }

        self._run_task("正在计算提示线…", task, self._hint_done)

    @pyqtSlot(object)
    def _hint_done(self, result):
        if not self._result_current(result):
            return
        try:
            show_overlay(self.bot, self.bot._gui_last_gold, self.bot._gui_last_red)
        except Exception as exc:
            self._auto = False
            self._task_failed(f"悬浮层创建失败：{exc}")
            return
        self._set_preview(result['path'])
        first = '我方' if self.bot._gui_last_turn == 'b' else '对手'
        reply = '对手' if self.bot._gui_last_turn == 'b' else '我方'
        text = (f"黄线（{first}当前最佳走法）：{result['gold']}\n"
                f"红线（{reply}随后应对）：{result['red']}\n"
                f"评估：{result['score']}；耗时：{result['elapsed']:.0f}ms")
        self.hint_value.setPlainText(text)
        self.status_value.setText(
            "提示计算完成；悬浮提示线已显示在目标窗口，复合截图已保存。")
        self._append(text.replace('\n', '；'))
        self._append(f"提示线截图：{result['path']}")

    def show_last_overlay(self):
        self._pause_requested = False
        self._auto = True
        self._accepted = self._candidate = None
        self._next_action = 'recognize'
        self.status_value.setText("正在核验当前局面后恢复自动提示…")

    def hide_current_overlay(self):
        self._pause_requested = True
        self._auto = False
        self._pending_command = None
        self._next_action = None
        if self.bot is not None:
            hide_overlay(self.bot)
        self.status_value.setText("提示线已隐藏。")
        self._append("提示线已隐藏")

    def clear_result(self):
        self._auto = False
        self._next_action = None
        self._accepted = self._candidate = None
        self.last_result = None
        if self.bot is not None:
            hide_overlay(self.bot)
        self.fen_value.clear()
        self.hint_value.clear()
        self.quality_value.setText("—")
        self.player_value.setText("未知")
        self.turn_value.setText("未知")
        self.status_value.setText("结果已清空；点击“重新识别”开始下一次检查。")

    def _window_text(self):
        if self.bot is None:
            return "未连接"
        return f"{self.bot.win_title}  ({self.bot.win_w}x{self.bot.win_h})"

    def closeEvent(self, event):
        self._timer.stop()
        self._auto = False
        if self._thread is not None:
            self._closing = True
            self.status_value.setText("正在等待当前后台操作结束并释放资源…")
            event.ignore()
            return
        if self.bot is not None:
            destroy_overlay(self.bot)
            if self.bot.engine is not None:
                self.bot.engine.close()
        event.accept()

    def _result_current(self, result):
        if self._closing or self._pending_command or not self._auto:
            return False
        geometry = win32_screen.get_client_rect(self.bot.win_hwnd)
        if result['geometry'] != geometry or self._visibility_paused():
            if result['geometry'] != geometry or not self._retain_obscured_hint():
                hide_overlay(self.bot)
            self._candidate = self._accepted = None
            self.status_value.setText("窗口位置或可见性变化，已丢弃过期结果。")
            return False
        return True

    def _visibility_changed(self, foreground_only):
        self.visibility_button.setText(
            "显示模式：仅前台" if foreground_only else "显示模式：始终显示")
        if self.bot is not None and self.bot.overlay is not None and self._auto:
            visible = (not self._visibility_paused() or self._retain_obscured_hint())
            self.bot.overlay.set_visible(visible)

    def _screenshot_changed(self, enabled):
        if enabled:
            self._screenshot_resume = self._auto
            self._screenshot_requested = True
            self._auto = False
            self.screenshot_button.setText("退出截图模式")
            self.status_value.setText("正在等待当前操作结束，准备保留提示线供截图…")
        else:
            if self._screenshot_active and self.bot is not None and self.bot.overlay is not None:
                try:
                    self.bot.overlay.set_capture_excluded(self._screenshot_old_exclusion)
                except Exception as exc:
                    self.status_value.setText(f"恢复截图排除失败：{exc}")
                    self.screenshot_button.blockSignals(True)
                    self.screenshot_button.setChecked(True)
                    self.screenshot_button.blockSignals(False)
                    return
            self._screenshot_requested = self._screenshot_active = False
            self._auto = self._screenshot_resume
            self._accepted = self._candidate = None
            self._next_action = None
            self.screenshot_button.setText("进入截图模式")
            self._set_busy(self._thread is not None, "正在恢复…")
            self.visibility_button.setEnabled(True)
            self.status_value.setText("已退出截图模式，恢复原来的运行状态。")

    def _prepare_screenshot(self):
        if self._thread is not None or self._screenshot_active:
            return
        if self.bot is None or self.bot.overlay is None:
            self.screenshot_button.setChecked(False)
            self.status_value.setText("请先校准并生成提示线，再进入截图模式。")
            return
        overlay = self.bot.overlay
        self._screenshot_old_exclusion = overlay.capture_excluded
        try:
            overlay.set_capture_excluded(False)
        except Exception as exc:
            self.screenshot_button.setChecked(False)
            self.status_value.setText(f"无法开启截图模式：{exc}")
            return
        self._screenshot_active = True
        self.turn_value.setText(self.turn_value.text() + "【截图快照，非实时】")
        if (self._calibrated_geometry == win32_screen.get_client_rect(self.bot.win_hwnd)
                and not win32_screen.is_iconic(self.bot.win_hwnd)):
            overlay.set_visible(True)
        for button in (self.capture_button, self.calibrate_button, self.recognize_button,
                       self.hint_button, self.show_overlay_button, self.hide_overlay_button,
                       self.clear_button, self.visibility_button):
            button.setEnabled(False)
        self.status_value.setText("截图模式已就绪：识别已暂停，提示线允许被截取。截图后点击“退出截图模式”。")

    def _visibility_paused(self):
        return (not self.bot.window_ready()[0] or
                (self.visibility_button.isChecked() and
                 not win32_screen.is_foreground(self.bot.win_hwnd)))

    def _retain_obscured_hint(self):
        # Keep the last render, but never feed an occluder's pixels to CNN.
        if self.bot is None or self.visibility_button.isChecked() or not self._auto:
            return False
        hwnd = self.bot.win_hwnd
        return (win32_screen.is_window_visible(hwnd) and
                not win32_screen.is_iconic(hwnd) and
                self._calibrated_geometry is not None and
                win32_screen.get_client_rect(hwnd) == self._calibrated_geometry and
                not self.bot.window_ready()[0])

    def _tick(self):
        if self._closing:
            return
        if self._screenshot_requested:
            self._prepare_screenshot()
            return
        if self._thread is None and self._pending_command:
            command, self._pending_command = self._pending_command, None
            if command == 'calibrate':
                self.start_calibration()
            else:
                self.start_recognition()
            return
        # Overlay belongs to the GUI thread and lives as long as its Qt loop.
        if self.bot is not None:
            if self.bot.overlay is not None:
                self.bot.overlay.pump()
            geometry = win32_screen.get_client_rect(self.bot.win_hwnd)
            if (self._calibrated_geometry is not None and
                    geometry != self._calibrated_geometry):
                hide_overlay(self.bot)
                self._auto = False
                self._next_action = None
                self._accepted = self._candidate = None
                self.status_value.setText("窗口已移动或缩放，请点击一次“校准棋盘”。")
                return
            if self._visibility_paused():
                if not self._retain_obscured_hint():
                    hide_overlay(self.bot)
                self.status_value.setText(
                    "识别暂停，保留已有提示（遮挡期间不会更新）" if self._retain_obscured_hint()
                    else "识别暂停，等待窗口可见 / 前台")
                return
        if not self._auto or self._thread is not None:
            return
        action, self._next_action = self._next_action, None
        if action == 'hint':
            self.start_hint()
        else:
            self._recognize_once()


def main():
    win32_screen.set_dpi_aware()
    app = QApplication(sys.argv)
    app.setApplicationName("Xiangqi Hint Diagnostics")
    panel = DebugPanel()
    panel.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
