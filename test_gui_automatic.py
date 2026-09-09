"""Read-only regressions: no live captures, mouse input, or game changes."""
import os
import unittest
from unittest.mock import patch, Mock
from types import SimpleNamespace

import win_hint_gui as gui
import win32_screen
from win_hint import WinHintBot


class AutomaticGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = gui.QApplication.instance() or gui.QApplication([])

    def setUp(self):
        self.panel = gui.DebugPanel()
        self.panel._timer.stop()

    def tearDown(self):
        self.panel._thread = None
        self.panel.bot = None
        self.panel.close()

    def test_window_selection_rejects_self_and_other_panel(self):
        windows = [(1, '天天象棋', (0, 0, 1100, 900)),
                   (2, '天天象棋提示调试面板', (0, 0, 1100, 900)),
                   (3, '天天象棋', (-2300, 200, 1692, 1020))]
        with patch.object(win32_screen, 'list_windows', return_value=windows), \
             patch.object(win32_screen, '_process_id', side_effect=lambda h: os.getpid() if h == 1 else -1):
            self.assertEqual(win32_screen.find_xiangqi_window()[0], 3)

    def test_drag_discards_old_result(self):
        p = self.panel
        p._auto = True
        p.bot = SimpleNamespace(win_hwnd=1, overlay=None)
        with patch.object(win32_screen, 'get_client_rect', return_value=(-1000, 0, 800, 900)):
            self.assertFalse(p._result_current({'geometry': (0, 0, 800, 900)}))
        self.assertIsNone(p._accepted)

    def test_button_queues_while_busy(self):
        p = self.panel
        p._thread = object()
        p.start_calibration()
        self.assertEqual(p._pending_command, 'calibrate')
        p._thread = None
        with patch.object(p, 'start_calibration') as execute:
            p._tick()
            execute.assert_called_once()

    def test_unchanged_board_keeps_overlay_and_skips_search(self):
        p = self.panel
        p._auto = True
        p._accepted = ('same', 'b', False)
        p.bot = SimpleNamespace(overlay=Mock())
        with patch.object(p, '_result_current', return_value=True):
            p._recognition_done({'fen': 'same', 'turn': 'b', 'playing_red': False})
        self.assertIsNone(p._next_action)
        p.bot.overlay.set_visible.assert_called_once_with(True)

    def test_arrowhead_is_filled_polygon(self):
        bot = WinHintBot(0, 'offline', (0, 0, 900, 1000), use_overlay=False)
        bot.cols_logical = [100+i*80 for i in range(9)]
        bot.rows_logical = [100+i*80 for i in range(10)]
        bot.cell_w = bot.cell_h = 80
        canvas = bot.render_arrows('a0b0', None, 0, 0, 900, 1000)
        # Between shaft endpoint and tip: absent when vertices are passed as
        # three separate one-point contours to fillPoly.
        self.assertGreater(int(canvas[100, 170, 3]), 0)


if __name__ == '__main__':
    unittest.main()
