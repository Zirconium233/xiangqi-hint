"""Run the Qt panel's capture/recognition/hint stages independently.

This probe never moves the mouse, sends clicks, or activates the target.  It
is intentionally small so a failure can be assigned to one stage instead of
being hidden inside the live loop.

Examples:
    python -X utf8 debug_gui_pipeline.py --step capture
    python -X utf8 debug_gui_pipeline.py --step recognize
    python -X utf8 debug_gui_pipeline.py --step hint --show-overlay
    python -X utf8 debug_gui_pipeline.py --step all --show-overlay
"""

from __future__ import annotations

import argparse
import time

import win32_screen
from gui_pipeline import annotate_board, destroy_overlay, save_step, show_overlay
from win_hint import WinHintBot


def make_bot():
    win32_screen.set_dpi_aware()
    found = win32_screen.find_xiangqi_window()
    if not found:
        raise RuntimeError("找不到“天天象棋”窗口")
    bot = WinHintBot(*found, start_turn="b", use_overlay=False)
    bot._set_client_geometry(
        win32_screen.get_client_rect(found[0]), apply_calibration=False)
    if not bot.load_calibration():
        raise RuntimeError("没有有效校准，请先在 GUI 点击“校准棋盘”")
    bot._grid_present = True
    bot.load_cnn()
    return bot


def capture(bot):
    started = time.perf_counter()
    image = bot._capture_settled_frame()
    path = save_step(annotate_board(bot, image), "01_capture_grid.png")
    print(f"capture: {image.shape[1]}x{image.shape[0]}  "
          f"{(time.perf_counter() - started) * 1000:.0f}ms  {path}")
    return image


def recognize(bot, image):
    started = time.perf_counter()
    board = bot.parse_board_cnn(image)
    quality = getattr(bot, "_last_parse_quality", {}) or {}
    if board is None:
        raise RuntimeError("CNN 没有返回棋盘")
    if not bot.detect_orientation_from_board(board):
        print("orientation: unknown (kings are incomplete or invalid)")
    fen = bot.board_to_fen(board)
    path = save_step(annotate_board(bot, image, board, quality),
                     "02_recognition.png")
    pieces = [(r, c, p) for r, row in enumerate(board)
              for c, p in enumerate(row) if p is not None]
    bot._gui_last_image = image.copy()
    bot._gui_last_board = board
    bot._gui_last_fen = fen
    bot._gui_last_quality = quality
    print(f"recognize: {len(pieces)}/90  fen={fen}  "
          f"repairs={len(quality.get('repairs', ())) }  {path}")
    print("pieces:", " ".join(f"{p}@{c}{r}" for r, c, p in pieces))
    return board, fen


def hint(bot, image, board, fen, show=False, seconds=2.0):
    from engine_session import EngineSession

    bot.engine = EngineSession(verbose=False)
    started = time.perf_counter()
    gold, red, info = bot._hint_for_position(board, fen, "b")
    annotated = annotate_board(
        bot, image, board, getattr(bot, "_last_parse_quality", {}), gold, red)
    path = save_step(annotated, "03_hint_lines.png")
    print(f"hint: gold={gold or '-'} red={red or '-'} "
          f"{bot.engine.score_str(info)}  "
          f"{(time.perf_counter() - started) * 1000:.0f}ms  {path}")
    if show and (gold or red):
        show_overlay(bot, gold, red)
        print(f"overlay: shown for {seconds:.1f}s on the target window")
        time.sleep(max(0.0, seconds))
    return gold, red


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=("capture", "recognize", "hint", "all"),
                        default="all")
    parser.add_argument("--show-overlay", action="store_true")
    parser.add_argument("--overlay-seconds", type=float, default=2.0)
    args = parser.parse_args()

    bot = make_bot()
    try:
        image = capture(bot)
        if args.step == "capture":
            return
        board, fen = recognize(bot, image)
        if args.step == "recognize":
            return
        hint(bot, image, board, fen, args.show_overlay, args.overlay_seconds)
    finally:
        destroy_overlay(bot)
        if bot.engine is not None:
            bot.engine.close()


if __name__ == "__main__":
    main()
