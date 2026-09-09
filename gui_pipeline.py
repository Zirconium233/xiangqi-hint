"""Small, testable rendering helpers used by the manual Qt diagnostics panel.

The helpers in this module deliberately have no Qt or input side effects.  A
capture, a CNN board annotation, an engine/PV annotation, and the real
click-through overlay can therefore be exercised independently before they
are wired together in :mod:`win_hint_gui`.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

try:
    from xiangqi_cnn import CLASS_TO_IDX
except Exception:  # pragma: no cover - keeps pure capture usable without CNN
    CLASS_TO_IDX = {}


DEFAULT_DIR = Path(__file__).resolve().parent / "debug" / "gui_pipeline"


def save_step(image, name, directory=DEFAULT_DIR):
    """Save one pipeline artifact and return its absolute path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    if not str(path).lower().endswith((".png", ".jpg", ".jpeg")):
        path = path.with_suffix(".png")
    if image is None or not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to save diagnostic image: {path}")
    return str(path)


def _cell_xy(bot, row, col):
    return (
        int(round(bot.cols_logical[col] - bot.win_x)),
        int(round(bot.rows_logical[row] - bot.win_y)),
    )


def _piece_confidence(bot, row, col, piece):
    probs = getattr(getattr(bot, "cnn", None), "_cell_probs", None)
    if probs is None or piece not in CLASS_TO_IDX:
        return None
    try:
        cell_probs = probs[row][col]
        if cell_probs is None:
            return None
        return float(cell_probs[CLASS_TO_IDX[piece]])
    except (IndexError, TypeError, ValueError):
        return None


def annotate_board(bot, image, board=None, quality=None,
                   gold_hint=None, red_hint=None):
    """Return a BGR diagnostic image with lattice, pieces, and optional PV.

    Coordinates are client-relative in the image and screen-relative in the
    bot, exactly like the production parser and overlay.  This makes the
    artifact useful for diagnosing either a bad CNN crop or a bad overlay
    transform without moving the mouse or activating a window.
    """
    out = image.copy()
    height, width = out.shape[:2]

    # A subtle lattice makes an off-by-one calibration obvious while keeping
    # the original screenshot visible underneath.
    for col in range(9):
        x, _ = _cell_xy(bot, 0, col)
        if 0 <= x < width:
            cv2.line(out, (x, 0), (x, height - 1), (95, 150, 210), 1,
                     cv2.LINE_AA)
    for row in range(10):
        _, y = _cell_xy(bot, row, 0)
        if 0 <= y < height:
            cv2.line(out, (0, y), (width - 1, y), (95, 150, 210), 1,
                     cv2.LINE_AA)

    if board is not None:
        for row in range(10):
            for col in range(9):
                piece = board[row][col]
                x, y = _cell_xy(bot, row, col)
                if not (0 <= x < width and 0 <= y < height):
                    continue
                cv2.circle(out, (x, y), 5, (255, 255, 255), -1,
                           cv2.LINE_AA)
                if piece is None:
                    continue
                # Uppercase labels are the bottom/player side in the raw
                # screen board; lowercase labels are the top/opponent side.
                color = (30, 80, 230) if piece.isupper() else (230, 100, 30)
                cv2.circle(out, (x, y), 9, color, 2, cv2.LINE_AA)
                conf = _piece_confidence(bot, row, col, piece)
                label = piece if conf is None else f"{piece}:{conf:.2f}"
                tx = max(2, min(width - 75, x - 23))
                ty = max(16, min(height - 3, y - 13))
                cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, color, 1, cv2.LINE_AA)

    if quality:
        text = (f"pieces={sum(p is not None for row in (board or []) for p in row)}/90  "
                f"min={quality.get('min_any', 0):.3f}  "
                f"repairs={len(quality.get('repairs', ())) }")
        cv2.putText(out, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (0, 255, 255), 2, cv2.LINE_AA)

    if gold_hint or red_hint:
        ox, oy, ow, oh = bot._overlay_rect()
        canvas = bot.render_arrows(gold_hint, red_hint, ox, oy, ow, oh)
        _alpha_composite(out, canvas, int(ox - bot.win_x),
                         int(oy - bot.win_y))

    return out


def _alpha_composite(base, overlay_bgra, left, top):
    """Composite a BGRA overlay onto a BGR client image, clipped safely."""
    if overlay_bgra is None or overlay_bgra.ndim != 3:
        return base
    oh, ow = overlay_bgra.shape[:2]
    bh, bw = base.shape[:2]
    x1, y1 = max(0, left), max(0, top)
    x2, y2 = min(bw, left + ow), min(bh, top + oh)
    if x1 >= x2 or y1 >= y2:
        return base
    sx1, sy1 = x1 - left, y1 - top
    patch = overlay_bgra[sy1:sy1 + y2 - y1, sx1:sx1 + x2 - x1]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    src = patch[:, :, :3].astype(np.float32)
    dst = base[y1:y2, x1:x2].astype(np.float32)
    base[y1:y2, x1:x2] = np.clip(src * alpha + dst * (1.0 - alpha),
                                 0, 255).astype(np.uint8)


def show_overlay(bot, gold_hint, red_hint):
    """Create/update the real click-through overlay and return its canvas data."""
    from win32_overlay import ClickThroughOverlay

    if bot.overlay is None:
        bot.overlay = ClickThroughOverlay()
    ox, oy, ow, oh = bot._overlay_rect()
    canvas = bot.render_arrows(gold_hint, red_hint, ox, oy, ow, oh)
    bot.overlay.update(ox, oy, ow, oh, canvas)
    bot.overlay.pump()
    return ox, oy, ow, oh, canvas


def hide_overlay(bot):
    overlay = getattr(bot, "overlay", None)
    if overlay is not None:
        overlay.set_visible(False)
        overlay.pump()


def destroy_overlay(bot):
    overlay = getattr(bot, "overlay", None)
    if overlay is not None:
        overlay.destroy()
        bot.overlay = None
