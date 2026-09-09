"""Win32 helpers for the Windows 天天象棋 hint bot.

- DPI awareness (all coordinates = physical pixels)
- window enumeration / 天天象棋 window finder
- window capture: PrintWindow (PW_RENDERFULLCONTENT) first — works even when
  the window is OCCLUDED and never steals focus; falls back to
  non-activating mss region grab. Coordinates are the window's CLIENT
  rect in screen space, so both paths are pixel-aligned.
- SendInput-based click / key press (pydirectinput, pyautogui fallback)
"""
import ctypes
import ctypes.wintypes as wt
import time
import os

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32

_ENUM_CB = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)

PW_RENDERFULLCONTENT = 2
GA_ROOT = 2

# The input/capture path is intentionally read-only. These bindings are used
# only to inspect foreground/occlusion state; they do not move the cursor or
# activate any window.
user32.GetForegroundWindow.restype = wt.HWND
user32.WindowFromPoint.argtypes = [wt.POINT]
user32.WindowFromPoint.restype = wt.HWND
user32.GetAncestor.argtypes = [wt.HWND, wt.UINT]
user32.GetAncestor.restype = wt.HWND
user32.GetWindowThreadProcessId.argtypes = [
    wt.HWND, ctypes.POINTER(wt.DWORD),
]
user32.GetWindowThreadProcessId.restype = wt.DWORD


# ----------------------------------------------------------------------
# DPI
# ----------------------------------------------------------------------

def set_dpi_aware():
    """Make the process DPI aware so all pixel math is physical, not logical."""
    try:
        if user32.SetProcessDpiAwarenessContext(-4):  # PER_MONITOR_AWARE_V2
            return True
    except Exception:
        pass
    try:
        if ctypes.windll.shcore.SetProcessDpiAwareness(2):
            return True
    except Exception:
        pass
    try:
        user32.SetProcessDPIAware()
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------
# Windows
# ----------------------------------------------------------------------

def list_windows():
    """All visible top-level windows: [(hwnd, title, (x, y, w, h)), ...]."""
    out = []
    def cb(hwnd, _lparam):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            r = wt.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(r))
            out.append((hwnd, buf.value,
                        (r.left, r.top, r.right - r.left, r.bottom - r.top)))
        return True
    user32.EnumWindows(_ENUM_CB(cb), 0)
    return out


def find_xiangqi_window():
    """Find the 天天象棋 window. Returns (hwnd, title, window_rect) or None.

    Priority: title contains 天天象棋 > 象棋 > WeChat/微信.
    """
    best = None
    for hwnd, title, rect in list_windows():
        # Never select this process or another copy of the diagnostics UI.
        if (_process_id(hwnd) == os.getpid() or
                any(word in title.lower() for word in
                    ('调试', '提示面板', 'diagnostic', 'codex')) or
                rect[2] < 300 or rect[3] < 300):
            continue
        if title.strip() == '天天象棋':
            s = 200
        elif '天天象棋' in title:
            s = 100
        else:
            continue
        if best is None or s > best[0]:
            best = (s, hwnd, title, rect)
    if best:
        return best[1], best[2], best[3]
    return None


def window_rect(hwnd):
    r = wt.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    return (r.left, r.top, r.right - r.left, r.bottom - r.top)


def get_client_rect(hwnd):
    """Client area in screen coordinates: (x, y, w, h)."""
    rc = wt.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rc))
    w, h = rc.right - rc.left, rc.bottom - rc.top
    pt = wt.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return (pt.x, pt.y, w, h)


def is_foreground(hwnd):
    return user32.GetForegroundWindow() == hwnd


def is_window_visible(hwnd):
    """Whether the target window is visible, without changing focus."""
    return bool(user32.IsWindowVisible(hwnd))


def _root_window(hwnd):
    if not hwnd:
        return None
    return user32.GetAncestor(hwnd, GA_ROOT) or hwnd


def _handle_value(hwnd):
    value = getattr(hwnd, 'value', hwnd)
    return int(value or 0)


def _process_id(hwnd):
    pid = wt.DWORD()
    if not hwnd or not user32.GetWindowThreadProcessId(
            hwnd, ctypes.byref(pid)):
        return None
    return int(pid.value)


def is_window_unobscured(hwnd, rect=None, ignore_hwnds=()):
    """Return whether representative client points belong to hwnd.

    This is a read-only occlusion check. WindowFromPoint may return a child
    window, so roots are compared; windows from the target process are also
    accepted. A caller can list known non-interactive overlays in
    ignore_hwnds. No focus, cursor, or z-order changes are performed.
    """
    if not hwnd or not is_window_visible(hwnd) or is_iconic(hwnd):
        return False
    if rect is None:
        rect = get_client_rect(hwnd)
    x, y, w, h = (int(v) for v in rect)
    if w <= 0 or h <= 0:
        return False

    target_root = _root_window(hwnd)
    target_root_value = _handle_value(target_root)
    target_pid = _process_id(hwnd)
    ignored = {_handle_value(hh) for hh in ignore_hwnds if hh}
    # Avoid borders where hit testing can legitimately resolve to a frame or
    # an adjacent window. Nine points are enough to catch normal occlusion
    # while keeping the check cheap at the hint-loop cadence.
    fractions = (0.15, 0.50, 0.85)
    for fy in fractions:
        for fx in fractions:
            pt = wt.POINT(
                int(round(x + (w - 1) * fx)),
                int(round(y + (h - 1) * fy)),
            )
            hit = user32.WindowFromPoint(pt)
            if not hit:
                return False
            if _handle_value(hit) in ignored:
                continue
            hit_root = _root_window(hit)
            if _handle_value(hit_root) == target_root_value:
                continue
            if target_pid is not None and _process_id(hit_root or hit) == target_pid:
                continue
            return False
    return True


def is_iconic(hwnd):
    """True if the window is minimized."""
    return bool(user32.IsIconic(hwnd))


def restore(hwnd):
    """Un-minimize a window (ShowWindow SW_RESTORE). Returns True if it was
    minimized. A programmatic restore does not steal foreground focus."""
    if not user32.IsIconic(hwnd):
        return False
    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    time.sleep(0.4)  # let it come back and repaint
    return True


def activate(hwnd):
    """Bring a window to the foreground (AttachThreadInput trick)."""
    fore = user32.GetForegroundWindow()
    if fore == hwnd:
        return True
    my_thread = kernel32.GetCurrentThreadId()
    fore_thread = user32.GetWindowThreadProcessId(fore, None)
    if fore_thread and fore_thread != my_thread:
        user32.AttachThreadInput(my_thread, fore_thread, True)
        user32.SetForegroundWindow(hwnd)
        user32.AttachThreadInput(my_thread, fore_thread, False)
    else:
        user32.SetForegroundWindow(hwnd)
    return True


# ----------------------------------------------------------------------
# Capture
# ----------------------------------------------------------------------

def _is_blank(bgr):
    """Heuristic: all-black / all-white / featureless => failed capture."""
    import numpy as np
    m = bgr.mean()
    if m < 12 and (bgr.max() - bgr.min()) < 24:
        return True   # black void
    if m > 245 and (bgr.max() - bgr.min()) < 24:
        return True   # blank white
    return False


def print_window(hwnd, w, h):
    """PrintWindow(hwnd, PW_RENDERFULLCONTENT) -> BGR numpy (client area),
    or None if the window refused / returned blank."""
    import numpy as np
    import cv2
    from win32_overlay import BITMAPINFO, BITMAPINFOHEADER

    hdc_wnd = user32.GetWindowDC(hwnd)
    if not hdc_wnd:
        return None
    hdc_mem = gdi32.CreateCompatibleDC(hdc_wnd)
    hbmp = gdi32.CreateCompatibleBitmap(hdc_wnd, w, h)
    try:
        if not hbmp:
            return None
        old = gdi32.SelectObject(hdc_mem, hbmp)
        ok = user32.PrintWindow(hwnd, hdc_mem, PW_RENDERFULLCONTENT)
        gdi32.SelectObject(hdc_mem, old)
        if not ok:
            return None
        bi = BITMAPINFO()
        bi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bi.bmiHeader.biWidth = w
        bi.bmiHeader.biHeight = -h  # top-down
        bi.bmiHeader.biPlanes = 1
        bi.bmiHeader.biBitCount = 32
        bi.bmiHeader.biCompression = 0
        bits = ctypes.create_string_buffer(w * h * 4)
        got = gdi32.GetDIBits(hdc_mem, hbmp, 1, h, bits, bi, 0)
        if got != h:
            return None
        arr = np.frombuffer(bits, np.uint8).reshape(h, w, 4)
        if float(arr[..., 3].mean()) < 8:
            return None  # fully transparent
        bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        if _is_blank(bgr):
            return None
        return bgr
    finally:
        if hbmp:
            gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(hwnd, hdc_wnd)


def _region_grab(x, y, w, h):
    """mss grab of an absolute screen rect -> BGR numpy (clamped)."""
    import mss
    import numpy as np
    import cv2
    vx = user32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
    vy = user32.GetSystemMetrics(77)
    vw = user32.GetSystemMetrics(78)
    vh = user32.GetSystemMetrics(79)
    x1, y1 = max(vx, int(x)), max(vy, int(y))
    x2, y2 = min(vx + vw, int(x) + int(w)), min(vy + vh, int(y) + int(h))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"window region off-screen: ({x},{y},{w},{h})")
    with mss.mss() as sct:
        mon = {'left': x1, 'top': y1, 'width': x2 - x1, 'height': y2 - y1}
        arr = np.asarray(sct.grab(mon), dtype=np.uint8)  # BGRA
    ox, oy = x1 - int(x), y1 - int(y)
    full = np.zeros((int(h), int(w), 4), dtype=np.uint8)
    full[oy:oy + (y2 - y1), ox:ox + (x2 - x1)] = arr
    return cv2.cvtColor(full, cv2.COLOR_BGRA2BGR)


def capture_window(hwnd, x, y, w, h, allow_fallback=True,
                   try_print=True, activate_fallback=False,
                   return_method=False):
    """Capture the window's client area (screen-rect x,y,w,h given).

    Tries PrintWindow first, then a non-activating region grab by default.
    Minimized windows are never restored by capture. Activation requires an
    explicit opt-in. `return_method=True` includes the capture method.
    """
    if is_iconic(hwnd):
        raise RuntimeError('window is minimized; waiting for user to restore it')
    if try_print:
        img = print_window(hwnd, w, h)
        if img is not None:
            return (img, "printwindow") if return_method else img
    if not allow_fallback:
        return (None, "printwindow_failed") if return_method else None
    if activate_fallback:
        activate(hwnd)
        time.sleep(0.35)  # let it come forward and repaint
    x2, y2, w2, h2 = get_client_rect(hwnd)  # may have moved
    img = _region_grab(x2, y2, w2, h2)
    if img is None or _is_blank(img):
        raise RuntimeError("capture failed (PrintWindow + region grab)")
    return (img, "screen") if return_method else img


# ----------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------

def click(x, y):
    """Real click at physical screen coords. Returns the backend used."""
    try:
        import pydirectinput
        pydirectinput.PAUSE = 0
        pydirectinput.moveTo(int(x), int(y))
        pydirectinput.click()
        return 'pydirectinput'
    except Exception:
        import pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.click(int(x), int(y))
        return 'pyautogui'


def press_esc():
    try:
        import pydirectinput
        pydirectinput.PAUSE = 0
        pydirectinput.press('esc')
        return True
    except Exception:
        import pyautogui
        pyautogui.press('esc')
        return True
