"""Click-through, topmost, per-pixel-transparent Win32 overlay.

The overlay is a disabled WS_EX_LAYERED popup whose pixels are supplied as a
BGRA buffer. It deliberately has no activation/focus behavior and returns
HTTRANSPARENT as an extra hit-test fallback. Disabling the popup is important:
HTTRANSPARENT alone only forwards hit testing within the same GUI thread, so
it is not sufficient when the game is another process.
The owner should call :meth:`pump` from its normal loop; this keeps the
window procedure responsive without creating a second GUI thread.
"""

import ctypes
from ctypes import wintypes as wt


# use_last_error is important here: ctypes.get_last_error() otherwise often
# reports the error from a previous unrelated Win32 call.
user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


WS_EX_TOPMOST = 0x00000008
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000
WS_POPUP = 0x80000000
WS_DISABLED = 0x08000000

ULW_ALPHA = 2
AC_SRC_ALPHA = 1
DIB_RGB_COLORS = 0
BI_RGB = 0
SW_HIDE = 0
SW_SHOWNOACTIVATE = 4
PM_REMOVE = 0x0001

WM_DESTROY = 0x0002
WM_MOUSEACTIVATE = 0x0021
WM_NCHITTEST = 0x0084
HTTRANSPARENT = -1
MA_NOACTIVATE = 3
ERROR_CLASS_ALREADY_EXISTS = 1410


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wt.UINT),
        ("style", wt.UINT),
        ("lpfnWndProc", ctypes.c_void_p),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wt.HINSTANCE),
        ("hIcon", wt.HICON),
        ("hCursor", wt.HANDLE),
        ("hbrBackground", wt.HBRUSH),
        ("lpszMenuName", wt.LPCWSTR),
        ("lpszClassName", wt.LPCWSTR),
        ("hIconSm", wt.HICON),
    ]


class POINT(ctypes.Structure):
    _fields_ = [("x", wt.LONG), ("y", wt.LONG)]


class SIZE(ctypes.Structure):
    _fields_ = [("cx", wt.LONG), ("cy", wt.LONG)]


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [
        ("BlendOp", wt.BYTE),
        ("BlendFlags", wt.BYTE),
        ("SourceConstantAlpha", wt.BYTE),
        ("AlphaFormat", wt.BYTE),
    ]


class RGBQUAD(ctypes.Structure):
    _fields_ = [
        ("rgbBlue", wt.BYTE),
        ("rgbGreen", wt.BYTE),
        ("rgbRed", wt.BYTE),
        ("rgbReserved", wt.BYTE),
    ]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wt.DWORD),
        ("biWidth", wt.LONG),
        ("biHeight", wt.LONG),
        ("biPlanes", wt.WORD),
        ("biBitCount", wt.WORD),
        ("biCompression", wt.DWORD),
        ("biSizeImage", wt.DWORD),
        ("biXPelsPerMeter", wt.LONG),
        ("biYPelsPerMeter", wt.LONG),
        ("biClrUsed", wt.DWORD),
        ("biClrImportant", wt.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", RGBQUAD * 1)]


class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", wt.HWND),
        ("message", wt.UINT),
        ("wParam", wt.WPARAM),
        ("lParam", wt.LPARAM),
        ("time", wt.DWORD),
        ("pt", POINT),
    ]


# A window callback is stdcall on Windows. Keep this object alive globally;
# passing the plain Python function to ctypes.cast causes the original
# "wrong type" failure.
_WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wt.HWND, wt.UINT,
                              wt.WPARAM, wt.LPARAM)


def _wndproc(_hwnd, msg, _wparam, _lparam):
    if msg == WM_NCHITTEST:
        return HTTRANSPARENT
    if msg == WM_MOUSEACTIVATE:
        return MA_NOACTIVATE
    if msg == WM_DESTROY:
        return 0
    # In particular, WM_NCCREATE must be passed to DefWindowProcW (or return
    # non-zero) or CreateWindowExW will create and immediately destroy the
    # popup.  The original stub returned 0 for every message.
    return user32.DefWindowProcW(_hwnd, msg, _wparam, _lparam)


_wndproc_ref = _WNDPROC(_wndproc)


def _bind_win32_api():
    """Declare the ABI for every Win32 function used by this module."""
    kernel32.GetModuleHandleW.argtypes = [wt.LPCWSTR]
    kernel32.GetModuleHandleW.restype = wt.HMODULE

    user32.RegisterClassExW.argtypes = [ctypes.POINTER(WNDCLASSEXW)]
    user32.RegisterClassExW.restype = wt.ATOM
    user32.CreateWindowExW.argtypes = [
        wt.DWORD, wt.LPCWSTR, wt.LPCWSTR, wt.DWORD,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wt.HWND, wt.HMENU, wt.HINSTANCE, ctypes.c_void_p,
    ]
    user32.CreateWindowExW.restype = wt.HWND
    user32.DestroyWindow.argtypes = [wt.HWND]
    user32.DestroyWindow.restype = wt.BOOL
    user32.EnableWindow.argtypes = [wt.HWND, wt.BOOL]
    user32.EnableWindow.restype = wt.BOOL
    user32.ShowWindow.argtypes = [wt.HWND, ctypes.c_int]
    user32.ShowWindow.restype = wt.BOOL
    user32.DefWindowProcW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
    user32.DefWindowProcW.restype = ctypes.c_ssize_t
    user32.GetDC.argtypes = [wt.HWND]
    user32.GetDC.restype = wt.HDC
    user32.ReleaseDC.argtypes = [wt.HWND, wt.HDC]
    user32.ReleaseDC.restype = ctypes.c_int
    user32.UpdateLayeredWindow.argtypes = [
        wt.HWND, wt.HDC, ctypes.POINTER(POINT), ctypes.POINTER(SIZE),
        wt.HDC, ctypes.POINTER(POINT), wt.COLORREF,
        ctypes.POINTER(BLENDFUNCTION), wt.DWORD,
    ]
    user32.UpdateLayeredWindow.restype = wt.BOOL
    user32.PeekMessageW.argtypes = [
        ctypes.POINTER(MSG), wt.HWND, wt.UINT, wt.UINT, wt.UINT,
    ]
    user32.PeekMessageW.restype = wt.BOOL
    user32.TranslateMessage.argtypes = [ctypes.POINTER(MSG)]
    user32.TranslateMessage.restype = wt.BOOL
    user32.DispatchMessageW.argtypes = [ctypes.POINTER(MSG)]
    user32.DispatchMessageW.restype = wt.LPARAM

    gdi32.CreateCompatibleDC.argtypes = [wt.HDC]
    gdi32.CreateCompatibleDC.restype = wt.HDC
    gdi32.DeleteDC.argtypes = [wt.HDC]
    gdi32.DeleteDC.restype = wt.BOOL
    gdi32.CreateDIBSection.argtypes = [
        wt.HDC, ctypes.POINTER(BITMAPINFO), wt.UINT,
        ctypes.POINTER(ctypes.c_void_p), wt.HANDLE, wt.DWORD,
    ]
    gdi32.CreateDIBSection.restype = wt.HBITMAP
    gdi32.SelectObject.argtypes = [wt.HDC, wt.HGDIOBJ]
    gdi32.SelectObject.restype = wt.HGDIOBJ
    gdi32.DeleteObject.argtypes = [wt.HGDIOBJ]
    gdi32.DeleteObject.restype = wt.BOOL


_bind_win32_api()


def _last_error(prefix):
    err = ctypes.get_last_error()
    return RuntimeError(f"{prefix} failed (Win32 error {err})")


class ClickThroughOverlay:
    def __init__(self, class_name="DshXiangqiHintOverlay"):
        self.class_name = class_name
        self.hwnd = None
        self._hdc_src = None
        self._hbitmap = None
        self._old_bitmap = None
        self._bits_ptr = None
        self._cur_size = None
        self._visible = False
        self.capture_excluded = False

        hinst = kernel32.GetModuleHandleW(None)
        if not hinst:
            raise _last_error("GetModuleHandleW")

        wc = WNDCLASSEXW()
        wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
        wc.lpfnWndProc = ctypes.cast(_wndproc_ref, ctypes.c_void_p)
        wc.hInstance = hinst
        wc.lpszClassName = class_name
        atom = user32.RegisterClassExW(ctypes.byref(wc))
        if not atom:
            err = ctypes.get_last_error()
            if err != ERROR_CLASS_ALREADY_EXISTS:
                raise RuntimeError(f"RegisterClassExW failed (Win32 error {err})")

        ex_style = (WS_EX_TOPMOST | WS_EX_TRANSPARENT | WS_EX_LAYERED |
                    WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE)
        self.hwnd = user32.CreateWindowExW(
            ex_style, class_name, "xiangqi-hint", WS_POPUP | WS_DISABLED,
            0, 0, 0, 0, None, None, hinst, None,
        )
        if not self.hwnd:
            raise _last_error("CreateWindowExW")
        # Keep the overlay on the monitor but out of recognition captures.
        # Older Windows treats 0x11 as a black rectangle: do not enable there.
        import sys
        if sys.getwindowsversion().build >= 19041:
            user32.SetWindowDisplayAffinity.argtypes = [wt.HWND, wt.DWORD]
            user32.SetWindowDisplayAffinity.restype = wt.BOOL
            self.capture_excluded = bool(user32.SetWindowDisplayAffinity(self.hwnd, 0x11))
        print('  overlay capture exclusion: ' + ('enabled' if self.capture_excluded
                                                 else 'unavailable; hide-on-capture fallback'))
        # Keep the popup non-interactive even if a later ShowWindow/update
        # path changes its visibility. This is the cross-process part of
        # click-through; HTTRANSPARENT by itself is only same-thread.
        user32.EnableWindow(self.hwnd, False)

        # Create the memory DC from the desktop DC.  A DC created with a NULL
        # template starts with a 1x1 monochrome surface on some Windows
        # builds; layered composition can then succeed while displaying no
        # pixels.  Using the desktop format avoids that silent failure.
        hdc_screen = user32.GetDC(None)
        if not hdc_screen:
            raise _last_error("GetDC")
        try:
            self._hdc_src = gdi32.CreateCompatibleDC(hdc_screen)
        finally:
            user32.ReleaseDC(None, hdc_screen)
        if not self._hdc_src:
            self.destroy()
            raise _last_error("CreateCompatibleDC")

    def set_capture_excluded(self, excluded):
        """Change OS screenshot exclusion without hiding the window."""
        user32.SetWindowDisplayAffinity.argtypes = [wt.HWND, wt.DWORD]
        user32.SetWindowDisplayAffinity.restype = wt.BOOL
        if not user32.SetWindowDisplayAffinity(self.hwnd, 0x11 if excluded else 0):
            raise _last_error("SetWindowDisplayAffinity")
        self.capture_excluded = bool(excluded)

    def update(self, x, y, w, h, bgra):
        """Position and display a BGRA uint8 canvas with per-pixel alpha."""
        import ctypes as _ct

        w, h = int(w), int(h)
        if w <= 0 or h <= 0:
            self.set_visible(False)
            return
        nbytes = w * h * 4
        if getattr(bgra, "shape", None) != (h, w, 4):
            raise ValueError(f"expected BGRA canvas {(h, w, 4)}, got "
                             f"{getattr(bgra, 'shape', None)}")

        if (w, h) != self._cur_size:
            self._release_bitmap()
            bi = BITMAPINFO()
            bi.bmiHeader.biSize = _ct.sizeof(BITMAPINFOHEADER)
            bi.bmiHeader.biWidth = w
            bi.bmiHeader.biHeight = -h  # top-down DIB
            bi.bmiHeader.biPlanes = 1
            bi.bmiHeader.biBitCount = 32
            bi.bmiHeader.biCompression = BI_RGB
            bits_ptr = _ct.c_void_p()
            self._hbitmap = gdi32.CreateDIBSection(
                self._hdc_src, _ct.byref(bi), DIB_RGB_COLORS,
                _ct.byref(bits_ptr), None, 0,
            )
            if not self._hbitmap or not bits_ptr.value:
                self._release_bitmap()
                raise _last_error("CreateDIBSection")
            self._old_bitmap = gdi32.SelectObject(
                self._hdc_src, self._hbitmap)
            if not self._old_bitmap:
                self._release_bitmap()
                raise _last_error("SelectObject")
            self._bits_ptr = bits_ptr
            self._cur_size = (w, h)

        # NumPy arrays expose their backing address through .ctypes.data.
        # Passing that address avoids ctypes trying to interpret the ndarray
        # object itself as a pointer.
        src = int(bgra.ctypes.data) if hasattr(bgra, "ctypes") else bgra
        _ct.memmove(self._bits_ptr, src, nbytes)

        pos = POINT(int(x), int(y))
        size = SIZE(w, h)
        src_pos = POINT(0, 0)
        blend = BLENDFUNCTION(0, 0, 255, AC_SRC_ALPHA)
        # Passing explicit DCs/points is more reliable than NULL here.  On
        # this Windows/DWM configuration the NULL form returns TRUE but the
        # layered surface remains visually empty.
        hdc_dst = user32.GetDC(None)
        if not hdc_dst:
            raise _last_error("GetDC")
        try:
            ok = user32.UpdateLayeredWindow(
                self.hwnd, hdc_dst, _ct.byref(pos), _ct.byref(size),
                self._hdc_src, _ct.byref(src_pos), 0,
                _ct.byref(blend), ULW_ALPHA,
            )
        finally:
            user32.ReleaseDC(None, hdc_dst)
        if not ok:
            raise _last_error(f"UpdateLayeredWindow at ({x},{y},{w},{h})")
        if not self._visible:
            user32.EnableWindow(self.hwnd, False)
            user32.ShowWindow(self.hwnd, SW_SHOWNOACTIVATE)
            self._visible = True

    def pump(self, max_messages=32):
        """Dispatch pending messages so hit-testing stays click-through."""
        msg = MSG()
        count = 0
        while count < max_messages and user32.PeekMessageW(
                ctypes.byref(msg), None, 0, 0, PM_REMOVE):
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
            count += 1
        return count

    def clear(self):
        self.set_visible(False)

    def set_visible(self, visible):
        if not self.hwnd:
            return
        if self._visible == bool(visible):
            return
        # Never enable the overlay: a disabled top-level window is ignored by
        # normal mouse hit testing, including when the target is another
        # process. The explicit HTTRANSPARENT response remains as a fallback.
        user32.EnableWindow(self.hwnd, False)
        user32.ShowWindow(self.hwnd, SW_SHOWNOACTIVATE if visible else SW_HIDE)
        self._visible = bool(visible)

    def _release_bitmap(self):
        if self._hbitmap:
            if self._old_bitmap and self._hdc_src:
                gdi32.SelectObject(self._hdc_src, self._old_bitmap)
            gdi32.DeleteObject(self._hbitmap)
        self._hbitmap = None
        self._old_bitmap = None
        self._bits_ptr = None
        self._cur_size = None

    def destroy(self):
        self.set_visible(False)
        self._release_bitmap()
        if self.hwnd:
            user32.DestroyWindow(self.hwnd)
            self.hwnd = None
        if self._hdc_src:
            gdi32.DeleteDC(self._hdc_src)
            self._hdc_src = None
