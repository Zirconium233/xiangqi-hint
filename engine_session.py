"""Persistent Pikafish UCI session for Windows.

The 2026-09-06 build is STRICT: an invalid FEN prints `info string CRITICAL
ERROR` and kills the process (exit 1). So every FEN is pre-validated here
BEFORE it ever reaches the engine, and the session respawns automatically if
the engine dies anyway.

Keeps ONE process alive across the game (no ~50MB net reload per query).
"""
import os
import queue
import re
import subprocess
import threading
import time

CREATE_NO_WINDOW = 0x08000000


class FenValidationError(ValueError):
    pass


def fen_prevalidate(fen):
    """Validate a FULL fen (piece rows + side token) before feeding the engine.

    Checks: 10 rows x 9 cells, legal piece chars, exactly one K and one k,
    kings not on the same file with an open line (face-off = illegal position
    the engine refuses to search).
    """
    parts = fen.split()
    if len(parts) < 2 or parts[1] not in ('w', 'b'):
        raise FenValidationError(f"bad fen structure: {fen!r}")
    legal = set('rnbakcpRNBACKP')
    rows = parts[0].split('/')
    if len(rows) != 10:
        raise FenValidationError(f"expected 10 rows, got {len(rows)}")
    grid = []
    for row in rows:
        cells = []
        n = 0
        for ch in row:
            if ch in '123456789':
                cells.extend([None] * int(ch))
                n += int(ch)
            elif ch in legal:
                cells.append(ch)
                n += 1
            else:
                raise FenValidationError(f"illegal char {ch!r} in {row!r}")
        if n != 9:
            raise FenValidationError(f"row {row!r} has {n} cells, want 9")
        grid.append(cells)
    def kings(ch):
        out = []
        for r, row in enumerate(grid):
            for c, p in enumerate(row):
                if p == ch:
                    out.append((r, c))
        return out
    kw, kb = kings('K'), kings('k')
    if len(kw) != 1 or len(kb) != 1:
        raise FenValidationError(f"king count K={len(kw)} k={len(kb)}")
    (wr, wc), (br, bc) = kw[0], kb[0]
    if wr == br and wc == bc:
        raise FenValidationError("kings on the same cell")
    if wc == bc:
        lo, hi = min(wr, br), max(wr, br)
        blocked = any(grid[r][wc] is not None for r in range(lo + 1, hi))
        if not blocked:
            raise FenValidationError("kings facing on open file")
    return True


class EngineSession:
    def __init__(self, exe=None, cwd=None, verbose=True):
        base = os.path.dirname(os.path.abspath(__file__))
        self.exe = exe or os.path.join(
            base, "pikafish" + (".exe" if os.name == "nt" else ""))
        self.cwd = cwd or base
        self.verbose = verbose
        self.lock = threading.Lock()
        self.proc = None
        self._stdout_queue = None
        self._reader_thread = None
        self._spawn()

    # ------------------------------------------------------------------
    def _spawn(self):
        if self.proc is not None and self.proc.poll() is None:
            return
        if self.proc is not None:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = subprocess.Popen(
            [self.exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, cwd=self.cwd,
            creationflags=CREATE_NO_WINDOW)
        self._stdout_queue = queue.Queue()

        def reader(proc, out_queue):
            try:
                for line in iter(proc.stdout.readline, ''):
                    out_queue.put(line.strip())
            except Exception:
                pass

        self._reader_thread = threading.Thread(
            target=reader, args=(self.proc, self._stdout_queue), daemon=True)
        self._reader_thread.start()
        self._write("uci")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            line = self._readline(timeout=max(0.01, deadline - time.monotonic()))
            if line and 'uciok' in line:
                break
        else:
            self._kill_process()
            raise RuntimeError("engine uci handshake timed out")
        self._write("isready")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            line = self._readline(timeout=max(0.01, deadline - time.monotonic()))
            if line and 'readyok' in line:
                break
        else:
            self._kill_process()
            raise RuntimeError("engine ready handshake timed out")
        if self.proc.poll() is not None:
            raise RuntimeError(f"engine died at startup rc={self.proc.returncode}")

    def _write(self, s):
        try:
            self.proc.stdin.write(s + "\n")
            self.proc.stdin.flush()
        except Exception as e:
            raise RuntimeError(f"engine write failed: {e}")

    def _readline(self, timeout=None):
        try:
            if self._stdout_queue is None:
                return ""
            return self._stdout_queue.get(timeout=timeout)
        except Exception:
            return ""

    def _drain(self):
        if self._stdout_queue is None:
            return
        while True:
            try:
                self._stdout_queue.get_nowait()
            except queue.Empty:
                return

    def _kill_process(self):
        proc = self.proc
        self.proc = None
        self._stdout_queue = None
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=1)
            except Exception:
                pass

    def _ensure(self):
        if self.proc is None or self.proc.poll() is not None:
            if self.verbose:
                print("  [engine] respawning...")
            self._spawn()

    # ------------------------------------------------------------------
    def _analyse_unlocked(self, fen, movetime_ms=1200, excluded=None,
                          pv_len=2):
        """Run one bounded search. Caller must hold self.lock."""
        fen_prevalidate(fen)
        self._ensure()
        self._drain()
        cmd = f"position fen {fen}"
        search_suffix = ""
        if excluded:
            allowed = [m for m in self._legal_unlocked(fen)
                       if m not in set(excluded)]
            if allowed:
                search_suffix = f" searchmoves {' '.join(allowed)}"
            else:
                return None, "", []
        self._write(cmd)
        self._write(f"go movetime {int(movetime_ms)}{search_suffix}")
        best, info, pv = None, "", []
        # The reader thread makes this a real timeout even when the engine
        # stops producing output. Keep the caller's five-second budget intact.
        deadline = time.monotonic() + max(0.8, min(4.0, movetime_ms / 1000 + 1.5))
        while time.monotonic() < deadline:
            line = self._readline(timeout=max(0.01, deadline - time.monotonic()))
            if not line:
                break
            if line.startswith('info'):
                if 'score' in line:
                    info = line
                fields = line.split()
                if 'pv' in fields:
                    i = fields.index('pv') + 1
                    pv = []
                    for m in fields[i:i + pv_len]:
                        if not re.fullmatch(r'[a-i][0-9][a-i][0-9]', m):
                            break
                        pv.append(m)
            elif line.startswith('bestmove'):
                fields = line.split()
                best = fields[1] if len(fields) > 1 else None
                break

        if best is None:
            # Stop and consume the terminal bestmove so a late result cannot
            # be mistaken for the next position's analysis.
            try:
                self._write("stop")
            except Exception:
                pass
            stop_deadline = time.monotonic() + 0.8
            while time.monotonic() < stop_deadline:
                line = self._readline(timeout=max(0.01, stop_deadline - time.monotonic()))
                if line.startswith('bestmove'):
                    fields = line.split()
                    best = fields[1] if len(fields) > 1 else None
                    break
            if best is None:
                self._kill_process()
        # A PV for a different root move is not a reply to bestmove.
        if not pv or pv[0] != best:
            pv = []
        return best, info, pv

    def analyse(self, fen, movetime_ms=1200, excluded=None, pv_len=2):
        """Return (bestmove, last_info_line, principal_variation)."""
        with self.lock:
            return self._analyse_unlocked(fen, movetime_ms, excluded, pv_len)

    def bestmove(self, fen, movetime_ms=1200, excluded=None):
        """Backward-compatible bestmove wrapper."""
        best, info, _pv = self.analyse(fen, movetime_ms, excluded, pv_len=1)
        return best, info

    def legal_moves(self, fen):
        """All legal moves via `go perft 1` (engine prints 'xxxx: n' per move)."""
        with self.lock:
            fen_prevalidate(fen)
            self._ensure()
            return self._legal_unlocked(fen)

    def _legal_unlocked(self, fen):
        self._drain()
        self._write(f"position fen {fen}")
        self._write("go perft 1")
        moves = []
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            line = self._readline(timeout=max(0.01, deadline - time.monotonic()))
            if not line:
                if self.proc.poll() is not None:
                    break
                continue
            match = re.fullmatch(r'([a-i][0-9][a-i][0-9]):\s*\d+', line)
            if match:
                moves.append(match.group(1))
            if line.startswith('Nodes'):
                return list(dict.fromkeys(moves))
        # An incomplete list cannot establish legality or uniqueness. Kill
        # this session so late perft lines cannot leak into the next request.
        self._kill_process()
        raise TimeoutError('legal-move query did not finish; result discarded')

    def score_str(self, info):
        if 'score cp' in info:
            p = info.split()
            try:
                return f"{int(p[p.index('cp') + 1]) / 100:+.1f}"
            except Exception:
                pass
        if 'score mate' in info:
            p = info.split()
            try:
                return f"M{p[p.index('mate') + 1]}"
            except Exception:
                pass
        return "?"

    def close(self):
        try:
            self._write("quit")
        except Exception:
            pass
        if self.proc is not None:
            try:
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None
        self._stdout_queue = None
