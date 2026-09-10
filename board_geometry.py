"""Piece-count-independent Xiangqi grid detection in client screenshots."""
import cv2
import numpy as np


def _edge_axis(profile, count):
    """Fit all grid lines jointly, allowing border lines at crop edges."""
    length = len(profile)
    profile = np.maximum.reduce([profile, np.roll(profile, 1), np.roll(profile, -1)])
    starts = np.arange(0, max(2, int(length*.15)))[:, None]
    best = None
    for span in np.arange(length*.78, length, .5):
        positions = starts + np.arange(count)[None, :] * span/(count-1)
        valid = positions[:, -1] < length
        positions = positions[valid]
        if not len(positions):
            continue
        values = profile[np.rint(positions).astype(int).clip(0, length-1)]
        mids = (positions[:, 1:] + positions[:, :-1])*.5
        between = profile[np.rint(mids).astype(int).clip(0, length-1)]
        hits = (values > .12).sum(axis=1)
        scores = np.minimum(values, .7).mean(axis=1) - .65*between.mean(axis=1)
        scores[hits < count-2] = -1
        i = int(scores.argmax())
        if best is None or scores[i] > best[0]:
            best = float(scores[i]), positions[i].tolist()
    return best if best and best[0] > .16 else None


def detect_edge_grid(img):
    """Find board frames without relying on wood color or occupied ranks.

    Rectangle edges propose crops; jointly fitted 9/10-line combs verify
    them. Requiring evidence on both axes rejects UI panels and avatars.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 25, 80)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (w < 180 or h < 200 or not .78 < w/h < 1.02 or
                w*h < img.shape[0]*img.shape[1]*.08):
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, .015*perimeter, True)
        if len(polygon) != 4 or abs(cv2.contourArea(contour))/(w*h) < .85:
            continue
        if any(abs(x-a)+abs(y-b)+abs(w-c)+abs(h-d) < 12 for a,b,c,d in boxes):
            continue
        boxes.append((x,y,w,h))
    best = None
    for x,y,w,h in sorted(boxes, key=lambda b:b[2]*b[3], reverse=True)[:12]:
        crop = gray[y:y+h, x:x+w]
        dark = cv2.morphologyEx(crop, cv2.MORPH_BLACKHAT, np.ones((9,9),np.uint8))
        ink = (dark > 7).astype(np.uint8)
        v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((max(15,h//12),1),np.uint8))
        z = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1,max(15,w//12)),np.uint8))
        cols, rows = _edge_axis(v.mean(axis=0),9), _edge_axis(z.mean(axis=1),10)
        if cols is None or rows is None:
            continue
        dx,dy = (cols[1][-1]-cols[1][0])/8, (rows[1][-1]-rows[1][0])/9
        if not .97 < dx/dy < 1.03:
            continue
        score = cols[0]+rows[0]
        if best is None or score > best[0]:
            best = score, [a+x for a in cols[1]], [b+y for b in rows[1]]
    return (best[1],best[2]) if best else None


def refine_grid_centers(img, cols, rows):
    """Refine a line fit with nearby piece circles, independent of layout.

    Match only within 0.4 cells of the proposed lattice. Unlike back-rank
    fitting this works with moved/captured pieces; sparse axes keep their
    line-derived spacing. Shadow-biased circle detections are median filtered.
    """
    cols, rows = np.asarray(cols, dtype=float), np.asarray(rows, dtype=float)
    step = float(min(np.median(np.diff(cols)), np.median(np.diff(rows))))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1.2, step * .6, param1=120,
        param2=28, minRadius=int(step * .30), maxRadius=int(step * .50))
    if circles is None:
        return cols.tolist(), rows.tolist()
    samples = []
    for x, y, _radius in circles[0]:
        c, r = int(np.argmin(abs(cols-x))), int(np.argmin(abs(rows-y)))
        if abs(cols[c]-x) < .4*step and abs(rows[r]-y) < .4*step:
            samples.append((c, r, float(x), float(y)))
    if len(samples) < 3:
        return cols.tolist(), rows.tolist()

    def fit_axis(original, index, coord):
        shifts = np.array([p[coord]-original[p[index]] for p in samples])
        median = np.median(shifts)
        kept = [p for p, d in zip(samples, shifts) if abs(d-median) < .10*step]
        ids = np.array([p[index] for p in kept])
        values = np.array([p[coord] for p in kept])
        if len(set(ids)) >= 3 and np.ptp(ids) >= (len(original)-1)*.5:
            slope, offset = np.polyfit(ids, values, 1)
            result = offset + slope*np.arange(len(original))
            if abs(slope/np.median(np.diff(original))-1) < .04:
                return result.tolist()
        return (original+median).tolist()

    return fit_axis(cols, 0, 2), fit_axis(rows, 1, 3)


def _fit_axis(profile, count):
    """Fit a regular comb inside the wooden margin; missing lines are allowed."""
    length = len(profile)
    # A line can occupy adjacent pixels; tolerate one-pixel rounding.
    profile = np.maximum.reduce([profile, np.roll(profile, 1), np.roll(profile, -1)])
    best = None
    for span in np.arange(length * .80, length * .95, .5):
        step = span / (count - 1)
        for start in range(int(length * .025), int(length * .14) + 1):
            positions = start + np.arange(count) * step
            if positions[-1] >= length - length * .02:
                continue
            values = profile[np.rint(positions).astype(int)]
            hits = np.count_nonzero(values > .16)
            if hits < count - 3:
                continue
            mids = profile[np.rint((positions[:-1] + positions[1:]) / 2).astype(int)]
            score = float(np.minimum(values, .65).mean() - .5 * mids.mean())
            if best is None or score > best[0]:
                best = score, positions
    return best if best and best[0] >= .18 else None


def _grid_quality(img, cols, rows):
    """Score a fitted grid using the dark line response in its crop."""
    if len(cols) != 9 or len(rows) != 10:
        return -1.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                            np.ones((11, 11), np.uint8))
    vp = dark.mean(axis=0)
    hp = dark.mean(axis=1)
    cv = np.asarray(np.rint(cols), dtype=int)
    rv = np.asarray(np.rint(rows), dtype=int)
    if (cv.min() < 0 or rv.min() < 0 or cv.max() >= vp.size or
            rv.max() >= hp.size):
        return -1.0
    values = float(np.minimum(vp[cv], 65).mean() +
                   np.minimum(hp[rv], 65).mean())
    mids = float(vp[np.rint((cv[:-1] + cv[1:]) / 2).astype(int)].mean() +
                 hp[np.rint((rv[:-1] + rv[1:]) / 2).astype(int)].mean())
    return values - .5 * mids


def _hough_boxes(img):
    """Yield rough board boxes when wood segmentation joins side panels."""
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 90)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 80,
        minLineLength=max(300, int(min(height, width) * .35)),
        maxLineGap=30,
    )
    horizontal, vertical = [], []
    for line in ([] if lines is None else lines):
        x1, y1, x2, y2 = (int(v) for v in line[0])
        if abs(y2 - y1) <= 5:
            lo, hi = sorted((x1, x2))
            if hi - lo >= max(300, int(min(height, width) * .35)):
                horizontal.append((int(round((y1 + y2) / 2)), lo, hi))
        elif abs(x2 - x1) <= 5:
            lo, hi = sorted((y1, y2))
            if hi - lo >= max(300, int(min(height, width) * .35)):
                vertical.append((int(round((x1 + x2) / 2)), lo, hi))

    def group(items):
        groups = []
        for item in sorted(items):
            if not groups or item[0] > groups[-1][0] + 6:
                groups.append([item[0], [item]])
            else:
                groups[-1][1].append(item)
                groups[-1][0] = int(round(np.mean([v[0] for v in groups[-1][1]])))
        result = []
        for pos, members in groups:
            result.append((pos, min(v[1] for v in members),
                           max(v[2] for v in members)))
        return result

    horizontal, vertical = group(horizontal), group(vertical)
    boxes = []

    def edge_fit(a, b, lo, hi):
        span = max(1, b - a)
        overlap = max(0, min(b, hi) - max(a, lo)) / span
        excess = max(0, (hi - lo) - span) / span
        return overlap - .35 * excess

    for top in horizontal:
        for bottom in horizontal:
            y1, y2 = top[0], bottom[0]
            if not 350 <= y2 - y1 <= int(height * .95):
                continue
            for left in vertical:
                for right in vertical:
                    x1, x2 = left[0], right[0]
                    # The application board is a bounded panel. Reject
                    # desktop/window border pairs spanning the whole client;
                    # they otherwise dominate the Hough candidate ranking.
                    if not 400 <= x2 - x1 <= min(900, int(width * .65)):
                        continue
                    aspect = (x2 - x1) / max(1, y2 - y1)
                    if not .76 <= aspect <= 1.05:
                        continue
                    # Expand from the perimeter lines so the inner 9x10
                    # lattice falls inside the normal detector's margin.
                    pad_x = max(18, int((x2 - x1) * .06))
                    pad_y = max(18, int((y2 - y1) * .06))
                    bx1, by1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                    bx2, by2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
                    score = (edge_fit(x1, x2, top[1], top[2]) +
                             edge_fit(x1, x2, bottom[1], bottom[2]) +
                             edge_fit(y1, y2, left[1], left[2]) +
                             edge_fit(y1, y2, right[1], right[2]))
                    boxes.append((score, bx1, by1, bx2, by2))
    seen = set()
    for _score, x1, y1, x2, y2 in sorted(boxes, reverse=True)[:24]:
        key = (x1 // 8, y1 // 8, x2 // 8, y2 // 8)
        if key in seen:
            continue
        seen.add(key)
        yield x1, y1, x2, y2


def detect_grid(img, _allow_hough=True):
    """Return (9 columns, 10 rows), or None when no grid is visible.

    Coordinates are relative to the supplied client image, never the desktop.
    Detect wood candidates, then require regularly spaced long lines on both
    axes. Menus containing wood and piece icons alone are not sufficient.
    """
    if _allow_hough:
        edge_fit = detect_edge_grid(img)
        if edge_fit is not None:
            return edge_fit
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    wood = cv2.inRange(hsv, (8, 25, 145), (35, 190, 255))
    wood = cv2.morphologyEx(wood, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    contours, _ = cv2.findContours(wood, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (.76 <= w/max(h, 1) <= 1.05 and w >= 250 and h >= 300):
            continue
        if w*h < img.shape[0]*img.shape[1]*.08:
            continue
        candidates.append((w*h, x, y, w, h))
    best = None
    for _, x, y, w, h in sorted(candidates, reverse=True)[:3]:
        gray = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, np.ones((11, 11), np.uint8))
        binary = (dark > 12).astype(np.uint8)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((max(25, h//6), 1), np.uint8))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((1, max(25, w//5)), np.uint8))
        cols = _fit_axis(vertical.mean(axis=0), 9)
        rows = _fit_axis(horizontal.mean(axis=1), 10)
        if cols is None or rows is None:
            continue
        cw, ch = np.mean(np.diff(cols[1])), np.mean(np.diff(rows[1]))
        if not .90 <= cw/ch <= 1.10:
            continue
        score = cols[0] + rows[0]
        if best is None or score > best[0]:
            best = score, (cols[1] + x).tolist(), (rows[1] + y).tolist()
    if best:
        return (best[1], best[2])

    if not _allow_hough:
        return None

    # Some layouts join the board's wood with a large side panel, so the
    # color contour has no 9:10 rectangle. Use long UI borders to propose
    # small board boxes, then reuse the stricter normal lattice detector.
    for x1, y1, x2, y2 in _hough_boxes(img):
        fit = detect_grid(img[y1:y2, x1:x2], _allow_hough=False)
        if fit is None:
            continue
        cols = [float(v) + x1 for v in fit[0]]
        rows = [float(v) + y1 for v in fit[1]]
        # The inner detector already requires a regular 9x10 lattice and
        # line contrast. Returning the first strict hit keeps first-frame
        # recovery bounded; later frames use _locate_grid's local fast path.
        return cols, rows
    return None
