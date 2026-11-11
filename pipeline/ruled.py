"""Line finder for ruled notebook paper: text lines are the bands between the printed rule lines.

Surya's detector picks up rules and margin numbers on notebook pages; on ruled paper the rules are a far better
signal. find_ruled_lines() returns line boxes (x1, y1, x2, y2) top to bottom, or None when the page has no regular rules
(then use the normal detector).
"""
from __future__ import annotations

import cv2
import numpy as np

MIN_RULES = 6
CROP_X = (0.03, 0.92)  # crop span as a share of page width: handwriting spills left; printed line numbers start at ~0.925


def _rule_rows(gray: np.ndarray) -> tuple[list[int], float, tuple[int, int]]:
    """y positions of the printed rules, their spacing, and the x range of the paper interior.

    Ink covers only part of a row but a rule spans all of it, so a low percentile over x of a row's darkness is a
    clean rule signal; the spacing comes from its autocorrelation."""
    h, w = gray.shape
    x1, x2 = int(w * 0.12), int(w * 0.88)
    prof = np.percentile(255.0 - gray[:, x1:x2], 20, axis=1)
    prof = prof - cv2.blur(prof.reshape(-1, 1), (1, 41)).ravel()  # drop slow lighting changes
    if prof.std() < 0.5:
        return [], 0.0, (x1, x2)
    ac = np.correlate(prof, prof, mode="full")[h - 1:]
    lo, hi = 40, min(h // 3, 500)
    if hi <= lo:
        return [], 0.0, (x1, x2)
    spacing = float(lo + int(np.argmax(ac[lo:hi])))
    thr = prof.std() * 1.2
    cand = [i for i in range(2, h - 2) if prof[i] > prof[i - 1] and prof[i] >= prof[i + 1] and prof[i] > thr]
    rules: list[int] = []
    for i in cand:  # merge double peaks of one thick rule: keep the strongest within 0.4 spacing
        if rules and i - rules[-1] < 0.4 * spacing:
            if prof[i] > prof[rules[-1]]:
                rules[-1] = i
        else:
            rules.append(i)
    return _extend(rules, prof, spacing, h), spacing, (x1, x2)


def _extend(rules: list[int], prof: np.ndarray, spacing: float, h: int) -> list[int]:
    """Follow the grid beyond the strong rules: down the page at the fixed spacing, up only while a rule is visible."""
    if not rules:
        return rules
    floor = prof.std() * 0.45
    out = sorted(rules)
    for step in (+1, -1):
        y = out[-1] if step > 0 else out[0]
        while True:
            guess = y + step * spacing
            lo, hi = int(guess - 0.2 * spacing), int(guess + 0.2 * spacing)
            if lo < 2 or hi > h - 2:
                break
            seg = prof[lo:hi]
            if seg.max() < floor and step < 0:
                break  # upward the header/binding starts: only follow visible rules
            y = lo + int(np.argmax(seg)) if seg.max() >= floor else int(guess)  # downward dense ink hides rules: keep the grid
            out.append(y) if step > 0 else out.insert(0, y)
    return out


def find_ruled_lines(rgb: np.ndarray) -> list[tuple[int, int, int, int]] | None:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rules, spacing, (x1, x2) = _rule_rows(gray)
    if len(rules) < MIN_RULES or spacing < 20:
        return None
    gaps = np.diff(rules)
    near = [abs(g / spacing - round(g / spacing)) <= 0.2 and round(g / spacing) >= 1 for g in gaps]  # a faint rule may be missing
    if float(np.mean(near)) < 0.6:
        return None  # not evenly ruled
    x1, x2 = int(gray.shape[1] * 0.04), int(gray.shape[1] * 0.90)
    background = cv2.medianBlur(gray, 51)
    ink = ((background.astype(np.int16) - gray.astype(np.int16)) > 35).astype(np.uint8) * 255
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    
    all_bands = []
    for top, bottom in zip(rules[:-1], rules[1:]):
        gap = bottom - top
        if gap > 1.8 * spacing:
            n_sub = int(round(gap / spacing))
            sub_step = gap / n_sub
            for k in range(n_sub):
                all_bands.append((int(top + k * sub_step), int(top + (k + 1) * sub_step)))
        else:
            all_bands.append((top, bottom))

    boxes = []
    pad = int(spacing * 0.25)
    for top, bottom in all_bands:
        y1, y2 = max(0, top - pad), min(gray.shape[0], bottom + pad)
        band = ink[top + 1:bottom - 1, x1:x2]
        if band.size == 0 or band.mean() / 255 < 0.005:  # empty band
            continue
        ys, xs = np.where(ink[y1:y2, x1:x2] > 0)
        if len(xs) == 0:
            continue
        min_x = max(x1, x1 + int(xs.min()) - 4)
        max_x = min(x2, x1 + int(xs.max()) + 4)
        min_y = max(0, y1 + int(ys.min()) - 2)
        max_y = min(gray.shape[0], y1 + int(ys.max()) + 2)
        boxes.append((min_x, min_y, max_x, max_y))
    return boxes or None
