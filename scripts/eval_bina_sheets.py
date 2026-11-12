#!/usr/bin/env python3
"""Score Bina-0.4-Rizeh's CTC line recognizer on numbered contact sheets of gold lines.

The sheets (one red number per row, gray rules between rows) hold resized copies of
dataset/lines_v3 crops; key.json maps each number to its gold text and document.
The recognizer is run in numpy (no torch): the forward pass mirrors
models/bina/stallion/indra-scale/recognizer.py, preprocessing mirrors
geometry.crop_line64/pad_batch, decoding mirrors rtl_reader.RTLReader (greedy CTC,
visual order flipped to logical). Weights: models/bina/model.safetensors.

    python scripts/eval_bina_sheets.py --key key.json --sheets 1.jpg 2.jpg ... --label bina_sheets
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
BINA = ROOT / "models" / "bina"
sys.path.insert(0, str(BINA / "stallion/bina04-indra-pilot-20261027/upstream"))

from indra_ocr.rtl import flip  # noqa: E402
from pipeline.normalizer import normalize_text  # noqa: E402
from training.eval.metrics import corpus_cer, corpus_wer  # noqa: E402


# ---------------------------------------------------------------- recognizer (numpy)
def hardswish(x):
    return x * np.clip(x + 3, 0, 6) / 6


def conv1x1(w, b, x):  # w (O,I,1), x (I,T)
    return w[:, :, 0] @ x + b[:, None]


class Recognizer:
    def __init__(self, path: Path):
        sd = load_file(str(path))
        self.p = {k[len("recognizer."):]: v.astype(np.float32) for k, v in sd.items() if k.startswith("recognizer.")}
        vocab = json.loads((BINA / "stallion/bina04-pagepath-v3-round3-clean-20261031/data/vocab.json").read_text())
        self.alphabet = vocab["chars"]
        assert self.p["head.weight"].shape[0] == len(self.alphabet) + 1, "vocab/head mismatch"

    def logits(self, x: np.ndarray, width: int) -> np.ndarray:
        """x: (64, Wp) float32 ink image, Wp a multiple of 8; returns (T, classes) for the valid frames."""
        p = self.p
        x = np.pad(x, ((0, 0), (0, 4)))  # F.pad(x,(0,4,0,0))
        win = np.lib.stride_tricks.sliding_window_view(x, 8, axis=1)[:, ::4, :]  # (64, T, 8)
        T = win.shape[1]
        z = np.einsum("ohk,htk->ot", p["stem.weight"][:, 0], win, optimize=True) + p["stem.bias"][:, None]
        n = (width + 3) // 4
        mask = (np.arange(T) < n).astype(np.float32)[None, :]
        z = (z - p["bn.running_mean"][:, None]) / np.sqrt(p["bn.running_var"][:, None] + 1e-5)
        z = hardswish(z * p["bn.weight"][:, None] + p["bn.bias"][:, None]) * mask
        for i in range(20):
            q = lambda s: p[f"blocks.{i}.{s}"]  # noqa: E731
            dw = q("dw.weight")[:, 0, :]  # (512,17)
            zp = np.pad(z, ((0, 0), (8, 8)))
            y = sum(dw[:, k:k + 1] * zp[:, k:k + T] for k in range(17)) + q("dw.bias")[:, None]
            h = hardswish(conv1x1(q("expand.weight"), q("expand.bias"), y))
            pool = (h * mask).sum(1, keepdims=True) / max(mask.sum(), 1)
            g = conv1x1(q("fc2.weight"), q("fc2.bias"), np.maximum(conv1x1(q("fc1.weight"), q("fc1.bias"), pool), 0))
            g = 1 / (1 + np.exp(-g))
            z = (z + conv1x1(q("project.weight"), q("project.bias"), h * g)) * mask
        out = conv1x1(p["head.weight"], p["head.bias"], z).T
        return out[:n]

    def read(self, gray: np.ndarray) -> tuple[str, float]:
        ink, width = crop_line64(gray)
        wp = math.ceil(width / 8) * 8
        x = np.zeros((64, wp), np.float32)
        x[:, :width] = ink
        lg = self.logits(x, width)
        e = np.exp(lg - lg.max(1, keepdims=True))
        pr = e / e.sum(1, keepdims=True)
        best, conf = pr.argmax(1), pr.max(1)
        keep = (best != 0) & np.concatenate(([True], best[1:] != best[:-1]))
        visual = "".join(self.alphabet[t - 1] for t in best[keep])
        return flip(visual, base_rtl=True), float(conf[keep].mean()) if keep.any() else 0.0


def is_dark_line(gray: np.ndarray) -> bool:
    lo, med, hi = np.percentile(gray, (2, 50, 98))
    return bool(hi - med > 2 * (med - lo) + 8)


def crop_line64(gray: np.ndarray, pad_y=.2, pad_x=.3):
    """geometry.crop_line64 with the box = the whole crop."""
    ch0, cw0 = gray.shape
    py, px = math.ceil(pad_y * ch0), math.ceil(pad_x * ch0)
    dark = is_dark_line(gray)
    canvas = np.full((ch0 + 2 * py, cw0 + 2 * px), 0 if dark else 255, np.uint8)
    canvas[py:py + ch0, px:px + cw0] = gray
    ch, cw = canvas.shape
    width = max(1, round(cw * 64 / ch))
    r = cv2.resize(canvas, (width, 64), interpolation=cv2.INTER_AREA if ch > 64 else cv2.INTER_LINEAR).astype(np.float32) / 255.
    return (r if dark else 1. - r), width


# ---------------------------------------------------------------- contact sheets
def sheet_rows(path: Path, label_col: int = 72):
    """Yield the line image of each row, top to bottom, with the red number column cut off."""
    a = cv2.imread(str(path))[:, :, ::-1].astype(int)
    g = a.mean(2)
    strip, red = g[:, 2:70], (a[:, 2:70, 0] - a[:, 2:70, 1]) > 60
    sep = np.nonzero(((strip < 235) & (strip > 150) & ~red).mean(1) > 0.9)[0]
    bounds, last = [], -10
    for y in sep:
        if y - last > 2:
            bounds.append(y)
        last = y
    top = 0
    for y in bounds:
        cell = a[top + 2:y - 1, label_col:]
        nonwhite = (cell < 245).any(2)
        ys, xs = np.nonzero(nonwhite)
        if len(ys):
            cell = cell[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        yield cv2.cvtColor(cell.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        top = y


def bucket(n: int) -> str:
    return "<=4" if n <= 4 else ("5-20" if n <= 20 else ">20")


def score(refs, hyps):
    c, w = corpus_cer(refs, hyps), corpus_wer(refs, hyps)
    return {"lines": len(refs), "cer": round(c, 4), "wer": round(w, 4),
            "char_accuracy_pct": round(100 * (1 - c), 2), "word_accuracy_pct": round(100 * (1 - w), 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--sheets", type=Path, nargs="+", required=True)
    ap.add_argument("--label", default="bina_rizeh_sheets")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "dataset" / "reports")
    ap.add_argument("--dump-crops", type=Path)
    args = ap.parse_args()

    key = json.loads(args.key.read_text(encoding="utf-8"))
    rec = Recognizer(BINA / "model.safetensors")
    preds, secs, n = {}, [], 0
    for sheet in args.sheets:
        for gray in sheet_rows(sheet):
            n += 1
            if args.dump_crops:
                args.dump_crops.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.dump_crops / f"{n:03d}.png"), gray)
            t0 = time.time()
            text, conf = rec.read(gray)
            secs.append(time.time() - t0)
            preds[str(n)] = {"pred": text, "confidence": round(conf, 3)}
    ids = [i for i in preds if i in key and key[i]["text"].strip()]
    refs = [normalize_text(key[i]["text"]) for i in ids]
    hyps = [normalize_text(preds[i]["pred"]) for i in ids]

    rng = random.Random(0)
    boots = sorted(corpus_cer(*zip(*[(refs[k], hyps[k]) for k in (rng.randrange(len(refs)) for _ in refs)]))
                   for _ in range(1000))
    by_doc, by_len = {}, {}
    for d in sorted({key[i]["doc"] for i in ids}):
        ix = [k for k, i in enumerate(ids) if key[i]["doc"] == d]
        by_doc[d] = score([refs[k] for k in ix], [hyps[k] for k in ix])
    for b in ("<=4", "5-20", ">20"):
        ix = [k for k in range(len(ids)) if bucket(len(refs[k])) == b]
        if ix:
            by_len[b] = score([refs[k] for k in ix], [hyps[k] for k in ix])
    report = {
        "name": args.label, "engine": "Bina-0.4-Rizeh recognizer only (95k Indra CTC, numpy port, greedy, no confidence filter)",
        "gold_lines": len(key), "answered": len(ids),
        "missing_ids": sorted((set(key) - set(preds)), key=int),
        "overall": score(refs, hyps),
        "cer_ci95_bootstrap": [round(boots[25], 4), round(boots[974], 4)],
        "exact_line_match_pct": round(100 * sum(a == b for a, b in zip(refs, hyps)) / len(refs), 2),
        "by_document": by_doc, "by_length": by_len,
        "seconds_per_line_median": round(sorted(secs)[len(secs) // 2], 3),
        "caveat": "lines cropped from resized contact sheets, not the original lines_v3 crops",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.label}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(args.out_dir / f"{args.label}.preds.jsonl", "w", encoding="utf-8") as f:
        for i in ids:
            f.write(json.dumps({"id": i, "doc": key[i]["doc"], "gold": key[i]["text"], **preds[i]}, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("answered", "overall", "cer_ci95_bootstrap", "by_document", "by_length")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
