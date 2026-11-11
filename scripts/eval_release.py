#!/usr/bin/env python3
"""Score an engine on the gold set through the exact code path the web app serves.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/eval_release.py --engine qwen3vl-fa --label release_qwen3vl_fa
    ... --engine qwen25vl-master-lora --label release_master_lora
    ... --engine qwen3vl-fa --label release_qwen3vl_fa_corr --correct   (needs Ollama qwen3:8b)

Every line crop goes through pipeline.ocr._fit_line_height and htr_eval.engines (the
worker the app spawns), loop outputs are dropped as in run_ocr, and, with --correct,
pipeline.corrector.correct_lines runs per source page as in the app.
Writes eval/reports/<label>.json and <label>.preds.jsonl (resumable).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from pipeline.normalizer import normalize_text  # noqa: E402
from training.eval.metrics import corpus_cer, corpus_wer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--gold", default="dataset/gold.jsonl")
    ap.add_argument("--correct", action="store_true")
    ap.add_argument("--data-root", type=Path, default=ROOT, help="folder that holds the gold JSONL and its line crops")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "eval" / "reports")
    args = ap.parse_args()

    from htr_eval.engines import load_engine
    from pipeline.ocr import _fit_line_height, _is_loop

    rows = [json.loads(l) for l in open(args.data_root / args.gold, encoding="utf-8") if l.strip()]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / f"{args.label}.preds.jsonl"
    done = {}
    if preds_path.exists():
        for l in open(preds_path, encoding="utf-8"):
            d = json.loads(l)
            done[d["image_path"]] = d
    engine = load_engine(args.engine)
    secs = []
    with open(preds_path, "a", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            if r["image_path"] in done:
                continue
            img = Image.open(args.data_root / r["image_path"]).convert("RGB")
            t0 = time.time()
            try:
                text = engine.recognize(_fit_line_height(img))["text"]
            except Exception as exc:  # the app marks such a block illegible; score it as empty
                if "out of memory" in str(exc).lower():
                    raise  # an environment problem, not a model result: stop, fix, resume
                print(f"recognize failed on {r['image_path']}: {exc}", flush=True)
                text = ""
            secs.append(time.time() - t0)
            if _is_loop(text):
                text = ""
            d = {"image_path": r["image_path"], "pred": text}
            done[r["image_path"]] = d
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"{i}/{len(rows)} {secs[-1]:.2f}s", flush=True)

    preds = {k: v["pred"] for k, v in done.items()}
    if args.correct:
        from pipeline.corrector import correct_lines
        by_page = defaultdict(list)
        for r in rows:
            by_page[r["source_image"]].append(r)
        for page_rows in by_page.values():
            page_rows.sort(key=lambda r: r["line_index"])
            texts = [preds[r["image_path"]] for r in page_rows]
            idx = [k for k, t in enumerate(texts) if t.strip()]
            fixed = correct_lines([texts[k] for k in idx]) if idx else []
            for k, t in zip(idx, fixed):
                preds[page_rows[k]["image_path"]] = t

    legible = [r for r in rows if r["text"].strip()]
    refs = [normalize_text(r["text"]) for r in legible]
    hyps = [normalize_text(preds[r["image_path"]]) for r in legible]
    cer_v, wer_v = corpus_cer(refs, hyps), corpus_wer(refs, hyps)

    rng = random.Random(0)
    boots = []
    for _ in range(1000):
        s = [rng.randrange(len(refs)) for _ in refs]
        boots.append(corpus_cer([refs[k] for k in s], [hyps[k] for k in s]))
    boots.sort()

    per_doc = {}
    for doc in sorted({r["source_document"] for r in legible}):
        ix = [k for k, r in enumerate(legible) if r["source_document"] == doc]
        per_doc[doc] = {"lines": len(ix),
                        "cer": round(corpus_cer([refs[k] for k in ix], [hyps[k] for k in ix]), 4),
                        "wer": round(corpus_wer([refs[k] for k in ix], [hyps[k] for k in ix]), 4)}
    illegible = [r for r in rows if not r["text"].strip()]
    report = {
        "label": args.label, "engine": args.engine, "post_correction": args.correct,
        "gold_rows": len(rows), "scored_lines": len(legible),
        "cer": round(cer_v, 4), "wer": round(wer_v, 4),
        "char_accuracy_pct": round(100 * (1 - cer_v), 2), "word_accuracy_pct": round(100 * (1 - wer_v), 2),
        "cer_ci95": [round(boots[25], 4), round(boots[974], 4)],
        "exact_line_match_pct": round(100 * sum(a == b for a, b in zip(refs, hyps)) / len(refs), 2),
        "illegible_rows_returned_empty_pct": round(100 * sum(not preds[r["image_path"]].strip() for r in illegible) / max(1, len(illegible)), 2),
        "seconds_per_line_median": round(statistics.median(secs), 2) if secs else None,
        "per_document": per_doc,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / f"{args.label}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("label", "cer", "wer", "char_accuracy_pct", "word_accuracy_pct", "cer_ci95")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
