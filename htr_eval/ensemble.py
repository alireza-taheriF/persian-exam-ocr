"""Fuse several engines' predictions for the same line crops: word-level vote around the line-level medoid.

    .venv/bin/python -m htr_eval.ensemble --out htr_eval/cache/fused.jsonl A.jsonl B.jsonl C.jsonl ...
Per line: the medoid is the candidate with the smallest total edit distance to the others; every other candidate is
aligned to it word by word, and a medoid word is replaced when more aligned candidates agree on another word.
Ties keep the medoid word. Inputs are the JSONL files written by htr_eval.run (same image_path keys).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from pipeline.normalizer import normalize_text
from training.eval.metrics import levenshtein


def load_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def fuse(cands: list[str]) -> str:
    med = min(cands, key=lambda x: sum(levenshtein(x, y) for y in cands)).split()
    votes = [[w] for w in med]
    for c in cands:
        words = c.split()
        if words == med:
            continue
        for op, i1, i2, j1, j2 in SequenceMatcher(None, med, words, autojunk=False).get_opcodes():
            if op == "equal":
                for i in range(i1, i2):
                    votes[i].append(med[i])
            elif op == "replace" and i2 - i1 == j2 - j1:  # same-length substitution: one vote per aligned word
                for k in range(i2 - i1):
                    votes[i1 + k].append(words[j1 + k])
    out = []
    for v in votes:
        word, n = Counter(v).most_common(1)[0]
        out.append(word if n > Counter(v)[v[0]] else v[0])
    return " ".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("preds", type=Path, nargs="+")
    a = ap.parse_args()
    runs = [{r["image_path"]: normalize_text(r.get("text") or "") for r in load_jsonl(p)} for p in a.preds]
    keys = [k for k in runs[0] if all(k in r for r in runs)]
    a.out.write_text("".join(json.dumps({"image_path": k, "text": fuse([r[k] for r in runs])}, ensure_ascii=False) + "\n" for k in keys), encoding="utf-8")
    print(f"{len(keys)} lines fused from {len(runs)} engines -> {a.out}")


if __name__ == "__main__":
    main()
