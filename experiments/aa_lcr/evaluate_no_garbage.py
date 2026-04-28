#!/usr/bin/env python3
"""Sanity check: SQuAD-F1 of uncompressed gpt-oss-120b answers against the
AA-LCR ground-truth answers. Just reports a distribution; does not drop
anything. The user wants 'a sense of garbage in / garbage out', not filtering.

Reads output_profiling.csv (has base_answer{i} + answer{i} columns) and
prints per-(context, question) F1 plus mean / median / fraction at 0.
"""

from __future__ import annotations

import argparse
import re
import string
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(s: str) -> str:
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES_RE.sub(" ", s)
    s = " ".join(s.split())
    return s


def token_f1(pred: str, ref: str) -> float:
    p = _normalize(pred).split()
    r = _normalize(ref).split()
    if not p or not r:
        return 0.0
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(r)
    return (2 * precision * recall) / (precision + recall)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True)
    ap.add_argument("--out", dest="out_csv", default=None,
                    help="If set, write per-(context,q) F1s as a CSV here.")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    base_cols = sorted(c for c in df.columns
                       if isinstance(c, str) and c.startswith("base_answer")
                       and c[len("base_answer"):].isdigit())
    if not base_cols:
        sys.exit("No base_answer{i} columns found")

    rows = []
    for _, r in df.iterrows():
        for col in base_cols:
            i = int(col[len("base_answer"):])
            ans_col = f"answer{i}"
            if ans_col not in df.columns:
                continue
            base = r[col]
            gt = r[ans_col]
            if pd.isna(base) or pd.isna(gt) or str(base).strip() == "":
                continue
            f1 = token_f1(str(base), str(gt))
            rows.append({
                "document_category": r.get("document_category"),
                "document_set_id": r.get("document_set_id"),
                "q_idx": i,
                "f1": f1,
                "len_pred": len(str(base)),
                "len_gt": len(str(gt)),
            })

    out = pd.DataFrame(rows)
    if args.out_csv:
        out.to_csv(args.out_csv, index=False)
        print(f"[SAVE] {args.out_csv}")

    if out.empty:
        print("no rows scored")
        return

    f1s = out["f1"]
    print(f"\nNo-garbage check (uncompressed gpt-oss-120b vs AA-LCR ground truth):")
    print(f"  n_pairs            = {len(f1s)}")
    print(f"  mean F1            = {f1s.mean():.3f}")
    print(f"  median F1          = {f1s.median():.3f}")
    print(f"  std F1             = {f1s.std():.3f}")
    print(f"  fraction at 0.0    = {(f1s == 0.0).mean():.3f}")
    print(f"  fraction >= 0.30   = {(f1s >= 0.30).mean():.3f}")
    print(f"  fraction >= 0.50   = {(f1s >= 0.50).mean():.3f}")
    print(f"  fraction >= 0.80   = {(f1s >= 0.80).mean():.3f}")
    qs = np.percentile(f1s, [10, 25, 50, 75, 90])
    print(f"  quantiles 10/25/50/75/90 = {qs.round(3).tolist()}")

    print("\nPer category:")
    for cat, sub in out.groupby("document_category"):
        print(f"  {cat:30s}  n={len(sub):3d}  mean={sub['f1'].mean():.3f}  median={sub['f1'].median():.3f}")


if __name__ == "__main__":
    main()
