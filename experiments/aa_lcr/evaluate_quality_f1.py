#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Token-level F1 quality evaluation (SQuAD convention).

Drop-in replacement for evaluate_quality.py (which used cosine similarity —
not conventional for systems-paper evaluation). The output CSV format is
identical:

  - Same column names as the input (no renames)
  - Text cells in `...answer{i}` columns are replaced with float scores in [0.0, 1.0]
  - Reference column set to 1.0
  - Empty hypothesis -> 0.0 (with --keep-empty-zero, default)
  - Bare `answer{i}` columns dropped in the final output

F1 uses the SQuAD-style normalization (lowercase, strip articles, strip
punctuation, collapse whitespace), then whitespace tokenization and
token-overlap F1 = 2*P*R/(P+R).

Behavior:
  * --use-base: reference = base_answer{i}
  * otherwise: reference = answer{i}
  * If reference is empty/NaN for a question on a row, all targets for that
    (row, q) are set to NaN and the reference itself to NaN.

Usage:
  python3 evaluate_quality_f1.py \\
      --in  different_part_of_context_0407/output_0418.csv \\
      --out different_part_of_context_0407/scores_0418_f1.csv \\
      --use-base
"""

import argparse
import re
import string
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


ANSWER_SUFFIX_RE = re.compile(r"answer(\d+)$")
BARE_ANSWER_RE = re.compile(r"^answer(\d+)$")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True, help="Path to input CSV.")
    ap.add_argument("--out", dest="out_csv", required=True, help="Path to output CSV.")
    ap.add_argument(
        "--keep-empty-zero",
        action="store_true",
        default=True,
        help="Empty hypothesis -> 0.0 (default: True).",
    )
    ap.add_argument(
        "--use-base",
        action="store_true",
        default=False,
        help=(
            "Use 'base_answer{i}' as gold reference. "
            "If not set, use 'answer{i}' as gold reference (if present)."
        ),
    )
    return ap.parse_args()


# ---------------- SQuAD-style normalization + token F1 ----------------

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(s: str) -> str:
    """SQuAD normalization: lowercase, remove articles, strip punctuation, collapse whitespace."""
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES_RE.sub(" ", s)
    s = " ".join(s.split())
    return s


def _tokens(s: str) -> List[str]:
    return _normalize(s).split()


def token_f1(pred: str, ref: str) -> float:
    """Token-level F1 between prediction and reference (SQuAD convention).

    Returns 0.0 if either side has no tokens after normalization, or if
    there is no token overlap. Returns 1.0 for exact post-normalization match.
    Handles the SQuAD edge case where empty-vs-empty returns 1.0 — here we
    still return 0.0 for empty hyp to match the old script's semantics.
    """
    pred_toks = _tokens(pred)
    ref_toks = _tokens(ref)
    if not pred_toks or not ref_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(ref_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(ref_toks)
    return (2.0 * precision * recall) / (precision + recall)


# ---------------- CSV-handling helpers (match old script) ----------------

def _is_empty(text) -> bool:
    """Treat None/NaN/empty-string as empty. Strings of only spaces are NOT empty."""
    if text is None:
        return True
    if isinstance(text, float):
        try:
            return pd.isna(text)
        except Exception:
            pass
    if isinstance(text, str):
        return text == ""
    return False


def _to_str(text) -> str:
    if isinstance(text, str):
        return text
    return str(text)


def collect_question_indices(columns: Iterable[str]) -> List[int]:
    """Find all i such that some column ends with 'answer{i}'."""
    idxs = []
    for col in columns:
        m = ANSWER_SUFFIX_RE.search(col)
        if m:
            idxs.append(int(m.group(1)))
    return sorted(set(idxs))


def columns_for_question(
    columns: Iterable[str], q_idx: int, use_base: bool
) -> Tuple[Optional[str], List[str]]:
    """Return (ref_col, other_cols) for a given question index."""
    cols = list(columns)
    suffix = f"answer{q_idx}"
    base_col = f"base_answer{q_idx}"
    bare_col = f"answer{q_idx}"

    if use_base:
        if base_col not in cols:
            return None, []
        ref_col = base_col
    else:
        if bare_col not in cols:
            return None, []
        ref_col = bare_col

    others = []
    for c in cols:
        if c.endswith(suffix) and c != ref_col:
            others.append(c)
    return ref_col, sorted(others)


# ---------------- Main ----------------

def main():
    args = parse_args()
    df = pd.read_csv(args.in_csv)
    cols = list(df.columns)

    q_indices = collect_question_indices(cols)
    if not q_indices:
        raise ValueError("No '...answer{i}' columns found in the CSV.")

    per_q_cols: Dict[int, Tuple[str, List[str]]] = {}
    total_targets = 0
    for qi in q_indices:
        ref_col, other_cols = columns_for_question(cols, qi, args.use_base)
        if ref_col is None:
            continue
        per_q_cols[qi] = (ref_col, other_cols)
        total_targets += len(other_cols)

    if not per_q_cols or total_targets == 0:
        print("No usable reference/target columns; writing input unchanged (minus bare answers).")
        final_drop = [c for c in df.columns if BARE_ANSWER_RE.fullmatch(c)]
        if final_drop:
            df = df.drop(columns=final_drop, errors="ignore")
        df.to_csv(args.out_csv, index=False)
        print(len(df.columns))
        return

    # Pre-compute column positions once to avoid repeated get_loc.
    col_pos = {c: df.columns.get_loc(c) for c in df.columns}

    for ridx in range(len(df)):
        row = df.iloc[ridx]
        for qi, (ref_col, other_cols) in per_q_cols.items():
            ref_text = row.get(ref_col, None)

            if _is_empty(ref_text):
                # Ref missing -> unusable; NaN for all targets and the ref itself.
                for oc in other_cols:
                    df.iat[ridx, col_pos[oc]] = float("nan")
                df.iat[ridx, col_pos[ref_col]] = float("nan")
                continue

            ref_str = _to_str(ref_text)

            for oc in other_cols:
                hyp_text = row.get(oc, None)
                if args.keep_empty_zero and _is_empty(hyp_text):
                    score = 0.0
                else:
                    hyp_str = _to_str(hyp_text)
                    score = token_f1(hyp_str, ref_str)
                df.iat[ridx, col_pos[oc]] = round(float(score), 4)

            df.iat[ridx, col_pos[ref_col]] = 1.0

        if (ridx + 1) % 50 == 0:
            print(f"Processed {ridx + 1}/{len(df)} rows...")

    final_drop = [c for c in df.columns if BARE_ANSWER_RE.fullmatch(c)]
    if final_drop:
        df = df.drop(columns=final_drop, errors="ignore")

    df.to_csv(args.out_csv, index=False)
    print(len(df.columns))
    print(f"Done. Wrote F1 scores to: {args.out_csv}")


if __name__ == "__main__":
    main()
