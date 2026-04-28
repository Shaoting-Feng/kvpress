#!/usr/bin/env python3
"""Split AA-LCR's 100 questions so EVERY (document_category, document_set_id)
appears in BOTH the profiling and testing halves — i.e. each context is shared,
only the questions differ.

For a set with k questions:
  - k >= 2: deterministically (by question_id) split into ceil(k/2) vs floor(k/2);
            assign the larger half to whichever side currently has fewer
            questions, to keep the two halves balanced.
  - k == 1: the singleton is DROPPED — its context cannot have distinct
            profiling and testing questions, so it is excluded entirely.

Outputs profiling.csv and testing.csv with the original schema.
"""

import csv
import sys
from collections import defaultdict, Counter
from pathlib import Path

csv.field_size_limit(sys.maxsize)

HERE = Path(__file__).resolve().parent
SRC = HERE / "AA-LCR_Dataset.csv"
OUT_PROFILING = HERE / "profiling.csv"
OUT_TESTING = HERE / "testing.csv"


def main() -> None:
    with SRC.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if fieldnames is None:
        raise RuntimeError("missing header in source CSV")

    by_set: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_set[(r["document_category"], r["document_set_id"])].append(r)

    for qs in by_set.values():
        qs.sort(key=lambda r: int(r["question_id"]))

    sets_sorted = sorted(by_set.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    sides: dict[str, list[dict]] = {"profiling": [], "testing": []}
    dropped_singletons: list[tuple[str, str]] = []

    for key, qs in sets_sorted:
        k = len(qs)
        if k == 1:
            dropped_singletons.append(key)
            continue

        big = (k + 1) // 2
        # Give the larger half to whichever side currently has fewer rows,
        # so totals converge toward balance.
        big_side = min(("profiling", "testing"), key=lambda s: (len(sides[s]), s))
        small_side = "testing" if big_side == "profiling" else "profiling"

        sides[big_side].extend(qs[:big])
        sides[small_side].extend(qs[big:])

    for name, out_path in (("profiling", OUT_PROFILING), ("testing", OUT_TESTING)):
        with out_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in sides[name]:
                w.writerow(r)

    print(f"total questions in source: {len(rows)}, total sets: {len(by_set)}")
    for name in ("profiling", "testing"):
        cat = Counter(r["document_category"] for r in sides[name])
        sets_present = {(r["document_category"], r["document_set_id"]) for r in sides[name]}
        print(
            f"{name:10s}  rows={len(sides[name]):3d}  unique_sets={len(sets_present):2d}  "
            f"by_category={dict(cat)}"
        )

    sets_p = {(r["document_category"], r["document_set_id"]) for r in sides["profiling"]}
    sets_t = {(r["document_category"], r["document_set_id"]) for r in sides["testing"]}
    only_p = sets_p - sets_t
    only_t = sets_t - sets_p
    assert not only_p and not only_t, f"context not on both sides: only_p={only_p}, only_t={only_t}"
    print(f"ok — every context appears in both halves ({len(sets_p & sets_t)} contexts shared)")
    if dropped_singletons:
        print(f"singletons dropped (k=1, no way to share context): {len(dropped_singletons)}")
        for k in dropped_singletons:
            print(f"  {k[0]}/{k[1]}")


if __name__ == "__main__":
    main()
