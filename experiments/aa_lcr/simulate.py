#!/usr/bin/env python3
"""Port of simulation_multi_tier.ipynb cells {2, 4, 6, 8, 10, 16} to a script,
adapted for the AA-LCR profiling/testing split on gpt-oss-120b.

Inputs (under HERE):
  scores_profiling.csv   — per-cell F1 scores, profiling questions per context
  scores_testing.csv     — per-cell F1 scores, testing questions per context
  speed_profile_gptoss120b.csv

Outputs (under HERE/intermediate_results/gpt-oss-120b/):
  results/ours.csv, results/{method}.csv, results/offload.csv, results/prefill.csv
  workload/...
  quality_vs_ttft.png

Naming bridge: AA-LCR's (document_category, document_set_id) is treated as
(dataset, index_in_dataset) so the StorageManager code in components.py works
unchanged.
"""

from __future__ import annotations

import contextlib
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))   # so `import components` resolves to the bundled copy
from components import StorageManager, Object  # noqa: E402

# =================== Config ===================
MODEL = "openai/gpt-oss-120b"
MODEL_DIR = MODEL.split("/")[-1].replace(".", "p")  # "gpt-oss-120b"

SCORES_PROFILING = HERE / "scores_profiling.csv"
SCORES_TESTING = HERE / "scores_testing.csv"
TTFT_CSV = HERE / "speed_profile_gptoss120b.csv"

# gb_per_token from end_to_end/components.py for openai/gpt-oss-120b
TOKENS_TO_GB = 0.0343 / 1000.0

METHOD_RATES = {
    "snapkv":  [0.3, 0.6, 0.9],
    "knorm":   [0.3, 0.6, 0.9],
    "keydiff": [0.3, 0.6, 0.9],
}
ALL_RATES = sorted({r for rs in METHOD_RATES.values() for r in rs})

INTER_DIR = HERE / "intermediate_results" / MODEL_DIR
RESULTS_DIR = INTER_DIR / "results"
WORKLOAD_DIR = INTER_DIR / "workload"


def rate_to_suffix(r: float) -> str:
    return f"{r:.1f}".replace(".", "p")


RATE_SUFFIXES = [rate_to_suffix(r) for r in ALL_RATES]
SUFFIX_TO_RATE = {rate_to_suffix(r): r for r in ALL_RATES}
RATE_MAP: dict[float, str] = {0.0: "0p0", **{float(r): rate_to_suffix(r) for r in ALL_RATES}}


def numeric_mean(df: pd.DataFrame, cols):
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(np.nan, index=df.index)
    sub = df[present].apply(pd.to_numeric, errors="coerce")
    return sub.mean(axis=1)


def detect_q_range(df: pd.DataFrame, prefix: str) -> list[int]:
    """Find indices i for which `{prefix}{i}` exists in df.columns. Sorted."""
    out = []
    plen = len(prefix)
    for c in df.columns:
        if isinstance(c, str) and c.startswith(prefix):
            suf = c[plen:]
            if suf.isdigit():
                out.append(int(suf))
    return sorted(set(out))


def load_and_rename(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    df = df.rename(columns={
        "document_category": "dataset",
        "document_set_id": "index_in_dataset",
    })
    return df


def build_size_score_df(df_prof: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, index_in_dataset) compute score_{suffix} (avg F1 over the
    PROFILING questions for that row), method_{suffix} (best method for that
    context at that rate), length_{suffix} (length * (1 - rate))."""
    train_qs = detect_q_range(df_prof, "base_answer")
    if not train_qs:
        raise RuntimeError("No base_answer{i} columns found in profiling scores CSV")

    df = df_prof.copy()
    df["length_0p0"] = pd.to_numeric(df["length"], errors="coerce")

    # rate=0 baseline
    base_cols = [f"base_answer{i}" for i in train_qs]
    df["score_0p0"] = numeric_mean(df, base_cols)

    # compressed rates
    for suffix in RATE_SUFFIXES:
        rate = SUFFIX_TO_RATE[suffix]
        method_means: list[pd.Series] = []
        avail: list[str] = []
        for method, mrates in METHOD_RATES.items():
            if rate not in mrates:
                continue
            cols = [f"{method}_{suffix}_answer{i}" for i in train_qs]
            method_means.append(numeric_mean(df, cols))
            avail.append(method)
        if method_means:
            stacked = pd.concat(method_means, axis=1)
            stacked.columns = avail
            best_scores = stacked.max(axis=1)
            best_methods = stacked.idxmax(axis=1)
            best_methods = best_methods.where(best_scores.notna(), np.nan)
            df[f"score_{suffix}"] = best_scores
            df[f"method_{suffix}"] = best_methods
        else:
            df[f"score_{suffix}"] = np.nan
            df[f"method_{suffix}"] = np.nan
        df[f"length_{suffix}"] = df["length_0p0"] * (1.0 - rate)

    out_cols = ["dataset", "index_in_dataset", "length_0p0", "score_0p0"]
    for suffix in RATE_SUFFIXES:
        out_cols += [f"length_{suffix}", f"score_{suffix}", f"method_{suffix}"]
    out_cols = [c for c in out_cols if c in df.columns]
    return df[out_cols].copy()


def pick_storage_sizes(size_score_df: pd.DataFrame) -> tuple[float, float]:
    """CPU/SSD sizes designed to put the curve's knee in view: target ~40% of
    total size on CPU, SSD = 10x CPU as in your prior notebook (capacity
    limit only really bites the offload baselines; ours uses utility)."""
    total_gb = float((size_score_df["length_0p0"] * TOKENS_TO_GB).sum())
    cpu_gb = round(total_gb * 0.40, 3)
    ssd_gb = round(cpu_gb * 10.0, 3)
    print(f"[CONFIG] total context KV size = {total_gb:.3f} GB "
          f"-> CPU = {cpu_gb} GB, SSD = {ssd_gb} GB")
    return cpu_gb, ssd_gb


def pick_alphas(size_score_df: pd.DataFrame, ttft_params: pd.DataFrame) -> list[float]:
    """Choose ALPHAS so utility = α·score - TTFT covers the regime where the
    optimum shifts between CPU/SSD/remote/evict.

    α = TTFT_at_remote / 1.0 puts the score-vs-ttft trade at parity for a
    typical context. Sweep ±10x around that midpoint for a clean curve."""
    typical_tokens = float(size_score_df["length_0p0"].median())
    ttft_p = ttft_params.set_index("device")
    a_remote = float(ttft_p.loc["remote", "A"])
    b_remote = float(ttft_p.loc["remote", "B"])
    midpoint = a_remote * typical_tokens + b_remote
    alphas = sorted(round(midpoint * f, 4) for f in (0.1, 0.3, 1.0, 3.0, 10.0))
    print(f"[CONFIG] typical_tokens={typical_tokens:.0f}, midpoint α={midpoint:.4f}, "
          f"sweep={alphas}")
    return alphas


def run_storage_policy(size_score_df: pd.DataFrame,
                       alphas: list[float],
                       cpu_gb: float, ssd_gb: float) -> dict[float, pd.DataFrame]:
    row_by_ctx = {(r["dataset"], r["index_in_dataset"]): r
                  for _, r in size_score_df.iterrows()}
    out: dict[float, pd.DataFrame] = {}

    for alpha in alphas:
        os.environ["CPU_SIZE"] = str(cpu_gb)
        os.environ["SSD_SIZE"] = str(ssd_gb)
        os.environ["ALPHA"] = str(alpha)
        os.environ["TTFT_CSV"] = str(TTFT_CSV)
        os.environ["MODEL"] = MODEL

        manager = StorageManager()
        for ctx, row in row_by_ctx.items():
            num_tokens = float(row["length_0p0"])
            rate_score: dict[float, float] = {}
            for rate, suf in RATE_MAP.items():
                col = f"score_{suf}"
                if col not in row or pd.isna(row[col]):
                    raise RuntimeError(f"missing/NaN {col} for {ctx}")
                rate_score[float(rate)] = float(row[col])

            obj = Object(
                name=f"{ctx[0]}__{ctx[1]}",
                num_context_tokens=num_tokens,
                rate=0.0,
                rate_score=rate_score,
            )
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                manager.policy(obj)

        # Build assignment_df
        records = []
        stored = set()
        for obj in manager.storages.values():
            ds, idx = obj.name.split("__", 1)
            ctx = (ds, idx)
            stored.add(ctx)
            row = row_by_ctx[ctx]
            rate = float(obj.rate)
            suf = RATE_MAP[rate]
            method = (row[f"method_{suf}"] if suf != "0p0" and f"method_{suf}" in row else "baseline")
            length_tokens = float(row[f"length_{suf}"])
            score = float(row[f"score_{suf}"])
            records.append({
                "dataset": ds, "index_in_dataset": idx,
                "device": obj.device, "rate": rate, "method": method,
                "size_GB": length_tokens * TOKENS_TO_GB, "score": score,
            })
        # contexts that didn't fit in any tier → prefill (no caching)
        for ctx in row_by_ctx.keys() - stored:
            row = row_by_ctx[ctx]
            length_tokens = float(row["length_0p0"])
            score = float(row["score_0p0"])
            records.append({
                "dataset": ctx[0], "index_in_dataset": ctx[1],
                "device": "prefill", "rate": 0.0, "method": "baseline",
                "size_GB": length_tokens * TOKENS_TO_GB, "score": score,
            })
        out[alpha] = pd.DataFrame(records)
        print(f"[α={alpha:.4f}] stored={len(stored)}, "
              f"cpu_used={out[alpha].query('device==\"cpu\"')['size_GB'].sum():.3f} GB, "
              f"ssd_used={out[alpha].query('device==\"ssd\"')['size_GB'].sum():.3f} GB, "
              f"by_dev={dict(out[alpha]['device'].value_counts())}")
    return out


def compute_ttft(ttft_params: pd.DataFrame, device: str, num_tokens: float, rate: float) -> float:
    p = ttft_params.set_index("device").loc[device]
    return float(p["A"]) * num_tokens * (1.0 - rate) + float(p["B"])


def evaluate_ours(assignments: dict[float, pd.DataFrame],
                  df_test: pd.DataFrame,
                  ttft_params: pd.DataFrame) -> pd.DataFrame:
    """Per α, join assignment to the testing scores using the assigned (method, rate),
    average all (context, q_id) test scores, and report total_ttft = sum(ctx_ttft) * NQ."""
    test_qs = detect_q_range(df_test, "base_answer")
    NQ = len(test_qs)
    df_test = df_test.copy()
    df_test["length_0p0"] = pd.to_numeric(df_test["length"], errors="coerce")
    df_test = df_test.set_index(["dataset", "index_in_dataset"])

    rows = []
    for alpha, assn in assignments.items():
        all_scores = []
        ttfts = []
        for _, r in assn.iterrows():
            ctx = (r["dataset"], r["index_in_dataset"])
            if ctx not in df_test.index:
                raise KeyError(f"testing CSV missing context {ctx}")
            t = df_test.loc[ctx]
            length = float(t["length_0p0"])
            rate = float(r["rate"])
            ttfts.append(compute_ttft(ttft_params, r["device"], length, rate))

            if rate == 0.0:
                cols = [f"base_answer{i}" for i in test_qs]
            else:
                method = r["method"]
                if not isinstance(method, str) or method == "" or method == "baseline":
                    raise ValueError(f"compressed row but invalid method: rate={rate} method={method}")
                cols = [f"{method}_{RATE_MAP[rate]}_answer{i}" for i in test_qs]
            for c in cols:
                if c not in df_test.columns:
                    continue  # column may not exist if max question count < this i
                v = t[c]
                if pd.isna(v):
                    continue
                all_scores.append(float(v))
        avg = float(np.mean(all_scores)) if all_scores else float("nan")
        rows.append({
            "alpha": alpha,
            "num_contexts": len(assn),
            "num_pairs": len(all_scores),
            "total_ttft": float(sum(ttfts) * NQ),
            "avg_score": avg,
        })
    return pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)


def evaluate_baselines(size_score_df: pd.DataFrame,
                       df_test: pd.DataFrame,
                       ttft_params: pd.DataFrame,
                       cpu_gb: float, ssd_gb: float) -> pd.DataFrame:
    """For each (method, rate) including rate=0 offload, greedy-fill CPU then SSD
    then remote (no per-context optimization, all contexts use same rate).
    Plus 'prefill' baseline (always prefill, never cache).
    """
    test_qs = detect_q_range(df_test, "base_answer")
    NQ = len(test_qs)
    df_test = df_test.copy()
    df_test["length_0p0"] = pd.to_numeric(df_test["length"], errors="coerce")

    base = size_score_df.merge(
        df_test.rename(columns={c: f"_test_{c}" for c in df_test.columns
                                if c not in ("dataset", "index_in_dataset")}),
        on=["dataset", "index_in_dataset"], how="left", validate="one_to_one"
    )

    rows = []

    def fill_devices(ddf: pd.DataFrame, rate: float) -> pd.Series:
        ddf = ddf.copy()
        ddf["size_GB"] = ddf["length_0p0"] * (1.0 - rate) * TOKENS_TO_GB
        cpu_used = ssd_used = 0.0
        devs = []
        for _, r in ddf.iterrows():
            s = float(r["size_GB"])
            if cpu_used + s <= cpu_gb:
                devs.append("cpu"); cpu_used += s
            elif ssd_used + s <= ssd_gb:
                devs.append("ssd"); ssd_used += s
            else:
                devs.append("remote")
        return pd.Series(devs, index=ddf.index)

    for method in METHOD_RATES:
        rates_for_method = sorted({0.0} | set(METHOD_RATES[method]))
        for rate in rates_for_method:
            ddf = base.copy()
            ddf["device"] = fill_devices(ddf, rate)
            ttft_per_ctx = ddf.apply(
                lambda r: compute_ttft(ttft_params, r["device"], float(r["length_0p0"]), rate),
                axis=1,
            )
            total_ttft = float((ttft_per_ctx * NQ).sum())
            if rate == 0.0:
                cols = [f"_test_base_answer{i}" for i in test_qs]
            else:
                cols = [f"_test_{method}_{rate_to_suffix(rate)}_answer{i}" for i in test_qs]
            scores = []
            for c in cols:
                if c in ddf.columns:
                    scores.extend(ddf[c].dropna().astype(float).tolist())
            avg = float(np.mean(scores)) if scores else float("nan")
            rows.append({
                "method": method,
                "rate": rate,
                "total_ttft": total_ttft,
                "avg_score": avg,
            })

    # prefill baseline (no cache)
    ddf = base.copy()
    ttft_per_ctx = ddf["length_0p0"].apply(
        lambda L: compute_ttft(ttft_params, "prefill" if "prefill" in ttft_params["device"].values else "None", float(L), 0.0)
    )
    # speed_profile uses 'None' for prefill row; remap
    if "None" in ttft_params["device"].values and "prefill" not in ttft_params["device"].values:
        a = float(ttft_params.set_index("device").loc["None", "A"])
        b = float(ttft_params.set_index("device").loc["None", "B"])
        ttft_per_ctx = ddf["length_0p0"].apply(lambda L: a * float(L) + b)
    total_ttft = float((ttft_per_ctx * NQ).sum())
    cols = [f"_test_base_answer{i}" for i in test_qs]
    scores = []
    for c in cols:
        if c in ddf.columns:
            scores.extend(ddf[c].dropna().astype(float).tolist())
    avg = float(np.mean(scores)) if scores else float("nan")
    rows.append({"method": "prefill", "rate": 0.0, "total_ttft": total_ttft, "avg_score": avg})

    return pd.DataFrame(rows)


def write_results(ours_df: pd.DataFrame, base_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ours_df.to_csv(out_dir / "ours.csv", index=False)
    print(f"[SAVE] {out_dir / 'ours.csv'}")
    # offload row (any method, rate=0): same since all_methods at rate 0 are identical
    offload_rows = base_df[base_df["rate"] == 0.0]
    if not offload_rows.empty:
        # Average tied rate-0 rows (they should be identical since rate=0 produces same answers)
        first = offload_rows.iloc[0]
        offload = pd.DataFrame([{
            "rate": 0.0, "total_ttft": first["total_ttft"], "avg_score": first["avg_score"],
        }])
        offload.to_csv(out_dir / "offload.csv", index=False)
    # per-method, drop rate=0 entry
    for method, sub in base_df.groupby("method"):
        if method == "prefill":
            df_m = sub
        else:
            df_m = sub[sub["rate"] != 0.0]
        df_m[["rate", "total_ttft", "avg_score"]].to_csv(out_dir / f"{method}.csv", index=False)


def plot_curve(out_dir: Path, fig_path: Path) -> None:
    csv_paths = sorted(out_dir.glob("*.csv"))
    fig, ax = plt.subplots(figsize=(8, 6))
    for p in csv_paths:
        df = pd.read_csv(p)
        if not {"total_ttft", "avg_score"}.issubset(df.columns):
            continue
        df_p = df[["total_ttft", "avg_score"]].dropna().sort_values("total_ttft")
        if df_p.empty:
            continue
        ax.plot(df_p["total_ttft"], df_p["avg_score"], marker="o", label=p.stem)
    ax.set_xlabel("Total TTFT (s)")
    ax.set_ylabel("Average score (F1)")
    ax.set_title(f"Quality vs. TTFT — {MODEL}")
    ax.grid(True)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=9)
        plt.subplots_adjust(top=0.78)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"[SAVE] {fig_path}")


def main() -> None:
    df_prof = load_and_rename(SCORES_PROFILING)
    df_test = load_and_rename(SCORES_TESTING)

    size_score_df = build_size_score_df(df_prof)
    print("size_score_df:", size_score_df.shape)
    print(size_score_df.head())

    cpu_gb, ssd_gb = pick_storage_sizes(size_score_df)

    ttft_params = pd.read_csv(TTFT_CSV)
    ttft_params["device"] = ttft_params["device"].fillna("None").astype(str).str.strip()

    alphas = pick_alphas(size_score_df, ttft_params)

    assignments = run_storage_policy(size_score_df, alphas, cpu_gb, ssd_gb)
    ours_df = evaluate_ours(assignments, df_test, ttft_params)
    print("\n===== ours =====")
    print(ours_df)

    base_df = evaluate_baselines(size_score_df, df_test, ttft_params, cpu_gb, ssd_gb)
    print("\n===== baselines =====")
    print(base_df)

    write_results(ours_df, base_df, RESULTS_DIR)
    plot_curve(RESULTS_DIR, INTER_DIR / "quality_vs_ttft.png")


if __name__ == "__main__":
    main()
