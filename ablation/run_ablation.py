#!/usr/bin/env python3
"""
R4 ablation: chunk size, evaluated across citation budgets k.

For every corpus variant in ablation/corpora/c<size>/ (see build_variants.py)
and every k, each question is sent through TABot.chat() in a fresh session and
scored with eval_dev.hit_rate — the same paper-level metric the harness uses.
k is the bot's `max_papers` setting (how many distinct papers it may cite).

Retrieval is BM25 and fully deterministic, so the mock endpoint gives the real
retrieval numbers. Pass --endpoint/--model to run against vLLM instead.

    python3 ablation/run_ablation.py                       # all sizes, k=1..5, mock
    python3 ablation/run_ablation.py --sizes 400 --ks 3 --endpoint http://localhost:8000/v1 --model ...

Writes ablation/results/per_question.csv and ablation/results/summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from eval_dev import hit_rate  # noqa: E402  (the harness's own metric)
from ta_bot import TABot  # noqa: E402

BASELINE_SIZE = 400  # the shipped corpus
N_BOOT = 2000


def load_questions(files: list[Path]) -> list[dict]:
    qs = []
    for path in files:
        qs += [json.loads(line) for line in path.open() if line.strip()]
    for item in map(json.loads, (ROOT / "dev_eval" / "single_turn.jsonl").open()):
        qs.append({**item, "style": "dev"})
    return qs


def cluster_ci(rows: list[dict], key: str, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap CI of mean(row[key]), resampling whole papers."""
    by_paper = defaultdict(list)
    for r in rows:
        by_paper[r["gold"]].append(r[key])
    papers = sorted(by_paper)
    rng = random.Random(seed)
    stats = []
    for _ in range(N_BOOT):
        vals = [v for p in (rng.choice(papers) for _ in papers) for v in by_paper[p]]
        stats.append(sum(vals) / len(vals))
    stats.sort()
    return stats[int(0.025 * N_BOOT)], stats[int(0.975 * N_BOOT) - 1]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[150, 250, 400, 600, 800])
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--questions", type=Path, nargs="*", default=sorted((HERE / "questions").glob("part*.jsonl")))
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--model", default="mock-model")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()

    server = None
    if args.endpoint:
        endpoint = args.endpoint
    else:
        from mock_endpoint import start_mock_endpoint, stop_mock_endpoint

        endpoint, server = start_mock_endpoint()

    questions = load_questions(args.questions)
    print(f"{len(questions)} questions; sizes={args.sizes}; ks={args.ks}; endpoint={endpoint}")

    per_q: list[dict] = []
    for size in args.sizes:
        corpus = HERE / "corpora" / f"c{size}"
        if not (corpus / "corpus.db").exists():
            sys.exit(f"missing {corpus}/corpus.db — run ablation/build_variants.py first")
        for k in args.ks:
            bot = TABot({
                "endpoint": endpoint, "model": args.model, "corpus_path": str(corpus),
                "temperature": 0.0, "seed": 42,
                "max_papers": k, "k_chunks": max(4, k),  # deployed k_chunks=4; never let it bind k
            })
            t0 = time.time()
            for q in questions:
                cited = bot.chat(f"abl-c{size}-k{k}-{q['id']}", "ablation_user", q["question"])["citations"]
                per_q.append({
                    "id": q["id"], "style": q["style"], "gold": q["gold_paper_ids"][0], "size": size, "k": k,
                    "hit": hit_rate(cited, q["gold_paper_ids"]), "cited": " ".join(cited),
                })
            print(f"  c{size} k={k}: {time.time() - t0:.1f}s")

    if server:
        stop_mock_endpoint(server)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "per_question.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_q[0]))
        w.writeheader()
        w.writerows(per_q)

    # Summary: hit-rate with paper-clustered 95% CI, per (style group, size, k).
    groups = {"named": {"named"}, "described": {"described"}, "all_new": {"named", "described"}, "dev": {"dev"}}
    summary = []
    for g, styles in groups.items():
        for size in args.sizes:
            for k in args.ks:
                rows = [r for r in per_q if r["style"] in styles and r["size"] == size and r["k"] == k]
                if not rows:
                    continue
                lo, hi = cluster_ci(rows, "hit")
                summary.append({"group": g, "size": size, "k": k, "n": len(rows),
                                "hit_rate": sum(r["hit"] for r in rows) / len(rows), "ci_lo": lo, "ci_hi": hi})
    with (args.out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    for g in groups:
        rows = [s for s in summary if s["group"] == g]
        if not rows:
            continue
        print(f"\n{g} (n={rows[0]['n']}) — hit-rate [95% CI]")
        print("  size " + "".join(f"{'k=' + str(k):>21}" for k in args.ks))
        for size in args.sizes:
            cells = {s["k"]: s for s in rows if s["size"] == size}
            print(f"  c{size:<4}" + "".join(
                f"{cells[k]['hit_rate']:>8.3f} [{cells[k]['ci_lo']:.2f},{cells[k]['ci_hi']:.2f}]" for k in args.ks))

    # Paired comparison against the shipped chunking, on the new questions.
    if BASELINE_SIZE in args.sizes and any(r["style"] != "dev" for r in per_q):
        print(f"\nPaired vs c{BASELINE_SIZE} (named+described): Δ hit-rate [95% CI], McNemar exact p")
        index = {(r["id"], r["size"], r["k"]): r for r in per_q if r["style"] != "dev"}
        ids = sorted({r["id"] for r in index.values()})
        for k in args.ks:
            for size in args.sizes:
                if size == BASELINE_SIZE:
                    continue
                pairs = [(index[(i, BASELINE_SIZE, k)], index[(i, size, k)]) for i in ids]
                diffs = [{"gold": b["gold"], "d": c["hit"] - b["hit"]} for b, c in pairs]
                lo, hi = cluster_ci(diffs, "d")
                worse = sum(1 for b, c in pairs if b["hit"] > c["hit"])
                better = sum(1 for b, c in pairs if b["hit"] < c["hit"])
                mean = sum(d["d"] for d in diffs) / len(diffs)
                print(f"  k={k} c{size:<4} Δ={mean:+.3f} [{lo:+.3f},{hi:+.3f}]  "
                      f"better={better:3d} worse={worse:3d}  p={mcnemar_exact(better, worse):.3f}")


if __name__ == "__main__":
    main()
