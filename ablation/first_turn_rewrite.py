#!/usr/bin/env python3
"""
Would first-turn retrieval improve if it also used the planning call's rewritten query?

The deployed bot uses the rewrite only on follow-ups (and when stored facts exist); a first
question is searched on its own words. This script measures the alternative on the ablation's
216 questions plus the 10 dev questions, on the shipped corpus (c400), at k = 1..5:

    raw      the question's own terms (deployed)
    add      the question's terms plus the rewrite's (what follow-ups use)
    replace  the rewrite's terms alone (for reference)

The rewrite needs the real model; retrieval does not. So the planning call runs once per
question per pass against the endpoint, and every variant is then scored offline. Two passes
measure how stable the rewrites are run to run (vLLM's prefix cache is cold in the first pass
and warm in the second).

    python3 ablation/first_turn_rewrite.py --endpoint http://localhost:8000/v1 --model <served name>
    python3 ablation/first_turn_rewrite.py --cached        # re-score saved rewrites, no endpoint

Writes ablation/results/first_turn_rewrites.json (the rewrites) and first_turn_rewrite.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from run_ablation import cluster_ci, load_questions, mcnemar_exact  # noqa: E402
from ta_bot import TABot  # noqa: E402

RESULTS = HERE / "results"
CACHE = RESULTS / "first_turn_rewrites.json"
VARIANTS = ("raw", "add", "replace")


def config(endpoint: str, model: str, **extra) -> dict:
    return {"endpoint": endpoint, "model": model, "corpus_path": str(ROOT / "corpus"),
            "temperature": 0.0, "seed": 42, **extra}


def get_rewrites(questions: list[dict], endpoint: str, model: str, passes: int) -> list[dict]:
    out = []
    for p in range(passes):
        bot = TABot(config(endpoint, model))
        rows = {}
        for i, q in enumerate(questions):
            plan = bot._route(q["question"], [], [])
            rows[q["id"]] = {"search_query": plan["search_query"] if plan else None,
                             "needs_retrieval": plan["needs_retrieval"] if plan else None}
            if (i + 1) % 50 == 0:
                print(f"  pass {p + 1}: {i + 1}/{len(questions)}", flush=True)
        out.append(rows)
    return out


def papers(sources: list[dict]) -> set[str]:
    return {s["paper_id"] for s in sources}


def score(questions: list[dict], rewrites: dict, k: int) -> list[dict]:
    bot = TABot(config("http://unused", "unused", max_papers=k, k_chunks=max(4, k)))  # retrieval only
    rows = []
    for q in questions:
        text, gold = q["question"], q["gold_paper_ids"][0]
        rw = rewrites[q["id"]]["search_query"] or ""
        got = {
            "raw": papers(bot._retrieve(text, [], "")[0]),
            "add": papers(bot._retrieve(text, [], rw)[0]),
            "replace": papers(bot._retrieve(rw, [], "")[0]) if rw else papers(bot._retrieve(text, [], "")[0]),
        }
        rows.append({"id": q["id"], "style": q["style"], "gold": gold, "k": k,
                     **{v: int(gold in got[v]) for v in VARIANTS}})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint")
    ap.add_argument("--model")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--cached", action="store_true", help="re-score saved rewrites")
    args = ap.parse_args()

    files = sorted((HERE / "questions").glob("part*.jsonl"))
    questions = load_questions(files)
    if args.cached:
        passes = json.loads(CACHE.read_text())["passes"]
    else:
        passes = get_rewrites(questions, args.endpoint, args.model, args.passes)
        RESULTS.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps({"model": args.model, "passes": passes}, indent=1, ensure_ascii=False))

    usable = sum(r["search_query"] is not None for r in passes[0].values())
    print(f"{len(questions)} questions; usable plans in pass 1: {usable}")
    if len(passes) > 1:
        same = sum(passes[0][i]["search_query"] == passes[1][i]["search_query"] for i in passes[0])
        print(f"rewrites identical across passes 1 and 2: {same}/{len(questions)}")

    out_rows = []
    for p, rewrites in enumerate(passes, 1):
        print(f"\n== pass {p}: hit-rate, and paired change vs raw (gained/lost, exact McNemar p)")
        for k in range(1, 6):
            rows = score(questions, rewrites, k)
            for style in ("named", "described", "dev", "all"):
                sub = [r for r in rows if style == "all" or r["style"] == style]
                line = [f"k={k} {style:>9} n={len(sub):3}"]
                for v in VARIANTS:
                    rate = sum(r[v] for r in sub) / len(sub)
                    lo, hi = cluster_ci(sub, v)
                    gained = sum(r[v] and not r["raw"] for r in sub)
                    lost = sum(r["raw"] and not r[v] for r in sub)
                    p_val = mcnemar_exact(gained, lost)
                    out_rows.append({"pass": p, "k": k, "style": style, "variant": v, "n": len(sub),
                                     "hit_rate": round(rate, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                                     "gained_vs_raw": gained, "lost_vs_raw": lost, "p_vs_raw": round(p_val, 4)})
                    line.append(f"{v} {rate:.3f}" + ("" if v == "raw" else f" (+{gained}/-{lost}, p={p_val:.3f})"))
                print("  " + " | ".join(line))

    with (RESULTS / "first_turn_rewrite.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {RESULTS / 'first_turn_rewrite.csv'}")


if __name__ == "__main__":
    main()
