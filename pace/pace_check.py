#!/usr/bin/env python3
"""
Run TABot against a real vLLM endpoint and save everything needed to review
the generation side: answers, the prompts the model saw, citations, latency,
token usage, truncation, and whether a second run reproduces the first.

    python3 pace/pace_check.py --endpoint http://localhost:8000/v1 --model <served name>

Writes pace/results/<model>.json and prints a summary. Retrieval and
citations do not depend on the model, so this is about answer behaviour.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval_dev import hit_rate  # noqa: E402
from ta_bot import TABot  # noqa: E402

# Behaviour probes with no gold paper: what the bot should do is in the note.
PROBES = [
    ("hi, thanks for the help!", "small talk: short reply, no citations"),
    ("What is the boiling point of water at sea level?", "off-topic: say it is outside the course papers"),
    ("What does the AlphaFold 3 paper say about diffusion models for protein structure?",
     "not in the corpus: say the course papers do not cover it, no invented details"),
    ("Can you compare RAG and Self-RAG in two sentences?", "brevity on request, both papers cited"),
]
_MARKER = re.compile(r"\[\d+(?:\s*[,;]\s*\d+)*\]")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def make_bot(args) -> tuple[TABot, list[dict]]:
    bot = TABot({"endpoint": args.endpoint, "model": args.model, "corpus_path": str(ROOT / "corpus"),
                 "temperature": 0.0, "seed": args.seed})
    calls: list[dict] = []
    post = bot._http.post

    def recording_post(url, **kw):  # keep the raw request/response of every LLM call
        t0 = time.perf_counter()
        resp = post(url, **kw)
        call = {"latency_s": round(time.perf_counter() - t0, 2), "status": resp.status_code,
                "messages": kw["json"]["messages"]}
        try:
            body = resp.json()
            call["usage"] = body.get("usage")
            call["finish_reason"] = body["choices"][0].get("finish_reason")
            call["raw"] = body["choices"][0]["message"]["content"]
        except Exception as exc:  # recorded, then TABot raises its own TimeoutError
            call["error"] = repr(exc)
        calls.append(call)
        return resp

    bot._http.post = recording_post
    return bot, calls


def turn(bot, calls, session, user, message, gold=None, note=None) -> dict:
    n_before = len(calls)
    t0 = time.perf_counter()
    try:
        out = bot.chat(session, user, message)
        rec = {"answer": out["answer"], "citations": out["citations"]}
    except TimeoutError as exc:
        rec = {"error": f"TimeoutError: {exc}", "answer": "", "citations": []}
    rec = {"message": message, **rec, "seconds": round(time.perf_counter() - t0, 2)}
    if gold is not None:
        rec["gold"], rec["hit"] = gold, hit_rate(rec["citations"], gold)
    if note:
        rec["expected"] = note
    rec["llm_calls"] = calls[n_before:]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-ablation", type=int, default=20, help="sampled ablation questions (half named)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    single = load_jsonl(ROOT / "dev_eval" / "single_turn.jsonl")
    dialogues = load_jsonl(ROOT / "dev_eval" / "multi_turn.jsonl")
    abl = [q for p in sorted((ROOT / "ablation" / "questions").glob("part*.jsonl")) for q in load_jsonl(p)]
    rng = random.Random(0)
    half = args.n_ablation // 2
    sample = (rng.sample([q for q in abl if q["style"] == "named"], half)
              + rng.sample([q for q in abl if q["style"] == "described"], args.n_ablation - half))

    bot, calls = make_bot(args)
    results: dict[str, list] = {"single_turn": [], "multi_turn": [], "ablation_sample": [], "probes": []}
    print(f"single-turn ({len(single)})", flush=True)
    for q in single:
        results["single_turn"].append(turn(bot, calls, f"st-{q['id']}", "u-st", q["question"], q["gold_paper_ids"]))
    print(f"multi-turn ({len(dialogues)} dialogues)", flush=True)
    for d in dialogues:
        results["multi_turn"].append({"id": d["id"], "turns": [
            turn(bot, calls, f"mt-{d['id']}", "u-mt", t["content"], t.get("gold_paper_ids", []))
            for t in d["turns"] if t["role"] == "user"]})
    print(f"ablation sample ({len(sample)})", flush=True)
    for q in sample:
        rec = turn(bot, calls, f"ab-{q['id']}", "u-ab", q["question"], q["gold_paper_ids"])
        results["ablation_sample"].append({"id": q["id"], "style": q["style"], **rec})
    print(f"probes ({len(PROBES)})", flush=True)
    for i, (message, note) in enumerate(PROBES):
        results["probes"].append(turn(bot, calls, f"probe-{i}", "u-probe", message, note=note))

    # Reproducibility: a fresh bot, same questions, temperature 0 and the same seed.
    print("re-run single-turn on a fresh bot", flush=True)
    bot2, calls2 = make_bot(args)
    rerun = [turn(bot2, calls2, f"st-{q['id']}", "u-st", q["question"]) for q in single]
    same = [a["answer"] == b["answer"] for a, b in zip(results["single_turn"], rerun)]
    results["rerun_single_turn"] = [{"message": r["message"], "answer": r["answer"], "identical": s}
                                    for r, s in zip(rerun, same)]

    turns = (results["single_turn"] + [t for d in results["multi_turn"] for t in d["turns"]]
             + results["ablation_sample"] + results["probes"])
    llm = [c for t in turns for c in t["llm_calls"]]
    prompt_toks = [c["usage"]["prompt_tokens"] for c in llm if c.get("usage")]
    answer_toks = [c["usage"]["completion_tokens"] for c in llm if c.get("usage")]
    with_sources = [t for t in turns if t["llm_calls"] and "Excerpts from the course papers" in
                    t["llm_calls"][-1]["messages"][-1]["content"]]

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    summary = {
        "hit_rate_single_turn": mean([t["hit"] for t in results["single_turn"]]),
        "hit_rate_multi_turn_last": mean([d["turns"][-1]["hit"] for d in results["multi_turn"]]),
        "hit_rate_multi_turn_all": mean([t["hit"] for d in results["multi_turn"] for t in d["turns"]]),
        "hit_rate_ablation_sample": mean([t["hit"] for t in results["ablation_sample"]]),
        "errors": sum(1 for t in turns if "error" in t),
        "latency_mean_s": round(mean([t["seconds"] for t in turns]), 2),
        "latency_max_s": max(t["seconds"] for t in turns),
        "prompt_tokens_max": max(prompt_toks, default=None),
        "answer_tokens_mean": round(mean(answer_toks), 1),
        "truncated_at_max_tokens": sum(1 for c in llm if c.get("finish_reason") == "length"),
        "answers_with_markers": f"{sum(1 for t in with_sources if _MARKER.search(t['llm_calls'][-1].get('raw') or ''))}"
                                f"/{len(with_sources)}",
        "rerun_identical": f"{sum(same)}/{len(same)}",
    }
    meta = {
        "model": args.model, "endpoint": args.endpoint, "seed": args.seed,
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(), "host": platform.node(),
        "ta_bot_sha256": hashlib.sha256((ROOT / "ta_bot.py").read_bytes()).hexdigest()[:12],
    }
    out = Path(args.out) if args.out else ROOT / "pace" / "results" / f"{re.sub(r'[^A-Za-z0-9.-]+', '_', args.model)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "summary": summary, **results}, indent=1, ensure_ascii=False))

    print("\nSUMMARY")
    for key, value in summary.items():
        print(f"  {key:28} {value:.3f}" if isinstance(value, float) else f"  {key:28} {value}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
