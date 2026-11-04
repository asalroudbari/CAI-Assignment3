#!/usr/bin/env python3
"""
Does an LLM judge notice what is wrong with an answer? A check of our own judge, for the
README's "judge caveat" bonus.

The course judge's prompt is not public (and we do not try to reconstruct it). This script uses
a judge of our own: Llama-3.1-8B-Instruct on vLLM (one of the models the course endpoint serves;
the README does not name the judge model), scoring 1-5 on the four criteria the README names (accuracy, relevance, completeness,
grounding). It sees the question, the excerpts our bot was shown, and the answer.

Items: the final PACE run's single-turn answers (10 dev, 20 ablation sample, 4 behaviour probes),
from pace/results/run2-memory-router/Llama-3.1-8B-Instruct.json. Each is judged as written and
in controlled variants whose correct score is known in advance:

    numbers    every quantity in the answer multiplied by 10, so it contradicts the excerpts
               (years, course numbers and list markers are left alone; only answers that
               contain a quantity). A good judge scores it lower.
    uncited    the [paper_id] markers removed. Grounding should drop.
    padded     a generic closing paragraph added that says nothing. Should not score higher.
    swapped    the answer to a different question. Should score 1.

    python3 pace/judge_check.py --endpoint http://localhost:8000/v1 --model <served name>

Writes pace/results/judge/judge_check-<model>.json and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ta_bot import TABot  # noqa: E402

RUN = ROOT / "pace/results/run2-memory-router/Llama-3.1-8B-Instruct.json"
CRITERIA = ("accuracy", "relevance", "completeness", "grounding")

JUDGE_PROMPT = """You grade answers from a course TA chatbot. The chatbot answers students' questions about the course's research papers, using excerpts retrieved from those papers. You get the question, the excerpts the chatbot was given, and its answer. The answer cites papers by id, like [L13-06]; each excerpt header shows its paper id.

Score the answer from 1 (very poor) to 5 (excellent) on each criterion:
- accuracy: the claims are correct and agree with the excerpts.
- relevance: it answers the question that was asked.
- completeness: it covers what the question asks for.
- grounding: its claims are supported by the excerpts and attributed to the right paper; invented details lower this.
If the question is not covered by the course papers, the best answer says so briefly. Small talk deserves a short friendly reply.

Reply with one JSON object and nothing else:
{"reason": "<one or two sentences>", "accuracy": n, "relevance": n, "completeness": n, "grounding": n, "overall": n}"""

PAD = ("\n\nOverall, this is a rich and important line of research. The ideas described above have "
       "been very influential, and they continue to shape how researchers and practitioners think "
       "about building capable and reliable language technologies.")
_CITE = re.compile(r"\[L\d\d-\d\d(?:\s*,\s*L\d\d-\d\d)*\]")
# A number, possibly with an "x"/"%" suffix; not part of a name such as GPT-3, T5X or Llama-2.
_NUM = re.compile(r"(?<![\w.])(?<![A-WYZa-wyz]-)(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![A-WYZa-wyz0-9_])(?!\.\d)")


def times_ten(answer: str) -> str:
    """Multiply every number outside citation markers by 10; years are left alone."""
    parts, last = [], 0
    for m in _CITE.finditer(answer):
        parts.append(("text", answer[last:m.start()]))
        parts.append(("cite", m.group(0)))
        last = m.end()
    parts.append(("text", answer[last:]))

    def scale(m: re.Match) -> str:
        s, after = m.group(1), m.string[m.end():m.end() + 2]
        if re.fullmatch(r"(19|20)\d\d", s) or s in ("6256", "6604"):  # years, course numbers
            return s
        if s.isdigit() and int(s) <= 20 and re.match(r"[.)](\s|$)", after + " "):  # list markers "1." "2)"
            return s
        before = m.string[max(0, m.start() - 30):m.start()]
        if s.isdigit() and int(s) <= 20 and re.search(r"\b[A-Z][\w-]* $", before):  # "AlphaFold 3", "Table 2"
            return s
        if "," in s:
            return f"{int(s.replace(',', '')) * 10:,}"
        if "." in s:
            return f"{float(s) * 10:g}"
        return str(int(s) * 10)

    return "".join(_NUM.sub(scale, t) if kind == "text" else t for kind, t in parts)


def judge(endpoint: str, model: str, question: str, excerpts: str, answer: str) -> dict | None:
    user = f"Question: {question}\n\nExcerpts given to the chatbot:\n{excerpts}\n\nChatbot answer:\n{answer}"
    r = requests.post(f"{endpoint.rstrip('/')}/chat/completions", timeout=120, json={
        "model": model, "temperature": 0.0, "seed": 42, "max_tokens": 300,
        "messages": [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user}]})
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]
    start = raw.find("{")
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw[start:]) if start >= 0 else (None, 0)
    except ValueError:
        obj = None
    if not isinstance(obj, dict) or not all(isinstance(obj.get(c), (int, float)) for c in (*CRITERIA, "overall")):
        return {"raw": raw, "parsed": False}
    return {**{c: obj[c] for c in (*CRITERIA, "overall")}, "reason": obj.get("reason", ""), "parsed": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    run = json.loads(RUN.read_text())
    items = ([{"set": "dev", **x} for x in run["single_turn"]]
             + [{"set": "ablation", **x} for x in run["ablation_sample"]]
             + [{"set": "probe", **x} for x in run["probes"]])

    # The excerpts the bot saw: first-turn retrieval is deterministic and uses the question alone.
    bot = TABot({"endpoint": "http://unused", "model": "unused", "corpus_path": str(ROOT / "corpus"),
                 "temperature": 0.0, "seed": 42})
    for it in items:
        sources = bot._retrieve(it["message"], [], "")[0] if it["citations"] else []
        it["excerpt_papers_match"] = {s["paper_id"] for s in sources} == set(it["citations"])
        it["excerpts"] = "\n\n".join(
            f"[{i}] ({s['paper_id']}) {bot._papers.get(s['paper_id'], '')}\n{s['text'][:bot.chunk_chars]}"
            for i, s in enumerate(sources, 1)) or "(none: the chatbot did not retrieve for this message)"

    results = []
    for i, it in enumerate(items):
        variants = {"original": it["answer"], "uncited": _CITE.sub("", it["answer"]),
                    "padded": it["answer"] + PAD,
                    "swapped": items[(i + 7) % len(items)]["answer"]}
        if times_ten(it["answer"]) != it["answer"]:
            variants["numbers"] = times_ten(it["answer"])
        scores = {v: judge(args.endpoint, args.model, it["message"], it["excerpts"], a)
                  for v, a in variants.items()}
        scores["original_repeat"] = judge(args.endpoint, args.model, it["message"], it["excerpts"], it["answer"])
        results.append({"set": it["set"], "id": it.get("id"), "question": it["message"],
                        "citations": it["citations"], "excerpt_papers_match": it["excerpt_papers_match"],
                        "variants": variants, "scores": scores})
        print(f"  {i + 1}/{len(items)}", flush=True)

    out = ROOT / "pace/results/judge" / f"judge_check-{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "judge_prompt": JUDGE_PROMPT, "items": results},
                              indent=2, ensure_ascii=False))

    ok = lambda r, v: r["scores"].get(v) and r["scores"][v].get("parsed")  # noqa: E731
    print(f"\nexcerpts recomputed to the same papers: {sum(r['excerpt_papers_match'] for r in results)}/{len(results)}")
    print(f"parsed judge replies: {sum(bool(ok(r, v)) for r in results for v in r['scores'])}"
          f"/{sum(len(r['scores']) for r in results)}")
    print("\nmean overall (and grounding) per variant:")
    for v in ("original", "original_repeat", "numbers", "uncited", "padded", "swapped"):
        rs = [r for r in results if ok(r, v)]
        if rs:
            ov = sum(r["scores"][v]["overall"] for r in rs) / len(rs)
            gr = sum(r["scores"][v]["grounding"] for r in rs) / len(rs)
            print(f"  {v:16} n={len(rs):2}  overall {ov:.2f}  grounding {gr:.2f}")
    print("\npaired with the original (overall): lower / same / higher")
    for v in ("original_repeat", "numbers", "uncited", "padded", "swapped"):
        rs = [r for r in results if ok(r, v) and ok(r, "original")]
        d = [r["scores"][v]["overall"] - r["scores"]["original"]["overall"] for r in rs]
        print(f"  {v:16} {sum(x < 0 for x in d):2} / {sum(x == 0 for x in d):2} / {sum(x > 0 for x in d):2}")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
