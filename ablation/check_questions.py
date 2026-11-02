#!/usr/bin/env python3
"""
Sanity checks for the ablation question set (ablation/questions/part*.jsonl).

Uses only titles and abstracts — never the retriever — so it cannot bias the
experiment toward any retrieval setting.

  1. structure: two questions per paper (named + described), valid gold ids
  2. name leak: a "described" question that contains the paper's name, an
     acronym from its title, or a word unique to its title among all 108
  3. copying: questions sharing a 4-word run with the paper's abstract
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB = HERE.parent / "corpus" / "corpus.db"
WORD = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
STOP = set("a an the of to in on for and or is are was were be by with as at from that this it its how does do "
           "what which why who when can could would should into their they them than more most each".split())


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def name_terms(title: str, title_df: Counter) -> set[str]:
    terms = set()
    head = title.split(":")[0] if ":" in title else ""
    if head and len(head.split()) <= 3:  # "MemGPT: Towards ..." -> "memgpt"
        terms.add(head.strip().lower())
    for tok in re.findall(r"[A-Za-z0-9][\w-]*", title):
        if sum(ch.isupper() for ch in tok) >= 2 or re.search(r"[A-Za-z]\d|\d[A-Za-z]", tok):
            terms.add(tok.lower())  # BERT, GPT-3, LoRA, T5
    terms |= {w for w in words(title) if title_df[w] == 1 and len(w) >= 5 and w not in STOP}
    return terms


def main() -> None:
    con = sqlite3.connect(f"{DB.resolve().as_uri()}?mode=ro", uri=True)
    papers = {pid: (title, abstract or "") for pid, title, abstract in con.execute("SELECT paper_id, title, abstract FROM papers")}
    title_df = Counter(w for title, _ in papers.values() for w in set(words(title)))

    qs = [json.loads(line) for p in sorted((HERE / "questions").glob("part*.jsonl")) for line in p.open() if line.strip()]
    by_paper = Counter((q["gold_paper_ids"][0], q["style"]) for q in qs)
    bad_gold = [q["id"] for q in qs if q["gold_paper_ids"][0] not in papers]
    missing = [pid for pid in papers for s in ("named", "described") if by_paper[(pid, s)] != 1]
    print(f"{len(qs)} questions; bad gold ids: {bad_gold or 'none'}; papers missing a style: {missing or 'none'}")

    leaks, copies = [], []
    for q in qs:
        pid = q["gold_paper_ids"][0]
        title, abstract = papers.get(pid, ("", ""))
        text = " " + " ".join(words(q["question"])) + " "
        if q["style"] == "described":
            hit = sorted(t for t in name_terms(title, title_df) if f" {' '.join(words(t))} " in text)
            if hit:
                leaks.append((q["id"], hit, q["question"]))
        qw, aw = words(q["question"]), words(abstract)
        grams = {tuple(aw[i:i + 4]) for i in range(len(aw) - 3)}
        shared = [" ".join(g) for g in (tuple(qw[i:i + 4]) for i in range(len(qw) - 3))
                  if g in grams and not set(g) <= STOP]
        if shared:
            copies.append((q["id"], shared, q["question"]))

    print(f"\nname leaks in described questions: {len(leaks)}")
    for qid, hit, text in leaks:
        print(f"  {qid}: {hit}\n      {text}")
    print(f"\nquestions sharing a 4-word run with their abstract: {len(copies)}")
    for qid, shared, text in copies:
        print(f"  {qid}: {shared[:3]}")
    sys.exit(1 if bad_gold or missing else 0)


if __name__ == "__main__":
    main()
