#!/usr/bin/env python3
"""
Mechanism check for the chunk-size result: how many of a question's search
terms co-occur in the gold paper's best-matching chunk, per chunk size.

If paraphrased ("described") questions lose recall with small chunks because
their terms are spread across the paper, the count should fall as chunks
shrink, while it stays high for "named" questions.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ta_bot import TABot  # noqa: E402

SIZES = [150, 250, 400, 600, 800]


def main() -> None:
    qs = [json.loads(line) for p in sorted((HERE / "questions").glob("part*.jsonl")) for line in p.open() if line.strip()]
    print("mean # of the question's terms present in the gold paper's best chunk (and as % of its terms)")
    print("  size   " + "".join(f"{s:>18}" for s in ("named", "described")))
    for size in SIZES:
        bot = TABot({"endpoint": "http://unused", "model": "-", "corpus_path": str(HERE / "corpora" / f"c{size}"),
                     "temperature": 0.0, "seed": 42})
        cells = []
        for style in ("named", "described"):
            counts, fracs = [], []
            for q in (q for q in qs if q["style"] == style):
                terms = bot._terms(q["question"])
                best = bot._search(terms, limit=1, paper_id=q["gold_paper_ids"][0])
                n = 0
                if best:
                    for t in terms:  # same porter-stemmed FTS matching the retriever uses
                        n += bot._db.execute(
                            "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH ? AND chunk_id = ?",
                            (f'"{t}"', best[0]["chunk_id"]),
                        ).fetchone()[0] > 0
                counts.append(n)
                fracs.append(n / len(terms) if terms else 0)
            cells.append(f"{statistics.mean(counts):6.2f} ({statistics.mean(fracs):4.0%})")
        print(f"  c{size:<5}" + "".join(f"{c:>18}" for c in cells))


if __name__ == "__main__":
    main()
