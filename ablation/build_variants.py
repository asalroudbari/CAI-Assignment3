#!/usr/bin/env python3
"""
Rebuild the corpus at several chunk sizes for the R4 ablation.

Each variant lives in ablation/corpora/c<size>/ with its own corpus.db, built by
the course's corpus/build_corpus.py (copied in, because it writes next to
itself). The extracted paper text is shared through a symlink; the original
corpus/ directory is never modified.

Run with a Python that has `tiktoken`, so token counts use cl100k_base exactly
like the shipped corpus (without it the build falls back to words x 1.3 and
the c400 variant will not match corpus/corpus.db).

    python3 ablation/build_variants.py                    # 150 250 400 600 800
    python3 ablation/build_variants.py --sizes 250 400
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parent / "corpus"
OUT = HERE / "corpora"


def build(size: int, overlap: float) -> Path:
    d = OUT / f"c{size}"
    if d.exists():
        if (d / "text").is_symlink():
            (d / "text").unlink()  # never let rmtree near the shared text
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for name in ("build_corpus.py", "manifest.json", "manifest.csv"):
        shutil.copy2(CORPUS / name, d / name)
    try:
        (d / "text").symlink_to(CORPUS / "text", target_is_directory=True)
    except OSError:  # e.g. Windows without symlink rights
        shutil.copytree(CORPUS / "text", d / "text")
    subprocess.run(
        [sys.executable, "build_corpus.py", "chunk", "db",
         "--chunk-tokens", str(size), "--overlap", str(overlap)],
        cwd=d, check=True,
    )
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[150, 250, 400, 600, 800])
    ap.add_argument("--overlap", type=float, default=0.15)
    args = ap.parse_args()
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        print("WARNING: tiktoken not installed; token counts will be estimated (words x 1.3).")
    for size in args.sizes:
        print(f"== building c{size}")
        build(size, args.overlap)


if __name__ == "__main__":
    main()
