# R4 ablation: chunk size, evaluated across citation budgets

**Axis:** chunk size (150 / 250 / 400 / 600 / 800 tokens, 15% overlap). **Measured at:** citation
budget k = 1…5 (k = `max_papers`, the number of distinct papers the bot may cite). The deployed
system is **c400, k = 3**. The c400 rebuild matches the shipped `corpus/corpus.db` chunk for chunk.

## Reproduce

```bash
pip install tiktoken matplotlib             # tiktoken: same cl100k token counts as the shipped corpus
python3 ablation/build_variants.py          # ~15 s -> ablation/corpora/c<size>/ (git-ignored, ~170 MB)
python3 ablation/check_questions.py         # structure + leakage checks on the question set
python3 ablation/run_ablation.py            # ~70 s, mock endpoint -> results/summary.csv, per_question.csv
python3 ablation/plot_ablation.py           # -> results/ablation.png / .pdf
python3 ablation/mechanism.py               # term co-occurrence check (below)
```

Retrieval is BM25 and deterministic, so the mock endpoint gives exact retrieval numbers. Every
question goes through `TABot.chat()` in a fresh session and is scored with `eval_dev.hit_rate`,
the harness's own metric. To run against vLLM, add `--endpoint … --model …`.

## Data

| Set | n | What it is |
|---|---|---|
| named | 108 | One per paper. Names the system or method, like the dev set ("How does RoPE…") |
| described | 108 | One per paper. Describes the idea without its name, acronym or distinctive title words |
| dev | 10 | `dev_eval/single_turn.jsonl`, reported for reference only |

The questions were written by Claude (an AI assistant) from each paper's title and abstract. The
writers did not test any question against the retriever. `check_questions.py` results:

- **Name leaks:** 2 described questions flagged, both false positives ("LLM"; the capitalised
  "NOT" in "How NOT To Evaluate…").
- **4-word runs shared with the abstract:** 45 of 108 named questions (they use the paper's own
  terminology) and 3 of 108 described questions (two numbers and a quoted phrase).

**Statistics:**
- **CIs:** 95% percentile bootstrap, 2,000 resamples. Papers are resampled rather than questions,
  because a paper's two questions are not independent.
- **Paired tests vs c400:** exact McNemar test on the same questions.

## Results (hit-rate)

| | k=1 | k=2 | k=3 (shipped) | k=4 | k=5 |
|---|---|---|---|---|---|
| **named**, all sizes | 0.991–1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **described** c150 | 0.759 | 0.861 | 0.935 | 0.963 | 0.991 |
| described c250 | 0.833 | 0.926 | 0.944 | 0.963 | 0.991 |
| described **c400** | 0.852 | 0.917 | **0.963** | 1.000 | 1.000 |
| described c600 | 0.843 | 0.944 | 0.963 | 0.972 | 0.981 |
| described c800 | 0.880 | 0.954 | 0.963 | 0.972 | 0.981 |

CIs are in `results/summary.csv`; the plot is `results/ablation.png`.

**Paired vs c400** (named + described, n = 216):
- **At k = 3, no size differs from c400.** Δ ranges from −1.4 to 0.0 points, and every p ≥ 0.375.
- **At k = 1, c150 is the only nominally significant difference.** Δ = −5.1 points, 95% CI
  [−8.8, −0.9], 15 questions worse vs 4 better, p = 0.019. It does not survive a Bonferroni
  correction over the 20 comparisons (threshold 0.0025).

**Mechanism** (`mechanism.py`): the share of a question's search terms that co-occur in the gold
paper's best chunk.

| | c150 | c250 | c400 | c600 | c800 |
|---|---|---|---|---|---|
| described | 46% | 52% | 56% | 58% | 61% |
| named | 75% | 81% | 82% | 85% | 85% |

## Interpretation

1. **For questions that name their paper, chunk size doesn't matter.** Hit-rate is at ceiling for
   every size, at every k ≥ 2. The dev set, and probably the hidden test, is written in this style.
   So retrieval is unlikely to be what separates systems on hit-rate; answer quality is.
2. **For paraphrased questions, small chunks hurt when the budget is tight.** At k = 1, described
   hit-rate climbs from 0.76 (c150) to 0.88 (c800). The mechanism table explains why: a paraphrase
   shares fewer words with the paper, and a 150-token chunk holds fewer of them at once, so a
   distractor that concentrates a few terms can outrank the gold paper. The effect is consistent in
   direction, but only c150 vs c400 is nominally significant, and not after correction.
3. **The citation budget matters more than chunk size.** At c400, going from k = 1 to k = 3 adds
   11.1 points on described questions. At k = 3, the whole range across chunk sizes is 2.8 points,
   and none of it is significant.
4. **Decision: keep c400 and k = 3.** No size beats the shipped chunking at the deployed budget.
   c150 is the only size that looks worse.

## Limitations

- **Retrieval only.** Answer quality is not measured. Larger chunks cost prompt length: four
  excerpts are about 1.4k tokens at c400 and 2.8k at c800, and `chunk_chars = 2000` truncates
  c600/c800 excerpts. An answer-quality comparison on PACE should raise `chunk_chars`.
- **Hit-rate only counts recall.** The cost side of a larger k (noisier context, "lost in the
  middle") does not appear in this metric.
- **AI-written questions, from abstracts.** Every chunking keeps the abstract text, which may favour
  some sizes. One BM25 retriever was tested; dense retrieval could change the curve.
