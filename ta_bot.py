"""
CS 6604 / ECE 6256 — Assignment 3: The Course TA-Bot
Conversational RAG with Memory

Do NOT rename the class or change the method signatures.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from pathlib import Path

import requests

SYSTEM_PROMPT = """You are the teaching assistant for Georgia Tech's Conversational AI course (ECE 6256 / CS 6604). Students ask you about the papers on the course reading list.

How to answer:
- Questions come with numbered excerpts from the course papers. Base your answer on those excerpts and cite the ones you use inline by number, like [1] or [2][3].
- Name the paper or method you are describing.
- If the excerpts do not contain the answer, say that the course papers you found do not cover it. Do not make up details.
- The excerpts come from different papers and some may be off-topic. Use only the ones that help answer the question, and keep each claim attributed to the paper it came from.
- Messages without excerpts (greetings, questions about the student) can be answered directly.
- Be accurate and complete, but concise: a few short paragraphs, no filler."""

# Dropped from search queries: function words plus question boilerplate that
# appears in nearly every question ("explain", "paper", ...).
STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before being
    below between both but by can could did do does doing done down during each few for from further
    had has have having he her here hers him his how i if in into is it its itself just let me more
    most my no nor not now of off on once only or other our out over own same she should so some such
    than that the their them then there these they this those through to too under until up very was
    we were what when where which while who whom why will with would you your
    explain describe discuss tell summarize summarise please paper papers approach approaches method
    methods work works use used uses using according mean means
    """.split()
)
_WORD = re.compile(r"\w+")

# Follow-up cues. L13: "Is it refundable?" retrieves nothing until you know what "it" is.
_SINGULAR_REF = re.compile(
    r"\b(it|its|(this|that|the same) (paper|approach|method|model|technique|framework|system|work|one))\b",
    re.IGNORECASE,
)
_PLURAL_REF = re.compile(
    r"\b(they|them|their|these|those|both|all (two|three|four|five|of them|of these)|"
    r"we (discussed|covered|talked about)|discussed|mentioned|so far|earlier|previous(ly)?)\b",
    re.IGNORECASE,
)
_TOPIC_SWITCH = re.compile(
    r"\b(switch(ing)? topics?|change (of )?topics?|new topic|different topic|unrelated)\b", re.IGNORECASE
)
_SMALL_TALK = frozenset(
    "hi hello hey thanks thank thx ok okay great cool nice awesome perfect bye goodbye sure yes yeah no got it "
    "help helpful helped much lot appreciate appreciated good morning afternoon evening later see "
    "makes sense clear understood welcome".split()
)
# Questions about the user, answered from memory rather than the corpus.
_PERSONAL = re.compile(
    r"\b(about me|do you remember|what do you know|my name|who am i)\b", re.IGNORECASE
)
# Inline citation markers the model emits: [1], [2, 3], [1; 4], and runs of them such as
# [1][2] or [1] and [3], which become one [paper_id, ...] (two excerpts can share a paper).
_CITE_GROUP = r"\[\d+(?:\s*[,;]\s*\d+)*\]"
_CITE_MARK = re.compile(rf"[ \t]*{_CITE_GROUP}(?:(?:[ \t]*,[ \t]*|[ \t]+and[ \t]+|[ \t]*){_CITE_GROUP})*")


class TABot:
    """Course TA-Bot with RAG retrieval and cross-session memory."""

    def __init__(self, config: dict) -> None:
        """
        Initialise the TA-Bot.

        Parameters
        ----------
        config : dict
            Required keys set by the harness (do not hard-code their values):
                endpoint   : str   – vLLM-compatible base URL
                model      : str   – model name served at the endpoint
                corpus_path: str   – path to the paper corpus directory
                temperature: float – generation temperature (0 for graded runs)
                seed       : int   – random seed for reproducibility
            You may add custom keys for your own use.
        """
        self.config = config
        self.model = config["model"]
        self.temperature = float(config["temperature"])
        self.seed = int(config["seed"])

        # Custom keys (optional, with defaults).
        self.max_tokens = int(config.get("max_tokens", 512))
        self.history_turns = int(config.get("history_turns", 3))
        # (connect, read) seconds. The read timeout bounds a whole non-streamed
        # completion, since vLLM sends nothing until generation finishes.
        self.timeout = (10.0, float(config.get("timeout", 60.0)))
        # Retrieval knobs.
        self.k_chunks = int(config.get("k_chunks", 4))  # excerpts in the prompt
        self.max_papers = int(config.get("max_papers", 3))  # distinct papers among them
        self.per_paper = int(config.get("per_paper_chunks", 2))
        self.title_weight = float(config.get("title_weight", 1.0))  # BM25 weight of the title column
        self.min_quality = float(config.get("min_quality", 0.0))  # drop table/equation chunks below this
        self.carry_turns = int(config.get("carry_turns", 3))  # how far back "these"/"all three" reach
        self.chunk_chars = int(config.get("chunk_chars", 2000))  # excerpt length cap in the prompt

        # The mock hands us "http://host:port", the README's real example is
        # "http://host:8000/v1" — accept both.
        base = config["endpoint"].rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        self._api = base + "/v1"
        self._http = requests.Session()  # keep-alive: one TCP connection, many turns

        # Corpus: opened read-only so nothing we do can modify it.
        corpus = Path(config["corpus_path"])
        db_path = corpus if corpus.is_file() else corpus / "corpus.db"
        if not db_path.exists():
            raise FileNotFoundError(f"corpus database not found at {db_path}")
        self._db = sqlite3.connect(
            f"{db_path.resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False
        )
        self._papers: dict[str, str] = dict(
            self._db.execute("SELECT paper_id, title FROM papers")
        )

        # session_id -> [{"role": "user"|"assistant", "content": str}, ...]
        self._sessions: dict[str, list[dict]] = {}
        # session_id -> the top paper of each retrieval turn: a one-slot dialogue
        # state that lets "it" / "these" resolve to papers already discussed.
        self._focus: dict[str, list[str]] = {}
        # user_id -> [{"text": str, "session_id": str, "turn": int}, ...]
        # Only "text" is exposed by get_memory(); the rest is provenance.
        self._memory: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------
    # Public interface (called by the harness)
    # ------------------------------------------------------------------

    def chat(
        self,
        session_id: str,
        user_id: str,
        message: str,
    ) -> dict:
        """
        Process one user turn and return an answer with citations.

        Parameters
        ----------
        session_id : str
            Conversation scope — a new session_id starts a fresh
            conversation but keeps the user's cross-session memory.
        user_id : str
            Memory scope — identifies the user whose memory store
            to read from and write to.
        message : str
            The user's message.

        Returns
        -------
        dict
            {"answer": str, "citations": list[str]}
            - answer: the bot's response (complete, no streaming)
            - citations: list of paper_id strings from the corpus
              that were used to produce the answer. May be empty
              if no retrieval was needed.

        Raises
        ------
        TimeoutError
            You MUST raise TimeoutError if the endpoint is
            unreachable or a request exceeds your timeout.
            The harness catches it and marks this turn as 0.
            Do NOT let the process hang — raise, don't wait.
        """
        history = self._sessions.setdefault(session_id, [])
        focus = self._focus.setdefault(session_id, [])

        sources, top_paper = self._retrieve(message, focus)
        messages = self._build_prompt(message, history, sources)
        raw = self._complete(messages)
        answer, cited = self._render_citations(raw, sources)

        # Recorded only after a successful call, so a failed turn never leaves
        # two consecutive user messages in the history.
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
        if top_paper:
            focus.append(top_paper)

        # Papers cited inline first, then the rest of what the model was shown.
        used = cited + [s["paper_id"] for s in sources]
        return {"answer": answer, "citations": self._clean_citations(used)}

    def reset_memory(self, user_id: str) -> None:
        """
        Clear all stored memory for the given user.

        The harness calls this before each memory-probe sequence
        to ensure a clean slate. After this call, ``get_memory(user_id)``
        must return an empty list.

        Parameters
        ----------
        user_id : str
            The user whose memory to clear.
        """
        self._memory.pop(user_id, None)

    def get_memory(self, user_id: str) -> list[str]:
        """
        Return the current memory store for the given user as a
        list of human-readable fact strings.

        This is read-only — calling it must not change the store.
        The harness uses it to inspect what your bot chose to remember
        (and what it chose not to remember).

        Any internal representation is fine as long as each entry
        serialises to a readable string.

        Parameters
        ----------
        user_id : str
            The user whose memory to inspect.

        Returns
        -------
        list[str]
            Stored facts as readable strings. Empty list if nothing
            is stored (or after ``reset_memory``).
        """
        return [record["text"] for record in self._memory.get(user_id, [])]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve(self, message: str, focus: list[str]) -> tuple[list[dict], str | None]:
        """
        Pick the excerpts for this turn.

        Returns (sources in prompt order, top paper of this turn's own search).
        """
        terms = self._terms(message)
        # L13: not every turn needs a lookup ("thanks!" -> no retrieval).
        if not terms or set(terms) <= _SMALL_TALK or _PERSONAL.search(message):
            return [], None

        ranked = self._search(terms, limit=100)
        top_paper = ranked[0]["paper_id"] if ranked else None

        # Papers an anaphor points back to ("it", "these", "all three we discussed").
        carried: list[str] = []
        if focus and not _TOPIC_SWITCH.search(message):
            if _PLURAL_REF.search(message):
                carried = list(dict.fromkeys(reversed(focus[-self.carry_turns :])))[::-1]
            elif _SINGULAR_REF.search(message):
                carried = [focus[-1]]

        carried_chunks = [c for pid in carried if (c := self._best_chunk(terms, pid))]

        # Fill the remaining slots from this turn's own search, MaxP-style:
        # a paper ranks by its best chunk; at most per_paper chunks each.
        budget = max(1, self.k_chunks - len(carried_chunks))
        top_papers = [p for p in dict.fromkeys(c["paper_id"] for c in ranked) if p not in carried]
        top_papers = set(top_papers[: self.max_papers])
        own: list[dict] = []
        per_paper: Counter = Counter()
        # Breadth first: the best chunk of each top paper, then second chunks
        # while slots remain — more distinct papers in the same budget.
        for cap in range(1, self.per_paper + 1):
            for c in ranked:
                if len(own) == budget:
                    break
                pid = c["paper_id"]
                if pid in top_papers and per_paper[pid] < cap and c not in own:
                    own.append(c)
                    per_paper[pid] += 1

        # Relevance order: this turn's best hit, then what the user referred
        # back to, then the rest.
        by_relevance = own[:1] + carried_chunks + own[1:]
        # L13 "lost in the middle": best first, second-best last.
        return by_relevance[0::2] + by_relevance[1::2][::-1], top_paper

    def _terms(self, text: str) -> list[str]:
        words = (w for w in _WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 1)
        return list(dict.fromkeys(words))[:40]

    def _search(self, terms: list[str], limit: int, paper_id: str | None = None) -> list[dict]:
        """BM25 over chunks via FTS5. Every term is quoted, so user text can never be FTS syntax."""
        match = " OR ".join(f'"{t}"' for t in terms)
        sql = f"""
            SELECT c.chunk_id, c.paper_id, c.section, c.text,
                   -bm25(chunks_fts, 0, 0, {self.title_weight}, 0, 1.0) AS score
            FROM chunks_fts JOIN chunks c USING (chunk_id)
            WHERE chunks_fts MATCH ? AND c.quality >= ? {"AND c.paper_id = ?" if paper_id else ""}
            ORDER BY score DESC LIMIT ?"""
        params = [match, self.min_quality] + ([paper_id] if paper_id else []) + [limit]
        cols = ("chunk_id", "paper_id", "section", "text", "score")
        return [dict(zip(cols, row)) for row in self._db.execute(sql, params)]

    def _best_chunk(self, terms: list[str], paper_id: str) -> dict | None:
        """The chunk of a given paper that best matches this turn; its opening chunk if none match."""
        hits = self._search(terms, limit=1, paper_id=paper_id)
        if hits:
            return hits[0]
        row = self._db.execute(
            "SELECT chunk_id, paper_id, section, text FROM chunks WHERE paper_id = ? "
            "ORDER BY chunk_index LIMIT 1",
            (paper_id,),
        ).fetchone()
        return dict(zip(("chunk_id", "paper_id", "section", "text"), row)) if row else None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _build_prompt(self, message: str, history: list[dict], sources: list[dict]) -> list[dict]:
        recent = history[-2 * self.history_turns :] if self.history_turns > 0 else []
        # Past answers are clipped: they are context for "it"/"that", not evidence.
        recent = [
            {**m, "content": m["content"][:600]} if m["role"] == "assistant" else m for m in recent
        ]

        if sources:
            blocks = []
            # Title only: the model copied paper ids and section names into its answers.
            for i, s in enumerate(sources, 1):
                title = self._papers.get(s["paper_id"], s["paper_id"])
                blocks.append(f"[{i}] {title}\n{s['text'][: self.chunk_chars]}")
            user = "Excerpts from the course papers:\n\n" + "\n\n".join(blocks)
            user += f"\n\nQuestion: {message}"
        else:
            user = message

        return [{"role": "system", "content": SYSTEM_PROMPT}, *recent, {"role": "user", "content": user}]

    def _render_citations(self, answer: str, sources: list[dict]) -> tuple[str, list[str]]:
        """Rewrite [n] markers as [paper_id] and collect the papers cited inline."""
        cited: list[str] = []

        def to_ids(m: re.Match) -> str:
            pids = [
                sources[int(n) - 1]["paper_id"]
                for n in re.findall(r"\d+", m.group(0))
                if 1 <= int(n) <= len(sources)
            ]
            cited.extend(pids)
            lead = "" if m.start() == 0 or answer[m.start() - 1] == "\n" else " "
            return lead + "[" + ", ".join(dict.fromkeys(pids)) + "]" if pids else ""

        return _CITE_MARK.sub(to_ids, answer), cited

    # ------------------------------------------------------------------
    # Endpoint
    # ------------------------------------------------------------------

    def _complete(self, messages: list[dict], max_tokens: int | None = None) -> str:
        """One chat completion. Any endpoint failure surfaces as TimeoutError."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": max_tokens or self.max_tokens,
        }
        try:
            resp = self._http.post(
                f"{self._api}/chat/completions", json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except requests.HTTPError as exc:
            raise TimeoutError(
                f"endpoint returned HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        except requests.RequestException as exc:  # timeouts, refused connections, ...
            raise TimeoutError(f"request to endpoint failed: {exc}") from exc
        except (ValueError, KeyError, IndexError, TypeError) as exc:  # malformed reply
            raise TimeoutError(f"malformed response from endpoint: {exc!r}") from exc
        return (content or "").strip()

    def _clean_citations(self, paper_ids: list[str]) -> list[str]:
        """Keep only real paper_ids from the papers table, de-duplicated, in order."""
        return [pid for pid in dict.fromkeys(paper_ids) if pid in self._papers]
