"""
Offline tests for TABot's planning call, memory and interface contract.

The mock endpoint returns the same canned text for every request, so it cannot exercise
anything that depends on the model's output. Here the endpoint call is replaced by a scripted
fake: the planning call gets a chosen JSON plan, the verifier call a chosen verdict, and the
answer call a fixed reply. Retrieval runs for real against the shipped corpus.

What these tests cover is the code around the model: wiring, fail-closed behaviour, follow-up
carry-over, updates, deletion and isolation. Whether the model's judgements are right (which facts it
proposes, which it refuses) is checked against a real endpoint by pace/memory_check.py.

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import ta_bot  # noqa: E402
from ta_bot import TABot  # noqa: E402

DEFAULT_PLAN = {"about_student": False, "needs_retrieval": True, "search_query": "", "refers_to": [],
                "facts": [], "forget": []}


class FakeLLM:
    """Stands in for TABot._complete.

    plans:   latest message -> plan dict, or a raw reply string (e.g. not JSON)
    verdict: None -> an unusable verifier reply; True/False -> that verdict for every candidate
    """

    def __init__(self, plans: dict | None = None, answer: str = "Here is an answer [1].", verdict=None):
        self.plans = plans or {}
        self.answer = answer
        self.verdict = verdict
        self.answer_prompts: list[list[dict]] = []  # messages of every answer call
        self.verify_requests: list[dict] = []  # what the verifier was shown
        self.route_inputs: list[str] = []  # what the planning call was shown

    def __call__(self, messages, max_tokens=None):
        if messages[0]["content"] == ta_bot.VERIFY_PROMPT:
            request = json.loads(messages[1]["content"])
            self.verify_requests.append(request)
            if self.verdict is None:
                return "I cannot help with that."
            return json.dumps({"verdicts": [{"reason": "scripted", "store": self.verdict}
                                            for _ in request["candidates"]]})
        if messages[0]["content"] == ta_bot.ROUTER_PROMPT:
            self.route_inputs.append(messages[1]["content"])
            latest = messages[1]["content"].rsplit("Latest message: ", 1)[1]
            plan = self.plans.get(latest, DEFAULT_PLAN)
            return plan if isinstance(plan, str) else json.dumps(plan)
        self.answer_prompts.append(messages)
        return self.answer


def make_bot(fake: FakeLLM | None = None, **extra) -> TABot:
    bot = TABot({"endpoint": "http://127.0.0.1:9", "model": "test", "corpus_path": str(ROOT / "corpus"),
                 "temperature": 0.0, "seed": 42, **extra})
    if fake is not None:
        bot._complete = fake
    return bot


def plan(**kw) -> dict:
    return {**DEFAULT_PLAN, **kw}


def fact(ftype: str, value: str) -> dict:
    return {"type": ftype, "value": value}


class ContractTests(unittest.TestCase):
    def test_chat_shape_and_valid_citations(self):
        bot = make_bot(FakeLLM())
        out = bot.chat("s", "u", "How does LoRA reduce the number of trainable parameters?")
        self.assertIsInstance(out["answer"], str)
        self.assertTrue(out["citations"])
        self.assertTrue(all(pid in bot._papers for pid in out["citations"]))

    def test_unreachable_endpoint_raises_timeout(self):
        bot = make_bot(timeout=2)  # port 9 (discard) is not listening
        with self.assertRaises(TimeoutError):
            bot.chat("s", "u", "What is RAG?")

    def test_unusable_plan_still_answers_and_stores_nothing(self):
        msg = "My name is Robin. What is retrieval-augmented generation?"
        bot = make_bot(FakeLLM({msg: "Sure! I think you should search."}, verdict=True))
        self.assertTrue(bot.chat("s", "u", msg)["citations"])
        self.assertEqual(bot.get_memory("u"), [])

    def test_get_memory_is_read_only(self):
        msg = "My name is Morgan."
        bot = make_bot(FakeLLM({msg: plan(facts=[fact("name", "Morgan")])}, verdict=True))
        bot.chat("s", "u", msg)
        bot.get_memory("u").append("tampered")
        self.assertEqual(bot.get_memory("u"), ["Name: Morgan"])


class RetrievalDecisionTests(unittest.TestCase):
    def test_without_a_plan_every_message_retrieves(self):
        bot = make_bot(use_router=False)
        bot._complete = FakeLLM()
        for q in ("What do you know about RLHF?", "Do you remember the DPO paper? How does it differ from PPO?"):
            self.assertTrue(bot.chat("s-" + q, "u", q)["citations"], q)
        unusable = FakeLLM({"How does LoRA work?": "Sure, let me search."})
        self.assertTrue(make_bot(unusable).chat("s", "u", "How does LoRA work?")["citations"])

    def test_question_about_the_student_skips_retrieval(self):
        q = "What is my name?"
        for needs in (False, True):  # the real model often sets both flags
            fake = FakeLLM({q: plan(about_student=True, needs_retrieval=needs)})
            self.assertEqual(make_bot(fake).chat("s", "u", q)["citations"], [])

    def test_plan_decides_retrieval(self):
        q = "thanks, that helped a lot!"
        fake = FakeLLM({q: plan(needs_retrieval=False)})
        self.assertEqual(make_bot(fake).chat("s", "u", q)["citations"], [])

    def test_follow_up_rewrite_finds_the_topic(self):
        first = "How does vLLM's PagedAttention manage KV cache memory?"
        fake = FakeLLM({"Can you elaborate?": plan(search_query="vLLM PagedAttention KV cache memory management")})
        bot = make_bot(fake)
        bot.chat("s", "u", first)
        self.assertIn("L11-03", bot.chat("s", "u", "Can you elaborate?")["citations"])

    def test_planner_sees_discussed_papers_and_carries_what_it_points_to(self):
        first = "How does vLLM's PagedAttention manage KV cache memory?"
        follow = "What are its limitations?"
        fake = FakeLLM({follow: plan(refers_to=[1, 7])})  # 7 is out of range and ignored
        bot = make_bot(fake)
        bot.chat("s", "u", first)
        out = bot.chat("s", "u", follow)
        self.assertIn("Papers discussed so far:\n[1] " + bot._papers["L11-03"], fake.route_inputs[-1])
        self.assertIn("L11-03", out["citations"])

    def test_pronoun_alone_carries_nothing(self):
        # "it" here starts a new topic; the planner says so and the earlier paper stays out.
        first = "How does vLLM's PagedAttention manage KV cache memory?"
        follow = "What is DPO and how does it differ from RLHF?"
        bot = make_bot(FakeLLM({follow: plan(refers_to=[])}))
        bot.chat("s", "u", first)
        self.assertNotIn("L11-03", bot.chat("s", "u", follow)["citations"])

    def test_first_turn_ignores_the_rewrite(self):
        # Keeps first-turn retrieval identical to the system measured in the ablation.
        q = "How does LoRA reduce the number of trainable parameters?"
        fake = FakeLLM({q: plan(search_query="Whisper speech recognition transcripts")})
        no_router = make_bot(use_router=False)
        no_router._complete = FakeLLM()
        self.assertEqual(make_bot(fake).chat("s", "u", q)["citations"], no_router.chat("s", "u", q)["citations"])


class MemoryWriteTests(unittest.TestCase):
    def test_approved_facts_are_stored_and_reach_a_new_session(self):
        msg = "I'm Alex, and my semester project is on speculative decoding for LLM inference."
        fake = FakeLLM({msg: plan(facts=[fact("name", "Alex"),
                                         fact("project", "speculative decoding for LLM inference")])},
                       verdict=True)
        bot = make_bot(fake)
        bot.chat("s1", "alex", msg)
        self.assertEqual(bot.get_memory("alex"),
                         ["Name: Alex", "Project: speculative decoding for LLM inference"])
        bot.chat("s2", "alex", "What do you remember about me?")
        self.assertIn("speculative decoding for LLM inference", fake.answer_prompts[-1][0]["content"])

    def test_verifier_refusal_is_respected(self):
        msg = "I'm interested in speech recognition."
        bot = make_bot(FakeLLM({msg: plan(facts=[fact("interest", "speech recognition")])}, verdict=False))
        bot.chat("s", "u", msg)
        self.assertEqual(bot.get_memory("u"), [])

    def test_unusable_verdict_stores_nothing(self):
        msg = "I'm interested in speech recognition."
        bot = make_bot(FakeLLM({msg: plan(facts=[fact("interest", "speech recognition")])}, verdict=None))
        bot.chat("s", "u", msg)
        self.assertEqual(bot.get_memory("u"), [])

    def test_verifier_sees_the_whole_message(self):
        msg = "Please don't save this. My project is about tokenization."
        fake = FakeLLM({msg: plan(facts=[fact("project", "tokenization")])}, verdict=False)
        make_bot(fake).chat("s", "u", msg)
        self.assertEqual(fake.verify_requests, [{"message": msg, "candidates": [fact("project", "tokenization")]}])

    def test_verifier_sees_only_the_message_not_excerpts_or_history(self):
        first = "How does vLLM's PagedAttention manage KV cache memory?"
        msg = "Can you explain how RLHF works?"
        fake = FakeLLM({msg: plan(facts=[fact("interest", "PagedAttention")])}, verdict=False)
        bot = make_bot(fake)
        bot.chat("s", "u", first)
        bot.chat("s", "u", msg)
        self.assertEqual(fake.verify_requests, [{"message": msg, "candidates": [fact("interest", "PagedAttention")]}])
        self.assertEqual(bot.get_memory("u"), [])

    def test_unknown_fact_types_never_reach_the_verifier(self):
        msg = "My name is Jo. My favourite colour is blue."
        fake = FakeLLM({msg: plan(facts=[fact("name", "Jo"), fact("color", "blue")])}, verdict=True)
        bot = make_bot(fake)
        bot.chat("s", "u", msg)
        self.assertEqual(fake.verify_requests[0]["candidates"], [fact("name", "Jo")])
        self.assertEqual(bot.get_memory("u"), ["Name: Jo"])

    def test_no_candidates_no_verifier_call(self):
        fake = FakeLLM(verdict=True)
        make_bot(fake).chat("s", "u", "How does LoRA work?")
        self.assertEqual(fake.verify_requests, [])


class MemoryUpdateTests(unittest.TestCase):
    def test_project_update_overwrites(self):
        a, b = "My project is about dense retrieval.", "Update: my project is now about memory for chat agents."
        fake = FakeLLM({a: plan(facts=[fact("project", "dense retrieval")]),
                        b: plan(facts=[fact("project", "memory for chat agents")])}, verdict=True)
        bot = make_bot(fake)
        bot.chat("s", "u", a)
        bot.chat("s", "u", b)
        self.assertEqual(bot.get_memory("u"), ["Project: memory for chat agents"])

    def test_restated_interest_is_not_duplicated(self):
        a, b = "I'm interested in evaluation.", "I'm really interested in evaluation and benchmarking."
        fake = FakeLLM({a: plan(facts=[fact("interest", "evaluation")]),
                        b: plan(facts=[fact("interest", "evaluation and benchmarking")])}, verdict=True)
        bot = make_bot(fake)
        bot.chat("s", "u", a)
        bot.chat("s", "u", b)
        self.assertEqual(bot.get_memory("u"), ["Interested in: evaluation and benchmarking"])

    def test_forget_one_type_and_everything(self):
        a = "My name is Riley and my project is about tokenization."
        fake = FakeLLM({a: plan(facts=[fact("name", "Riley"), fact("project", "tokenization")]),
                        "Please forget my project.": plan(needs_retrieval=False, forget=["project"]),
                        "Forget everything about me.": plan(needs_retrieval=False, forget=["all"])},
                       verdict=True)
        bot = make_bot(fake)
        bot.chat("s", "u", a)
        bot.chat("s", "u", "Please forget my project.")
        self.assertEqual(bot.get_memory("u"), ["Name: Riley"])
        bot.chat("s", "u", "Forget everything about me.")
        self.assertEqual(bot.get_memory("u"), [])

    def test_users_are_isolated_and_reset_clears(self):
        msg = "My name is Avery."
        fake = FakeLLM({msg: plan(facts=[fact("name", "Avery")])}, verdict=True)
        bot = make_bot(fake)
        bot.chat("shared-session", "a", msg)
        bot.chat("shared-session", "b", "What is my name?")
        self.assertEqual(bot.get_memory("b"), [])
        self.assertNotIn("avery", json.dumps(fake.answer_prompts[-1]).lower())  # no history or memory leak
        bot.reset_memory("a")
        self.assertEqual(bot.get_memory("a"), [])


if __name__ == "__main__":
    unittest.main()
