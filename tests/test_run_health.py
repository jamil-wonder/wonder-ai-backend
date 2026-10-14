"""J6 / A4: a failed model answer is missing data, never a zero.

All model responses here are hand-made fixtures. Nothing calls an AI
provider or the database, so these cost nothing and are safe to run anywhere.
"""
import unittest
from unittest import mock

from agents.phase5 import analysis
from agents.phase5.helpers import _estimate_target_visibility_score
from agents.phase5.scoring import (
    compute_provider_score,
    compute_run_health,
    provider_result_state,
)

PROVIDERS = ("perplexity", "chatgpt", "claude", "gemini")


def ok_answer(mentioned: bool) -> dict:
    return {
        "mentioned": mentioned,
        "position": 3 if mentioned else None,
        "sources": ["example.com"],
        "cited": True,
        "status": "Mentioned" if mentioned else "Not Mentioned",
        "fetch_status": "ok",
    }


def failed_answer(msg: str = "OpenAI rate limit reached. Please retry shortly.") -> dict:
    return {
        "mentioned": False, "position": None, "sources": [], "cited": False,
        "status": "Not Mentioned", "fetch_status": "failed", "error": msg,
    }


def not_run_answer() -> dict:
    return {
        "mentioned": False, "position": None, "sources": [], "cited": False,
        "status": "Not Mentioned", "fetch_status": "not_run",
    }


def build_run(per_provider: dict, questions: int = 20, mentioned_every: int = 4) -> dict:
    """per_provider maps provider name -> 'ok' | 'failed' | 'not_run'."""
    results = {}
    for q in range(questions):
        providers = {}
        for name in PROVIDERS:
            kind = per_provider.get(name, "ok")
            if kind == "failed":
                providers[name] = failed_answer()
            elif kind == "not_run":
                providers[name] = not_run_answer()
            else:
                providers[name] = ok_answer(q % mentioned_every == 0)
        results[str(q + 1)] = {"id": str(q + 1), "providers": providers}
    return results


class ResultStateTests(unittest.TestCase):
    def test_explicit_status_wins(self):
        self.assertEqual(provider_result_state({"fetch_status": "failed"}), "failed")
        self.assertEqual(provider_result_state({"fetch_status": "not_run"}), "not_run")
        self.assertEqual(provider_result_state({"fetch_status": "ok", "error": "x"}), "ok")

    def test_results_stored_before_the_field_existed(self):
        # Old jobs only carry an optional error string.
        self.assertEqual(provider_result_state({"error": "Gemini rate limit reached."}), "failed")
        self.assertEqual(provider_result_state({"status": "Not Mentioned"}), "ok")

    def test_garbage_is_failed_not_zero(self):
        self.assertEqual(provider_result_state(None), "failed")
        self.assertEqual(provider_result_state("nope"), "failed")


class ProviderScoreTests(unittest.TestCase):
    def test_broken_model_does_not_change_the_others(self):
        healthy = build_run({})
        one_broken = build_run({"chatgpt": "failed"})
        for name in ("perplexity", "claude", "gemini"):
            self.assertEqual(
                compute_provider_score(healthy, name),
                {**compute_provider_score(one_broken, name)},
            )

    def test_failed_model_is_unavailable_not_zero_of_twenty(self):
        score = compute_provider_score(build_run({"gemini": "failed"}), "gemini")
        self.assertFalse(score["available"])
        self.assertEqual(score["total"], 0)
        self.assertEqual(score["failed"], 20)
        self.assertEqual(score["mentioned"], 0)

    def test_not_run_model_is_unavailable_but_not_failed(self):
        score = compute_provider_score(build_run({"gemini": "not_run"}), "gemini")
        self.assertFalse(score["available"])
        self.assertEqual(score["failed"], 0)
        self.assertEqual(score["not_run"], 20)

    def test_partial_failures_only_drop_the_failed_questions(self):
        run = build_run({})
        for qid in ("1", "2", "3", "4", "5"):
            run[qid]["providers"]["claude"] = failed_answer()
        score = compute_provider_score(run, "claude")
        self.assertTrue(score["available"])
        self.assertEqual(score["total"], 15)
        self.assertEqual(score["failed"], 5)

    def test_two_runs_on_the_same_answers_give_the_same_score(self):
        # A5 regression: aggregation itself must be deterministic.
        run = build_run({"chatgpt": "failed"})
        self.assertEqual(compute_provider_score(run, "claude"), compute_provider_score(run, "claude"))


class RunHealthTests(unittest.TestCase):
    def test_healthy_run_is_complete(self):
        health = compute_run_health(build_run({}))
        self.assertFalse(health["incomplete"])
        self.assertEqual(health["failed"], 0)

    def test_the_real_incident_chatgpt_and_gemini_dead_is_incomplete(self):
        health = compute_run_health(build_run({"chatgpt": "failed", "gemini": "failed"}))
        self.assertTrue(health["incomplete"])
        self.assertEqual(health["failed_fraction"], 0.5)
        self.assertEqual(health["failed_by_provider"], {"chatgpt": 20, "gemini": 20})

    def test_exactly_a_quarter_failed_is_still_complete(self):
        # one of four models down = 25%, the limit is "more than 25%"
        self.assertFalse(compute_run_health(build_run({"gemini": "failed"}))["incomplete"])

    def test_just_over_a_quarter_is_incomplete(self):
        run = build_run({"gemini": "failed"})
        run["1"]["providers"]["claude"] = failed_answer()
        self.assertTrue(compute_run_health(run)["incomplete"])

    def test_not_run_is_not_counted_as_a_failure(self):
        health = compute_run_health(build_run({"gemini": "not_run"}))
        self.assertFalse(health["incomplete"])
        self.assertEqual(health["failed"], 0)
        self.assertEqual(health["not_run"], 20)

    def test_everything_failed_is_incomplete(self):
        health = compute_run_health(build_run({n: "failed" for n in PROVIDERS}))
        self.assertTrue(health["incomplete"])

    def test_empty_results_do_not_crash(self):
        self.assertFalse(compute_run_health({})["incomplete"])
        self.assertFalse(compute_run_health(None)["incomplete"])


class SafeProviderResultTests(unittest.TestCase):
    def test_exception_becomes_a_failed_result_with_the_reason(self):
        result = analysis._safe_provider_result(RuntimeError("OpenAI rate limit reached."))
        self.assertEqual(result["fetch_status"], "failed")
        self.assertIn("rate limit", result["error"])

    def test_non_dict_result_is_failed(self):
        self.assertEqual(analysis._safe_provider_result(None)["fetch_status"], "failed")

    def test_normal_answer_is_ok_and_not_run_passes_through(self):
        self.assertEqual(
            analysis._safe_provider_result({"status": "Mentioned", "position": 2})["fetch_status"], "ok"
        )
        self.assertEqual(
            analysis._safe_provider_result({"fetch_status": "not_run", "status": "Not Mentioned"})["fetch_status"],
            "not_run",
        )


class TargetScoreTests(unittest.TestCase):
    def test_questions_every_model_failed_are_not_counted_as_misses(self):
        good = build_run({}, questions=5, mentioned_every=1)
        padded = dict(good)
        for i in range(6, 21):
            padded[str(i)] = {"id": str(i), "providers": {n: failed_answer() for n in PROVIDERS}}
        self.assertEqual(
            _estimate_target_visibility_score(good),
            _estimate_target_visibility_score(padded),
        )


class RetryOnceTests(unittest.IsolatedAsyncioTestCase):
    async def _run_multi(self, backoff):
        with mock.patch.object(analysis, "_fetch_page_context", new=mock.AsyncMock(return_value={})), \
             mock.patch.object(analysis, "_run_with_backoff", new=backoff), \
             mock.patch.object(analysis, "PHASE5_VOTE_SAMPLING", False), \
             mock.patch.object(analysis.asyncio, "sleep", new=mock.AsyncMock()):
            return await analysis.analyze_single_question_multi("https://example.com", {"id": "1", "text": "q"})

    async def test_a_model_that_fails_once_is_retried_and_recovers(self):
        calls = {"openai": 0}

        async def backoff(url, question, model_provider, include_competitors, target_context):
            if model_provider == "openai":
                calls["openai"] += 1
                if calls["openai"] == 1:
                    raise RuntimeError("OpenAI rate limit reached.")
            return {"id": "1", "status": "Mentioned", "position": 1}

        out = await self._run_multi(backoff)
        self.assertEqual(calls["openai"], 2)
        self.assertEqual(out["providers"]["chatgpt"]["fetch_status"], "ok")

    async def test_a_model_that_keeps_failing_is_recorded_failed_after_one_retry(self):
        calls = {"openai": 0}

        async def backoff(url, question, model_provider, include_competitors, target_context):
            if model_provider == "openai":
                calls["openai"] += 1
                raise RuntimeError("OpenAI rate limit reached.")
            return {"id": "1", "status": "Mentioned", "position": 1}

        out = await self._run_multi(backoff)
        self.assertEqual(calls["openai"], 2)  # first try + exactly one retry
        self.assertEqual(out["providers"]["chatgpt"]["fetch_status"], "failed")
        self.assertEqual(out["providers"]["claude"]["fetch_status"], "ok")


if __name__ == "__main__":
    unittest.main()
