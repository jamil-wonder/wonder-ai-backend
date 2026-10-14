"""A5: the same question asked repeatedly gives a stable answer (majority vote)."""
import unittest
from unittest import mock

from agents.phase5 import analysis


def s(mentioned, status="ok", **kw):
    return {"mentioned": mentioned, "fetch_status": status, **kw}


class PickMajorityTests(unittest.TestCase):
    def test_majority_of_three(self):
        self.assertTrue(analysis.pick_majority_sample([s(True), s(False), s(True)])["mentioned"])
        self.assertFalse(analysis.pick_majority_sample([s(True), s(False), s(False)])["mentioned"])

    def test_vote_is_recorded(self):
        self.assertEqual(analysis.pick_majority_sample([s(True), s(False), s(True)])["vote"], {"mentioned": 2, "of": 3})

    def test_failed_samples_do_not_vote(self):
        out = analysis.pick_majority_sample([s(False, "failed"), s(True), s(True)])
        self.assertTrue(out["mentioned"])
        self.assertEqual(out["vote"]["of"], 2)

    def test_two_voters_that_disagree_keep_the_first(self):
        self.assertTrue(analysis.pick_majority_sample([s(True), s(False)])["mentioned"])
        self.assertFalse(analysis.pick_majority_sample([s(False), s(True)])["mentioned"])

    def test_all_failed_returns_a_failed_answer(self):
        self.assertEqual(analysis.pick_majority_sample([s(False, "failed")])["fetch_status"], "failed")

    def test_input_is_not_mutated(self):
        first = s(True)
        analysis.pick_majority_sample([first, s(True)])
        self.assertNotIn("vote", first)


class VotingFlowTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, per_call):
        """per_call: list of results returned by successive provider calls (same for every provider)."""
        counters = {}

        async def fake_backoff(*, model_provider, **kw):
            i = counters.get(model_provider, 0)
            counters[model_provider] = i + 1
            r = per_call[min(i, len(per_call) - 1)]
            if isinstance(r, Exception):
                raise r
            return dict(r)

        with mock.patch.object(analysis, "_run_with_backoff", fake_backoff), \
                mock.patch.object(analysis, "_fetch_page_context", mock.AsyncMock(return_value={})), \
                mock.patch.object(analysis, "_safe_provider_result", lambda raw: raw):
            out = await analysis.analyze_single_question_multi("https://x.com", {"id": "1", "text": "q"})
        return out, counters

    async def test_agreeing_answers_cost_two_calls_per_model(self):
        out, counters = await self._run([s(True), s(True)])
        self.assertTrue(all(c == 2 for c in counters.values()))
        self.assertTrue(out["providers"]["chatgpt"]["mentioned"])

    async def test_disagreement_triggers_a_third_call_and_majority_wins(self):
        out, counters = await self._run([s(True), s(False), s(False)])
        self.assertTrue(all(c == 3 for c in counters.values()))
        self.assertFalse(out["providers"]["claude"]["mentioned"])

    async def test_a_failed_extra_sample_keeps_the_first_answer(self):
        out, _ = await self._run([s(True), RuntimeError("boom")])
        self.assertTrue(out["providers"]["gemini"]["mentioned"])

    async def test_a_failed_first_answer_is_not_voted_on(self):
        with mock.patch("asyncio.sleep", mock.AsyncMock()):
            out, counters = await self._run([s(False, "failed")])
        self.assertEqual(out["providers"]["gemini"]["fetch_status"], "failed")
        self.assertTrue(all(c == 2 for c in counters.values()))  # first try + the A4 retry, no votes

    async def test_can_be_switched_off(self):
        with mock.patch.object(analysis, "PHASE5_VOTE_SAMPLING", False):
            out, counters = await self._run([s(True)])
        self.assertTrue(all(c == 1 for c in counters.values()))


class VoteSurvivesNormalisationTests(unittest.IsolatedAsyncioTestCase):
    """Regression: the real _safe_provider_result used to drop `vote`, so vote-share
    scoring and the score margin silently never ran (tests that mocked it hid this)."""

    async def test_vote_reaches_the_stored_answer_and_the_score(self):
        from agents.phase5.scoring import compute_mention_rate, compute_mention_rate_margin

        n = {}

        async def fake_backoff(*, model_provider, **kw):
            i = n.get(model_provider, 0)
            n[model_provider] = i + 1
            return {"status": "Mentioned" if i % 2 == 0 else "Not Mentioned", "fetch_status": "ok"}

        with mock.patch.object(analysis, "_run_with_backoff", fake_backoff),                 mock.patch.object(analysis, "_fetch_page_context", mock.AsyncMock(return_value={})):
            out = await analysis.analyze_single_question_multi("https://x.com", {"id": "1", "text": "q", "category": "branded"})
        cell = out["providers"]["chatgpt"]
        self.assertEqual(cell["vote"], {"mentioned": 3, "of": 5})
        qs = [{"id": "1", "category": "branded"}]
        self.assertAlmostEqual(compute_mention_rate({"1": out}, qs), 60.0, places=1)   # 3 of 5, not a hard yes
        self.assertNotEqual(compute_mention_rate_margin({"1": out}, qs), compute_mention_rate_margin(
            {"1": {"providers": {p: {"mentioned": True, "fetch_status": "ok"} for p in out["providers"]}}}, qs))


class SampleMentionedRuleTests(unittest.TestCase):
    def test_same_rule_as_the_mention_rate(self):
        m = analysis._sample_mentioned
        self.assertTrue(m({"status": "Mentioned"}))
        self.assertTrue(m({"status": "Not Mentioned", "position": 3}))
        self.assertFalse(m({"status": "Not Mentioned", "target_site": {"status": "matched"}}))  # relevance != mention
        self.assertFalse(m({"status": "Not Mentioned", "target_site": {"status": "no_match"}}))
        self.assertFalse(m({"status": "Not Mentioned"}))
        self.assertFalse(m(None))


if __name__ == "__main__":
    unittest.main()
