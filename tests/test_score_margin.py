"""A5: branded questions get up to 5 votes, and the score carries a +/- margin."""
import unittest
from unittest import mock

from agents.phase5 import analysis
from agents.phase5.scoring import compute_mention_rate_margin, wonder_margin

PROVIDERS = ("perplexity", "chatgpt", "claude", "gemini")
QS = [{"id": str(i), "category": "branded" if i <= 5 else "non-branded"} for i in range(1, 21)]


def rows(vote=None, mentioned=False, failed=()):
    out = {}
    for q in QS:
        provs = {}
        for p in PROVIDERS:
            if p in failed:
                provs[p] = {"mentioned": False, "fetch_status": "failed"}
            else:
                cell = {"mentioned": mentioned, "fetch_status": "ok"}
                if vote:
                    cell["vote"] = {"mentioned": vote[0], "of": vote[1]}
                provs[p] = cell
        out[q["id"]] = {"id": q["id"], "providers": provs}
    return out


class MarginTests(unittest.TestCase):
    def test_more_votes_means_a_smaller_margin(self):
        two = compute_mention_rate_margin(rows(vote=(1, 2)), QS)
        five = compute_mention_rate_margin(rows(vote=(3, 5)), QS)
        self.assertGreater(two, five)

    def test_more_answered_models_means_a_smaller_margin(self):
        all_four = compute_mention_rate_margin(rows(vote=(1, 2)), QS)
        one = compute_mention_rate_margin(rows(vote=(1, 2), failed=("chatgpt", "claude", "gemini")), QS)
        self.assertGreater(one, all_four)

    def test_none_when_nobody_answered(self):
        self.assertIsNone(compute_mention_rate_margin(rows(failed=PROVIDERS), QS))
        self.assertIsNone(compute_mention_rate_margin({}, QS))

    def test_never_claims_zero_noise(self):
        self.assertGreater(compute_mention_rate_margin(rows(vote=(0, 2)), QS), 0)

    def test_wonder_margin_is_60_percent_of_it(self):
        self.assertEqual(wonder_margin(10.0), 6.0)
        self.assertIsNone(wonder_margin(None))


class BrandedVotesTests(unittest.IsolatedAsyncioTestCase):
    async def _calls(self, category, answers):
        n = {}

        async def fake_backoff(*, model_provider, **kw):
            i = n.get(model_provider, 0)
            n[model_provider] = i + 1
            return {"mentioned": answers[min(i, len(answers) - 1)], "fetch_status": "ok"}

        with mock.patch.object(analysis, "_run_with_backoff", fake_backoff), \
                mock.patch.object(analysis, "_fetch_page_context", mock.AsyncMock(return_value={})), \
                mock.patch.object(analysis, "_safe_provider_result", lambda raw: raw):
            out = await analysis.analyze_single_question_multi("https://x.com", {"id": "1", "text": "q", "category": category})
        return out["providers"]["chatgpt"], n["openai"]

    async def test_branded_unanimous_costs_three_calls(self):
        res, calls = await self._calls("branded", [True])
        self.assertEqual((calls, res["vote"]), (3, {"mentioned": 3, "of": 3}))

    async def test_branded_split_goes_to_five_votes(self):
        res, calls = await self._calls("branded", [True, False, True, False, True])
        self.assertEqual(calls, 5)
        self.assertTrue(res["mentioned"])
        self.assertEqual(res["vote"], {"mentioned": 3, "of": 5})

    async def test_non_branded_still_two_then_a_tiebreaker(self):
        _, calls = await self._calls("non-branded", [True])
        self.assertEqual(calls, 2)
        _, calls = await self._calls("non-branded", [True, False, False])
        self.assertEqual(calls, 3)


if __name__ == "__main__":
    unittest.main()
