"""E1: the shared report is built from stored COMPLETE runs and says only what
those runs support."""
import os
import unittest
from datetime import datetime, timedelta
from unittest import mock

from report_selection import pick_previous_score, pick_previous_tracker_job_id

os.environ["PHASE5_ALWAYS_RUN_DEEP"] = "false"


def iso(days_ago=0.0):
    return (datetime.utcnow() - timedelta(days=days_ago)).isoformat() + "Z"


def entry(run_id, score, days_ago, job):
    return {"run_id": run_id, "score": score, "created_at": iso(days_ago), "tracker_job_id": job}


class PickPreviousScoreTests(unittest.TestCase):
    def test_needs_a_complete_run_at_least_six_days_older(self):
        hist = [entry("a", 30, 1, "j1"), entry("b", 33, 0, "j2")]
        self.assertIsNone(pick_previous_score(hist, "b", iso(0)))

    def test_uses_the_most_recent_run_that_is_old_enough(self):
        hist = [entry("a", 20, 20, "j1"), entry("b", 25, 8, "j2"), entry("c", 27, 3, "j3"), entry("d", 30, 0, "j4")]
        got = pick_previous_score(hist, "d", iso(0))
        self.assertEqual((got["run_id"], got["score"]), ("b", 25))

    def test_exactly_six_days_counts(self):
        hist = [entry("a", 20, 6.001, "j1"), entry("b", 30, 0, "j2")]
        self.assertEqual(pick_previous_score(hist, "b", iso(0))["run_id"], "a")

    def test_junk_is_ignored(self):
        self.assertIsNone(pick_previous_score(None, "x", iso(0)))
        self.assertIsNone(pick_previous_score([{"score": "n/a"}, None, {}], "x", iso(0)))
        self.assertIsNone(pick_previous_score([entry("a", 1, 10, "j")], "x", None))


class PickPreviousJobTests(unittest.TestCase):
    def test_needs_an_older_complete_run_with_different_tracker_data(self):
        runs = [entry("b", 33, 0, "j2"), entry("a", 30, 1, "j1")]       # newest first
        self.assertEqual(pick_previous_tracker_job_id(runs, "b", "j2"), "j1")

    def test_same_tracker_data_is_not_a_comparison(self):
        runs = [{"run_id": "b", "tracker_job_id": "j1"}, {"run_id": "a", "tracker_job_id": "j1"}]
        self.assertIsNone(pick_previous_tracker_job_id(runs, "b", "j1"))

    def test_a_run_without_tracker_data_is_skipped_not_guessed(self):
        runs = [{"run_id": "c", "tracker_job_id": "j3"}, {"run_id": "b"}, {"run_id": "a", "tracker_job_id": "j1"}]
        self.assertEqual(pick_previous_tracker_job_id(runs, "c", "j3"), "j1")

    def test_only_one_run_means_nothing_to_compare(self):
        self.assertIsNone(pick_previous_tracker_job_id([{"run_id": "a", "tracker_job_id": "j1"}], "a", "j1"))
        self.assertIsNone(pick_previous_tracker_job_id([], "a", "j1"))


try:
    import main  # noqa: E402
except Exception as exc:  # pragma: no cover
    main = None
    _skip = f"backend app could not be imported here: {exc}"
else:
    _skip = None

if main is not None:
    from bson import ObjectId  # noqa: E402

    from tests.test_scan_run import FakeCol  # noqa: E402


def job(job_id, mentioned_by, total=20, days_ago=0):
    return {"job_id": job_id, "business_id": "b1", "job_type": "core", "model": "multi", "status": "completed",
            "created_at": iso(days_ago), "questions": [{"id": "1", "text": "q one"}], "results": {},
            "provider_scores": {p: {"mentioned": 7 if p in mentioned_by else 0, "total": total} for p in ("chatgpt", "perplexity", "claude", "gemini")}}


@unittest.skipIf(main is None, _skip)
class PublicReportEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def _report(self, history, latest_run_id, jobs, score=30.0):
        biz = {"_id": ObjectId(), "report_share_token": "tok", "businessName": "Cornus", "url": "https://cornus.co.uk",
               "latest_wonder_score": score, "latest_wonder_run_id": latest_run_id, "latest_wonder_at": history[-1]["created_at"] if history else None,
               "latest_wonder_inputs": {"mention_rate": 20, "technical_score": 80, "market_position": 0}, "wonder_scores": history}
        with mock.patch.object(main, "businesses_col", FakeCol([biz])), mock.patch.object(main, "phase5_jobs_col", FakeCol(jobs)):
            return await main.api_public_report("tok")

    async def test_first_complete_run_has_no_week_comparison_and_no_what_moved(self):
        r = await self._report([entry("a", 30, 0, "j1")], "a", [job("j1", {"chatgpt"})])
        self.assertEqual(r["score"], 30)
        self.assertIsNone(r["previousScore"])
        self.assertFalse(r["whatMovedAvailable"])
        self.assertEqual(r["whatMoved"], [])

    async def test_a_run_from_minutes_ago_is_not_last_week(self):
        hist = [entry("a", 28, 0.01, "j1"), entry("b", 30, 0, "j2")]
        r = await self._report(hist, "b", [job("j1", {"chatgpt"}), job("j2", {"chatgpt"})])
        self.assertIsNone(r["previousScore"])
        self.assertTrue(r["whatMovedAvailable"])        # two complete runs with their own tracker data

    async def test_a_complete_run_a_week_older_is_the_comparison(self):
        hist = [entry("a", 25, 8, "j1"), entry("b", 30, 0, "j2")]
        r = await self._report(hist, "b", [job("j1", {"chatgpt"}), job("j2", {"chatgpt"})])
        self.assertEqual(r["previousScore"], 25)
        self.assertEqual(r["previousScoreAt"], hist[0]["created_at"])

    async def test_per_model_comes_from_the_run_behind_the_score_not_the_newest_job(self):
        hist = [entry("a", 30, 0, "j1")]
        newest_but_not_in_a_complete_run = job("j9", {"claude"})
        r = await self._report(hist, "a", [job("j1", {"chatgpt"}), newest_but_not_in_a_complete_run])
        self.assertEqual({m["model"] for m in r["perModel"] if m["mentioned"] > 0}, {"ChatGPT"})

    async def test_no_complete_run_means_no_score(self):
        biz = {"_id": ObjectId(), "report_share_token": "tok", "businessName": "X", "url": "https://x.co.uk", "wonder_scores": []}
        with mock.patch.object(main, "businesses_col", FakeCol([biz])), mock.patch.object(main, "phase5_jobs_col", FakeCol([])):
            r = await main.api_public_report("tok")
        self.assertIsNone(r["score"])
        self.assertFalse(r["visibilityVerified"])
        self.assertIsNone(r["previousScore"])


if __name__ == "__main__":
    unittest.main()
