"""A5: the score is measured from a stored snapshot of what all four models said.
A re-run inside the window re-uses it (identical score, no AI calls); the weekly
run / an explicit fresh run asks the models again."""
import os
import unittest
from datetime import datetime, timedelta
from unittest import mock

os.environ["PHASE5_ALWAYS_RUN_DEEP"] = "false"

try:
    import main  # noqa: E402
except Exception as exc:
    raise unittest.SkipTest(f"backend app could not be imported here: {exc}")

from bson import ObjectId  # noqa: E402

from tests.test_scan_run import FakeCol  # noqa: E402  (same in-memory collection stand-in)

PROVIDERS = ("perplexity", "chatgpt", "claude", "gemini")
QUESTIONS = [{"id": str(i), "text": f"question {i}", "category": "branded" if i <= 5 else "non-branded"} for i in range(1, 21)]


def cell(mentioned=False, state="ok"):
    if state == "failed":
        return {"mentioned": False, "status": "Not Mentioned", "fetch_status": "failed", "error": "x"}
    return {"mentioned": mentioned, "status": "Mentioned" if mentioned else "Not Mentioned", "fetch_status": "ok"}


def stored_row(qid, mentioned_by=(), failed=()):
    return {"id": qid, "providers": {p: cell(p in mentioned_by, "failed" if p in failed else "ok") for p in PROVIDERS}}


def iso(days_ago=0):
    return (datetime.utcnow() - timedelta(days=days_ago)).isoformat() + "Z"


class SnapshotLookupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        p = mock.patch.object(main, "PHASE5_SNAPSHOT_DAYS", 7)  # opt-in feature
        p.start()
        self.addCleanup(p.stop)

    def _jobs(self, days_ago=1, questions=QUESTIONS):
        return FakeCol([{
            "job_id": "old", "business_id": "b1", "job_type": "core", "model": "multi", "status": "completed",
            "created_at": iso(days_ago), "questions": questions,
            "results": {q["id"]: stored_row(q["id"], mentioned_by=("claude",)) for q in questions},
        }])

    async def _load(self, jobs, **kw):
        with mock.patch.object(main, "phase5_jobs_col", jobs):
            return await main._load_answer_snapshot("b1", QUESTIONS, **kw)

    async def test_recent_complete_run_is_re_used(self):
        snap, meta = await self._load(self._jobs(days_ago=2))
        self.assertEqual(len(snap), 20)
        self.assertEqual(meta["job_id"], "old")

    async def test_older_than_the_window_is_not_used(self):
        snap, meta = await self._load(self._jobs(days_ago=9))
        self.assertEqual((snap, meta), ({}, None))

    async def test_fresh_request_ignores_the_snapshot(self):
        self.assertEqual(await self._load(self._jobs(), fresh=True), ({}, None))

    async def test_different_questions_do_not_match(self):
        other = [{"id": str(i), "text": f"other {i}", "category": "branded"} for i in range(1, 21)]
        self.assertEqual(await self._load(self._jobs(questions=other)), ({}, None))

    async def test_incomplete_runs_are_never_a_snapshot(self):
        jobs = self._jobs()
        jobs.docs[0]["status"] = "incomplete"
        self.assertEqual(await self._load(jobs), ({}, None))

    async def test_window_counts_from_when_answers_were_collected_not_when_re_used(self):
        jobs = self._jobs(days_ago=1)           # a recent job that only re-used old answers ...
        jobs.docs[0]["answers_collected_at"] = iso(9)   # ... collected 9 days ago
        self.assertEqual(await self._load(jobs), ({}, None))


class DefaultIsOffTests(unittest.IsolatedAsyncioTestCase):
    async def test_by_default_every_run_asks_all_four_models_again(self):
        self.assertEqual(main.PHASE5_SNAPSHOT_DAYS, 0)
        jobs = FakeCol([{
            "job_id": "old", "business_id": "b1", "job_type": "core", "model": "multi", "status": "completed",
            "created_at": iso(0), "questions": QUESTIONS,
            "results": {q["id"]: stored_row(q["id"]) for q in QUESTIONS},
        }])
        with mock.patch.object(main, "phase5_jobs_col", jobs):
            self.assertEqual(await main._load_answer_snapshot("b1", QUESTIONS), ({}, None))


class MergeTests(unittest.TestCase):
    def test_good_stored_answers_are_kept_failed_ones_replaced(self):
        cached = stored_row("1", mentioned_by=("claude",), failed=("gemini",))
        self.assertEqual(main._snapshot_missing_providers(cached), {"gemini"})
        fresh = {"providers": {"gemini": cell(True)}}
        row, reused = main._merge_snapshot_row({"id": "1"}, cached, fresh)
        self.assertEqual(reused, 3)
        self.assertTrue(row["providers"]["gemini"]["mentioned"])
        self.assertTrue(row["providers"]["claude"]["mentioned"])

    def test_stored_row_is_copied_not_shared(self):
        cached = stored_row("1")
        row, _ = main._merge_snapshot_row({"id": "1"}, cached, None)
        row["providers"]["claude"]["mentioned"] = True
        self.assertFalse(cached["providers"]["claude"]["mentioned"])


class WorkerReuseTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        p = mock.patch.object(main, "PHASE5_SNAPSHOT_DAYS", 7)  # opt-in feature
        p.start()
        self.addCleanup(p.stop)

    async def _run(self, jobs, biz, fresh=False, asked=None):
        runs = FakeCol()
        job = {"job_id": "new", "job_type": "core", "model": "multi", "url": "https://example.com", "user_id": "u1",
               "business_id": "b1", "status": "running", "seed_results": {}, "questions": QUESTIONS, "fresh": fresh}
        jobs.docs.append(job)
        asked = asked if asked is not None else []

        async def fake_multi(url, question, include_competitors=False, only_providers=None):
            asked.append((question["id"], None if only_providers is None else set(only_providers)))
            names = PROVIDERS if only_providers is None else only_providers
            return {"id": question["id"], "providers": {p: cell(True) for p in names}}

        with mock.patch.object(main, "businesses_col", biz), mock.patch.object(main, "phase5_jobs_col", jobs), \
                mock.patch.object(main, "scan_runs_col", runs), mock.patch.object(main, "analyze_single_question_multi", fake_multi), \
                mock.patch.object(main, "generate_deep_competitor_scores", mock.AsyncMock(return_value=[])), \
                mock.patch.object(main, "generate_brand_perception_summary", mock.AsyncMock(return_value=None)), \
                mock.patch.object(main, "_upsert_user_business", mock.AsyncMock(return_value=None)), \
                mock.patch.object(main, "_log_ai_usage_event", mock.AsyncMock()):
            await main._process_phase5_job(job)
        return job, asked

    def _setup(self, days_ago=1, failed=()):
        jobs = FakeCol([{
            "job_id": "old", "business_id": "b1", "job_type": "core", "model": "multi", "status": "completed",
            "created_at": iso(days_ago), "questions": QUESTIONS,
            "results": {q["id"]: stored_row(q["id"], mentioned_by=("claude", "chatgpt"), failed=failed) for q in QUESTIONS},
        }])
        biz = FakeCol([{"_id": ObjectId(), "latest_phase1_score": 91}])
        return jobs, biz

    async def test_a_rerun_inside_the_window_asks_no_model_and_gives_identical_answers(self):
        jobs, biz = self._setup()
        job, asked = await self._run(jobs, biz)
        self.assertEqual(asked, [])
        self.assertEqual(job["status"], "completed")
        old_rate = main.compute_mention_rate(jobs.docs[0]["results"], QUESTIONS)
        self.assertEqual(main.compute_mention_rate(job["results"], QUESTIONS), old_rate)

    async def test_the_new_run_keeps_the_original_collection_date(self):
        jobs, biz = self._setup(days_ago=3)
        job, _ = await self._run(jobs, biz)
        self.assertEqual(job["answers_collected_at"], jobs.docs[0]["created_at"])
        self.assertEqual(job["snapshot_from_job"], "old")

    async def test_only_the_missing_model_is_asked_again(self):
        jobs, biz = self._setup(failed=("gemini",))
        job, asked = await self._run(jobs, biz)
        self.assertEqual(len(asked), 20)
        self.assertTrue(all(only == {"gemini"} for _, only in asked))
        self.assertEqual(job["status"], "completed")

    async def test_a_fresh_run_asks_all_four_models(self):
        jobs, biz = self._setup()
        job, asked = await self._run(jobs, biz, fresh=True)
        self.assertEqual(len(asked), 20)
        self.assertTrue(all(only is None for _, only in asked))
        self.assertNotEqual(job["answers_collected_at"], jobs.docs[0]["created_at"])

    async def test_an_expired_snapshot_asks_all_four_models(self):
        jobs, biz = self._setup(days_ago=10)
        _, asked = await self._run(jobs, biz)
        self.assertEqual(len(asked), 20)
        self.assertTrue(all(only is None for _, only in asked))


if __name__ == "__main__":
    unittest.main()
