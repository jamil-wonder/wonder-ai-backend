"""J6: scan-run lifecycle. A Wonder Score is written ONLY when a run completes,
never mid-run, never from a failed / incomplete run.

Runs the real backend code against an in-memory fake database with the AI
models mocked - nothing touches Mongo or any provider. Needs the backend's
normal environment (.env) to import `main`; skipped if that isn't available."""
import os
import unittest
from unittest import mock

os.environ["PHASE5_ALWAYS_RUN_DEEP"] = "false"  # never trigger real competitor / summary AI calls

try:
    import main  # noqa: E402  (executes backend/main_parts/*)
except Exception as exc:  # missing env vars, dependencies, ...
    raise unittest.SkipTest(f"backend app could not be imported here: {exc}")

from bson import ObjectId  # noqa: E402

PROVIDERS = ("perplexity", "chatgpt", "claude", "gemini")


def _set_path(doc, path, value):
    parts = path.split(".")
    cur = doc
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


class FakeCol:
    """Just enough of a Mongo collection for the scan-run code."""

    def __init__(self, docs=None):
        self.docs = docs if docs is not None else []

    def _match(self, d, q):
        for k, v in q.items():
            if isinstance(v, dict) and "$in" in v:
                if d.get(k) not in v["$in"]:
                    return False
            elif isinstance(v, dict) and "$ne" in v:
                if d.get(k) == v["$ne"]:
                    return False
            elif isinstance(v, dict) and "$gte" in v:
                if not (d.get(k) or "") >= v["$gte"]:
                    return False
            elif isinstance(v, dict) and "$exists" in v:
                if (k in d) != v["$exists"]:
                    return False
            elif d.get(k) != v:
                return False
        return True

    async def find_one(self, q, projection=None, sort=None, **kw):
        hits = [d for d in self.docs if self._match(d, q)]
        if sort:
            hits.sort(key=lambda d: d.get(sort[0][0]) or "", reverse=sort[0][1] == -1)
        return hits[0] if hits else None

    async def insert_one(self, doc):
        self.docs.append(doc)

    async def count_documents(self, q, **kw):
        return len([d for d in self.docs if self._match(d, q)])

    async def find_one_and_update(self, q, upd, upsert=False, **kw):
        d = await self.find_one(q)
        if d is None:
            if not upsert:
                return None
            d = {k: v for k, v in q.items() if not isinstance(v, dict)}
            for k, v in (upd.get("$setOnInsert") or {}).items():
                _set_path(d, k, v)
            self.docs.append(d)
        for k, v in (upd.get("$set") or {}).items():
            _set_path(d, k, v)
        return d

    async def update_one(self, q, upd, **kw):
        d = await self.find_one(q)
        if not d:
            return
        for k, v in (upd.get("$set") or {}).items():
            _set_path(d, k, v)
        for k, v in (upd.get("$inc") or {}).items():
            d[k] = d.get(k, 0) + v


def answer(m):
    return {"mentioned": m, "status": "Mentioned" if m else "Not Mentioned", "fetch_status": "ok"}


def tracker_results(mentioned_qids):
    return {str(i): {"id": str(i), "providers": {p: answer(i in mentioned_qids) for p in PROVIDERS}} for i in range(1, 21)}


QUESTIONS = [{"id": str(i), "category": "branded" if i <= 5 else "non-branded"} for i in range(1, 21)]


class ScanRunTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bid = ObjectId()
        self.biz = FakeCol([{"_id": self.bid, "latest_phase1_score": 91, "user_id": "u1"}])
        self.jobs = FakeCol([{
            "business_id": str(self.bid), "job_type": "core", "model": "multi", "status": "completed",
            "created_at": "2026-10-01", "results": tracker_results({1, 6, 7, 8, 9, 10}), "questions": QUESTIONS,
        }])
        self.runs = FakeCol()
        patchers = [
            mock.patch.object(main, "businesses_col", self.biz),
            mock.patch.object(main, "phase5_jobs_col", self.jobs),
            mock.patch.object(main, "scan_runs_col", self.runs),
        ]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)

    async def _new_run(self, trigger="manual_analyzer", legs=("crawl",)):
        return await main._scan_run_create(business_id=str(self.bid), user_id="u1", url="x.com", trigger=trigger, legs=list(legs))

    async def test_score_is_not_written_while_the_run_is_running(self):
        rid = await self._new_run()
        await main._scan_run_set_leg(rid, "crawl", "running")
        await main._scan_run_finalize(rid)
        self.assertEqual(self.runs.docs[0]["status"], "running")
        self.assertNotIn("latest_wonder_score", self.biz.docs[0])

    async def test_manual_analyzer_recrawl_recomputes_the_score(self):
        rid = await self._new_run()
        await main._scan_run_set_leg(rid, "crawl", "ok", score=80)
        await main._scan_run_finalize(rid)
        rate = main.compute_mention_rate(tracker_results({1, 6, 7, 8, 9, 10}), QUESTIONS)
        self.assertEqual(self.runs.docs[0]["status"], "complete")
        self.assertEqual(self.biz.docs[0]["latest_wonder_score"], main.compute_wonder_score(rate, 80, 0))

    async def test_finalizing_twice_does_not_duplicate_history(self):
        rid = await self._new_run()
        await main._scan_run_set_leg(rid, "crawl", "ok", score=80)
        await main._scan_run_finalize(rid)
        await main._scan_run_finalize(rid)
        self.assertEqual(len(self.biz.docs[0]["wonder_scores"]), 1)

    async def test_incomplete_tracker_writes_no_score_and_keeps_the_previous_one(self):
        self.biz.docs[0].update({"latest_wonder_score": 55.0, "wonder_scores": [{"run_id": "old", "score": 55.0}]})
        rid = await self._new_run("manual_tracker", ("tracker",))
        await main._scan_run_set_leg(rid, "tracker", "incomplete")
        await main._scan_run_finalize(rid)
        self.assertIsNone(self.runs.docs[0]["wonder_score"])
        self.assertEqual(self.biz.docs[0]["latest_wonder_score"], 55.0)
        self.assertEqual(len(self.biz.docs[0]["wonder_scores"]), 1)

    async def test_a_successful_retry_finally_produces_the_score(self):
        rid = await self._new_run("manual_tracker", ("tracker",))
        await main._scan_run_set_leg(rid, "tracker", "incomplete")
        await main._scan_run_finalize(rid)
        await main._scan_run_restart(rid)
        await main._scan_run_set_leg(rid, "tracker", "ok", mention_rate=40.0)
        await main._scan_run_finalize(rid)
        self.assertEqual(self.runs.docs[0]["status"], "complete")
        self.assertEqual(self.biz.docs[0]["latest_wonder_score"], main.compute_wonder_score(40.0, 91, 0))

    async def test_a_failed_plan_leg_never_blocks_the_score(self):
        rid = await self._new_run("weekly", ("crawl", "tracker", "plan"))
        await main._scan_run_set_leg(rid, "crawl", "ok", score=91)
        await main._scan_run_set_leg(rid, "tracker", "ok", mention_rate=30.0)
        await main._scan_run_set_leg(rid, "plan", "failed", error="blog failed")
        await main._scan_run_finalize(rid)
        self.assertEqual(self.runs.docs[0]["status"], "complete")
        self.assertEqual(self.biz.docs[0]["latest_wonder_score"], main.compute_wonder_score(30.0, 91, 0))

    async def test_a_failed_website_scan_gives_no_score(self):
        rid = await self._new_run("weekly", ("crawl", "tracker"))
        await main._scan_run_set_leg(rid, "crawl", "failed", error="The site may be unreachable")
        await main._scan_run_set_leg(rid, "tracker", "ok", mention_rate=50.0)
        await main._scan_run_finalize(rid)
        self.assertEqual(self.runs.docs[0]["status"], "incomplete")
        self.assertNotIn("latest_wonder_score", self.biz.docs[0])

    async def test_a_fresh_account_with_only_a_crawl_has_no_score(self):
        self.jobs.docs.clear()  # never ran the tracker
        rid = await self._new_run()
        await main._scan_run_set_leg(rid, "crawl", "ok", score=91)
        await main._scan_run_finalize(rid)
        self.assertIsNone(self.runs.docs[0]["wonder_score"])
        self.assertNotIn("latest_wonder_score", self.biz.docs[0])

    async def test_anonymous_callers_get_no_run(self):
        self.assertIsNone(await main._scan_run_create(business_id=None, user_id=None, url="x", trigger="manual_tracker", legs=["tracker"]))

    async def test_existing_business_is_backfilled_once_not_left_blank(self):
        docs = list(self.biz.docs)
        await main._ensure_wonder_scores(docs)
        first = self.biz.docs[0].get("latest_wonder_score")
        self.assertIsInstance(first, float)
        await main._ensure_wonder_scores(list(self.biz.docs))
        self.assertEqual(len(self.biz.docs[0]["wonder_scores"]), 1)
        self.assertEqual(self.biz.docs[0]["latest_wonder_score"], first)

    async def test_business_with_nothing_measured_gets_no_backfill(self):
        other = {"_id": ObjectId(), "user_id": "u1"}
        self.biz.docs.append(other)
        await main._ensure_wonder_scores([other])
        self.assertNotIn("latest_wonder_score", other)


class TrackerJobDrivesTheRunTests(unittest.IsolatedAsyncioTestCase):
    """The real _process_phase5_job, with mocked models."""

    @staticmethod
    def _prov(kind, m=False):
        if kind == "failed":
            return {"mentioned": False, "sources": [], "cited": False, "status": "Not Mentioned",
                    "fetch_status": "failed", "error": "OpenAI rate limit reached."}
        return {"mentioned": m, "position": 2 if m else None, "sources": ["x.com"], "cited": True,
                "status": "Mentioned" if m else "Not Mentioned", "fetch_status": "ok"}

    async def _run(self, dead):
        bid = ObjectId()
        biz = FakeCol([{"_id": bid, "latest_phase1_score": 91, "user_id": "u1"}])
        jobs, runs = FakeCol(), FakeCol()
        job = {"job_id": "j9", "job_type": "core", "model": "multi", "url": "https://example.com", "user_id": "u1",
               "business_id": str(bid), "status": "running", "seed_results": {},
               "questions": [{"id": str(i), "text": f"q{i}", "category": "branded" if i <= 5 else "non-branded"} for i in range(1, 21)]}
        jobs.docs.append(job)

        async def fake_multi(url, question, include_competitors=False):
            m = int(question["id"]) % 4 == 0
            return {"id": question["id"], "providers": {p: self._prov("failed" if p in dead else "ok", m) for p in PROVIDERS}}

        with mock.patch.object(main, "businesses_col", biz), mock.patch.object(main, "phase5_jobs_col", jobs), \
                mock.patch.object(main, "scan_runs_col", runs), mock.patch.object(main, "analyze_single_question_multi", fake_multi), \
                mock.patch.object(main, "generate_deep_competitor_scores", mock.AsyncMock(return_value=[])), \
                mock.patch.object(main, "generate_brand_perception_summary", mock.AsyncMock(return_value=None)), \
                mock.patch.object(main, "_upsert_user_business", mock.AsyncMock(return_value=None)), \
                mock.patch.object(main, "_log_ai_usage_event", mock.AsyncMock()):
            # exactly what POST /api/phase5/start-job does
            run_id = await main._scan_run_create(business_id=str(bid), user_id="u1", url="x", trigger="manual_tracker", legs=["tracker"])
            job["scan_run_id"], job["scan_run_finalize"] = run_id, True
            await main._scan_run_set_leg(run_id, "tracker", "running", job_id="j9")
            self.assertEqual(runs.docs[0]["status"], "running")
            await main._process_phase5_job(job)
        return runs.docs[0], biz.docs[0], job

    async def test_healthy_run_completes_with_a_score(self):
        run, biz, job = await self._run(set())
        self.assertEqual((run["status"], job["status"]), ("complete", "completed"))
        self.assertIsInstance(biz["latest_wonder_score"], float)

    async def test_one_model_down_still_completes(self):
        run, biz, job = await self._run({"gemini"})
        self.assertEqual(run["status"], "complete")
        self.assertIn("latest_wonder_score", biz)

    async def test_the_real_incident_chatgpt_and_gemini_down_gives_no_score(self):
        run, biz, job = await self._run({"chatgpt", "gemini"})
        self.assertEqual(job["status"], "incomplete")
        self.assertNotEqual(run["status"], "complete")
        self.assertNotIn("latest_wonder_score", biz)


if __name__ == "__main__":
    unittest.main()
