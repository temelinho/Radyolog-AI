"""manifest.py + runner.py testleri — F12, F10."""

import json
import tempfile
import unittest
from pathlib import Path

from pipeline import manifest as MAN
from pipeline import runner as R
from pipeline.config import Status


class TestManifestF12(unittest.TestCase):
    def test_from_pos_neg_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pan.json"
            p.write_text(json.dumps({"POS": ["a", "b"], "NEG": ["c"]}), encoding="utf-8")
            cases = MAN.from_pos_neg_json(p, organ="pankreas", task="Task007")
            self.assertEqual(len(cases), 3)
            self.assertEqual(sum(c.group == "POS" for c in cases), 2)
            self.assertEqual(sum(c.group == "NEG" for c in cases), 1)
            self.assertTrue(all(c.task == "Task007" for c in cases))

    def test_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            cases = [MAN.Case("p1", "p1", "Task003", "karaciger", "POS"),
                     MAN.Case("p2", "p2", "Task007", "pankreas", "NEG", role="test")]
            csv_path = Path(d) / "cohort.csv"
            MAN.save_csv(cases, csv_path)
            loaded = MAN.load_csv(csv_path)
            self.assertEqual(loaded, cases)

    def test_patient_collision(self):
        cases = [MAN.Case("P", "s1", "Task003", "karaciger", "POS"),
                 MAN.Case("P", "s2", "Task007", "pankreas", "POS"),
                 MAN.Case("Q", "s3", "Task003", "karaciger", "NEG")]
        coll = MAN.find_patient_collisions(cases)
        self.assertIn("P", coll)
        self.assertNotIn("Q", coll)

    def test_sha256_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.bin"
            p.write_bytes(b"merlin-ct-bytes")
            self.assertEqual(MAN.file_sha256(p), MAN.file_sha256(p))


class TestRunnerF10(unittest.TestCase):
    def test_cache_key_sensitivity(self):
        k1 = R.cache_key(input_hash="aaa", weight_hash="w")
        k2 = R.cache_key(input_hash="aaa", weight_hash="w")
        k3 = R.cache_key(input_hash="bbb", weight_hash="w")
        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, k3)

    def test_atomic_write_and_complete(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "bulgular.json"
            key = R.cache_key(input_hash="h1")
            R.atomic_write_json(out, {"organ": "karaciger"}, key=key)
            self.assertTrue(out.exists())
            self.assertTrue(R.is_complete(out, key))
            # Farklı anahtar -> tamamlanmış sayılmaz (kod/girdi değişti)
            self.assertFalse(R.is_complete(out, R.cache_key(input_hash="h2")))

    def test_missing_meta_not_complete(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "x.json"
            out.write_text('{"a":1}', encoding="utf-8")  # meta yok
            self.assertFalse(R.is_complete(out, "anykey"))

    def test_run_jobs_isolates_failures(self):
        with tempfile.TemporaryDirectory() as d:
            def process(job_id):
                if job_id == "bad":
                    raise RuntimeError("API 500")
                return ({"id": job_id}, Status.SUCCESS, {})
            jobs = [("ok1", Path(d) / "ok1.json", R.cache_key(input_hash="1")),
                    ("bad", Path(d) / "bad.json", R.cache_key(input_hash="2")),
                    ("ok2", Path(d) / "ok2.json", R.cache_key(input_hash="3"))]
            res = R.run_jobs(jobs, process)
            by = {r.job_id: r.status for r in res}
            self.assertEqual(by["ok1"], Status.SUCCESS)
            self.assertEqual(by["ok2"], Status.SUCCESS)   # bad diğerlerini düşürmedi
            self.assertEqual(by["bad"], Status.FAILED)

    def test_run_jobs_resume_skips(self):
        with tempfile.TemporaryDirectory() as d:
            key = R.cache_key(input_hash="1")
            out = Path(d) / "ok1.json"
            calls = []
            def process(job_id):
                calls.append(job_id)
                return ({"id": job_id}, Status.SUCCESS, {})
            jobs = [("ok1", out, key)]
            R.run_jobs(jobs, process)          # ilk koşu yazar
            R.run_jobs(jobs, process)          # ikinci koşu atlamalı
            self.assertEqual(calls.count("ok1"), 1)


if __name__ == "__main__":
    unittest.main()
