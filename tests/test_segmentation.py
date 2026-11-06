"""segmentation.py testleri — C8/C10/C11 (GPU gerekmez, subprocess mock)."""

import tempfile
import unittest
from pathlib import Path

from pipeline import segmentation as S
from pipeline.config import Status, SEGMENTATION


class FakeProc:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr


def _make_model(tmp, task, folds=(0, 1, 2, 3, 4)):
    md = Path(tmp) / task
    for f in folds:
        d = md / f"fold_{f}"
        d.mkdir(parents=True)
        (d / "checkpoint_final.pth").write_bytes(b"x")
    return Path(tmp)


class TestFoldCheckC8(unittest.TestCase):
    def test_all_folds_present(self):
        with tempfile.TemporaryDirectory() as t:
            res = _make_model(t, "Task003_Liver")
            r = S.check_folds_present("Task003_Liver", results_dir=res)
            self.assertTrue(r["ok"])
            self.assertEqual(r["eksik_foldlar"], [])

    def test_missing_fold_detected(self):
        with tempfile.TemporaryDirectory() as t:
            res = _make_model(t, "Task007_Pancreas", folds=(0, 1, 2))  # 3,4 eksik
            r = S.check_folds_present("Task007_Pancreas", results_dir=res)
            self.assertFalse(r["ok"])
            self.assertEqual(r["eksik_foldlar"], [3, 4])

    def test_command_uses_five_folds(self):
        cmd = S.build_nnunet_command("in", "out", "Task003_Liver")
        # -f 0 1 2 3 4 olmalı; tek fold DEĞİL (C8)
        i = cmd.index("-f")
        self.assertEqual(cmd[i + 1:i + 6], ["0", "1", "2", "3", "4"])


class TestRunNnunetC10(unittest.TestCase):
    def test_missing_folds_failed(self):
        with tempfile.TemporaryDirectory() as t:
            res = _make_model(t, "Task003_Liver", folds=(0, 1))
            r = S.run_nnunet("in", str(Path(t) / "out"), "Task003_Liver", results_dir=res)
            self.assertEqual(r.status, Status.FAILED)
            self.assertEqual(r.detail["neden"], "eksik_fold")

    def test_nonzero_exit_failed(self):
        with tempfile.TemporaryDirectory() as t:
            res = _make_model(t, "Task003_Liver")
            out = Path(t) / "out"
            r = S.run_nnunet("in", str(out), "Task003_Liver", results_dir=res,
                             _runner=lambda *a, **k: FakeProc(returncode=1, stderr="boom"))
            self.assertEqual(r.status, Status.FAILED)
            self.assertEqual(r.detail["neden"], "cikis_kodu")

    def test_success_when_output_present(self):
        with tempfile.TemporaryDirectory() as t:
            res = _make_model(t, "Task003_Liver")
            out = Path(t) / "out"

            def runner(cmd, **k):
                out.mkdir(parents=True, exist_ok=True)
                (out / "vaka.nii.gz").write_bytes(b"m")
                return FakeProc(returncode=0)

            r = S.run_nnunet("in", str(out), "Task003_Liver", results_dir=res, _runner=runner)
            self.assertEqual(r.status, Status.SUCCESS)

    def test_exit_zero_but_no_output_failed(self):
        with tempfile.TemporaryDirectory() as t:
            res = _make_model(t, "Task003_Liver")
            out = Path(t) / "out"
            r = S.run_nnunet("in", str(out), "Task003_Liver", results_dir=res,
                             _runner=lambda *a, **k: FakeProc(returncode=0))
            self.assertEqual(r.status, Status.FAILED)
            self.assertEqual(r.detail["neden"], "cikti_yok")


class TestTotalSegC11(unittest.TestCase):
    def test_return_value_used_nonzero(self):
        with tempfile.TemporaryDirectory() as t:
            r = S.run_totalsegmentator("ct.nii.gz", str(Path(t) / "v"),
                                       _runner=lambda *a, **k: FakeProc(returncode=2, stderr="x"))
            self.assertEqual(r.status, Status.FAILED)  # dönüş AÇIKÇA kullanıldı (C11)

    def test_exit_zero_no_roi_is_empty(self):
        with tempfile.TemporaryDirectory() as t:
            r = S.run_totalsegmentator("ct.nii.gz", str(Path(t) / "v"),
                                       _runner=lambda *a, **k: FakeProc(returncode=0))
            self.assertEqual(r.status, Status.EMPTY_MASK)

    def test_success_when_roi_present(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "v"

            def runner(cmd, **k):
                out.mkdir(parents=True, exist_ok=True)
                (out / f"{SEGMENTATION.totalseg_roi[0]}.nii.gz").write_bytes(b"m")
                return FakeProc(returncode=0)

            r = S.run_totalsegmentator("ct.nii.gz", str(out), _runner=runner)
            self.assertEqual(r.status, Status.SUCCESS)


if __name__ == "__main__":
    unittest.main()
