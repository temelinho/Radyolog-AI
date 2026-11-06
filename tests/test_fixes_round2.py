"""Round-2 düzeltme testleri (Codex) — #1 izolasyon, #3 fold yapısı, #4 0-byte."""

import tempfile
import unittest
from pathlib import Path

from pipeline import segmentation as S
from pipeline import run as R
from pipeline import manifest as man
from pipeline.config import Status


class TestOutputValidationZeroByte(unittest.TestCase):
    def test_zero_byte_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / "bos.nii.gz").write_bytes(b"")       # 0 byte
            self.assertFalse(S.validate_nnunet_output(t)["ok"])

    def test_nonzero_accepted(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / "dolu.nii.gz").write_bytes(b"xxxx")
            r = S.validate_nnunet_output(t)
            self.assertTrue(r["ok"])
            self.assertEqual(len(r["maskeler"]), 1)


class TestFoldNestedStructure(unittest.TestCase):
    def test_v1_deep_structure(self):
        """v1 …/Task/nnUNetTrainerV2__Plans/fold_X — derinlemesine bulunmalı (#3)."""
        with tempfile.TemporaryDirectory() as t:
            task = "Task003_Liver"
            deep = Path(t) / task / "nnUNetTrainerV2__nnUNetPlansv2.1"
            for f in range(5):
                d = deep / f"fold_{f}"
                d.mkdir(parents=True)
                (d / "model_final_checkpoint.model").write_bytes(b"x")
            r = S.check_folds_present(task, results_dir=Path(t))
            self.assertTrue(r["ok"], r)

    def test_missing_deep_fold(self):
        with tempfile.TemporaryDirectory() as t:
            task = "Task007_Pancreas"
            deep = Path(t) / task / "trainer"
            for f in (0, 1, 2):
                d = deep / f"fold_{f}"; d.mkdir(parents=True)
                (d / "checkpoint_final.pth").write_bytes(b"x")
            r = S.check_folds_present(task, results_dir=Path(t))
            self.assertFalse(r["ok"])
            self.assertEqual(r["eksik_foldlar"], [3, 4])


class TestSegmentCaseIsolation(unittest.TestCase):
    """#1: nnU-Net'e izole tek vaka verilir; çıktı study_id ile seçilir."""

    def test_correct_case_mask_selected(self):
        with tempfile.TemporaryDirectory() as t:
            # Vaka girdi dosyası
            ct = Path(t) / "AC_B.nii.gz"
            ct.write_bytes(b"ctdata")
            case = man.Case(patient_id="AC_B", study_id="AC_B", task="Task003_Liver",
                            organ="karaciger", group="POS", input_path=str(ct))

            calls = {}

            def fake_nnunet(in_dir, out_dir, task, **k):
                # Gerçek nnU-Net gibi: in_dir'deki <id>_0000.nii.gz -> <id>.nii.gz
                staged = list(Path(in_dir).glob("*_0000.nii.gz"))
                calls["staged"] = [p.name for p in staged]
                sid = staged[0].name.replace("_0000.nii.gz", "")
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                (Path(out_dir) / f"{sid}.nii.gz").write_bytes(b"mask")
                return S.SegResult(Status.SUCCESS, out_dir, {"maskeler": [f"{sid}.nii.gz"]})

            def fake_totalseg(path, out_dir, **k):
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                return S.SegResult(Status.SUCCESS, out_dir, {})

            orig_nn, orig_ts = R.segmentation.run_nnunet, R.segmentation.run_totalsegmentator
            R.segmentation.run_nnunet = fake_nnunet
            R.segmentation.run_totalsegmentator = fake_totalseg
            try:
                res = R.segment_case(case, Path(t) / "work")
            finally:
                R.segmentation.run_nnunet = orig_nn           # orijinalleri geri yükle
                R.segmentation.run_totalsegmentator = orig_ts

            self.assertEqual(res["status"], Status.SUCCESS)
            self.assertTrue(res["mask_path"].endswith("AC_B.nii.gz"))  # DOĞRU vaka
            self.assertEqual(calls["staged"], ["AC_B_0000.nii.gz"])    # izole tek dosya


if __name__ == "__main__":
    unittest.main()
