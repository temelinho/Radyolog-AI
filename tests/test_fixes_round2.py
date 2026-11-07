"""Round-2 düzeltme testleri (Codex) — #1 izolasyon, #3 fold yapısı, #4 0-byte."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

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
            deep = Path(t) / "nnUNet" / "3d_fullres" / task / "nnUNetTrainerV2__nnUNetPlansv2.1"
            for f in range(5):
                d = deep / f"fold_{f}"
                d.mkdir(parents=True)
                (d / "model_final_checkpoint.model").write_bytes(b"x")
                (d / "model_final_checkpoint.model.pkl").write_bytes(b"x")
            r = S.check_folds_present(task, results_dir=Path(t))
            self.assertTrue(r["ok"], r)

    def test_missing_deep_fold(self):
        with tempfile.TemporaryDirectory() as t:
            task = "Task007_Pancreas"
            deep = Path(t) / "nnUNet" / "3d_fullres" / task / "nnUNetTrainerV2__nnUNetPlansv2.1"
            for f in (0, 1, 2):
                d = deep / f"fold_{f}"; d.mkdir(parents=True)
                (d / "model_final_checkpoint.model").write_bytes(b"x")
                (d / "model_final_checkpoint.model.pkl").write_bytes(b"x")
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
                nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), np.eye(4)),
                         Path(out_dir) / f"{sid}.nii.gz")
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

    def test_previous_attempt_mask_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ct = root / "B.nii.gz"
            ct.write_bytes(b"ct")
            case = man.Case(patient_id="B", study_id="B", task="Task003_Liver",
                            organ="karaciger", group="POS", input_path=str(ct))
            work = root / "work"

            def first_prediction(_in_dir, out_dir, _task):
                Path(out_dir).mkdir()
                nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), np.eye(4)),
                         Path(out_dir) / "B.nii.gz")
                return S.SegResult(Status.SUCCESS)

            with patch.object(S, "run_totalsegmentator", return_value=S.SegResult(Status.SUCCESS)) as vessels:
                with patch.object(S, "run_nnunet", side_effect=first_prediction):
                    first = R.segment_case(case, work)
                self.assertEqual(first["status"], Status.SUCCESS)
                with patch.object(S, "run_nnunet", return_value=S.SegResult(Status.SUCCESS)):
                    second = R.segment_case(case, work)
                self.assertEqual(second["detail"]["neden"], "beklenen_maske_gecersiz")
                self.assertTrue(Path(first["mask_path"]).exists())
                vessels.assert_called_once()


if __name__ == "__main__":
    unittest.main()
