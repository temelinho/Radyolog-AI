"""run.py testleri — findings_for_masks (GPU-suz uçtan uca çekirdek, C34)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import nibabel as nib

from pipeline import run as R
from pipeline.config import SCHEMA_VERSION


class TestFindingsForMasks(unittest.TestCase):
    def test_mask_to_findings(self):
        with tempfile.TemporaryDirectory() as t:
            # Tümör maskesi: 1=organ, 2=tümör (kompakt küp)
            d = np.zeros((40, 40, 20), dtype=np.uint8)
            d[5:35, 8:16, 4:16] = 1
            d[10:20, 9:16, 6:14] = 2       # tümör (>100 voxel, hacim eşiğini aşar)
            aff = np.eye(4)
            mp = Path(t) / "tumor.nii.gz"
            nib.save(nib.Nifti1Image(d, aff), str(mp))

            # Damar: aorta (tümöre bitişik)
            vdir = Path(t) / "vessels"
            vdir.mkdir()
            v = np.zeros((40, 40, 20), dtype=np.uint8)
            v[20:24, 9:16, 6:14] = 1
            nib.save(nib.Nifti1Image(v, aff), str(vdir / "aorta.nii.gz"))

            findings = R.findings_for_masks(str(mp), "karaciger", str(vdir))

            self.assertEqual(findings["schema_version"], SCHEMA_VERSION)
            self.assertEqual(findings["organ"], "karaciger")
            self.assertGreaterEqual(findings["lezyon_sayisi"], 1)
            self.assertIn("aorta", findings["damar"])
            # Çap aksiyel, pozitif
            enb = findings["en_buyuk_cap_lezyon"]
            self.assertIsNotNone(enb)
            self.assertGreater(enb["cap_mm"], 0)

    def test_no_vessel_dir(self):
        with tempfile.TemporaryDirectory() as t:
            d = np.zeros((20, 20, 10), dtype=np.uint8)
            d[5:9, 5:9, 3:7] = 2
            nib.save(nib.Nifti1Image(d, np.eye(4)), str(Path(t) / "m.nii.gz"))
            findings = R.findings_for_masks(str(Path(t) / "m.nii.gz"), "pankreas")
            self.assertEqual(findings["damar"], {})
            self.assertIn("boyut_kategorisi", findings)


if __name__ == "__main__":
    unittest.main()
