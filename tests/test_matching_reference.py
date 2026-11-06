"""matching.py + reference.py testleri — F18 (eşleştirme + GT denetim izi)."""

import tempfile
import unittest
from pathlib import Path

from pipeline import matching as MT
from pipeline import reference as RF
from pipeline.reference import ReferenceMeasurement as RM


LESIONS = [
    {"id": 1, "cap_mm": 30.0, "cap_mm_raw": 30.0, "hacim_cm3": 5.0, "hacim_cm3_raw": 5.0,
     "centroid_ijk": [10, 10, 5]},
    {"id": 2, "cap_mm": 12.0, "cap_mm_raw": 12.0, "hacim_cm3": 20.0, "hacim_cm3_raw": 20.0,
     "centroid_ijk": [40, 40, 8]},
]


class TestMatching(unittest.TestCase):
    def test_policy_diameter_vs_volume(self):
        self.assertEqual(MT.select_target(LESIONS, "largest_diameter")["id"], 1)
        self.assertEqual(MT.select_target(LESIONS, "largest_volume")["id"], 2)

    def test_empty_target_none(self):
        self.assertIsNone(MT.select_target([], "largest_diameter"))

    def test_centroid_match(self):
        near = MT.match_by_centroid(LESIONS, [41, 39, 8], max_dist_voxel=10)
        self.assertEqual(near["id"], 2)
        far = MT.match_by_centroid(LESIONS, [200, 200, 200], max_dist_voxel=10)
        self.assertIsNone(far)

    def test_build_pairs_separates_denominators(self):
        records = [
            {"study_id": "a", "lesions": LESIONS, "reference_mm": 28.0},   # eşleşir
            {"study_id": "b", "lesions": [], "reference_mm": 15.0},         # pipeline yok
            {"study_id": "c", "lesions": LESIONS, "reference_mm": None},    # referans yok
        ]
        res = MT.build_pairs(records, policy="largest_diameter")
        self.assertEqual(res.pairs, [(30.0, 28.0)])
        self.assertEqual(res.skipped_no_pipeline, ["b"])
        self.assertEqual(res.skipped_no_reference, ["c"])
        s = res.summary()
        self.assertEqual(s["n_eslesen"], 1)
        self.assertEqual(s["n_pipeline_yok"], 1)
        self.assertEqual(s["n_referans_yok"], 1)

    def test_build_pairs_centroid_used_when_available(self):
        records = [{"study_id": "a", "lesions": LESIONS,
                    "reference_mm": 11.0, "reference_centroid_ijk": [41, 39, 8]}]
        res = MT.build_pairs(records)
        # Koordinat verildi -> lezyon 2 (çap 12) eşleşmeli, en büyük (30) değil
        self.assertEqual(res.pairs, [(12.0, 11.0)])


class TestReferenceAudit(unittest.TestCase):
    def test_validate_flags(self):
        refs = [
            RM("a", "karaciger", "POS", target_diameter_mm="30",
               source_sentence="Karaciğerde 30 mm lezyon.", report_date="2026-01-01"),
            RM("b", "karaciger", "POS", target_diameter_mm="25",
               source_sentence=""),                     # ölçü var, cümle yok
            RM("c", "pankreas", "POS", target_diameter_mm=""),  # POS ama ölçü yok
            RM("d", "pankreas", "POS", target_diameter_mm="40",
               source_sentence="40 mm", revision="2", uncertainty_note=""),  # revize, not yok
        ]
        v = RF.validate_references(refs)
        self.assertIn("b", v["kaynak_cumlesi_eksik"])
        self.assertIn("c", v["POS_olcusu_eksik"])
        self.assertIn("d", v["gerekcesiz_revizyon"])
        self.assertNotIn("a", v["kaynak_cumlesi_eksik"])

    def test_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as dtmp:
            refs = [RM("a", "karaciger", "POS", target_diameter_mm="30",
                       source_sentence="30 mm lezyon", report_date="2026-01-01",
                       transcriber="kullanici")]
            p = Path(dtmp) / "gt.csv"
            RF.save_csv(refs, p)
            self.assertEqual(RF.load_csv(p), refs)

    def test_reference_mm_parse(self):
        self.assertEqual(RF.reference_mm(RM("a", "k", "POS", target_diameter_mm="12,5")), 12.5)
        self.assertIsNone(RF.reference_mm(RM("a", "k", "NEG", target_diameter_mm="")))


if __name__ == "__main__":
    unittest.main()
