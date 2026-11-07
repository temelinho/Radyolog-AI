"""select_cohort.py classify_finding testleri — sentetik raporlar (gerçek veri yok)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from select_cohort import classify_finding  # noqa: E402
from finalize_cohort import finalize_from_rows  # noqa: E402
from extract_diameters import extract_diameter  # noqa: E402


class TestClassifyLiver(unittest.TestCase):
    def c(self, text):
        return classify_finding(text, "liver")[0]

    def test_pos_mass(self):
        self.assertEqual(self.c("There is a hepatic mass measuring 3 cm in the right lobe."), "POS")

    def test_pos_metastasis(self):
        self.assertEqual(self.c("Multiple liver metastases are present."), "POS")

    def test_neg_normal(self):
        self.assertEqual(self.c("The liver is normal and unremarkable in appearance."), "NEG")

    def test_neg_benign_cyst(self):
        self.assertEqual(self.c("A simple hepatic cyst is noted, otherwise unremarkable."), "NEG")

    def test_indeterminate_hedge(self):
        self.assertEqual(self.c("Hepatic lesion; cannot exclude malignancy, follow-up recommended."),
                         "INDETERMINATE")

    def test_negation_not_pos(self):
        # Olumsuzlanmış ifade POZİTİF sayılmamalı (güvenlik)
        self.assertNotEqual(self.c("No evidence of hepatic metastasis."), "POS")
        self.assertEqual(self.c("No hepatic mass or focal lesion."), "NEG")

    def test_skip_no_organ(self):
        self.assertEqual(self.c("The lungs are clear. No pleural effusion."), "SKIP")


class TestClassifyPancreas(unittest.TestCase):
    def c(self, text):
        return classify_finding(text, "pancreas")[0]

    def test_pos_adenoca(self):
        self.assertEqual(self.c("Pancreatic adenocarcinoma in the head of the pancreas."), "POS")

    def test_neg_normal(self):
        self.assertEqual(self.c("The pancreas is normal without focal lesion."), "NEG")

    def test_indeterminate(self):
        self.assertEqual(self.c("Pancreatic mass, indeterminate, recommend MRI correlation."),
                         "INDETERMINATE")

    def test_benign_explainer_not_pos(self):
        # Kitle terimi geçse de apse/yağ/atrofi açıklaması → POS değil (kullanıcı geri bildirimi)
        self.assertNotEqual(self.c("Pancreatic mass, likely an abscess."), "POS")
        self.assertNotEqual(self.c("Fatty infiltration of the pancreas, no discrete mass."), "POS")
        self.assertNotEqual(self.c("Pancreatic atrophy; mass not identified."), "POS")


class TestFinalize(unittest.TestCase):
    ROWS = [
        {"draw_order": "5", "study id": "P3", "onay_pos_neg": "POS", "cap_mm": "40", "kaynak_cumle": "m3"},
        {"draw_order": "1", "study id": "P1", "onay_pos_neg": "POS", "cap_mm": "20", "kaynak_cumle": "m1"},
        {"draw_order": "2", "study id": "N1", "onay_pos_neg": "NEG"},
        {"draw_order": "3", "study id": "P2", "onay_pos_neg": "POS", "cap_mm": "", "kaynak_cumle": "m2"},
        {"draw_order": "4", "study id": "X", "onay_pos_neg": ""},   # onaysız → atlanır
        {"draw_order": "6", "study id": "N2", "onay_pos_neg": "NEG"},
        {"draw_order": "7", "study id": "N3", "onay_pos_neg": "NEG"},
    ]

    def test_locks_first_target_by_draw_order(self):
        cases, gt, uyari = finalize_from_rows(self.ROWS, "liver", target=2)
        self.assertEqual(cases["POS"], ["P1", "P2"])   # draw_order sırası; P3 dışarıda
        self.assertEqual(cases["NEG"], ["N1", "N2"])   # N3 dışarıda
        self.assertEqual(len(gt), 4)

    def test_warns_missing_diameter(self):
        _, _, uyari = finalize_from_rows(self.ROWS, "liver", target=2)
        self.assertTrue(any("P2" in u for u in uyari))  # POS ama cap boş

    def test_pool_short_warns(self):
        cases, _, uyari = finalize_from_rows(self.ROWS, "liver", target=20)
        self.assertTrue(any("POS" in u and "yetmedi" in u for u in uyari))


class TestExtractDiameter(unittest.TestCase):
    def test_x_notation_takes_longest(self):
        cap, s = extract_diameter("There is a hepatic mass measuring 2.7 x 1.9 cm.", "liver")
        self.assertEqual(cap, 27.0)              # 2.7 cm = 27 mm (en uzun eksen)
        self.assertIn("mass", s.lower())

    def test_mm(self):
        cap, _ = extract_diameter("27 mm liver lesion in segment 7.", "liver")
        self.assertEqual(cap, 27.0)

    def test_pancreas_cm(self):
        cap, _ = extract_diameter("Pancreatic adenocarcinoma, 3.5 cm in the head.", "pancreas")
        self.assertEqual(cap, 35.0)

    def test_none_when_no_measure(self):
        cap, s = extract_diameter("The liver is normal and unremarkable.", "liver")
        self.assertIsNone(cap)
        self.assertEqual(s, "")


if __name__ == "__main__":
    unittest.main()
