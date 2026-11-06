"""evaluation.py testleri — F13, F15, F17, F18 (dogrulama.json tekrarları)."""

import unittest

from pipeline import evaluation as E
from pipeline.evaluation import DetectionRecord as DR


class TestDetectionF13(unittest.TestCase):
    def test_raw_threshold_no_rounding_flip(self):
        """9.96 mm ham negatif; 10.0'a yuvarlanıp pozitif OLMAMALI (F13)."""
        self.assertFalse(E.is_positive(9.96, 10.0))
        self.assertTrue(E.is_positive(10.0, 10.0))
        self.assertTrue(E.is_positive(10.01, 10.0))

    def test_skipped_counted_separately(self):
        recs = [DR(True, 25.0), DR(True, 5.0), DR(False, 2.0), DR(True, None)]
        m = E.detection_metrics(recs)
        self.assertEqual(m["TP"], 1)
        self.assertEqual(m["FN"], 1)
        self.assertEqual(m["TN"], 1)
        self.assertEqual(m["atlanan"], 1)          # None ayrı sayıldı (F18)
        self.assertEqual(m["degerlendirilen"], 3)

    def test_wilson_ci_bounds(self):
        lo, hi = E.wilson_ci(17, 20)
        self.assertTrue(0 <= lo <= hi <= 1)


class TestParseF17(unittest.TestCase):
    def test_decimal_preserved(self):
        ms = E.extract_measurements("Aorta: distance 41.2 mm")
        self.assertEqual(len(ms), 1)
        self.assertAlmostEqual(ms[0].value, 41.2)
        self.assertEqual(ms[0].category, "length")

    def test_sentence_split_keeps_decimal(self):
        s = E.split_sentences("Lezyon 41.2 mm. Başka bulgu yok")
        self.assertEqual(s, ["Lezyon 41.2 mm", "Başka bulgu yok"])

    def test_cm_to_mm(self):
        ms = E.extract_measurements("çap 2 cm")
        self.assertAlmostEqual(ms[0].value, 20.0)
        self.assertEqual(ms[0].category, "length")

    def test_volume_unit_is_volume(self):
        ms = E.extract_measurements("hacim 2 cm^3")
        self.assertEqual(ms[0].category, "volume")
        self.assertAlmostEqual(ms[0].value, 2.0)


class TestFidelityF15(unittest.TestCase):
    def test_wrong_context_not_credited(self):
        """Tümör çapı 20; '20' yalnız aorta mesafe cümlesinde -> sadık DEĞİL (F15)."""
        findings = {"en_buyuk_cap_lezyon": {"cap_mm": 20.0}}
        report = "Aorta mesafe 20 mm olarak ölçüldü"
        r = E.numeric_fidelity(report, findings)
        self.assertEqual(r.supported, 0)
        self.assertIn("en_buyuk_cap", r.missing)

    def test_wrong_unit_not_credited(self):
        """'2 cm^3' hacim; tümör çapı (length) olarak sayılmamalı (F15)."""
        findings = {"en_buyuk_cap_lezyon": {"cap_mm": 20.0}}
        report = "Lezyonun çapı bölgesinde 2 cm^3 ifadesi"
        r = E.numeric_fidelity(report, findings)
        self.assertEqual(r.supported, 0)

    def test_correct_context_credited(self):
        findings = {"en_buyuk_cap_lezyon": {"cap_mm": 20.0, "hacim_cm3": 8.4}}
        report = ("En büyük lezyonun aksiyel çapı 20 mm. "
                  "Tümör hacmi 8.4 cm3 olarak hesaplandı.")
        r = E.numeric_fidelity(report, findings)
        self.assertEqual(r.supported, 2)
        self.assertEqual(r.total, 2)

    def test_vessel_distance_right_vessel(self):
        findings = {"damar": {"aorta": {"mesafe_mm": 28.4}}}
        report = "Aorta ile mesafe 28.4 mm."
        r = E.numeric_fidelity(report, findings)
        self.assertEqual(r.supported, 1)


class TestAgreementF18(unittest.TestCase):
    def test_denominators_explicit(self):
        pairs = [(20.0, 22.0), (30.0, 28.0), (15.0, 15.0)]
        r = E.measurement_agreement(pairs, n_skipped=5)
        self.assertEqual(r["n_eslesen"], 3)
        self.assertEqual(r["n_atlanan"], 5)       # gizli paydaya katılmadı
        self.assertAlmostEqual(r["MAE"], (2 + 2 + 0) / 3, places=2)
        self.assertAlmostEqual(r["yanlilik"], (-2 + 2 + 0) / 3, places=2)

    def test_empty_pairs(self):
        r = E.measurement_agreement([], n_skipped=3)
        self.assertIsNone(r["MAE"])
        self.assertEqual(r["n_atlanan"], 3)


if __name__ == "__main__":
    unittest.main()
