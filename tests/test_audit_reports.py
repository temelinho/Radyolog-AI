"""audit_reports: gerçek model cümlelerinden dayanaksız ekleme / eksik bulgu testleri."""

import unittest

import audit_reports as A
from pipeline import reporting as R


def _liver():
    f = R.build_findings(
        organ="karaciger",
        summary={"lezyon_sayisi": 5, "toplam_tumor_hacim_cm3": 45.32,
                 "en_buyuk_cap_lezyon": {"id": 1, "cap_mm": 43.26, "hacim_cm3": 42.07},
                 "en_buyuk_hacim_lezyon": {"id": 1, "cap_mm": 43.26, "hacim_cm3": 42.07}},
        vessels={"aorta": {"mesafe_mm": 31.4, "sarma_derece": 0},
                 "vci": {"mesafe_mm": 17.3, "sarma_derece": 0},
                 "portal_ven": {"mesafe_mm": 11.6, "sarma_derece": 0}})
    return f, R.model_findings(f)


class TestAdditions(unittest.TestCase):
    def test_template_adds_nothing(self):
        f, mf = _liver()
        r = A.denetle(R.template_report(f), mf, f)
        for k in A.EK:
            self.assertEqual(r[k], 0, k)
        for k in A.EKSIK:
            self.assertEqual(r[k], 0, k)

    def test_gemma_invented_diagnosis(self):
        f, mf = _liver()
        t = ("Multifocal hepatic lesions consistent with hepatocellular carcinoma are seen. "
             "Histopathological confirmation is necessary to establish a final diagnosis.")
        self.assertEqual(A.denetle(t, mf, f)["ek_tani"], 1)

    def test_llama_invented_location(self):
        f, mf = _liver()
        t = ("The findings suggest one of multiple lesions in the left liver measuring "
             "43.26 mm in diameter and 42.07 cm3 in volume.")
        self.assertEqual(A.denetle(t, mf, f)["ek_konum_morfoloji"], 1)

    def test_invented_number(self):
        f, mf = _liver()
        t = "Axial images were reconstructed at 5 mm slice thickness. The lesion measures 43.26 mm."
        self.assertEqual(A.denetle(t, mf, f)["ek_sayi"], 1)

    def test_rounded_cm_is_not_invented(self):
        f, mf = _liver()
        t = "The largest lesion measures 4.3 cm; the aorta lies 3.1 cm away."
        self.assertEqual(A.denetle(t, mf, f)["ek_sayi"], 0)

    def test_category_threshold_is_not_invented(self):
        f, mf = _liver()   # liver thresholds 20 and 50 mm
        t = "Size category 2-5 cm (lesions larger than 5 cm form a separate category)."
        self.assertEqual(A.denetle(t, mf, f)["ek_sayi"], 0)

    def test_negated_statements_are_not_additions(self):
        f, mf = _liver()
        t = ("The imaging phase was not specified. Resectability should not be determined from "
             "these measurements. A negative result does not exclude a small cystic lesion.")
        r = A.denetle(t, mf, f)
        self.assertEqual((r["ek_faz"], r["ek_rezektabilite"], r["ek_tani"],
                          r["ek_konum_morfoloji"]), (0, 0, 0, 0))

    def test_portal_venous_vessel_is_not_phase(self):
        f, mf = _liver()
        t = "There is portal venous encasement of 0 degrees and the distance is 11.6 mm."
        self.assertEqual(A.denetle(t, mf, f)["ek_faz"], 0)

    def test_phase_claim_when_phase_null(self):
        f, mf = _liver()
        t = "Contrast-enhanced CT in the portal venous phase was performed."
        self.assertEqual(A.denetle(t, mf, f)["ek_faz"], 1)

    def test_resectability_claim(self):
        f, mf = _liver()
        self.assertEqual(A.denetle("The tumor appears resectable.", mf, f)["ek_rezektabilite"], 1)


class TestOmissions(unittest.TestCase):
    def test_missing_category_count_and_organ(self):
        f, mf = _liver()
        t = "Lesions are present. The largest measures 43.26 mm."
        r = A.denetle(t, mf, f)
        self.assertEqual(r["eksik_boyut_kategorisi"], 1)
        self.assertEqual(r["eksik_lezyon_sayisi"], 1)
        self.assertEqual(r["eksik_organ_adi"], 1)

    def test_count_as_word_and_category(self):
        f, mf = _liver()
        t = "Five hepatic lesions are seen. They fall within the 2–5 cm size category."
        r = A.denetle(t, mf, f)
        self.assertEqual((r["eksik_lezyon_sayisi"], r["eksik_boyut_kategorisi"],
                          r["eksik_organ_adi"]), (0, 0, 0))


class TestRealSentencesRound2(unittest.TestCase):
    """İlk gerçek koşuda görülen yanlış alarmlar (öneri cümleleri, HTML birim, e.g.)."""

    def _r(self, t):
        f, mf = _liver()
        return A.denetle(t, mf, f)

    def test_recommendations_are_not_additions(self):
        for t in [
            "Further imaging with a contrast-enhanced CT or MRI is recommended to assess for pancreatic pathology.",
            "If clinical suspicion for hepatic pathology persists, consider dedicated multiphase contrast-enhanced imaging or MRI as clinically appropriate.",
            "Histopathological confirmation is recommended to assess for malignancy.",
            "The distance to major vessels should be cross-referenced to further evaluate for resectability.",
            "Management decisions, including resectability, should be made in multidisciplinary tumor board discussion.",
        ]:
            r = self._r(t)
            with self.subTest(t=t[:40]):
                self.assertEqual((r["ek_faz"], r["ek_tani"], r["ek_rezektabilite"]), (0, 0, 0))

    def test_eg_split_keeps_negation(self):
        t = ("Lesion morphology, enhancement pattern, background liver parenchyma (e.g. cirrhosis) "
             "and lesion location are not provided in the metrics.")
        self.assertEqual(self._r(t)["ek_konum_morfoloji"], 0)

    def test_html_superscript_volume(self):
        r = self._r("The total tumor volume is 45.32 cm<sup>3</sup>.")
        self.assertEqual(r["ek_sayi"], 0)

    def test_zero_comparison_not_invented(self):
        r = self._r("The distances from these vessels are all greater than 0 mm.")
        self.assertEqual(r["ek_sayi"], 0)

    def test_real_additions_still_caught(self):
        self.assertEqual(self._r("Two hypodense liver lesions are present.")["ek_konum_morfoloji"], 1)
        self.assertEqual(self._r("CT segmentation reveals a single enhancing lesion.")["ek_konum_morfoloji"], 1)
        self.assertEqual(self._r("The segmentation identifies a significant hepatocellular neoplasm.")["ek_tani"], 1)
        self.assertEqual(self._r("The largest lesion has a diameter of 24.88 mm.")["ek_sayi"], 1)
        self.assertEqual(self._r("The tumor is borderline resectable.")["ek_rezektabilite"], 1)


class TestRealSentencesRound3(unittest.TestCase):
    def _r(self, t):
        f, mf = _liver()
        return A.denetle(t, mf, f)

    def test_derived_number_is_not_invented(self):
        # toplam 45.32 - en büyük 42.07 = 3.25 cm3
        r = self._r("By subtraction, the remaining lesions account for approximately 3.25 cm3.")
        self.assertEqual((r["ek_sayi"], r["turetilmis_sayi"]), (0, 1))

    def test_required_confirmation_is_recommendation(self):
        r = self._r("Histopathological confirmation would be required for any suspected malignancy.")
        self.assertEqual(r["ek_tani"], 0)

    def test_real_diagnosis_still_caught(self):
        self.assertEqual(self._r("The findings are consistent with multifocal liver metastases.")["ek_tani"], 1)


if __name__ == "__main__":
    unittest.main()
