"""compare_reference rapor-beyanı ayıklayıcısı: gerçek model cümlelerinden regresyon testleri.

İlk koşuda damar cümleleri ("distance of the VCI to the tumor is 47.2 mm") ve sıfır
sayıları ("0 hepatic lesions") yanlışlıkla 'lezyon var' sayılıyordu (17 yanlış-pozitif).
"""

import unittest

import compare_reference as C

NEGATIVE = [
    "There are no lesions identified in the pancreas. The total tumor volume is 0.0 cm3. "
    "The distance of the vci to the tumor is 47.2 mm, and its encasement status is not assessed.",
    "FINDINGS: No pancreatic lesions are identified. Total tumor volume is 0.0 cm3. "
    "IMPRESSION AND RECOMMENDATION: No evidence of pancreatic neoplasm is identified on this CT scan.",
    "Number of segmented lesions: 0. Total tumor volume: 0.0 cm3. "
    "The segmentation did not identify any focal pancreatic lesion. No tumor volume was measured.",
    "The aorta is not encased by tumor, with a distance of 28.1 mm and an encasement angle of "
    "0.0 degrees.",
    "FINDINGS: There are 0 hepatic lesions identified. Total tumor volume is 0.0 cm3.",
    "No masses or lesions are identified within the pancreas. - **Aorta:** Distance and "
    "encasement by any pancreatic mass are not assessed.",
]

POSITIVE = [
    ("1 lesion(s) observed in the liver parenchyma.\nThe largest lesion has an axial diameter of "
     "32.0 mm.\n\n3. VASCULAR\naorta: distance 28.4 mm, encasement 0 degrees.", 32.0),
    ("A hepatic mass measures 3.1 cm in diameter. The lesion is 28.4 mm from the aorta.", 31.0),
    ("The largest pancreatic lesion measures 41.2 mm and abuts the portal vein.", 41.2),
    ("Two lesions are present in the liver; the largest measures 25 mm.", 25.0),
]


class TestReportClaims(unittest.TestCase):
    def test_negative_reports(self):
        for t in NEGATIVE:
            with self.subTest(t=t[:50]):
                self.assertEqual(C.report_claims(t), (False, None))

    def test_positive_reports_and_size(self):
        for t, mm in POSITIVE:
            with self.subTest(t=t[:50]):
                pos, got = C.report_claims(t)
                self.assertTrue(pos)
                self.assertAlmostEqual(got, mm, places=2)


class TestOrganNaming(unittest.TestCase):
    def test_wrong_organ_flagged(self):
        q = ("The CT segmentation analysis was performed on the right kidney (karaciger) "
             "without identifying any discrete lesions.")
        self.assertEqual(C.organ_naming(q, "liver"), (0, 1))

    def test_correct_organ(self):
        self.assertEqual(C.organ_naming("No focal hepatic lesion.", "liver"), (1, 0))


class TestSentences(unittest.TestCase):
    def test_split_keeps_decimals_and_digit_endings(self):
        t = "There is a 3.0 cm mass in segment 7. The pancreas is unremarkable."
        self.assertEqual(C.sentences(t),
                         ["There is a 3.0 cm mass in segment 7", "The pancreas is unremarkable"])


if __name__ == "__main__":
    unittest.main()
