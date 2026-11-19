"""Sayısal sadakat: GERÇEK model raporlarından regresyon testleri.

İlk İngilizce JSON koşusunda Gemma'nın tamamen doğru bir raporu 0/9 aldı: birimler
"millimeters" / "cubic centimeters" diye açık yazılmıştı ve 0° sarma "no encasement
noted" diye ifade edilmişti. Ölçüm bunları tanımıyordu (model hatası değil).
"""

import unittest

import compare_reference as C
from pipeline import evaluation as E
from pipeline import reporting as R

GEMMA_REPORT = """1. TECHNIQUE:

Computed tomography

2. FINDINGS:

CT segmentation of the liver reveals five hepatic lesions, the largest measuring 43.26 millimeters in axial diameter and 42.07 cubic centimeters in volume.  The total tumor volume is 45.32 cubic centimeters.  The lesions fall within the 2–5 cm size category.

3. VASCULAR:

The aorta is 31.4 millimeters from the largest hepatic lesion, with no encasement noted.  The inferior vena cava is 17.3 millimeters from the largest hepatic lesion, with no encasement noted.  The portal vein and splenic vein are 11.6 millimeters from the largest hepatic lesion, with no encasement noted.
"""


def _findings():
    return R.build_findings(
        organ="karaciger",
        summary={"lezyon_sayisi": 5, "toplam_tumor_hacim_cm3": 45.32,
                 "en_buyuk_cap_lezyon": {"id": 1, "cap_mm": 43.26, "hacim_cm3": 42.07},
                 "en_buyuk_hacim_lezyon": {"id": 1, "cap_mm": 43.26, "hacim_cm3": 42.07}},
        vessels={"aorta": {"mesafe_mm": 31.4, "sarma_derece": 0},
                 "vci": {"mesafe_mm": 17.3, "sarma_derece": 0},
                 "portal_ven": {"mesafe_mm": 11.6, "sarma_derece": 0}})


class TestFidelityRealReports(unittest.TestCase):
    def test_gemma_spelled_units_and_no_encasement(self):
        fid = E.numeric_fidelity(GEMMA_REPORT, _findings())
        self.assertEqual((fid.supported, fid.total), (9, 9), fid.missing)

    def test_spelled_units_parsed(self):
        m = {(x.category, x.value) for x in E.extract_measurements(
            "43.26 millimeters, 42.07 cubic centimeters, 3.1 centimeters, 5 milliliters")}
        self.assertIn(("length", 43.26), m)
        self.assertIn(("volume", 42.07), m)
        self.assertIn(("length", 31.0), m)
        self.assertIn(("volume", 5.0), m)

    def test_adjective_vessel_names(self):
        f = R.build_findings(organ="pankreas", summary={"lezyon_sayisi": 1},
                             vessels={"aorta": {"mesafe_mm": 28.4, "sarma_derece": 0}})
        t = "The lesion lies 28.4 mm from the aortic wall, without aortic encasement."
        fid = E.numeric_fidelity(t, f)
        self.assertEqual((fid.supported, fid.total), (2, 2), fid.missing)

    def test_dimensionless_angle(self):
        f = R.build_findings(organ="pankreas", summary={"lezyon_sayisi": 1},
                             vessels={"aorta": {"mesafe_mm": None, "sarma_derece": 120}})
        fid = E.numeric_fidelity("Aorta: encasement angle 120 (research-grade).", f)
        self.assertEqual((fid.supported, fid.total), (1, 1))

    def test_not_assessed_is_not_zero_angle(self):
        f = R.build_findings(organ="pankreas", summary={"lezyon_sayisi": 1},
                             vessels={"aorta": {"mesafe_mm": None, "sarma_derece": 0}})
        fid = E.numeric_fidelity("Aortic encasement status is not assessed.", f)
        self.assertEqual(fid.supported, 0)

    def test_wrong_value_still_fails(self):
        f = _findings()
        wrong = GEMMA_REPORT.replace("43.26 millimeters", "53.26 millimeters")
        fid = E.numeric_fidelity(wrong, f)
        self.assertIn("en_buyuk_cap", fid.missing)


class TestOrganConfusionVesselNames(unittest.TestCase):
    def test_splenic_vein_is_not_spleen(self):
        t = ("A distance of 4.0 mm from the tumor to the portal vein and splenic vein is "
             "reported with no vascular encasement observed.")
        self.assertEqual(C.organ_naming(t, "pancreas"), (0, 0))   # organ yok ama karışıklık da yok

    def test_real_spleen_still_flagged(self):
        self.assertEqual(C.organ_naming("A lesion is seen in the spleen.", "pancreas"), (0, 1))


if __name__ == "__main__":
    unittest.main()
