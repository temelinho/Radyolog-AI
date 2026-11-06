"""clinical_rules.py + reporting.py testleri — F07, F19, F22 ve entegrasyon."""

import unittest

from pipeline import clinical_rules as CR
from pipeline import reporting as REP
from pipeline import evaluation as E


class TestClinicalRulesF07(unittest.TestCase):
    def test_pancreas_size_bucket(self):
        r = CR.size_category(30.0, "pankreas")
        self.assertEqual(r["kategori"], "2–4 cm")
        self.assertEqual(r["taban"], "yalnız boyut")  # evre DEĞİL

    def test_liver_not_pancreas_thresholds(self):
        """Karaciğere pankreas eşiği uygulanmamalı (F07)."""
        self.assertEqual(CR.size_category(30.0, "pankreas")["kategori"], "2–4 cm")
        self.assertEqual(CR.size_category(30.0, "karaciger")["kategori"], "2–5 cm")

    def test_missing_input_not_assessed(self):
        self.assertEqual(CR.size_category(None, "pankreas")["durum"], "değerlendirilemez")

    def test_unknown_organ(self):
        self.assertEqual(CR.size_category(30.0, "akciger")["durum"], "değerlendirilemez")

    def test_vascular_no_definitive_verdict(self):
        v = CR.vascular_assessment({"aorta": {"mesafe_mm": 0.0, "sarma_derece": 200}}, "pankreas")
        self.assertTrue(v["damarlar"][0]["research_grade"])
        self.assertIn("uzman", v["uyari"].lower())


class TestReportingF19F22(unittest.TestCase):
    def test_anonymize_drops_identity(self):
        f = {"organ": "karaciger", "tc": "123", "ad_soyad": "X Y", "lezyon_sayisi": 2}
        temiz = REP.anonymize_findings(f)
        self.assertIn("organ", temiz)
        self.assertNotIn("tc", temiz)
        self.assertNotIn("ad_soyad", temiz)

    def test_llm_response_validity(self):
        self.assertFalse(REP.LLMResponse("", "p", "m").is_valid())
        self.assertFalse(REP.LLMResponse("rapor", "p", "m", truncated=True).is_valid())
        self.assertTrue(REP.LLMResponse("rapor", "p", "m").is_valid())

    def test_template_provider_no_api(self):
        findings = {"organ": "karaciger", "lezyon_sayisi": 0}
        resp = REP.call_llm("template", "rule-based", REP.SYSTEM_PROMPT, "x", findings=findings)
        self.assertTrue(resp.is_valid())
        self.assertIn("lezyon", resp.text.lower())


class TestIntegrationTemplateFidelity(unittest.TestCase):
    """build_findings -> template_report -> numeric_fidelity ~1.0 (uçtan uca)."""

    def test_template_fully_faithful(self):
        summary = {
            "lezyon_sayisi": 1,
            "toplam_tumor_hacim_cm3": 8.4,
            "en_buyuk_cap_lezyon": {"id": 1, "cap_mm": 20.0, "hacim_cm3": 8.4},
            "en_buyuk_hacim_lezyon": {"id": 1, "cap_mm": 20.0, "hacim_cm3": 8.4},
            "status": "success",
        }
        vessels = {"aorta": {"mesafe_mm": 28.4, "sarma_derece": 0}}
        findings = REP.build_findings(organ="pankreas", summary=summary, vessels=vessels)

        self.assertEqual(findings["schema_version"], REP.config.SCHEMA_VERSION)
        self.assertEqual(findings["boyut_kategorisi"]["kategori"], "≤2 cm")

        report = REP.template_report(findings)
        fid = E.numeric_fidelity(report, findings)
        # Şablon %100 sadık olmalı: tüm gerçekler doğru bağlamda geçmeli
        self.assertEqual(fid.supported, fid.total, f"eksik: {fid.missing}")
        self.assertEqual(fid.ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
