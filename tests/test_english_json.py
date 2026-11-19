"""English JSON, shared input and cache isolation regressions. No network."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import compare_llm
from pipeline import reporting as R


class TestEnglishJSON(unittest.TestCase):
    def setUp(self):
        self.findings = R.build_findings(
            organ="karaciger",
            summary={"lezyon_sayisi": 1, "toplam_tumor_hacim_cm3": 8.4,
                     "en_buyuk_cap_lezyon": {"id": 1, "cap_mm": 20.0, "hacim_cm3": 8.4,
                                            "tc": "PRIVATE"},
                     "en_buyuk_hacim_lezyon": {"id": 1, "cap_mm": 20.0, "hacim_cm3": 8.4}},
            vessels={"portal_ven": {"mesafe_mm": 5.0, "sarma_derece": None,
                                    "status": "not_assessed"}},
            provenance={"mask": "/private/patient.nii.gz"})

    def test_json_preserves_values_and_null_without_private_fields(self):
        before = copy.deepcopy(self.findings)
        prompt = R.build_prompt(self.findings)
        payload = json.JSONDecoder().raw_decode(prompt.split("\n", 1)[1])[0]
        self.assertEqual(payload["organ"], "liver")
        self.assertIsNone(payload["imaging_phase"])
        self.assertEqual(payload["largest_lesion_by_diameter"]["axial_diameter_mm"], 20.0)
        vessel = payload["vessels"]["portal vein and splenic vein"]
        self.assertEqual(vessel["distance_mm"], 5.0)
        self.assertIsNone(vessel["encasement_angle_deg"])
        self.assertNotIn("PRIVATE", prompt)
        self.assertNotIn("patient.nii.gz", prompt)
        self.assertNotIn("karaciger", prompt)
        self.assertEqual(self.findings, before)

    def test_request_identity_changes_with_input_or_settings(self):
        prompt = R.build_prompt(self.findings)
        first = R.request_identity("local_hf", "model", R.SYSTEM_PROMPT, prompt)
        self.assertNotEqual(first, R.request_identity("local_hf", "other-model",
                                                     R.SYSTEM_PROMPT, prompt))
        self.assertNotEqual(first, R.request_identity("local_hf", "model",
                                                     R.SYSTEM_PROMPT, prompt + "changed"))
        with patch.dict(R.GEN, {"temperature": 0.3}):
            self.assertNotEqual(first, R.request_identity("local_hf", "model",
                                                         R.SYSTEM_PROMPT, prompt))

    def test_old_cache_is_rejected_and_current_cache_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, output = root / "input", root / "output"
            inputs.mkdir()
            source = inputs / "case.json"
            source.write_text(json.dumps(self.findings), encoding="utf-8")
            with patch.object(compare_llm, "FINDINGS_DIR", str(inputs)), \
                 patch.object(compare_llm, "OUTPUT_DIR", str(output)), \
                 patch.dict("os.environ", {"RADYOLOG_MODELS": "Template"}), \
                 redirect_stdout(io.StringIO()):
                compare_llm.main()
                with patch.object(R, "call_llm", side_effect=AssertionError("cached")):
                    compare_llm.main()
                record = output / "case__Template.uretim.json"
                record.unlink()
                with self.assertRaisesRegex(RuntimeError, "new experiment directory"):
                    compare_llm.main()


class TestTemplateIsControl(unittest.TestCase):
    """Şablon: yalnız modele giden JSON'u kullanır; eksik değer 'yok' yazılmaz."""

    def _f(self, summary):
        return R.build_findings(organ="pankreas", summary=summary,
                                vessels={"aorta": {"mesafe_mm": None, "sarma_derece": None}})

    def test_missing_lesion_count_is_not_written_as_absent(self):
        f = self._f({"toplam_tumor_hacim_cm3": None})
        f.pop("lezyon_sayisi", None)
        self.assertIsNone(R.model_findings(f)["lesion_count"])
        t = R.template_report(f)
        self.assertIn("not assessed", t)
        self.assertNotIn("No measurable focal lesion", t)

    def test_zero_lesions_written_as_absent(self):
        t = R.template_report(self._f({"lezyon_sayisi": 0, "toplam_tumor_hacim_cm3": 0.0}))
        self.assertIn("No measurable focal lesion detected in the pancreas", t)

    def test_template_uses_same_values_as_model_json(self):
        f = R.build_findings(
            organ="karaciger",
            summary={"lezyon_sayisi": 2, "toplam_tumor_hacim_cm3": 9.1,
                     "en_buyuk_cap_lezyon": {"id": 1, "cap_mm": 31.5, "hacim_cm3": 7.2},
                     "en_buyuk_hacim_lezyon": {"id": 1, "cap_mm": 31.5, "hacim_cm3": 7.2}},
            vessels={"vci": {"mesafe_mm": 12.0, "sarma_derece": 0}})
        c, t = R.model_findings(f), R.template_report(f)
        for v in ("31.5 mm", "7.2 cm3", "9.1 cm3", "inferior vena cava (IVC)", "12.0 mm"):
            self.assertIn(v, t)
        self.assertEqual(c["vessels"]["inferior vena cava (IVC)"]["distance_mm"], 12.0)
        self.assertNotIn("karaciger", t)


class TestExperimentIsolation(unittest.TestCase):
    def test_generation_and_comparison_use_same_report_dir(self):
        import os
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RADYOLOG_REPORTS_DIR", None)
            os.environ.pop("RADYOLOG_REPORTS", None)
            self.assertEqual(R.config.LAYOUT.reports.name, "reports_en_json_v3")

    def test_measurement_cache_version_unchanged(self):
        # ölçüm önbellek anahtarı PROMPT_VERSION'a bağlı; rapor istemi ayrı sürüm
        self.assertEqual(R.config.PROMPT_VERSION, "2.0.0")
        self.assertEqual(R.config.REPORT_PROMPT_VERSION, "3.0.0")


class TestSingleAttemptInvalidOutput(unittest.TestCase):
    """Geçersiz (kesik/boş) çıktı saklanır, özete 'gecersiz' yazılır, TEKRAR denenmez."""

    def test_invalid_output_recorded_and_not_retried(self):
        f = R.build_findings(organ="karaciger",
                             summary={"lezyon_sayisi": 0, "toplam_tumor_hacim_cm3": 0.0},
                             vessels={})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, output = root / "input", root / "output"
            inputs.mkdir()
            (inputs / "case.json").write_text(json.dumps(f), encoding="utf-8")
            bad = R.LLMResponse("partial text", "local_hf", "m", finish_reason="length",
                                truncated=True)
            with patch.object(compare_llm, "FINDINGS_DIR", str(inputs)),                  patch.object(compare_llm, "OUTPUT_DIR", str(output)),                  patch.dict("os.environ", {"RADYOLOG_MODELS": "Llama"}),                  redirect_stdout(io.StringIO()):
                with patch.object(R, "call_llm", return_value=bad) as call:
                    compare_llm.main()
                    self.assertEqual(call.call_count, 1)
                self.assertTrue((output / "case__Llama.gecersiz.txt").exists())
                rec = json.loads((output / "case__Llama.gecersiz.json").read_text(encoding="utf-8"))
                self.assertEqual(rec["neden"], "kesik")
                self.assertFalse((output / "case__Llama.txt").exists())
                with patch.object(R, "call_llm", side_effect=AssertionError("retried")):
                    compare_llm.main()          # ikinci koşu: tekrar çağırmamalı
                import csv as _csv
                rows = list(_csv.DictReader(open(output / "ozet.csv", encoding="utf-8-sig")))
                self.assertEqual([r["yontem"] for r in rows], ["gecersiz-kesik"])
