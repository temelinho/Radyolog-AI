"""
test_security_fixes.py — dış inceleme (Codex) sonrası iki düzeltmenin regresyonu:
  1) provenance (maske yolu = hasta study_id'si) dış LLM istemine SIZMAMALI.
  2) '<ad>.json.meta.json' cache dosyaları bulgu sanılıp sahte vaka ÜRETMEMELİ.
"""

from __future__ import annotations

import json
import os
import tempfile

import compare_llm
from pipeline import reporting


def _ornek_findings():
    return reporting.build_findings(
        organ="karaciger",
        summary={"lezyon_sayisi": 1, "toplam_tumor_hacim_cm3": 5.0,
                 "en_buyuk_cap_lezyon": {"id": 1, "cap_mm": 20.0, "hacim_cm3": 5.0},
                 "en_buyuk_hacim_lezyon": {"id": 1, "cap_mm": 20.0, "hacim_cm3": 5.0}},
        provenance={"mask": "/content/drive/MyDrive/Radyolog-AI-2/masks/AC423d579.nii.gz",
                    "quality_flags": [], "geometry": {}})


def test_provenance_istemde_yok_study_id_sizmaz():
    findings = _ornek_findings()
    assert "provenance" in findings                      # tam JSON'da yerelde kalır
    anon = reporting.anonymize_findings(findings)
    assert "provenance" not in anon                      # anonimde yok
    prompt = reporting.build_prompt(findings)
    # Dış API'ye giden istemde ne study_id ne maske yolu bulunmalı
    assert "AC423d579" not in prompt
    assert ".nii.gz" not in prompt
    assert "drive" not in prompt.lower()


def test_meta_json_bulgu_sanilmaz():
    with tempfile.TemporaryDirectory() as d:
        meta = os.path.join(d, "X__Task003_Liver.json.meta.json")
        with open(meta, "w", encoding="utf-8") as f:
            json.dump({"cache_key": "abc"}, f)
        assert compare_llm._load_findings(meta) is None   # cache dosyası -> atla

        ok = os.path.join(d, "X__Task003_Liver.json")
        with open(ok, "w", encoding="utf-8") as f:
            json.dump({"schema_version": "2.0.0", "organ": "karaciger",
                       "lezyon_sayisi": 0}, f)
        assert compare_llm._load_findings(ok) is not None  # gerçek bulgu -> geçerli
