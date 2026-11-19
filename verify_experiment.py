"""
verify_experiment.py — rapor deneyinin bütünlük denetimi (API/GPU kullanmaz).

Her rapor (ve geçersiz çıktı) kaydındaki `request_sha256` değerini, GÜNCEL koddan
aynı bulgu dosyası ve aynı model ayarlarıyla YENİDEN hesaplar ve karşılaştırır.
Eşleşiyorsa: o rapor, şu anki girdi + talimat + üretim koşuluyla üretilmiştir
(hangi commit'te koşulmuş olursa olsun). Ayrıca model başına sayım, Claude'da
düşünme bloğu, raporlarda Türkçe organ adı kontrolü yapar.

Kullanım:
    RADYOLOG_BASE=... RADYOLOG_REPORTS_DIR=... python verify_experiment.py
Çıkış kodu: sorun yoksa 0, varsa 1.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import re
import sys

from pipeline import config, reporting

import compare_llm


def main() -> int:
    exp = str(config.LAYOUT.reports)
    findings_dir = str(config.LAYOUT.findings)
    cases = {}
    for p in sorted(glob.glob(os.path.join(findings_dir, "*.json"))):
        if p.endswith(".meta.json"):
            continue
        f = compare_llm._load_findings(p)
        if f is not None:
            cases[os.path.splitext(os.path.basename(p))[0]] = f
    print(f"deney: {exp}\nvaka: {len(cases)} | REPORT_PROMPT_VERSION {config.REPORT_PROMPT_VERSION}")

    sorun = collections.Counter()
    for etiket, cfg in reporting.MODELS.items():
        gecerli, gecersiz, uyusmayan, eksik_kayit, dusundu, turkce = 0, 0, [], 0, 0, 0
        for ad, f in cases.items():
            prompt = reporting.build_prompt(f)
            beklenen = reporting.request_identity(cfg["provider"], cfg["model"],
                                                  reporting.SYSTEM_PROMPT, prompt)["request_sha256"]
            txt = os.path.join(exp, f"{ad}__{etiket}.txt")
            bad = os.path.join(exp, f"{ad}__{etiket}.gecersiz.json")
            if os.path.exists(txt):
                gecerli += 1
                rec_p = txt[:-4] + ".uretim.json"
                if not os.path.exists(rec_p):
                    eksik_kayit += 1
                    continue
                rec = json.load(open(rec_p, encoding="utf-8"))
                if rec.get("request_sha256") != beklenen:
                    uyusmayan.append(ad)
                if etiket == "Claude" and (rec.get("usage") or {}).get("thinking_blocks"):
                    dusundu += 1
                if re.search(r"karaci|pankreas", open(txt, encoding="utf-8").read(), re.I):
                    turkce += 1
            elif os.path.exists(bad):
                gecersiz += 1
                rec = json.load(open(bad, encoding="utf-8"))
                if rec.get("request_sha256") != beklenen:
                    uyusmayan.append(ad + " (gecersiz)")
        denenen = gecerli + gecersiz
        if denenen == 0:
            continue
        print(f"  {etiket:16} geçerli {gecerli:2} | geçersiz {gecersiz} | denenen {denenen}/{len(cases)}"
              f" | kimlik uyuşmayan {len(uyusmayan)} | kayıt eksik {eksik_kayit}"
              f" | Türkçe organ {turkce}" + (f" | düşündü {dusundu}" if etiket == "Claude" else ""))
        for k, v in (("eksik", denenen != len(cases)), ("kimlik", bool(uyusmayan)),
                     ("kayit", bool(eksik_kayit)), ("turkce", bool(turkce)), ("dusundu", bool(dusundu))):
            if v:
                sorun[f"{etiket}_{k}"] += 1
        if uyusmayan:
            print("     uyuşmayan:", uyusmayan[:10])
    print("SONUÇ:", "TEMİZ" if not sorun else dict(sorun))
    return 0 if not sorun else 1


if __name__ == "__main__":
    sys.exit(main())
