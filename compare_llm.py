"""
compare_llm.py — çoklu LLM grounded rapor karşılaştırması (ÇEKİRDEĞE BAĞLI).

Eskiden kendi promptu, kendi kaba sayı-sadakati, yalnız Claude+Gemini ve sabit
`C:\\Users\\ASUS\\...` yolu vardı (C28–C31). Artık TEK kaynak olarak
`pipeline/` kullanılıyor:
  * Prompt/şema/anonimleştirme/model seti -> pipeline.reporting
  * Bağlam+birim eşleşmeli sadakat        -> pipeline.evaluation.numeric_fidelity
  * Yollar                                 -> pipeline.config (Colab/Drive)

Kullanım:
    python compare_llm.py                 # findings klasöründeki tüm JSON'lar
    python compare_llm.py a.json b.json   # seçili dosyalar
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from pipeline import config, reporting, evaluation

FINDINGS_DIR = os.environ.get("RADYOLOG_FINDINGS", str(config.LAYOUT.findings))
OUTPUT_DIR = str(config.LAYOUT.reports)


def _load_findings(path: str) -> dict | None:
    """Dosyadan bulguları al. 'bulgular' yoksa tüm JSON'a DÜŞME (C31)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    findings = data.get("bulgular", data)
    if not isinstance(findings, dict):
        return None
    # İzinli-alan anonimleştirme (kimlik/geçmiş asla gönderilmez)
    return reporting.anonymize_findings(findings)


def main(dosyalar: list[str] | None = None) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if dosyalar:
        paths = [d if os.path.isabs(d) else os.path.join(FINDINGS_DIR, d) for d in dosyalar]
    else:
        paths = sorted(glob.glob(os.path.join(FINDINGS_DIR, "*.json")))
    if not paths:
        print(f"⚠️  {FINDINGS_DIR} içinde JSON yok."); return

    ozet = []
    for path in paths:
        ad = os.path.splitext(os.path.basename(path))[0]
        try:
            findings = _load_findings(path)
        except Exception as e:
            print(f"❌ {ad}: okunamadı ({e})"); continue
        if findings is None:
            print(f"⏭️  {ad}: geçerli bulgular yok, atlandı."); continue

        prompt = reporting.build_prompt(findings)
        print(f"\n=== {ad} ({findings.get('organ', '?')}) ===")
        karsilastirma = {"hasta": ad, "bulgular": findings, "raporlar": {}, "sadakat": {}}

        for etiket, cfg in reporting.MODELS.items():
            try:
                resp = reporting.call_llm(cfg["provider"], cfg["model"],
                                          reporting.SYSTEM_PROMPT, prompt, findings=findings)
            except NotImplementedError:
                print(f"  ⏭️  {etiket}: açık model Colab'da çalışır, atlandı."); continue
            except Exception as e:
                print(f"  ❌ {etiket}: {e}"); continue
            if not resp.is_valid():  # boş/kesik yanıt başarı sayılmaz (F19)
                print(f"  ⚠️  {etiket}: geçersiz/kesik yanıt, atlandı."); continue

            fid = evaluation.numeric_fidelity(resp.text, findings)  # bağlam+birim (C28)
            karsilastirma["raporlar"][etiket] = resp.text
            karsilastirma["sadakat"][etiket] = fid.as_dict()
            print(f"  ✅ {etiket}: sadakat {fid.supported}/{fid.total} (oran {fid.ratio})")

            with open(os.path.join(OUTPUT_DIR, f"{ad}__{etiket}.txt"), "w", encoding="utf-8") as f:
                f.write(resp.text)
            ozet.append({"hasta": ad, "organ": findings.get("organ", ""), "model": etiket,
                         "sadakat_gecen": fid.supported, "sadakat_toplam": fid.total,
                         "oran": fid.ratio, "yontem": "otomatik-proxy"})

        with open(os.path.join(OUTPUT_DIR, f"{ad}__karsilastirma.json"), "w", encoding="utf-8") as f:
            json.dump(karsilastirma, f, ensure_ascii=False, indent=2)

    if ozet:
        csv_path = os.path.join(OUTPUT_DIR, "ozet.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(ozet[0].keys()))
            w.writeheader(); w.writerows(ozet)
        print(f"\n📊 Özet: {csv_path}")
    print(f"📁 Çıktılar: {OUTPUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else None)
