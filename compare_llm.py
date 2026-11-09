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
    # Model seçimi: RADYOLOG_MODELS=Gemini,Qwen,Gemma,Llama,Template (Claude hariç gibi).
    # Boşsa tüm MODELS koşar. Bilinmeyen etiket yok sayılır.
    secili = os.environ.get("RADYOLOG_MODELS")
    models = reporting.MODELS
    if secili:
        istenen = {s.strip() for s in secili.split(",") if s.strip()}
        models = {k: v for k, v in reporting.MODELS.items() if k in istenen}
        bilinmeyen = istenen - set(reporting.MODELS)
        if bilinmeyen:
            print(f"⚠️  Bilinmeyen model etiketi yok sayıldı: {sorted(bilinmeyen)}")
    print(f"🧪 Koşulacak modeller: {list(models)}")
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

        for etiket, cfg in models.items():
            out_txt = os.path.join(OUTPUT_DIR, f"{ad}__{etiket}.txt")
            # DEVAM ETTİRME: bu model+vaka zaten üretildiyse API'yi TEKRAR çağırma.
            # Kota korunur (Gemini vb. günlük limit); kayıtlı metinden sadakat yeniden
            # hesaplanır, böylece ozet.csv tam kalır. Yeniden-üretmek istersen .txt'yi sil.
            cached = os.path.exists(out_txt) and os.path.getsize(out_txt) > 0
            if cached:
                with open(out_txt, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                try:
                    resp = reporting.call_llm(cfg["provider"], cfg["model"],
                                              reporting.SYSTEM_PROMPT, prompt, findings=findings)
                except NotImplementedError:
                    print(f"  ⏭️  {etiket}: açık model Colab'da çalışır, atlandı."); continue
                except Exception as e:
                    print(f"  ❌ {etiket}: {e}"); continue
                if not resp.is_valid():  # boş/kesik yanıt başarı sayılmaz (F19)
                    print(f"  ⚠️  {etiket}: geçersiz/kesik yanıt, atlandı."); continue
                text = resp.text
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write(text)
                # Dakikalık limit (RPM) için API çağrıları arasına bekleme (saniye).
                bekle = float(os.environ.get("RADYOLOG_LLM_DELAY", "0") or 0)
                if bekle and cfg["provider"] in ("google", "anthropic"):
                    import time
                    time.sleep(bekle)

            fid = evaluation.numeric_fidelity(text, findings)  # bağlam+birim (C28)
            karsilastirma["raporlar"][etiket] = text
            karsilastirma["sadakat"][etiket] = fid.as_dict()
            print(f"  {'⏭️ mevcut' if cached else '✅'} {etiket}: "
                  f"sadakat {fid.supported}/{fid.total} (oran {fid.ratio})")
            ozet.append({"hasta": ad, "organ": findings.get("organ", ""), "model": etiket,
                         "sadakat_gecen": fid.supported, "sadakat_toplam": fid.total,
                         "oran": fid.ratio,
                         "yontem": "otomatik-proxy-cache" if cached else "otomatik-proxy"})

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
