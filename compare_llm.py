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
OUTPUT_DIR = str(config.LAYOUT.reports)   # compare_reference ile AYNI klasör


def _load_findings(path: str) -> dict | None:
    """Dosyadan bulguları al. 'bulgular' yoksa tüm JSON'a DÜŞME (C31)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    findings = data.get("bulgular", data)
    if not isinstance(findings, dict):
        return None
    # Geçerli bulgu şeması değilse atla (ör. '<ad>.json.meta.json' cache dosyası
    # yalnız {"cache_key": ...} tutar; bulgu sanılıp boş vaka üretmesin).
    if "organ" not in findings and "schema_version" not in findings:
        return None
    # İzinli-alan anonimleştirme (kimlik/geçmiş asla gönderilmez)
    return reporting.anonymize_findings(findings)


def _kosullari_kaydet(models: dict) -> None:
    """Her modele GERÇEKTE uygulanan üretim koşullarını OUTPUT_DIR/kosullar.json'a
    yazar (model-model koşuda birikimli). Aynı-koşul iddiası bu kayıtla denetlenir."""
    import hashlib
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    yol = os.path.join(OUTPUT_DIR, "kosullar.json")
    kayit = {}
    if os.path.exists(yol):
        try:
            with open(yol, "r", encoding="utf-8") as f:
                kayit = json.load(f)
        except Exception:
            kayit = {}
    if kayit and kayit.get("prompt_version") != reporting.config.REPORT_PROMPT_VERSION:
        raise RuntimeError("Output directory belongs to another prompt version. "
                           "Set RADYOLOG_REPORTS_DIR to a new experiment directory.")
    kayit["prompt_version"] = reporting.config.REPORT_PROMPT_VERSION
    kayit["system_prompt_sha256"] = hashlib.sha256(
        reporting.SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    kayit["gen"] = reporting.GEN
    kayit.setdefault("modeller", {})
    for etiket, cfg in models.items():
        kayit["modeller"][etiket] = reporting.generation_conditions(cfg["provider"], cfg["model"])
    with open(yol, "w", encoding="utf-8") as f:
        json.dump(kayit, f, ensure_ascii=False, indent=2)


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
    # Varsayılan dışlama: Gemini (günlük kota -> eksik koşu). Kapatmak için
    # RADYOLOG_EXCLUDE_MODELS="" ver.
    exclude = {m.strip() for m in os.environ.get(
        "RADYOLOG_EXCLUDE_MODELS", "Gemini").split(",") if m.strip()}
    if exclude:
        models = {k: v for k, v in models.items() if k not in exclude}
    print(f"🧪 Koşulacak modeller: {list(models)}")
    _kosullari_kaydet(models)
    if dosyalar:
        paths = [d if os.path.isabs(d) else os.path.join(FINDINGS_DIR, d) for d in dosyalar]
    else:
        # '*.json' glob'u '<ad>.json.meta.json' cache dosyalarını da yakalar;
        # onları ele (yoksa 80 vaka -> 160 girdi, boşa API çağrısı).
        paths = sorted(p for p in glob.glob(os.path.join(FINDINGS_DIR, "*.json"))
                       if not p.endswith(".meta.json"))
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
        # Model-model koşuda (her çağrı tek model) ÖNCEKİ modellerin sonuçlarını EZME:
        # varsa mevcut karşılaştırmayı yükle, üstüne bu koşunun modelini ekle.
        kpath = os.path.join(OUTPUT_DIR, f"{ad}__karsilastirma.json")
        karsilastirma = {"hasta": ad, "bulgular": findings,
                         "model_input": reporting.model_findings(findings),
                         "raporlar": {}, "sadakat": {}}
        if os.path.exists(kpath):
            try:
                with open(kpath, "r", encoding="utf-8") as f:
                    eski = json.load(f)
                karsilastirma["raporlar"] = eski.get("raporlar", {}) or {}
                karsilastirma["sadakat"] = eski.get("sadakat", {}) or {}
            except Exception:
                pass

        for etiket, cfg in models.items():
            out_txt = os.path.join(OUTPUT_DIR, f"{ad}__{etiket}.txt")
            # DEVAM ETTİRME: bu model+vaka zaten üretildiyse API'yi TEKRAR çağırma.
            # Kota korunur (Gemini vb. günlük limit); kayıtlı metinden sadakat yeniden
            # hesaplanır, böylece ozet.csv tam kalır. Yeniden-üretmek istersen .txt'yi sil.
            identity = reporting.request_identity(cfg["provider"], cfg["model"],
                                                  reporting.SYSTEM_PROMPT, prompt)
            record_path = out_txt[:-4] + ".uretim.json"
            cached = False
            if os.path.exists(out_txt) and os.path.getsize(out_txt) > 0:
                try:
                    with open(record_path, encoding="utf-8") as f:
                        previous = json.load(f)
                    cached = (previous.get("request_sha256") == identity["request_sha256"])
                except (OSError, ValueError):
                    pass
                if not cached:
                    raise RuntimeError(
                        f"Cached report has different or missing generation conditions: {out_txt}. "
                        "Set RADYOLOG_REPORTS_DIR to a new experiment directory.")
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
                    # repr: boş str'li hatalarda bile tip görünür; boşsa tam traceback.
                    print(f"  ❌ {etiket}: {e!r}")
                    if not str(e):
                        import traceback
                        traceback.print_exc()
                    continue
                if not resp.is_valid():  # boş/kesik yanıt başarı sayılmaz (F19)
                    print(f"  ⚠️  {etiket}: geçersiz/kesik yanıt, atlandı."); continue
                text = resp.text
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write(text)
                # Rapor başına üretim kaydı (bitiş nedeni, token, Claude'da düşünme bloğu var mı)
                with open(out_txt[:-4] + ".uretim.json", "w", encoding="utf-8") as f:
                    json.dump({**identity, "finish_reason": resp.finish_reason, "usage": resp.usage,
                               "kosul": reporting.generation_conditions(cfg["provider"],
                                                                        cfg["model"])},
                              f, ensure_ascii=False, indent=2)
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

        with open(kpath, "w", encoding="utf-8") as f:
            json.dump(karsilastirma, f, ensure_ascii=False, indent=2)

    if ozet:
        csv_path = os.path.join(OUTPUT_DIR, "ozet.csv")
        # BİRİKİMLİ: mevcut ozet.csv'yi oku, (hasta, model) anahtarıyla birleştir ki
        # model-model koşuda son model öncekileri EZMESİN (480 satır tam kalsın).
        birlesik: dict[tuple, dict] = {}
        if os.path.exists(csv_path):
            try:
                with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
                    for row in csv.DictReader(f):
                        birlesik[(row.get("hasta"), row.get("model"))] = row
            except Exception:
                pass
        for row in ozet:
            birlesik[(row["hasta"], row["model"])] = row
        rows = sorted(birlesik.values(), key=lambda r: (r.get("model", ""), r.get("hasta", "")))
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(ozet[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n📊 Özet: {csv_path}  (toplam {len(rows)} satır)")
    print(f"📁 Çıktılar: {OUTPUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else None)
