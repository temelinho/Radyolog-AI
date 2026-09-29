"""
compare_llm.py
================
Makale icin: Ayni segmentasyon metrikleri (JSON) uzerinden birden fazla LLM ile
(Claude Opus 4.8 ve Gemini 2.5 Flash) grounded radyoloji raporu uretip yan yana kaydeder.

Amac: "Grounded raporlamada LLM karsilastirmasi" (girdi birebir sabit, sadece model degisken).

Kullanim:
    python compare_llm.py                      # patients_db'deki tum JSON'lari isler
    python compare_llm.py hasta1.json hasta2.json   # secili dosyalar

Cikti: llm_karsilastirma/ klasorune her hasta icin:
    - <hasta>_<Model>.txt   (her modelin raporu)
    - <hasta>_karsilastirma.json  (metrikler + tum raporlar + sadakat kontrolu)
    - ozet.csv  (tum hastalar icin model x metrik-sadakati tablosu)

NOT: API'ye kimlik gonderilmez; sadece klinik metrikler yollanir (KVKK/gizlilik).
"""

import os
import re
import json
import csv
import glob
import datetime

# .env yukle
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
PATIENTS_DB_DIR = os.environ.get("PATIENTS_DB_DIR", r"C:\Users\ASUS\PycharmProjects\TemelProje\patients_db")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_karsilastirma")

# Karsilastirilacak modeller
MODELS = {
    "Claude-Opus-4.8": {"provider": "anthropic", "model": "claude-opus-4-8"},
    "Gemini-2.5-Flash": {"provider": "google", "model": "gemini-2.5-flash"},
}

SYSTEM_PROMPT = """Sen 15 yıllık deneyimli bir abdominal radyologsun.
Sana verilen BT segmentasyon bulgularını ACR ve ESR standartlarına uygun şekilde raporla.
ÖNEMLİ KURALLAR:
- Sadece verilen metriklere dayanarak rapor yaz. Metriklerde olmayan hiçbir sayı/bulgu UYDURMA.
- Karaciğer ve Pankreas gibi tümörlü vakalarda tümör boyutuna göre T evrelemesi yap (T1: ≤2cm, T2: 2-4cm, T3: >4cm).
- Vasküler temas varsa NCCN kriterlerine göre rezektabilite değerlendir (180° üzeri sarma lokal ileri/irresektabl anlamına gelir).
- Kesin tanı koyma, histopatolojik doğrulama önerilir de.
- Raporu Türkçe yaz, kısa ve öz tut. Markdown KULLANMA, düz metin yaz.
RAPOR YAPISI:
1. TEKNİK
2. BULGULAR
3. VASKÜLER DEĞERLENDİRME (varsa)
4. EVRELEME VE REZEKTABİLİTE (varsa)
5. SONUÇ VE ÖNERİ"""


def call_llm(provider, model, system_prompt, user_prompt, temperature=0.3, max_tokens=2500):
    if provider == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY tanimli degil (.env).")
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            system=system_prompt, messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    elif provider == "google":
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY tanimli degil (.env).")
        from google import genai
        client = genai.Client(api_key=GOOGLE_API_KEY)
        resp = client.models.generate_content(
            model=model, contents=user_prompt,
            config=genai.types.GenerateContentConfig(system_instruction=system_prompt, temperature=temperature),
        )
        return resp.text
    raise ValueError(provider)


def anonymize(bulgular):
    """Sadece klinik metrikleri birak; olasi kimlik alanlarini cikar."""
    drop = {"ad_soyad", "tc", "hasta", "isim", "name"}
    return {k: v for k, v in bulgular.items() if k.lower() not in drop}


def build_prompt(bulgular):
    temiz = anonymize(bulgular)
    return ("BT Segmentasyon Bulguları:\n"
            + json.dumps(temiz, ensure_ascii=False, indent=2)
            + "\n\nLütfen standart radyoloji raporu oluştur.")


# ---- Basit metrik-sadakati kontrolu (halusinasyon proxy) ----
def numeric_fidelity(rapor, bulgular):
    """
    Metriklerdeki sayisal degerlerin raporda gecip gecmedigini kontrol eder (grounding gostergesi).
    Dondurur: (raporda_gecen, toplam_sayisal_metrik, oran)
    Not: Kaba bir otomatik gostergedir; nihai degerlendirme radyolog tarafindan yapilmalidir.
    """
    sayisal = {}
    for k, v in bulgular.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            sayisal[k] = float(v)
    if not sayisal:
        return 0, 0, None
    rapor_sayilar = set(re.findall(r"\d+\.?\d*", rapor.replace(",", ".")))
    rapor_floats = set()
    for s in rapor_sayilar:
        try:
            rapor_floats.add(round(float(s), 1))
        except ValueError:
            pass
    gecen = 0
    for v in sayisal.values():
        if round(v, 1) in rapor_floats or round(v, 0) in {round(x, 0) for x in rapor_floats}:
            gecen += 1
    return gecen, len(sayisal), round(gecen / len(sayisal), 2)


def main(dosyalar=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if dosyalar:
        paths = [os.path.join(PATIENTS_DB_DIR, d) if not os.path.isabs(d) else d for d in dosyalar]
    else:
        paths = sorted(glob.glob(os.path.join(PATIENTS_DB_DIR, "*.json")))

    if not paths:
        print(f"⚠️  {PATIENTS_DB_DIR} içinde JSON bulunamadı.")
        return

    ozet_satirlar = []
    for path in paths:
        ad = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ {ad}: okunamadı ({e})")
            continue

        bulgular = data.get("bulgular") or data  # dogrudan metrik JSON da desteklenir
        if not isinstance(bulgular, dict):
            print(f"⏭️  {ad}: 'bulgular' yok, atlandı.")
            continue

        prompt = build_prompt(bulgular)
        print(f"\n=== {ad} ({bulgular.get('organ', '?')}) ===")
        karsilastirma = {"hasta": ad, "bulgular": bulgular, "raporlar": {}, "sadakat": {}}

        for etiket, cfg in MODELS.items():
            try:
                rapor = call_llm(cfg["provider"], cfg["model"], SYSTEM_PROMPT, prompt)
            except Exception as e:
                print(f"  ❌ {etiket}: {e}")
                continue
            gecen, toplam, oran = numeric_fidelity(rapor, bulgular)
            karsilastirma["raporlar"][etiket] = rapor
            karsilastirma["sadakat"][etiket] = {"gecen": gecen, "toplam": toplam, "oran": oran}
            print(f"  ✅ {etiket}: metrik sadakati {gecen}/{toplam} (oran {oran})")

            with open(os.path.join(OUTPUT_DIR, f"{ad}__{etiket}.txt"), "w", encoding="utf-8") as f:
                f.write(rapor)
            ozet_satirlar.append({
                "hasta": ad, "organ": bulgular.get("organ", ""), "model": etiket,
                "metrik_sadakati_gecen": gecen, "metrik_sadakati_toplam": toplam, "oran": oran,
            })

        with open(os.path.join(OUTPUT_DIR, f"{ad}__karsilastirma.json"), "w", encoding="utf-8") as f:
            json.dump(karsilastirma, f, ensure_ascii=False, indent=2)

    # Ozet CSV
    if ozet_satirlar:
        csv_path = os.path.join(OUTPUT_DIR, "ozet.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(ozet_satirlar[0].keys()))
            writer.writeheader()
            writer.writerows(ozet_satirlar)
        print(f"\n📊 Özet tablo: {csv_path}")
    print(f"📁 Tüm çıktılar: {OUTPUT_DIR}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] if len(sys.argv) > 1 else None)
