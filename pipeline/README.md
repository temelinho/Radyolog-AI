# pipeline/ — Radyolog AI ortak çekirdeği (v2.0.0)

UI'dan (Streamlit) ve notebook'lardan **bağımsız**, import edilebilir çekirdek.
Amaç: ölçüm / klinik kural / rapor / değerlendirme mantığının **tek kaynaktan**
gelmesi. 29 Ekim 2026 incelemesindeki deney-kritik P1 hataları hedeflenmiştir.

> Paketi import etmek GPU çıkarımı, model indirme veya API çağrısı **yapmaz**
> (yan etkisiz import — F11).

## Modüller ve çözdükleri hatalar

| Modül | Çözdüğü bulgu | Özet |
|---|---|---|
| `config.py` | F11, F19, F20 | Merkezî yollar/sürümler; sabit `C:\Users\ASUS\...` kaldırıldı; durum sözlüğü (success/empty_mask/not_assessed/failed) |
| `imaging.py` | F03, F06 | Dünya-koordinat mesafesi; origin/spacing/direction doğrulama; nearest-neighbor resample; oblik bayrağı |
| `measurements.py` | F01, F04, F05, F14 | Lezyon bazlı **tek** aksiyel çap + ayrı hacim; 26-komşuluk; deterministik; **yerel** damar sarma açısı; null≠0 |
| `evaluation.py` | F13, F15, F17, F18 | RAW eşik (yuvarlama yok); bağlam+birim eşleşmeli sadakat; ondalık koruyan ayrıştırma; ayrı paydalar + Wilson GA |
| `manifest.py` | F12, F18 | Tek kaynak vaka listesi; POS/NEG→şema; aynı-kişi çakışma denetimi |
| `runner.py` | F10, F19 | Hash'li cache anahtarı; atomik yaz-doğrula-rename; iş izolasyonu |
| `clinical_rules.py` | F07 | Organ-özel eşikler; "boyut kategorisi" (≠ T-evresi); eksikte "değerlendirilemez" |
| `reporting.py` | F07, F19, F22 | Ortak şema + prompt; izinli-alan anonimleştirme; kural-tabanlı şablon (%100 sadık baseline); yanıt doğrulama (boş/kesik≠başarı) |

## Test

```bash
python -m unittest discover -s tests -v
```

51 test; her biri 29 Ekim incelemesindeki bir repro'yu (ör. 100 mm ötelenmiş
maske, 9.96→10.0 eşik, "2 cm³"→20 mm, uzak kesit açıyı değiştirme) hedefler.

## Çalışma sırası

1. **Yerel / CPU (GPU yok):** çekirdek + testler, manifest üretimi, değerlendirme.
2. **Colab / GPU:** nnU-Net 5-fold + TotalSegmentator → maskeler.
3. Maskeler gelince: `measurements` → `bulgular.json` → `reporting` (API) → `evaluation`.

## Sürümleme

`config.SCHEMA_VERSION` / `PIPELINE_VERSION` / `PROMPT_VERSION` değişince cache
geçersizleşir (F10). Şema veya ölçüm mantığı değişince bu sürümler **artırılmalı**.
