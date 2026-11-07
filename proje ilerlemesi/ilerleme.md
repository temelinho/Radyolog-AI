# Proje İlerlemesi — Radyolog AI

**Son güncelleme:** 8 Kasım 2026
**Amaç:** Bu oturumda (6–8 Kasım 2026) baştan sona ne yaptığımızın detaylı, kronolojik kaydı. Nerede kaldığımız ve sıradaki adım da sonda.

---

## 0. Bir cümlede proje
Abdominal BT'den otomatik çıkarılan geometrik tümör metrikleriyle (grounded) LLM radyoloji raporu üretmek ve bunu **gerçek radyolog raporuyla** nicel kıyaslayan, **tekrarlanabilir** bir pipeline + değerlendirme çerçevesi. Veri: Stanford **Merlin** (gerçek radyolog raporları). Segmentasyon: MSD'de eğitilmiş **nnU-Net v1**, Merlin'de **zero-shot**.

---

## 1. Başlangıç durumu ve teşhis
- İki doküman vardı: `radyolog_ai_proje_dokumani.md` (iyimser) ve `inceleme/2026-10-29_pipeline_inceleme_ve_plan.md` (23 hata: **F01–F23**).
- Eski kodun ana sorunu: aynı ölçüm/değerlendirme mantığının `radyoloji_app.py` + Colab notebook'larında **kopyalanıp birbirinden sapması.**
- En kritik eski hatalar: çap = hacim-eşdeğeri küre çapı (aksiyel değil), damar mesafesi aynı-grid sanılıp indeksleniyor (fiziksel uzay yok), sarma açısı global merkez, tek-fold, sabit `C:\Users\ASUS\...` yolları.

## 2. Verilen kararlar
- **Mimari:** UI'dan bağımsız **ortak çekirdek** (`pipeline/`).
- **Kapsam:** deney-kritik P1 hataları.
- **Kohort:** eski seçimleri at, **sıfırdan tabakalı-random** (20 POS + 20 NEG / organ).
- **nnU-Net:** v1 (elimizdeki Zenodo MSD ağırlıkları v1), 5-fold ensemble.
- **Veri:** Merlin kalır (gerçek radyolog raporu = işin kalbi); AbdomenAtlas gibi alternatifler yan-kanıt olabilir ama ana referans Merlin.

## 3. Ortak çekirdek `pipeline/` (13 modül)
| Modül | İş | Çözdüğü |
|---|---|---|
| `config.py` | Colab/Drive tek-kök yollar, 5-fold ayarı, sürümler | F11/F20 |
| `imaging.py` | Dünya-koordinat mesafesi, oblik bayrağı, bellek-güvenli resample | F03/F06 |
| `measurements.py` | Lezyon-bazlı aksiyel çap + ayrı hacim, yerel sarma açısı, `measure_case` | F01/F04/F14 |
| `evaluation.py` | RAW eşik, bağlam+birim sadakat proxy'si, Wilson GA | F13/F15/F17/F18 |
| `matching.py` | Lezyon eşleştirme, eşleşmeyen ayrı | F18 |
| `reference.py` | GT denetim izi (kaynak cümle/tarih/revizyon) | F18 |
| `clinical_rules.py` | Organ-özel kural, boyut kategorisi (≠T evresi) | F07 |
| `reporting.py` | Ortak şema/prompt, izinli-alan anonim, kural-tabanlı şablon | F07/F19/F22 |
| `segmentation.py` | nnU-Net v1/v2 adaptörü, 5-fold, fold kontrolü, çıktı doğrulama | F10/F20 |
| `run.py` | Uçtan uca orkestratör (BT→maske→ölçüm→bulgular.json) | F12 |
| `manifest.py`, `runner.py` | Tek-kaynak vaka listesi, hash'li cache | F10/F12 |

**Doğrulama:** her düzeltme sentetik veriyle test ediliyor.

## 4. Sağlamlaştırma (eleştirel denetim)
- K1 resample bellek, K2 oblik kapısı, K3 temas tanımı, K4 bağlantılılık, K5 kenar durum.
- `matching.py` + `reference.py` eklendi (lezyon eşleştirme + GT denetim izi).
- Sadakat çıktısı "otomatik-proxy" etiketlendi.

## 5. 🔴 Gizli veri olayı (çözüldü)
- Merlin verisi (16 MB `reports_final.xlsx` dahil) GitHub'a (private) push'lanmıştı → **RUA riski.**
- `git filter-branch` + gc ile **tüm geçmişten silindi**, `master` force-push (46ccc3f→158ece9), `merlin/` gitignore. 24.6 MB yedek alındı. Codex bağımsız doğruladı: temiz commit'lerde merlin yok.

## 6. App ve compare_llm'i çekirdeğe bağlama (C1–C34)
- `radyoloji_app.py`: `extract_metrics` → `measure_case`; segmentasyon → `pipeline.segmentation`; prompt → `reporting.SYSTEM_PROMPT`; boylamsal organ-filtreli; RECIST → "doğrulanmamış çap eğilimi"; atomik kayıt + PHI'siz dosya adı.
- `compare_llm.py`: kendi kaba puanlaması yerine `evaluation.numeric_fidelity` + 6 model seti.
- `main.py`: nnU-Net yaması kaldırıldı, tek protokol.
- CI (`.github/workflows/tests.yml`) eklendi.

## 7. Codex bağımsız incelemeleri (2 tur, hepsi giderildi)
- **Tur 2:** #1 yanlış vaka maskesi, #2 kimlik karışması, #3 nnU-Net sürüm uyumu, #4 0-byte/cache, #5 kullanılabilirlik.
- **Tur 3:** kimlik aliasing (yüklemede ayrı kopya), bayat maske (her koşu taze `attempt-` dizini), v1 fold yolu (`nnUNet/3d_fullres/Task.../trainer`) + env sürüme göre.
- Hepsi testlerle doğrulandı.

## 8. Tekrarlanabilir kohort seçimi (yeni)
- **`scripts/select_cohort.py`:** `reports_final.xlsx` (Findings) üzerinde organ-özel regex → POS/NEG/BELİRSİZ; seed'li; her sınıftan 30 aday (buffer); BELİRSİZ kohorta girmez ama `akis_*.csv`'de id+sıra+sebep ile loglanır (STARD); `provenance_*.json` (seed + reports SHA256 + terim sürümü) → tam tekrarlanabilir; `metadata.csv`'den gerçek contrast/phase.
- **`scripts/finalize_cohort.py`:** insan onayından draw_order sırasına göre ilk 20+20'yi kilitler + `reference_gt.csv` (çap + kaynak cümle).
- Çıktılar gitignore'da (RUA).

## 9. Colab'da ilk gerçek koşu (8 Kasım)
- Kendi Drive (`MyDrive/Radyolog-AI-2`) + repo klonu + `reports_final.xlsx`/`metadata.csv` yüklendi.
- **`select_cohort` çalıştı:** karaciğer **30 POS + 30 NEG**, pankreas **30 POS + 30 NEG** aday; **23.887 rapor** tarandı (kontrastlı filtre). Çıktılar `Radyolog-AI-2/protocol/`.

---

## 10. NEREDE KALDIK (sıradaki adım)
1. **Onay (manuel, şu an burada):** `onay_liver.csv` + `onay_pancreas.csv`'yi Excel'de doldur — her satırda `Findings` oku → `onay_pos_neg` (POS/NEG), POS'ta `cap_mm` + `kaynak_cumle`. Her sınıfta 20 geçerli dolunca dur.
2. **Finalize:** `finalize_cohort.py` → `liver_cases.json`, `pancreas_cases.json` (kesin 20+20) + `reference_gt.csv`.
3. **Manifest:** `build_cohort.py` → `cohort.csv`.
4. **Segmentasyon (GPU):** nnU-Net v1 ağırlıkları (Zenodo Task003/Task007) + TotalSegmentator → `python -m pipeline.run`.
5. **Raporlama:** API anahtarları → `compare_llm.py`.
6. **Değerlendirme + makale tabloları.**

## 11. Dürüst sınırlamalar (Limitations'a yazılacak)
- **Segmentasyon gerçek-referansı (maske) YOK** → Dice hesaplanamaz; ölçüm "radyologla uyum", "doğruluk" değil.
- **Zero-shot domain shift** (MSD→Merlin).
- GT = radyolog raporundan transkripsiyon (bağımsız re-okuma değil); patient_id yok → kişi-düzeyi bağımsızlık garanti değil.
- Zaman-içi lezyon eşleştirme ve kullanıcı erişim kontrolü (demo) kapsam dışı.

## 12. Git durumu
- **PR #1** dalı: `claude/radyolog-ai-proje-dokumani-764226` (tüm iş burada, push'lu).
- `master`: `158ece9` (merlin'siz temiz taban; pipeline PR merge edilince geçer).
- **13 modül, 95 birim testi**, hepsi yeşil. Ağırlık/hasta verisi/API anahtarı GitHub'da **yok**.

*İlgili belgeler: `PROJE_TAM_DURUM.md`, `kod_zayifliklari.md`, `savunulabilirlik_notu.md`, `pipeline/README.md`.*
