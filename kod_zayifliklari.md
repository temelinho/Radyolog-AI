# Kod Zayıflıkları — tam statik denetim (çalıştırmadan önce)

**Tarih:** 7 Kasım 2026
**Kapsam:** Yalnız kendi kodumuz (makale/veri/çerçeve değil). Satır referansları `radyoloji_app.py`, `compare_llm.py`, `main.py` üzerinden; hepsi **okunarak**, çalıştırılmadan tespit edildi.

**Durum anahtarı:** 🟢 çekirdekte düzeltildi+test · 🟡 çekirdekte çözüm var AMA app/compare_llm hâlâ eskiyi kullanıyor · 🔴 hiç yazılmadı/kullanılmıyor

---

## 0. En büyük sorun — ENTEGRASYON BOŞLUĞU (🟡 hepsinin kökü)

`pipeline/` çekirdeği (63 test) **hiçbir yerden çağrılmıyor.** `radyoloji_app.py` ve `compare_llm.py` kendi eski fonksiyonlarını kullanıyor. Yani aşağıdaki 🟢'ların app karşılığı hâlâ bozuk. Bu düzeltilmeden "hatasız çalıştırdık" diyemeyiz.

---

## 1. Ölçüm / geometri (radyoloji_app.py)

| # | Yer | Sorun | Durum | Bağlanacağı çekirdek |
|---|---|---|---|---|
| C1 | `extract_metrics` **L157** | `tumor_cap_mm = (hacim·6/π)^⅓·10` → küre-eşdeğeri çap, aksiyel değil (F01) | 🟡 | `measurements.measure_case` |
| C2 | **L155-157** | Birden çok lezyon tek hacme/çapa iniyor; lezyon ayrımı yok (F14) | 🟡 | `measurements.extract_lesions` |
| C3 | **L159-163** | `baş/gövde/kuyruk` z-konumundan; karaciğere de uygulanıyor (F02) | 🟡 | çekirdek `null` veriyor |
| C4 | **L172-180** | `aorta_dist[tumor_arr].min()` iki maskeyi aynı grid sanıp indeksliyor; fiziksel hiza yok (F03) | 🟡 | `imaging.min_surface_distance_mm` |
| C5 | `calculate_encasement` **L186-204** | Tüm damarın **tek** ağırlık merkezi; kesitler karışıyor; portal+splenik birleşik (F04) | 🟡 | `measurements.vessel_wrap_angle_deg` |
| C6 | **L144** | Sabit `"Portal venöz faz BT"` (F06) | 🟡 | çekirdek `"bilinmiyor"` |
| C7 | **L191-204** | Temas/açı ölçülemezse `0.0` dönüyor → "ölçülemedi" ile "0" karışıyor (F05) | 🟡 | çekirdek `null` + durum |

## 2. Segmentasyon / kurulum (radyoloji_app.py, main.py)

| # | Yer | Sorun | Durum |
|---|---|---|---|
| C8 | app **L113** | `-f 0` → **tek fold** (main.py:28 ise `folds=[0,1,2,3,4]` → iki akış tutarsız) (F20) | 🔴 segmentation.py yok |
| C9 | app **L24,105,125** | Sabit `C:\Users\ASUS\...` yolları (F11) | 🔴 app'te duruyor |
| C10 | app **L121-122** | Başarı = sabit `hasta_001.nii.gz` var + boyut>0; `subprocess` çıkış kodu yok sayılıyor (F10) | 🔴 |
| C11 | app **L670** | `run_totalsegmentator` dönüşü **kullanılmıyor**; damar seg başarısızsa sessizce devam (F05) | 🔴 |
| C12 | main.py **L9-17** | Kurulu nnU-Net dosyasını import'ta metinle yamалıyor; kırılgan, tekrar-üretilemez (F11/F20) | 🔴 |
| C13 | main.py **L22** | Yalnız Task007 (pankreas) sabit; genel değil | 🔴 |

## 3. Boylamsal takip / RECIST (radyoloji_app.py)

| # | Yer | Sorun | Durum |
|---|---|---|---|
| C14 | `get_patient_history` **L509-528** | Geçmiş **yalnız TC** ile eşleşiyor; organ/lezyon/tetkik ayrımı yok → aynı kişinin karaciğer+pankreası tek seride karışıyor (F08) | 🔴 |
| C15 | **L756-820** | Boylamsal grafik/karşılaştırma karışık organları prev vs curr diye kıyaslıyor (F08) | 🔴 |
| C16 | `recist_yanit` **L206-228** | Küre-çapı serisi; `guncel<=0 → CR`; hedef lezyon eşleştirmesi/çap toplamı/yeni lezyon yok (F09) | 🔴 |
| C17 | **L774** | `tumor_cap_mm or 0` → eksik çap **0** oluyor, RECIST'i yanlış **CR**'a sürüklüyor (F09) | 🔴 |
| C18 | **L770-771** | `p_vol==0`/`p_cap==0` iken yüzde değişim `0` gösteriliyor (eksik ≠ değişim yok) | 🔴 |

## 4. Rapor / kayıt / atomiklik (radyoloji_app.py)

| # | Yer | Sorun | Durum |
|---|---|---|---|
| C19 | akış **L676-702** | Ölçüm→rapor→JSON→maske kopyası sıralı; maske kopyası (L692-702) başarısızsa rapor var ama 3D eksik kayıt kalıyor; atomik değil (F10) | 🔴 |
| C20 | `save_patient` **L492-493** | JSON doğrudan yazılıyor (tmp+atomik değil); yarıda kesilirse bozuk kayıt (F10) | 🔴 |
| C21 | **L676 vs L688** | `st.session_state.bulgular` rapordan önce yazılıyor; API hatası (L664 st.stop) eski raporu yeni bulgularla bırakabilir (F21) | 🔴 |

## 5. Kimlik / gizlilik (radyoloji_app.py) — yalnız kod davranışı

| # | Yer | Sorun | Durum |
|---|---|---|---|
| C22 | `save_patient` **L482** | Dosya adında **TC + ad_soyad** (PHI dosya adında) (F22) | 🔴 |
| C23 | `generate_report` **L412** | Anonim modda bile `gecmis_bulgular` tüm geçmiş JSON'u (iç içe kimlik olabilir) gönderiliyor (F22) | 🔴 |
| C24 | sohbet **L887-897** | Chat, **tüm hasta geçmişini** alan-süzmesi olmadan LLM'e yolluyor; rapor yolundaki `ANONYMIZE_FOR_API` burada yok (F22) | 🔴 |

## 6. Klinik kural / prompt (radyoloji_app.py, compare_llm.py)

| # | Yer | Sorun | Durum | Çekirdek |
|---|---|---|---|---|
| C25 | app **L384** | Aynı `≤2/2-4/>4 cm` T eşiği karaciğer+pankreasa (F07) | 🟡 | `clinical_rules` organ-özel |
| C26 | app **L385** | ">180° = irresektabl" blanket kural (F07) | 🟡 | `clinical_rules` kesin karar yok |
| C27 | **3 ayrı prompt** app L380, compare_llm L46, pipeline reporting | Üç farklı sistem promptu → tek kaynak yok (F19) | 🟡 | `reporting.SYSTEM_PROMPT` |

## 7. compare_llm.py (makale yolu)

| # | Yer | Sorun | Durum | Çekirdek |
|---|---|---|---|---|
| C28 | **L100-123** | `numeric_fidelity` her sayıyı sayıyor; alan/birim/bağlam yok; "2 cm³"→20 mm; tam-sayı yuvarlama eşliyor (F15) | 🟡 | `evaluation.numeric_fidelity` |
| C29 | **L37** | `PATIENTS_DB_DIR = C:\Users\ASUS\...` sabit (F11) | 🔴 |
| C30 | **L41-44** | Yalnız Claude+Gemini (6 set değil); model ID app'ten farklı (F19) | 🟡 | `reporting.MODELS` |
| C31 | **L148** | `bulgular = data.get("bulgular") or data` → "bulgular" yoksa tüm hasta JSON'una (kimlik) düşüyor (F22) | 🟡 | `reporting.anonymize_findings` |

## 8. Test / doğrulama altyapısı

| # | Sorun | Durum |
|---|---|---|
| C32 | 63 birim testi var ama **CI yok** (GitHub Actions); otomatik koşmuyor | 🔴 |
| C33 | Uçtan uca (dosya→segmentasyon→rapor→kayıt) **entegrasyon testi yok** | 🔴 |
| C34 | `pipeline/segmentation.py` ve `run.py` **yok** → gerçek kohort koşusu yapılamıyor | 🔴 |

---

## Özet sayılar
- 🟢 Çekirdekte düzeltilmiş + test: ölçüm/mesafe/açı/eşik/sadakat/kural/anonim mantığı (ama **app kullanmıyor**).
- 🟡 Çekirdekte çözüm var, app/compare_llm hâlâ eski: **C1-C7, C25-C28, C30-C31** (≈14).
- 🔴 Hiç yazılmamış / app'te duran: **C8-C24, C29, C32-C34** (≈20).

## Öncelik sırası (çalıştırmadan önce)
1. **C8/C9/C10/C12 (segmentasyon):** `pipeline/segmentation.py` — 5-fold + config yolları + varlık/çıkış-kodu kontrolü.
2. **C1-C7 (app ölçümü):** `extract_metrics`/`calculate_encasement` yerine `measure_case`.
3. **C28/C30/C31 (compare_llm):** çekirdek değerlendirme/şema.
4. **C19-C24 (kayıt/kimlik):** atomik kayıt + alan-süzmeli prompt.
5. **C14-C18 (boylamsal/RECIST):** tetkik+lezyon kimliği; eksik≠0; RECIST'i doğrula veya "araştırma" etiketle.
6. **C32-C34:** CI + uçtan uca runner.
