# Radyolog AI — Tam Durum Raporu (başka bir AI'ya sunmak için)

**Tarih:** 7 Kasım 2026
**Amaç:** Bu belge, projenin mevcut durumunu, yapılan tüm işi, bilinen zayıflıkları ve açık kararları **bağımsız bir AI'nın değerlendirebilmesi** için eksiksiz toplar. Belgenin sonunda "sorular" bölümü var.

---

## 0. Bir cümlede proje

Abdominal BT'den otomatik çıkarılan geometrik tümör metrikleriyle (grounded) LLM radyoloji raporu üretmek ve bu raporu gerçek radyolog raporuyla **nicel** kıyaslayan tekrarlanabilir bir pipeline + değerlendirme çerçevesi. Merkez iddia bir "klinik ürün" değil, **farkı ölçen bir çerçeve.**

**Akış:** BT → iki paralel segmentasyon (nnU-Net tümör/organ + TotalSegmentator damar) → maskelerden geometrik metrikler (`bulgular.json`) → 6 LLM seti aynı grounded prompt'la rapor üretir → raporlar radyolog raporuyla üç eksende (tespit/sadakat/içerik) kıyaslanır.

**Veri:** Stanford Merlin abdominal BT + radyolog raporları (RUA altında, gizli). Segmentasyon modeli MSD'de eğitilmiş nnU-Net, Merlin'de **zero-shot** çalışıyor.

---

## 1. Başlangıç durumu ve tespit edilen 23 hata

Projede iki temel doküman vardı:
- `radyolog_ai_proje_dokumani.md` (6 Kasım) — iyimser, bazı kararları "kilitli" sayan proje özeti.
- `inceleme/2026-10-29_pipeline_inceleme_ve_plan.md` (29 Ekim) — çok daha temkinli bir kod incelemesi: **"mevcut kodun doğrudan yeniden çalıştırılması önerilmez"** ve **23 deney-kritik hata (F01–F23).**

Eski kodun (`radyoloji_app.py`, `compare_llm.py`, Colab notebook'ları) ana sorunu: aynı mantığın birden çok yerde kopyalanıp **birbirinden sapması**. Örnek kritik hatalar:
- **F01/F14:** LLM'e verilen "çap" aslında aksiyel çap değil, **hacim-eşdeğeri küre çapı**ydı; vakaların %15–16'sında T-kategorisini değiştiriyordu.
- **F03:** Maske/BT fiziksel uzayı (origin/spacing/yön) kontrol edilmiyordu; 100 mm ötelenmiş damar maskesi mesafeyi **0 mm** veriyordu.
- **F04:** Damar sarma açısı tüm damarın tek ağırlık merkezinden; uzak voxeller açıyı 90°→16° değiştiriyordu.
- **F13:** Eşik öncesi yuvarlama (9.96 mm → 10.0 → yanlış pozitif) + `df.gt` pandas çakışması.
- **F15:** Sadakat ölçer yanlış bulgudaki/birimdeki sayıyı doğru sayıyordu ("2 cm³" → 20 mm).
- **F18:** Ölçüm uyumu MAE'si **farklı lezyonları** kıyaslıyordu (radyolog hedef lezyon, pipeline en büyük bağlı kütle) → karaciğer korelasyon 0.19'un gerçek sebebi.

---

## 2. Verilen kararlar

| Karar | Seçim |
|---|---|
| Mimari | **Ortak çekirdek** (`pipeline/` modülleri), UI'dan bağımsız |
| Kapsam | **Deney-kritik P1** (F01, F03, F04, F10–F15, F17, F18) + F07/F19/F22 |
| Kohort | Eski 80 vaka = **pilot yeniden-değerlendirme** (bağımsız dış test diye sunulmayacak) |
| Çap | Aksiyel en uzun çap (küre-çapı bug'ı düzeltildi) |
| Segmentasyon | 5-fold ensemble (Colab/GPU'da çalışacak) |

---

## 3. Yapılan iş — `pipeline/` ortak çekirdeği

**11 modül, ~1512 satır, 63 birim testi (stdlib `unittest`), hepsi yeşil.** Her test 29 Ekim incelemesindeki gerçek bir hata senaryosunu (repro) hedefler. Paket import'u yan etkisizdir (GPU/model/API çağırmaz).

| Modül | Çözdüğü hata | Ne yapar |
|---|---|---|
| `config.py` | F11, F19, F20 | Merkezî yol/sürüm; sabit `C:\Users\ASUS\...` yolları kalktı; durum sözlüğü (success/empty_mask/not_assessed/failed) |
| `imaging.py` | F03, F06 | Dünya-koordinat mesafesi (KDTree); origin/spacing/yön doğrulama; oblik bayrağı; bellek-güvenli resample (`affine_transform`) |
| `measurements.py` | F01, F04, F05, F14 | Lezyon-bazlı **tek** aksiyel çap + ayrı hacim; 26-komşuluk; deterministik (rastgele örnekleme yok); yerel damar sarma açısı; `measure_case` (BT→ölçüm tek giriş, oblik kapısı) |
| `evaluation.py` | F13, F15, F17, F18 | RAW değer eşiği; bağlam+birim eşleşmeli sadakat **proxy'si**; ondalık koruyan ayrıştırma; ayrı paydalar + Wilson GA |
| `matching.py` | F18 | Lezyon eşleştirme (açık politika / konum-bazlı); eşleşmeyenler ayrı sayılır |
| `reference.py` | F18 | GT referans tablosu: **kaynak cümle + tarih + transkribe eden + revizyon** (denetim izi) |
| `clinical_rules.py` | F07 | Organ-özel eşikler; "boyut kategorisi" (≠ T-evresi); eksikte "değerlendirilemez"; rezektabilite kesin karar vermez |
| `reporting.py` | F07, F19, F22 | Ortak şema + prompt; **izinli-alan** anonimleştirme; kural-tabanlı şablon (%100 sadık baseline); boş/kesik yanıt ≠ başarı; Google'a max_tokens aktarımı |
| `manifest.py` | F12, F18 | Tek kaynak vaka listesi; POS/NEG→şema; aynı-kişi çakışma denetimi |
| `runner.py` | F10, F19 | Hash'li cache anahtarı (girdi+ağırlık+sürüm); atomik yaz-doğrula-rename; iş izolasyonu |

**Doğrulanmış repro örnekleri (test olarak):** 100 mm ötelenmiş maske → mesafe 100 (0 değil); 9.96 → negatif; "aorta 20 mm" tümör çapını onaylamıyor; "2 cm³" ≠ 20 mm; 41.2 bölünmüyor; uzak kesit sarma açısını değiştirmiyor; şablon raporu %100 sadakat alıyor.

**Gerçek veri kontrolü:** `imaging.load_nifti` gerçek bir Merlin BT'sini sorunsuz okudu (512×512×424, 0.63×0.63×1.0 mm, oblik 0°, HU −1024..1766).

---

## 4. İkinci tur — sağlamlaştırma (eleştirel denetim sonrası)

Kendi kodumu eleştirel inceledikten sonra bulunan ve giderilen zayıflıklar:
- **K1:** `resample` tüm grid'i üretiyordu (~5.3 GB). `affine_transform` ile bellek-güvenli hale getirildi.
- **K2:** Ölçüm oblik kontrolü yapmıyordu. `measure_case` artık güçlü oblik hacimde yanlış çap üretmek yerine `not_assessed` veriyor.
- **K3:** Temas mesafesi "yüzey-voxel-merkezi" olarak açıkça etiketlendi.
- **K4:** Lezyon sayısı artık 6- ve 26-komşulukla **ayrı** raporlanıyor (tanım farkı görünür).
- **K5:** `run_jobs` boş-obj kenar durumu düzeltildi.
- **Proxy etiketi:** Sadakat çıktısı "otomatik-proxy, nihai karar radyolog" olarak işaretlendi.

---

## 5. DÜRÜST ZAYIFLIKLAR — elimizi zayıflatan şeyler

### 5a. Kod tarafı (çoğu giderildi; kalan tanımsal seçimler)
- Temas mesafesi **merkez-bazlı** (bitişik voxel 1 mm okunur, 0 değil) — tanım olarak belgelendi.
- Lezyon bağlantılılığı **26-komşuluk** seçildi; 6 vs 26 sonucu etkiler — artık ikisi de raporlanıyor ama "seçim" bir karardır.
- Sadakat ölçer **otomatik proxy**; anahtar-kelime örtüşmesi hem fazla hem eksik kredi verebilir.

### 5b. Bilimsel / yapısal (kod çözmez — çerçeve/veri kararı)
1. **Segmentasyon gerçek-referansı YOK.** Merlin'de tümör maskesi olmadığından Dice/segmentasyon doğruluğu ölçülemez. Tüm "ölçüm doğruluğu" radyolog **raporundaki sayıya** karşı. → zero-shot, betimleyici çerçeve.
2. **GT bağımsız re-okuma değil;** radyolog raporundan kullanıcı transkripsiyonu. Referans boyutlar sürümler arası ciddi değişmiş (ör. 4.3→6.1, 3.5→8.7 cm) — bu yüzden `reference.py` denetim izi eklendi.
3. **Kohort bağımsızlığı:** hataları bulup düzelttiğimiz aynı 80 vakada test → pilot yeniden-değerlendirme; bağımsız dış test değil.
4. **Pankreas eksik segmentasyon (−14 mm bias):** model tarafı; zero-shot'ın yapısal sınırı.
5. **Domain shift:** nnU-Net MSD'de eğitildi, Merlin'de zero-shot. Vaka eklemek çözmez.
6. **Rezektabilite/lokasyon doğrulanmadı:** kohortta damar tutulumu nadir (1/80); lokasyon (F02) kapsam dışı (kod artık yanlış üretmek yerine null veriyor).
7. **Model karşılaştırması confounded:** açık modeller 4-bit/Colab, Claude/Gemini API; token/kuantizasyon farkları.

---

## 6. 🔴 Veri gizliliği olayı ve giderilmesi (ÖNEMLİ)

**Bulgu:** Stanford Merlin veri seti dosyaları (16 MB `reports_final.xlsx` = radyolog raporları + GT CSV'leri) **GitHub'a (private repo) commit edilip push edilmişti.** Merlin RUA altında — üçüncü-taraf dağıtımı yasak.

**Yapılan (kullanıcı onayıyla, repo private):**
- İzole klonda `git filter-branch` + `gc` ile **tüm geçmişten silindi** → 0 referans, `reports_final.xlsx` blob'u ulaşılamaz.
- `master` force-push: `46ccc3f` → `158ece9` (temiz).
- `merlin/` artık `.gitignore`'da.
- 24.6 MB tam yedek bundle alındı (yerel scratchpad'de).

**Açık uç:** Eski commit `46ccc3f` GitHub sunucu tarafında bir süre SHA ile erişilebilir kalabilir (GitHub GC'ye kadar); private olduğu için risk düşük. **RUA gereği danışmana bildirim önerildi.**

---

## 7. Mevcut git durumu (7 Kasım 2026)

**GitHub `master` = `158ece9`** (Merlin'siz; ama müsvedde scriptler hâlâ master'da; pipeline henüz master'da değil).

**Çalışılan dal (lokal + yedek):** `claude/radyolog-ai-proje-dokumani-764226`
```
0aa7505 chore: musvedde scriptleri kaldir (deneme/debug/damarbul)
758a830 chore: merlin/ gitignore (RUA)
50d223d feat(pipeline): hatasiz ortak cekirdek + degerlendirme (F01-F22)
8e8e5fe feat: LLM karsilastirma... (merlin temizlenmis)
8943653 Initial commit
```

| Yer | Merlin verisi | Müsvedde scriptler | Pipeline çekirdeği |
|---|---|---|---|
| GitHub `master` | ✅ silindi | ⏳ hâlâ var | ❌ yok |
| Lokal feature dalı | ✅ yok | ✅ silindi | ✅ var (63 test) |

**Bekleyen işler:**
- Pipeline dalı GitHub'da değil (yalnız lokal + yedek) → PR ile master'a taşınabilir (master'ı da temizler).
- **Ana klasör (main checkout) resync gerekiyor** — eski geçmişte; resync öncesi `merlin/` klasörü repo dışına yedeklenmeli (tracked dosyalar silinir, .nii.gz gitignore'da güvende).

---

## 8. Açık kararlar (danışmanla netleşecek)

- [ ] **Merkez iddia cümlesi** — "grounded pipeline + nicel değerlendirme çerçevesi; hata LLM'den değil pipeline ölçümünden" çerçevesi onaylansın mı?
- [ ] **Claude Opus mu Sonnet mi** (raporlama aşamasında lazım).
- [ ] **Rezektabilite ve lokasyon** makalede yer alacak mı, yoksa "gelecek çalışma" mı?
- [ ] **Kohort:** eski 80 (pilot yeniden-değerlendirme) kesinleşsin mi?

---

## 9. Kalan yol haritası

1. **`segmentation.py` adaptörü** — nnU-Net 5-fold + TotalSegmentator için temiz çağrı arayüzü (çalışması Colab/GPU).
2. **Uçtan uca `run.py`** — BT → maske → ölçüm → `bulgular.json` → rapor → değerlendirme zinciri (manifest üzerinden).
3. **Gerçek kohort manifesti** — yerel POS/NEG + karaciğer JSON'larından `cohort.csv`.
4. **Colab'da segmentasyon** → maskeler (GPU).
5. **LLM raporlama** (API anahtarı + Opus/Sonnet kararı).
6. **Değerlendirme (A/B/C)** + makale tabloları.

> Not: Gerçek uçtan uca **koşu** iki şeyi bekliyor — Colab'da üretilmiş maskeler ve `.env`'de API anahtarları. Çekirdek + değerlendirme ise şu an tam çalışır ve testlidir.

---

## 10. Başka AI'ya sorular (görüş almak için)

1. Çekirdek mimarisi ve F01–F22 düzeltmeleri sağlam mı; gözden kaçan bir hata sınıfı var mı?
2. **Segmentasyon gerçek-referansı olmadan** ölçüm doğruluğu iddiası nasıl daha savunulabilir çerçevelenir?
3. **Lezyon eşleştirme politikası** (konum yoksa en-büyük-çap) kabul edilebilir mi; daha iyi bir yöntem var mı?
4. **Merkez iddia** (hata LLM'den değil pipeline'dan) otomatik sadakat proxy'sine dayanıyor — bu iddiayı güçlendirmenin yolu nedir?
5. Kohort bağımsızlığı / pilot yeniden-değerlendirme çerçevesi hakem için yeterince dürüst mü?
6. Model karşılaştırması (4-bit açık modeller vs API) confounding'ini nasıl ele almalı?
7. RUA veri sızıntısı giderme adımları (force-push + geçmiş temizliği) yeterli mi; başka ne yapılmalı?
8. Eksik/yanlış bulduğun başka bir şey var mı?

---

*Bu rapor, 6–7 Kasım 2026 oturumunda yapılan işi eksiksiz özetler. Kod: 11 modül, 63 test, hepsi yeşil. Detaylı hata listesi için `inceleme/2026-10-29_pipeline_inceleme_ve_plan.md`, iddia/itiraz eşlemesi için `savunulabilirlik_notu.md`, modül dokümantasyonu için `pipeline/README.md`.*
