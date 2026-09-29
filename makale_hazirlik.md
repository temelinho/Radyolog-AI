# Radyoloji AI Sistemi — Dergi Makalesi Hazırlık Dokümanı

*Amaç: Hocanızla birlikte, Türkçe, hakemli dergi makalesini elle yazarken başvuracağınız temel çalışma dokümanı. Metin buradan yazılacak; bu doküman iskelet, içerik notları, kaynakça ve açık kararları içerir.*

---

## ★ ÇALIŞMANIN KONUSU VE ANA TEZİ (Kararlaştırıldı)

**Tek cümlelik konu:** Çok organlı BT görüntülerinde, segmentasyondan hekim etkileşimine kadar **uçtan uca entegre** bir yapay zeka platformu geliştirmek; ve bu platformun temel bilimsel iddiası olarak, büyük dil modelini **ham görüntü yerine yalnızca deterministik geometrik metriklerle** besleyerek (grounded / temellendirilmiş raporlama) klinik açıdan güvenilir, **halüsinasyonu azaltılmış** radyoloji raporları üretmek.

**Konumlanma:** Bir *sistem/platform makalesi*; klinik kapsam **6 organ genişliği** üzerinden sunulur (ana güçlü yön), **pankreas kanseri rezektabilitesi** ise en detaylı vitrin vaka olarak öne çıkarılır. Nicel doğrulama **public MSD/TCIA verisinde altı organ için segmentasyon Dice skoru** ile yapılır; geometrik ölçüm ve LLM raporları gösterim (demonstration) düzeyinde sunulur.

**Araştırma soruları:**
1. Segmentasyon → deterministik geometri → LLM boru hattı, 6 farklı organda tutarlı ve klinik olarak anlamlı metrikler (hacim, çap, damar mesafesi, sarma açısı) üretebiliyor mu?
2. Metriklerle temellendirilmiş LLM raporu, prompttaki klinik kurallara (NCCN rezektabilite, RECIST, T-evreleme) ne ölçüde sadık kalıyor — yani halüsinasyon ne kadar azalıyor?
3. Bu entegre sistem, gerçek pilot vakalarda klinik olarak yorumlanabilir ve tutarlı çıktı (rapor + 3D + takip) veriyor mu?

**Üç temel katkı (çıpa):**
1. **Silo yapısının kırılması** — segmentasyondan hekim-chatbot diyaloğuna uzanan tam entegre ilk platform.
2. **Grounded raporlama** — LLM yalnızca matematiksel metriklerle beslendiği için halüsinasyon minimize.
3. **Hekim-dostu açıklanabilirlik** — 3D Plotly görselleştirme + LLM chatbot ile karar sorgulanabilir.

**Ana rakip/karşılaştırma hattı:** Genişlik ve entegrasyonda PAN-VIQ (sadece pankreas), MedRegion-CT (pseudo-mask), LesionLocator (sadece takip), 3D-CT-GPT/RaDialog (sadece rapor/chatbot) — hiçbiri altı organı + geometri + rapor + takip + chatbot'u bir arada sunmuyor. (Detay: Tablo 1.)

**Not:** Yukarıdaki "grounded LLM raporlama" iddiası, istenirse aynı metrik JSON'ları üzerinden **Gemini vs Claude pilot kıyaslamasıyla** güçlendirilebilir (opsiyon — bkz. Bölüm 8).

---

## 0. Önce Netleştirilmesi Gereken Kritik Noktalar (Hocayla konuşulacak)

Bunlar makalenin gövdesini yazmaya başlamadan karara bağlanmalı. Yanlış giderse tüm bölümleri etkiler.

1. **LLM tutarsızlığı — EN ÖNEMLİ. (Kod incelemesiyle netleşti.)** Durum sanılanın tersi: **çalışan kod hâlâ tamamen Google Gemini 2.5 Flash kullanıyor**, Claude'a hiç geçilmemiş. Kanıt:
   - `radyoloji_app.py` → rapor üretimi (satır 349) ve chatbot (satır 824): `model="gemini-2.5-flash"`; `from google import genai`; arayüz başlığı (satır 489): "nnU-Net + TotalSegmentator + **Gemini AI**".
   - `deneme2.py` → yine Gemini 2.5 Flash.
   - Buna karşılık **sadece dokümanlar** Claude diyor: README, `requirements.txt` (anthropic satırı), literatür taraması PDF'i, `generate_pdf.py`.
   → Yani "Claude'a güncelleme" dokümanlara yansımış ama koda yansımamış. Karar: **(a)** kodu gerçekten Claude'a çevir + 4 hastayı yeniden çalıştır (makale Claude anlatır), **(b)** makaleyi dürüstçe Gemini 2.5 Flash üzerinden yaz (dokümanları Gemini'ye geri düzelt), **(c)** "LLM-agnostik mimari; mevcut sürüm Gemini 2.5 Flash, Claude ile de uyumlu" diye konumlandır. Öneri: **(a) veya (c)**. Hangisi seçilirse README/requirements/PDF ve kod **birbiriyle tutarlı** hâle getirilmeli — hakem bu çelişkiyi hemen görür.

2. **Deneysel doğrulama (Results) — KARARLAŞTIRILDI.** Veri kaynağı **public MSD/TCIA**; gerektiği kadar hasta, **altı organ için** segmentasyon çalıştırılabilir. Doğrulama **segmentasyon Dice skoru** (organ bazında, isteğe bağlı IoU) üzerinden yapılacak.
   - **Kapsam:** Makale artık "sadece 4 hasta pilot" değil; her organ için makul bir kohort (öneri: organ başına ~10–20 vaka) ile Dice tablosu verilebilir. İlk pankreas vakası (aşağıda) detaylı gösterim olarak kalır.
   - **Model kaynağı (netleşti):** Kendi eğitimimiz YOK. Isensee ve ekibinin **Zenodo'da yayımladığı hazır/donmuş nnU-Net ağırlıkları** kullanıldı (kayıt 4485926, v2.1, 2021; `Setup.py` bunları indiriyor). Yöntem'de "pretrained nnU-Net models (Isensee et al., Zenodo)" diye yazılmalı; "eğittik" DENMEMELİ.
   - **⚠️ Dice'ın anlamı — dürüst çerçeve:** Bu ağırlıklar MSD **eğitim** verisinde eğitildi ve MSD **resmi test** setinin etiketleri halka açık DEĞİL. Bu yüzden "kendi test setimizde biz doğruladık" DENEMEZ. İki geçerli yol:
     - **(a) Yayınlanmış Dice'ı alıntıla** (pratik, dürüst): "Kullandığımız hazır modellerin MSD'de bildirilen Dice performansı" → Isensee 2021 / MSD Decathlon (Antonelli 2022) kaynak gösterilir. Makalenin sistem/platform doğası buna uygun.
     - **(b) Dış-doğrulama** (daha güçlü, daha çok iş): Modelin görmediği farklı public sette (ör. karaciğer LiTS, pankreas NIH-Pancreas-CT) GT ile kendi Dice'ını hesapla. Etiket tanımları farklı olabilir, dikkat.
   - **Lisans:** Zenodo ağırlıkları **CC-BY-NC 4.0** (ticari olmayan). Makalede atıf zorunlu + "klinik/ticari kullanım ek lisans gerektirir" notu.
   - **Neyin doğrulandığı, neyin gösterildiği:** Bu tasarımda **segmentasyon katmanı nicel doğrulanır (Dice)**; geometrik ölçümler (mesafe, sarma açısı) ve LLM raporları ise **doğrulanmaz, gösterilir (demonstration)**. Bu ayrım Sınırlılıklar'da dürüstçe yazılmalı — aksi halde "ölçümleriniz/raporunuz doğru mu?" sorusuna cevabınız olmaz.
   - **Elde hazır somut örnek vaka** (`deneme2.py`'den): Portal venöz faz BT (kesit 2.5 mm, piksel 0.64 mm); pankreas hacmi 41.89 cm³; pankreas başında 37.3 mm solid lezyon (27.12 cm³); portal ven 293° sarma; aorta mesafesi 2.32 mm, sarma yok → T2, lokal ileri/irresektabl. Bulgular'da figürlü örnek olarak kullanılır.

3. **Hasta verisi ve etik.** Gerçek hasta BT'si kullanıldıysa etik kurul onayı / aydınlatılmış onam ifadesi gerekir. TC kimlik no gibi kişisel veriler işleniyor (`hasta_bilgi["tc"]`) — makalede anonimleştirme ve KVKK/etik beyanı şart.

4. **Hedef dergi.** Türkçe hakemli seçenekler: *Diagnostic and Interventional Radiology* (İngilizce), *Tıp Fakültesi dergileri*, *Türk Radyoloji Dergisi/TJR*, ya da mühendislik tarafında *bilişim/biyomedikal mühendisliği* dergileri. Dergi seçimi format ve uzunluğu belirler.

5. **API anahtarı güvenliği.** `radyoloji_app.py` içinde Google API anahtarı **düz metin (hardcoded)** yazılı. Makale/repo paylaşımından önce mutlaka iptal edilip `.env`'e taşınmalı. (Makale içeriğini etkilemez ama repo public olacaksa kritik.)

---

## 1. Çalışılan Başlık Önerileri

- ★ **(Seçilen eksene en uygun)** "Çoklu Organ BT Analizinde Uçtan Uca Yapay Zeka: Segmentasyon, Geometrik Ölçüm ve Temellendirilmiş (Grounded) LLM Raporlaması ile Entegre Bir Platform"
- "Halüsinasyonu Azaltan Temellendirilmiş Raporlama: Altı Organ İçin Segmentasyondan Hekim Etkileşimine Entegre Bir Radyoloji AI Platformu (Pilot Çalışma)"
- Kısa alternatif: "Radyoloji AI: BT'den Klinik Karara Entegre Bir Yapay Zeka Boru Hattı"

---

## 2. Makale İskeleti (IMRAD) ve Her Bölümde Ne Yazılacak

### Öz (Abstract) — ~250 kelime, yapılandırılmış
Amaç / Yöntem / Bulgular / Sonuç alt başlıklarıyla. En son yazılır. İçermeli: problem (VLM halüsinasyonu + silo çözümler), yaklaşım (segmentasyon → geometri → LLM), 6 organ kapsamı, temel katkı.
**Anahtar kelimeler:** yapay zeka, bilgisayarlı tomografi, segmentasyon, nnU-Net, büyük dil modeli, rezektabilite, pankreas kanseri, RECIST.

### 1. Giriş (Introduction)
- Radyolojide iş yükü ve raporlama darboğazı.
- Yapay zekanın segmentasyon başarısı (nnU-Net, TotalSegmentator) ama **klinik karara** dönüşememesi.
- VLM tabanlı doğrudan rapor üretiminin **halüsinasyon** riski.
- Literatürdeki **silo problemi**: kimi sadece segmentasyon, kimi sadece raporlama, kimi sadece takip.
- Boşluk (gap) cümlesi → bu çalışmanın amacı ve katkıları (madde madde 3 novelty).

### 2. İlgili Çalışmalar / Literatür (Related Work)
Literatür taramasındaki 5 kategoriyi buraya taşı (Bölüm 4'teki kaynakça hazır):
- 2.1 Temellendirilmiş LLM raporlama (CLarGen, MARCH, 3D-CT-GPT, RadFM/M3D, CT2Rep, Agentic PET/CT, MedRegion-CT)
- 2.2 Tümör-damar rezektabilite (PAN-VIQ, Viviers, DoDNet-CAD, Ochs)
- 2.3 Boylamsal takip (LongiSeg, Detect-Then-Track, Whole-Body Tracking, LesionLocator)
- 2.4 Segmentasyon altyapıları (nnU-Net, TotalSegmentator)
- 2.5 LLM değerlendirme ve chatbot (GREEN, CRIMSON, ReportQA, Mergen-RECIST, RaDialog, XrayGPT)
- Her alt başlık sonunda "bizden farkı" cümlesi (literatür taramasında hazır var).

### 3. Yöntem (Materials and Methods)
Sistem mimarisini anlat. Şekil 1 = mimari diyagram (README'deki akış).
- 3.1 **Genel mimari / boru hattı**: BT (.nii.gz) → nnU-Net → TotalSegmentator → geometrik hesaplama → LLM → 3D görselleştirme → arayüz.
- 3.2 **Segmentasyon**: 6 nnU-Net task'i (Task003 Karaciğer, 006 Akciğer, 007 Pankreas, 008 Hepatik Damar, 009 Dalak, 010 Kolon). **Kendi eğitimimiz yok**; Isensee ve ekibinin Zenodo'daki hazır/donmuş ağırlıkları kullanıldı (DOI: 10.5281/zenodo.4485926, CC-BY-NC 4.0). Damarlar için TotalSegmentator (aorta + portal ven). 3d_fullres, tek fold (f=0) — kodda `run_nnunet`.
- 3.3 **Geometrik analiz** (kodun matematiği — makalenin en özgün teknik kısmı):
  - Hacim: voxel sayısı × voxel hacmi (mm³→cm³).
  - Tümör çapı: eşdeğer küre yaklaşımı, `d = (V·6/π)^(1/3)`.
  - Tümör lokasyonu: organ ekseninde göreli konum (baş/gövde/kuyruk).
  - Damar mesafesi: Öklid mesafe dönüşümü (`distance_transform_edt`) ile mm cinsinden en yakın mesafe.
  - **Sarma açısı (encasement)**: damar merkezine göre temas voxellerinin açısal dağılımı; 360°'den en büyük boşluk çıkarılarak sarma derecesi. (Formül `calculate_encasement` fonksiyonundan.)
- 3.4 **Rezektabilite değerlendirmesi**: NCCN kriterleri, >180° sarma = lokal ileri/irresektabl (pankreas için).
- 3.5 **LLM raporlama**: sistem promptu (15 yıllık abdominal radyolog rolü), ACR/ESR standartları, sadece metriklere dayalı üretim (temellendirme), T-evreleme kuralları (T1 ≤2 cm, T2 2–4 cm, T3 >4 cm), NCCN rezektabilite kuralı (>180° sarma), organa özgü kurallar (dalak splenomegali derecesi, akciğer/kolon lezyon yorumu), düşük sıcaklık (0.3). **Mevcut kodda model = Gemini 2.5 Flash** — makalede yazılacak model adı Bölüm 0.1 kararına göre kesinleşecek ve kodla tutarlı olmalı.
- 3.6 **Boylamsal / RECIST takibi**: TC bazlı geçmiş sorgulama, hacim % değişimi, PR/SD/PD sınıflaması (küçülme ≤−30%, artış ≥+20%).
- 3.7 **3D görselleştirme**: marching cubes + Gaussian yumuşatma, Plotly Mesh3d.
- 3.8 **Etkileşimli chatbot**: rapor + tüm geçmiş bağlamıyla soru-cevap.
- 3.9 **Uygulama / arayüz**: Streamlit; hasta veritabanı (JSON); PDF çıktısı.

### 4. Bulgular (Results)
- **4.1 Segmentasyon doğruluğu (nicel — ana Results tablosu):** Altı organ için MSD/TCIA held-out veride Dice skoru (± std), gerekirse IoU. **Tablo 3** = organ × Dice. Modellerin literatürdeki yayınlanmış skorlarıyla kısa kıyas eklenebilir.
- **4.2 Sistem çıktılarının gösterimi:** Çalıştırılan kohortun özet tablosu — organ, tümör hacmi/çapı, damar mesafesi, sarma açısı, sistemin verdiği rezektabilite/RECIST kararı.
- **4.3 Örnek vaka (detaylı gösterim):** `deneme2.py`'deki pankreas vakası — 37.3 mm lezyon, 27.12 cm³, portal ven 293° sarma, T2 lokal ileri/irresektabl. Bu vaka için: (1) segmentasyon maskesi, (2) 3D rekonstrüksiyon figürü, (3) sistemin ürettiği tam rapor metni, (4) chatbot örnek diyaloğu.
- **4.4** (varsa) örnek boylamsal takip grafiği; işlem süresi (BT başına segmentasyon + rapor) — pratik uygulanabilirlik kanıtı.
- *Vurgu: Nicel doğrulama segmentasyon Dice ile sınırlı; geometrik ölçüm ve rapor kalitesi gösterim düzeyinde (bkz. Sınırlılıklar).*

### 5. Tartışma (Discussion)
- 3 novelty'yi literatür karşısında konumlandır (silo kırma, grounded/halüsinasyon azaltma, hekim-dostu açıklanabilirlik).
- Güçlü yönler.
- **Sınırlılıklar (dürüstçe)**: Nicel doğrulama **yalnızca segmentasyon Dice** ile sınırlı — geometrik ölçümler (mesafe, sarma açısı) ve LLM raporları radyolog altın standardıyla bağımsız doğrulanmadı, gösterim düzeyinde. Ayrıca: sadece 2 damar (aorta+portal ven), NCCN'in yalnız pankreasa özgü olması, public veri (klinik/prospektif değil), LLM çıktısının hâlâ hekim onayı gerektirmesi.
- Gelecek çalışmalar: prospektif validasyon, daha çok damar, çoklu merkez, radyolog kör okuma çalışması.

### 6. Sonuç (Conclusion)
Kısa; katkı + klinik potansiyel + "klinik kullanım için ek doğrulama gerekir" notu.

### Beyanlar
Etik onay, çıkar çatışması, yazar katkıları, finansman, veri/kod erişilebilirliği (GitHub linki).

---

## 3. Üç Temel Novelty (Tartışma ve Giriş için hazır)

1. **Silo yapısının kırılması** — segmentasyondan hekim-chatbot diyaloğuna uzanan tam entegre ilk platform.
2. **Temellendirilmiş raporlama (grounded reporting)** — LLM yalnız matematiksel metriklerle (hacim, çap, mesafe, açı) besleniyor; görüntü doğrudan modele verilmediği için halüsinasyon minimize ediliyor.
3. **Hekim-dostu açıklanabilirlik** — 3D Plotly görselleştirme + LLM chatbot ile hekim, yapay zekanın kararını sorgulayabiliyor.

---

## 4. Kaynakça Taslağı (literatürdeki 23 çalışma)

*Not: Aşağıdaki künyeler literatür taramasından çıkarıldı. Dergi formatına (Vancouver/APA) göre son hâli verilecek; her birinin tam sayfa no / DOI'si yazımda tamamlanmalı. Bazıları 2026 tarihli — yayın durumları (preprint mi, yayınlandı mı) kontrol edilmeli.*

**Segmentasyon altyapıları**
1. Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*. 2021;18(2):203-211.
1b. Isensee F, Jäger PF, Kohl SAA, Petersen J, Maier-Hein KH. Pretrained models for 3D semantic image segmentation with nnU-Net (Version 2.1) [Data set]. *Zenodo*. 2021. DOI: 10.5281/zenodo.4485926. (Kullanılan hazır ağırlıklar; CC-BY-NC 4.0)
1c. Antonelli M, et al. The Medical Segmentation Decathlon. *Nature Communications*. 2022;13:4128. (MSD veri setleri ve bildirilen performans)
2. Wasserthal J, et al. TotalSegmentator: Robust segmentation of 104 anatomical structures in CT images. *Radiology: Artificial Intelligence*. 2023;5(5):e230024.

**Grounded LLM raporlama**
3. Kyung S, et al. MedRegion-CT. *arXiv*. 2025.
4. Maye-Lasserre T, et al. CLarGen. *arXiv*. 2026.
5. Lin Y, et al. MARCH. *ACL*. 2026.
6. Choi H, et al. Agentic End-to-End PET/CT reporting. *Journal of Nuclear Medicine*. 2026.
7. Chen H, et al. 3D-CT-GPT. *arXiv*. 2024.
8. Wu C, et al. RadFM. *arXiv*. 2023.
9. Zhang G, et al. M3D. *arXiv*. 2024.
10. Zhang G, et al. CT2Rep. *arXiv*. 2024.

**Tümör-damar rezektabilite**
11. Zhang Y, et al. PAN-VIQ. *npj Digital Medicine*. 2025.
12. Viviers C, et al. Probabilistic tumor-vessel boundary uncertainty. *ICCV Workshop*. 2023.
13. Zhao T, et al. DoDNet-based CAD for pancreatic resectability. *International Journal of Surgery*. 2025.
14. Ochs V, et al. Resectability prediction with Swin-UNETR. *MIDL*. 2026.

**Boylamsal takip**
15. Kirchhoff Y, et al. LongiSeg. *arXiv*. 2024.
16. Cai J, et al. Detect-Then-Track. *IEEE TMI*. 2022.
17. Roy S, et al. Whole-Body Soft-Tissue Tracking. *MICCAI*. 2024.
18. Rokuss M, et al. LesionLocator. *arXiv*. 2025.

**LLM değerlendirme, RECIST ve chatbot**
19. Ostmeier S, et al. GREEN. *EMNLP*. 2024.
20. Baharoon M, et al. CRIMSON. *arXiv*. 2026.
21. ReportQA. *arXiv*. 2024.
22. Mergen M, et al. LLaMA-3.3 ile RECIST evrelemesi. *Scientific Reports*. 2026.
23. Lee KH, et al. 8-LLM klinik kıyaslama ve halüsinasyon analizi. *JMIR Medical Informatics*. 2025.
24. Liu Q, et al. Exploring the Boundaries of GPT-4 in Radiology. *arXiv*. 2023.
25. Pellegrini C, et al. RaDialog. *arXiv*. 2023/2025.
26. Thawkar O, et al. XrayGPT. *arXiv*. 2023.

**Ek (yazımda eklenecek):** NCCN Pankreas Kanseri Kılavuzu; RECIST 1.1 (Eisenhauer et al., Eur J Cancer 2009); Medical Segmentation Decathlon (Antonelli et al.).

---

## 5. Ana Karşılaştırma Tablosu (Makaledeki Tablo 1 taslağı)

Her satır bir çalışma; sütunlar: Segmentasyon | Organ kapsamı | Damar | Geometrik ölçüm | LLM rapor | Boylamsal takip | Chatbot/arayüz. "Bizim sistem" en altta tüm sütunlarda ✓ ile öne çıkar. (Detaylı içerik literatür taramasında mevcut; tabloya dönüştürülecek.)

| Çalışma | Seg. | Organ | Damar | Geometri (mm/°) | LLM Rapor | Takip | Chatbot |
|---|---|---|---|---|---|---|---|
| PAN-VIQ | ✓ | Pankreas | 5 damar | ✓ | ✗ | ✗ | ✗ |
| MedRegion-CT | pseudo | Çoklu | Büyük damar | kısmi | ✓ | ✗ | ✗ |
| LesionLocator | ✓ | Tüm vücut | ✗ | ✗ | ✗ | ✓ | ✗ |
| 3D-CT-GPT | ✗ | Akciğer | ✗ | ✗ | ✓ | ✗ | VQA |
| RaDialog | ✗ | Göğüs | ✗ | ✗ | ✓ | ✗ | ✓ |
| **Bu çalışma** | ✓ | **6 organ** | Aorta+Portal | ✓ | ✓ | ✓ RECIST | ✓ |

---

## 6. Gerekli Figürler ve Tablolar Listesi

- **Şekil 1**: Sistem mimarisi / boru hattı diyagramı.
- **Şekil 2**: Geometrik analiz şeması (mesafe + sarma açısı hesabının görsel açıklaması).
- **Şekil 3**: Örnek 3D rekonstrüksiyon (organ + tümör + damar).
- **Şekil 4**: Boylamsal tümör hacim trend grafiği + RECIST karar ekranı.
- **Şekil 5**: Örnek üretilmiş rapor + chatbot ekran görüntüsü.
- **Tablo 1**: Literatür karşılaştırma tablosu (Bölüm 5).
- **Tablo 2**: Desteklenen organlar ve nnU-Net task eşlemesi.
- **Tablo 3** (varsa): Doğrulama sonuçları / ölçüm metrikleri.

---

## 7. Önerilen Yazım Sırası (birlikte ilerlerken)

1. Bölüm 0 kararlarını hocayla netleştir (özellikle LLM ve Results).
2. Yöntem (Bölüm 3) — sistem sizde en net olan kısım, buradan başlamak kolay.
3. İlgili çalışmalar (Bölüm 2) — literatür taraması hazır, tabloya + prosaya dönüştür.
4. Giriş (Bölüm 1).
5. Bulgular + Tartışma.
6. Öz + Sonuç (en son).
7. Kaynakça son hâli + figürler.

---

---

## 8. Opsiyon: Grounded Raporlamada Gemini vs Claude Pilot Kıyaslaması

*Ana ekseni (grounded raporlama) güçlendirebilecek ikincil bir katkı. Zorunlu değil; hocayla karar.*

**Neden temiz bir tasarım:** Geometrik metrikler (hacim, çap, mesafe, sarma açısı) Python/nnU-Net/scipy'den çıkıyor — LLM'den bağımsız ve deterministik. Sistem her hastanın metriklerini `patients_db`'ye JSON kaydediyor. Aynı JSON iki modele de verildiğinde **girdi birebir sabit, sadece LLM değişken** olur → kontrollü karşılaştırma.

**Karşılaştırılabilecek boyutlar:**
- Metrik sadakati / halüsinasyon (verilen sayıları doğru yansıtma, uydurma bulgu eklememe).
- Prompttaki klinik kurallara uyum (>180° → irresektabl, T-evreleme, RECIST PR/SD/PD).
- Klinik kalite ve okunabilirlik (radyolog/hoca kısa Likert puanı).
- İsteğe bağlı otomatik metrik: GREEN veya CRIMSON (literatür Kategori 5).

**Literatür bağlantısı:** Kategori 5'e (GREEN, CRIMSON, 8-LLM halüsinasyon kıyası) doğrudan oturur.

**Uyarılar:** n=4 küçük → "pilot karşılaştırma", kesin üstünlük iddiası yok. Puanlama için referans/rubrik gerekir.

**Teknik yol:** Kodu tümden Claude'a çevirmeye gerek yok; mevcut Gemini akışının yanına, kayıtlı JSON'dan Claude (anthropic API) ile de rapor üreten küçük bir kıyas scripti eklenebilir → aynı 4 hasta iki modelde yan yana.

---

*Bir sonraki adım: Konu ve eksen netleşti. Artık birlikte hangi bölümden başlamak isterseniz (öneri: Yöntem) o bölümü açıp cümle cümle yazmaya geçebiliriz. Bölüm 0'daki LLM tutarlılık kararı da yazıma başlamadan verilmeli.*
