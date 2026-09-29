# YÖNTEM Bölümü — Yazım Kılavuzu

*Nasıl kullanılır: Her alt başlıkta "Ne yazacaksın" kısmını oku, "Kullanılacak sayılar/olgular"ı metnine yerleştir, "Örnek açılış cümlesi"ni kendi cümlenle değiştir. Kılavuzdaki cümleleri OLDUĞU GİBİ kopyalama — kendi sözcüklerinle yaz.*

**Genel yazım kuralları (Yöntem'e özgü):**
- Geçmiş zaman + kişisel olmayan dil: "…hesaplandı", "…kullanıldı" (─ "ben yaptım" değil).
- Her hazır araç/model için atıf ver (nnU-Net, TotalSegmentator, MSD).
- Parametreleri net yaz (kesit kalınlığı, model sürümü, eşik değerleri). Hakem tekrarlanabilirlik ister.
- İddia/yorum YOK — o Tartışma'ya ait. Burada sadece "ne yaptık, nasıl yaptık".
- "Eğittik" DEME (modelleri siz eğitmediniz); "hazır/önceden eğitilmiş model kullanıldı" de.

---

## NEREDEN BAŞLA? → 2.1 Genel Bakış paragrafı

En kolay giriş budur. **Tek paragraf**, ~5–6 cümle: tüm boru hattını baştan sona özetle. Detaya girme, sadece "girdi → adımlar → çıktı" akışını anlat ve Şekil 1'e (mimari diyagram) referans ver.

**Ne yazacaksın:** Sistemin girdiyi (BT) alıp hangi aşamalardan geçirerek çıktıyı (rapor + 3B + takip) ürettiğinin özeti.

**Örnek açılış cümlesi (kendi cümlenle yaz):**
> "Geliştirilen sistem, kontrastlı abdominal/toraks BT görüntülerini girdi olarak alan ve segmentasyondan otomatik raporlamaya uzanan uçtan uca bir işlem hattından oluşmaktadır (Şekil 1)."

Sonra sırayla adımları tek cümleyle say: nnU-Net segmentasyonu → TotalSegmentator damar segmentasyonu → Python geometrik ölçüm → LLM raporlama → 3B görselleştirme → hekim arayüzü.

---

## 2.2 Veri Seti

**Ne yazacaksın:** Hangi veriyi kullandın, kaç vaka, hangi organlar, görüntü tipi.

**Kullanılacak olgular/sayılar:**
- Kaynak: **public MSD (Medical Segmentation Decathlon) / TCIA** verileri.
- Organlar: karaciğer, akciğer, pankreas, hepatik damar, dalak, kolon (6 organ + tümör).
- Görüntü formatı: NIfTI (.nii/.nii.gz), portal venöz faz BT.
- Kaç vaka değerlendirildi (kohort sayısını buraya yaz — netleştir).
- Etik: public anonim veri kullanıldığından ek etik onam gerekmedi (bunu belirt).

**Örnek açılış:**
> "Bu çalışmada, Medical Segmentation Decathlon (MSD) kapsamında halka açık olarak paylaşılan çok organlı BT veri setleri kullanıldı."

---

## 2.3 Segmentasyon

### 2.3.1 Organ ve tümör segmentasyonu (nnU-Net)
**Ne yazacaksın:** Hangi model, kaç task, hangi ayarlar. Kendi eğitimin olmadığını açıkça belirt.

**Kullanılacak olgular/sayılar:**
- **Önceden eğitilmiş (pretrained) nnU-Net ağırlıkları** kullanıldı; Isensee ve ark.'nın Zenodo'da yayımladığı modeller (DOI: 10.5281/zenodo.4485926, CC-BY-NC 4.0). Kendi eğitimimiz yok.
- 6 task: Task003 karaciğer, Task006 akciğer, Task007 pankreas, Task008 hepatik damar, Task009 dalak, Task010 kolon.
- Konfigürasyon: `3d_fullres`, fold 0 (kodda `run_nnunet`).
- Etiketler: organ=1, tümör=2 (dalak/akciğer/kolonda değişken — bunu Tablo 2'de ver).

**Örnek açılış:**
> "Organ ve tümör segmentasyonu için, Isensee ve arkadaşları tarafından yayımlanan önceden eğitilmiş nnU-Net modelleri kullanıldı."

### 2.3.2 Damar segmentasyonu (TotalSegmentator)
**Ne yazacaksın:** Damarların nasıl segmente edildiği.

**Kullanılacak olgular:**
- **TotalSegmentator** ile **aorta** ve **portal ven** (portal_vein_and_splenic_vein) segmente edildi (kodda `run_totalsegmentator`, `--roi_subset`).
- Atıf: Wasserthal ve ark., Radiology: AI 2023.

**Örnek açılış:**
> "Tümör–damar ilişkisinin değerlendirilebilmesi için aorta ve portal ven, TotalSegmentator aracı ile ayrı olarak segmente edildi."

---

## 2.4 Geometrik Analiz (makalenin en özgün teknik kısmı — detaylı yaz)

**Ne yazacaksın:** Maskelerden hangi metriklerin, hangi formülle çıkarıldığı. Her metriği bir alt paragrafta anlat.

**Kullanılacak olgular/formüller (kodda `extract_metrics`, `calculate_encasement`):**
- **Hacim:** voxel sayısı × voxel hacmi; mm³ → cm³.
- **Tümör çapı:** eşdeğer küre yaklaşımı — d = (6V/π)^(1/3).
- **Tümör lokasyonu:** organ ekseni boyunca göreli konum → baş / gövde / kuyruk.
- **Tümör–damar mesafesi:** Öklid uzaklık dönüşümü (`distance_transform_edt`) ile en yakın mesafe (mm).
- **Sarma (encasement) açısı:** damar merkezine göre temas eden voxellerin açısal dağılımı; 360°'den en büyük açısal boşluk çıkarılarak derece cinsinden sarma hesaplanır.

**Örnek açılış:**
> "Segmentasyon maskeleri üzerinden, klinik karar için gerekli geometrik metrikler Python ortamında (NumPy, SciPy, SimpleITK) hesaplandı."

Sonra her metriği kendi cümlesiyle açıkla. Sarma açısı için Şekil 2'ye (şema) referans ver.

---

## 2.5 Rezektabilite Değerlendirmesi
**Ne yazacaksın:** Damar sarma açısından rezektabilite kararının nasıl verildiği.
**Olgular:** NCCN kriterleri; >180° sarma → lokal ileri/irresektabl (pankreas için). Tümör boyutuna göre T-evreleme (T1 ≤2 cm, T2 2–4 cm, T3 >4 cm).
**Örnek açılış:**
> "Pankreas olgularında rezektabilite, NCCN kriterleri temel alınarak damar sarma açısına göre değerlendirildi."

---

## 2.6 Temellendirilmiş (Grounded) LLM Raporlaması
**Ne yazacaksın:** Metriklerin LLM'e nasıl verildiği ve raporun nasıl üretildiği. Buradaki KİLİT nokta: LLM'e ham görüntü değil, yalnızca sayısal metrikler verilir (halüsinasyon azaltma).
**Olgular:**
- Hesaplanan metrikler JSON olarak, uzman radyolog rolü tanımlı bir sistem promptuyla LLM'e verildi.
- Model: **Claude Opus 4.8** (birincil); karşılaştırma için Gemini 2.5 Flash (bkz. 2.x kıyas). Sıcaklık 0.3.
- Prompt kuralları: sadece verilen metriklere dayan, ACR/ESR standardı, T-evreleme, NCCN rezektabilite, kesin tanı koyma.
- Gizlilik: kimlik alanları (TC/ad) API'ye gönderilmez (anonimleştirme).
**Örnek açılış:**
> "Rapor üretiminde, görüntünün doğrudan modele verilmesi yerine yalnızca hesaplanmış geometrik metrikler büyük dil modeline aktarıldı; böylece modelin görüntüyü yorumlamaktan kaynaklanan halüsinasyon riski tasarımsal olarak azaltıldı."

---

## 2.7 Boylamsal Takip (RECIST)
**Olgular:** Aynı hastanın önceki taramaları sorgulanır; tümör hacmindeki yüzde değişim hesaplanır; karar: küçülme ≤ −%30 → kısmi yanıt (PR), artış ≥ +%20 → progresyon (PD), arası → stabil (SD).
**Örnek açılış:**
> "Aynı hastaya ait ardışık taramalarda tümör hacmindeki yüzdesel değişim hesaplanarak tedavi yanıtı RECIST 1.1 çerçevesinde sınıflandırıldı."

---

## 2.8 3B Görselleştirme
**Olgular:** marching cubes ile yüzey çıkarımı + hafif Gaussian yumuşatma; Plotly Mesh3d ile organ/tümör/damar interaktif gösterimi.

## 2.9 Etkileşimli Chatbot
**Olgular:** Üretilen rapor + geçmiş bulgular bağlam olarak verilip hekimin doğal dilde soru sormasına olanak tanındı.

## 2.10 Uygulama / Sistem
**Olgular:** Streamlit web arayüzü; hasta verileri JSON veritabanı; PDF rapor çıktısı; Python 3.11.

---

## 2.x LLM Karşılaştırması (opsiyonel — grounded iddiayı güçlendirir)
**Olgular:** Aynı metrik JSON'ları hem Claude Opus 4.8 hem Gemini 2.5 Flash'a verilerek raporlar üretildi ve metrik sadakati açısından karşılaştırıldı (kodda `compare_llm.py`).

---

### İLK GÖREV
Sadece **2.1 Genel Bakış** paragrafını yaz (5–6 cümle). Bitince buraya yapıştır; birlikte akışını/dilini düzeltip 2.2'ye geçelim.
