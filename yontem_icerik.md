# YÖNTEM — Projenin Gerçek Bilgileriyle Doldurulmuş İçerik

*Nasıl kullanılır: Aşağıdaki her blok, o alt başlıkta anlatılacak gerçek teknik bilgiyi içerir. Bunları kendi cümlelerinle yeniden yaz. Köşeli parantez [ ] içindekiler senin dolduracağın/karar vereceğin yerlerdir.*

---

## 2.1 Genel Bakış

Sistem, kontrastlı BT görüntülerini (NIfTI, .nii/.nii.gz) girdi alan, uçtan uca otomatik bir işlem hattıdır. Akış: (1) nnU-Net ile organ ve tümör segmentasyonu, (2) TotalSegmentator ile aorta ve portal ven segmentasyonu, (3) Python ile geometrik ölçüm (hacim, çap, damar mesafesi, sarma açısı), (4) elde edilen sayısal metriklerin büyük dil modeline verilerek yapılandırılmış radyoloji raporu üretimi, (5) NCCN'e göre rezektabilite ve RECIST'e göre boylamsal takip, (6) Plotly ile 3B interaktif görselleştirme, (7) Streamlit tabanlı hekim arayüzü ve chatbot. Tüm hat Şekil 1'de gösterilmiştir.

---

## 2.2 Veri Seti

- Kaynak: Medical Segmentation Decathlon (MSD) kapsamında halka açık BT veri setleri (ve/veya TCIA).
- Kapsanan organlar (6): karaciğer, akciğer, pankreas, hepatik damar, dalak, kolon; ilgili tümör/lezyonlarla birlikte.
- Görüntüleme: portal venöz faz BT. Örnek bir vakada teknik parametreler: kesit kalınlığı 2.5 mm, piksel aralığı 0.64 mm (bunlar her görüntünün DICOM/NIfTI başlığındaki `spacing` değerinden otomatik okunur).
- Değerlendirilen vaka sayısı: [KOHORT — kaç vaka çalıştırdıysanız yazın; şu an pilot ~4 vaka, genişletilebilir].
- Etik: Yalnızca halka açık, anonimleştirilmiş veri kullanıldığından ek etik kurul onamı gerekmemiştir. [Kendi klinik verinizi eklerseniz bu cümle değişir + etik onay no gerekir.]

---

## 2.3 Segmentasyon

### 2.3.1 Organ ve Tümör Segmentasyonu (nnU-Net)
- Kendi model eğitimi yapılmamıştır. Isensee ve arkadaşları tarafından yayımlanan **önceden eğitilmiş (pretrained) nnU-Net** ağırlıkları kullanılmıştır (Zenodo, DOI: 10.5281/zenodo.4485926, sürüm 2.1; lisans CC-BY-NC 4.0).
- Altı ayrı MSD task modeli kullanıldı: Task003 (karaciğer), Task006 (akciğer), Task007 (pankreas), Task008 (hepatik damar), Task009 (dalak), Task010 (kolon).
- Çıkarım (inference) konfigürasyonu: `3d_fullres` mimarisi, tek katlama (fold 0).
- Maske etiketleri: organ = 1, tümör = 2 (dalak, akciğer ve kolon modellerinde etiket şeması farklıdır; ayrıntı Tablo 2'de verilir).
- nnU-Net atfı: Isensee ve ark., Nature Methods 2021.

### 2.3.2 Damar Segmentasyonu (TotalSegmentator)
- Tümör–damar ilişkisini değerlendirmek için aorta ve portal ven (portal_vein_and_splenic_vein) TotalSegmentator ile segmente edildi.
- TotalSegmentator, ilgilenilen yapılar alt kümesiyle (aorta ve portal ven) çalıştırıldı.
- Atıf: Wasserthal ve ark., Radiology: Artificial Intelligence 2023.

---

## 2.4 Geometrik Analiz

Segmentasyon maskeleri üzerinden klinik metrikler Python ortamında (NumPy, SciPy, SimpleITK) hesaplandı. Voxel hacmi, görüntünün üç eksendeki `spacing` değerlerinin çarpımından elde edildi (mm³, ardından cm³'e çevrildi).

- **Hacim (cm³):** ilgili etikete ait voxel sayısı × voxel hacmi. Hem organ hem tümör için hesaplandı.
- **Tümör çapı (mm):** tümör, eşdeğer hacimli bir küre kabul edilerek d = (6V/π)^(1/3) formülüyle hesaplandı.
- **Tümör lokalizasyonu:** tümör ağırlık merkezinin, organın uzun ekseni boyunca göreli konumu hesaplanarak sınıflandırıldı — göreli konum < 0.4 ise "baş", < 0.7 ise "gövde", aksi halde "kuyruk".
- **Tümör–damar mesafesi (mm):** damar maskesinin Öklid uzaklık dönüşümü (SciPy `distance_transform_edt`, voxel `spacing` ile ölçeklenmiş) alınarak, tümör voxelleri içindeki en küçük uzaklık en yakın mesafe olarak kaydedildi. Aorta ve portal ven için ayrı hesaplandı.
- **Sarma (encasement) açısı (°):** damar merkezine 2 mm'den yakın temas eden tümör voxellerinin, damar merkezine göre açısal konumları (arctan2) hesaplandı; bu açıların oluşturduğu dağılımda en büyük açısal boşluk 360°'den çıkarılarak sarma derecesi bulundu (0°–360°). Şema Şekil 2'de verilir.

---

## 2.5 Rezektabilite ve Evreleme

- Pankreas olgularında rezektabilite, NCCN kriterleri temel alınarak damar sarma açısına göre değerlendirildi: >180° sarma lokal ileri/irresektabl kabul edildi.
- Tümör boyutuna göre T-evreleme uygulandı: T1 ≤ 2 cm, T2 2–4 cm, T3 > 4 cm.
- Not: NCCN rezektabilite değerlendirmesi pankreas kanserine özgüdür; diğer organlarda volumetrik değerlendirme ve RECIST kriterleri kullanılır.

---

## 2.6 Temellendirilmiş (Grounded) LLM Raporlaması

- Hesaplanan metrikler JSON biçiminde, "15 yıllık deneyimli abdominal radyolog" rolü tanımlayan bir sistem promptuyla birlikte büyük dil modeline verildi.
- **Kilit ilke:** Görüntünün kendisi modele verilmez; yalnızca sayısal geometrik metrikler aktarılır. Böylece modelin görüntü yorumundan doğan halüsinasyon riski tasarımsal olarak azaltılır.
- Model: birincil olarak **Claude Opus 4.8** (Anthropic API); karşılaştırma amacıyla Gemini 2.5 Flash (Google). Örnekleme sıcaklığı 0.3.
- Sistem promptundaki kurallar: yalnızca verilen metriklere dayan, ACR ve ESR standartlarına uygun raporla, T-evreleme yap, vasküler temas varsa NCCN'e göre rezektabilite değerlendir, kesin tanı koyma (histopatolojik doğrulama öner), Türkçe ve düz metin yaz.
- Rapor yapısı: Teknik, Bulgular, (varsa) Karşılaştırmalı Değerlendirme, Vasküler Değerlendirme, Evreleme ve Rezektabilite, Sonuç ve Öneri.
- Gizlilik: Hasta kimliği (TC, ad-soyad) API'ye gönderilmez; yalnızca tarih ve klinik metrikler iletilir (anonimleştirme).

---

## 2.7 Boylamsal Takip (RECIST)

- Aynı hastaya ait önceki taramalar hasta kimliğine göre sorgulanır; güncel ve geçmiş tümör hacimleri karşılaştırılır.
- Tümör hacmindeki yüzdesel değişime göre tedavi yanıtı sınıflandırılır: değişim ≤ −%30 → kısmi yanıt (Partial Response, PR); ≥ +%20 → progresif hastalık (Progressive Disease, PD); ara değerler → stabil hastalık (Stable Disease, SD).
- Ayrıca damar mesafeleri ve sarma açılarındaki değişim de karşılaştırmalı olarak raporlanır.
- **Dürüstlük notu (yazarken dikkat):** Klasik RECIST 1.1 en uzun çap toplamı üzerinden değerlendirir; bu sistemde eşikler (−%30 / +%20) **hacim** değişimine uygulanmıştır. Metinde bunu "RECIST 1.1'den esinlenilmiş hacim-temelli yanıt sınıflaması" gibi dürüstçe ifade et; birebir "RECIST 1.1 uyumlu" deme.

---

## 2.8 Üç Boyutlu Görselleştirme

- Segmentasyon maskelerinden marching cubes algoritmasıyla yüzey ağı (mesh) çıkarıldı; sınırların yumuşatılması için hafif Gaussian filtre uygulandı.
- Performans için 2 kat alt-örnekleme (downsampling) yapıldı ve gerçek boyutlar `spacing` ile ölçeklendirildi.
- Organ, tümör, aorta ve portal ven ayrı renk/opaklıkta Plotly Mesh3d ile interaktif olarak gösterildi (döndürme, yakınlaştırma).

---

## 2.9 Etkileşimli Chatbot

- Üretilen rapor ve hastanın geçmiş bulguları bağlam olarak modele verilerek, hekimin doğal dilde soru sormasına olanak tanıyan bir soru-cevap arayüzü kuruldu.
- Model, yalnızca sağlanan veriye dayanır; veri yoksa "bu veri mevcut değil" yanıtını verecek şekilde yönlendirilir (uydurmayı önleme).

---

## 2.10 Uygulama ve Sistem

- Arayüz: Streamlit tabanlı web uygulaması (Python 3.11).
- Veri saklama: hasta bulguları, rapor ve sohbet geçmişi JSON biçiminde yerel veritabanında tutulur; segmentasyon maskeleri 3B rekonstrüksiyon için saklanır.
- Rapor çıktısı: ReportLab ile PDF üretimi.
- API anahtarları koda gömülmez, ortam değişkeninden (.env) okunur.

---

## 2.11 LLM Karşılaştırması (opsiyonel — grounded iddiayı güçlendirir)

- Aynı segmentasyon metrikleri (JSON) hem Claude Opus 4.8 hem Gemini 2.5 Flash'a birebir aynı promptla verilerek raporlar üretildi.
- Raporlar, metriklerdeki sayısal değerlerin metne doğru yansıması (metrik sadakati / halüsinasyon göstergesi) açısından karşılaştırıldı; nihai kalite değerlendirmesi radyolog tarafından yapılabilir.
- Bu karşılaştırma `compare_llm.py` ile otomatikleştirildi.

---

### Senin dolduracağın açık noktalar
1. **Kaç vaka** kullandın (2.2).
2. Hedef dergi kesinleşince atıf formatı (Vancouver/APA).
3. LLM kararı: makalede birincil model Claude Opus 4.8 mı, yoksa "Gemini ile geliştirildi, Claude'a taşındı" mı — tek anlatı seç.
