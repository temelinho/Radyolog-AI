# 2. Gereç ve Yöntem (taslak)

> Taslak, kodun ve koşuların gerçek değerlerinden yazılmıştır (dal `claude/radyolog-ai-proje-dokumani-764226`).
> `[KONTROL: …]` işaretli yerler sizin ya da hocanızın teyit etmesi gereken noktalardır.
> Eski `yontem_icerik.md` önceki plana (6 organ, eşdeğer küre çapı, tek fold) göre yazılmıştı; bu dosya onun yerine geçer.

---

## 2.1 Çalışma tasarımı

Bu çalışma, retrospektif ve halka açık bir veri seti üzerinde yürütülen bir yöntem değerlendirme çalışmasıdır. Değerlendirilen işlem hattı dört ardışık aşamadan oluşur: (1) önceden eğitilmiş modellerle organ, tümör ve damar segmentasyonu; (2) segmentasyon maskelerinden deterministik geometrik ölçüm; (3) yalnızca bu sayısal ölçümlerle beslenen büyük dil modelleri (BDM) ile yapılandırılmış radyoloji raporu üretimi; (4) her aşamanın çıktısının radyolog raporuyla karşılaştırılması. Büyük dil modellerine görüntü verilmemiştir; modeller yalnızca hesaplanmış metrikleri görür (temellendirilmiş, *grounded* raporlama).

Çalışmanın amacı iki yönlüdür: (i) metriklerle temellendirilmiş BDM'lerin verilen ölçümleri rapora ne ölçüde sadık aktardığını belirlemek; (ii) uçtan uca doğruluğun hangi aşamada sınırlandığını, radyolog raporunu referans alarak ortaya koymak.

## 2.2 Veri seti ve kohort

**Veri kaynağı.** Stanford Merlin abdominal BT veri seti kullanılmıştır. Veri seti, BT hacimlerini ve bunlara ait radyolog raporlarını (Findings bölümü) içerir. Veri, sağlayıcının araştırma kullanım sözleşmesi (RUA) kapsamında kullanılmış; görüntüler, rapor metinleri ve çalışma kimlikleri üçüncü taraflarla paylaşılmamıştır. [KONTROL: Merlin atfı ve sürümü; etik kurul muafiyeti cümlesi.]

**Kohort seçimi.** Karaciğer ve pankreas için ayrı ayrı, tümörlü (POS) ve tümörsüz (NEG) olmak üzere her organdan 20+20, toplam 80 vaka seçilmiştir. Seçim tabakalı ve rastgeledir:

1. Rapor Findings metinleri, organa özgü düzenli ifadelerle sınıflandırılmıştır: organ terimi ile malignite terimi (ör. *mass, metastasis, carcinoma*) aynı cümlede ve olumsuzlanmadan geçiyorsa POS; organ normal olarak tanımlanıyorsa NEG. Malign olmayan durumlar (kist, hemanjiyom, apse, steatoz, pankreatit, postoperatif değişiklik) ve belirsiz ifadeler (*cannot exclude, indeterminate, possible*) POS sayılmamıştır. Belirsiz vakalar kohorta alınmamış, ancak gerekçesiyle kayıt altına alınmıştır.
2. Adaylar sabit bir rastgelelik tohumu (seed = 42) ile karıştırılmış ve her sınıftan aday listesi çekilmiştir.
3. Adaylar rapor metni okunarak elle onaylanmış; POS vakalarda radyoloğun raporda belirttiği hedef lezyon çapı (mm) transkribe edilmiştir. [KONTROL: onayı ve çap transkripsiyonunu kim yaptı; tek okuyucu mu?]

Seçim kodu, girdi dosyasının özeti (SHA-256), terim sürümü ve tüm parametrelerle birlikte kaydedildiği için seçim tekrarlanabilirdir. Hiçbir vaka, işlem hattının çıktısına bakılarak seçilmemiştir.

**Referans standart.** Vaka düzeyinde tümör varlığı için kohort etiketi (POS/NEG), çap uyumu için radyoloğun rapordaki çapı, rapor kıyası için radyolog raporunun ilgili organa ait cümleleri referans alınmıştır. Radyolog raporlarında lezyon koordinatı bulunmadığından voksel düzeyinde referans maske mevcut değildir.

## 2.3 Segmentasyon

**Organ ve tümör.** Kendi model eğitimi yapılmamıştır. Medical Segmentation Decathlon (MSD) üzerinde eğitilmiş, halka açık nnU-Net (v1) ağırlıkları kullanılmıştır: karaciğer için Task003_Liver, pankreas için Task007_Pancreas (Isensee ve ark., 2021; Zenodo DOI 10.5281/zenodo.4485926). Çıkarım `3d_fullres` yapılandırmasında, beş katlamanın (fold 0–4) topluluk ortalaması ve test zamanı veri çoğaltma (TTA) ile yapılmıştır. Modeller Merlin verisinde yeniden eğitilmemiş veya ince ayarlanmamıştır (*zero-shot* dış veri kullanımı). Maske etiketleri: organ = 1, tümör = 2.

**Damarlar.** Aort, portal ven–splenik ven ve inferior vena kava, TotalSegmentator ile ilgili yapı alt kümesi (`--roi_subset`) seçilerek segmente edilmiştir (Wasserthal ve ark., 2023).

Çıkarım Google Colab ortamında tek GPU (NVIDIA T4) üzerinde yapılmıştır. [KONTROL: nnU-Net ve TotalSegmentator sürüm numaraları.]

## 2.4 Geometrik ölçüm

Ölçümler, maskelerin voksel aralığı (spacing) kullanılarak fiziksel birimlerde, Python (NumPy, SciPy) ile hesaplanmıştır.

- **Lezyon tanımı.** Tümör maskesi 26-komşuluk ile bağlı bileşenlere ayrılmış; 0.1 cm³'ten küçük bileşenler atılmıştır. Her kalan bileşen bir lezyon olarak kabul edilmiştir.
- **Hacim.** Voksel sayısı × voksel hacmi (cm³); lezyon başına ve toplam.
- **Çap.** Aksiyel en uzun çap: her aksiyel kesitte lezyonun en uzun kirişi (dışbükey zarf üzerinden), kesitler arasında en büyük değer. Eşdeğer küre çapı kullanılmamıştır.
- **Hedef lezyon.** Raporlama ve kıyas için en büyük aksiyel çaplı lezyon hedef alınmıştır. En büyük hacimli lezyon duyarlılık analizi olarak ayrıca raporlanmıştır.
- **Tümör–damar mesafesi.** Tümör ve damar yüzey voksel merkezleri arasındaki en kısa mesafe (mm).
- **Sarma açısı.** Her aksiyel kesitte, damara 2 mm'den yakın tümör vokselleri damarın o kesitteki merkezine göre açısal olarak değerlendirilmiş, temas eden açı aralığı derece olarak hesaplanmıştır. Bu ölçüm araştırma amaçlıdır ve tek başına rezektabilite kararı için kullanılmamıştır.
- **Boyut kategorisi.** En büyük çapa göre organa özgü kategori (pankreas ≤2 / 2–4 / >4 cm; karaciğer ≤2 / 2–5 / >5 cm). Bu bir T evresi değildir.

**Ölçüm kodunun doğrulanması.** Ölçüm fonksiyonları, gerçek boyutu analitik olarak bilinen sentetik fantomlarla (küre, elipsoid, iki ayrı küre, anizotrop voksel aralığı) test edilmiştir. Toplam 112 otomatik birim testi bulunmaktadır.

## 2.5 Temellendirilmiş rapor üretimi

**Girdi.** Her vaka için modele yalnızca organ adı ve klinik sayısal metrikler (lezyon sayısı, toplam tümör hacmi, en büyük lezyonun çapı ve hacmi, damar mesafeleri ve sarma açıları, boyut kategorisi, görüntüleme fazı) etiketli bir metin bloğu olarak verilmiştir. İzinli alan listesi (allowlist) uygulanmış; veri seti adı, segmentasyon modelinin kimliği, çalışma kimliği ve dosya yolları modele gönderilmemiştir. Görüntü verilmemiştir.

**Sistem istemi.** Tüm modellere aynı sistem istemi verilmiştir. İstem, yalnızca verilen metriklere dayanmayı, verilmeyen sayı veya bulgu üretmemeyi, ölçülmemiş alanı ölçülmüş gibi sunmamayı, boyut kategorisini T evresi olarak sunmamayı, sarma açısından tek başına rezektabilite kararı çıkarmamayı ve histopatolojik doğrulama önermeyi şart koşar. Rapor İngilizce ve standart yapıdadır (Teknik, Bulgular, Vasküler, Boyut kategorisi, Sonuç ve öneri). Rapor uzunluğu bilerek kısıtlanmamıştır.

**Modeller.** Tüm modeller aynı koşullarda çalıştırılmıştır (sıcaklık 0.3, en fazla 3500 çıktı tokeni):

| Etiket | Model | Erişim |
|---|---|---|
| Claude | Claude Sonnet 5.5 (`claude-sonnet-5-5`) | Anthropic API |
| Qwen | Qwen2.5-14B-Instruct | Yerel, 4-bit NF4 nicemleme |
| Gemma | Gemma-2-9B-it | Yerel, 4-bit NF4 nicemleme |
| Llama | Llama-3.1-8B-Instruct | Yerel, 4-bit NF4 nicemleme |
| Template | Kural tabanlı şablon | Kontrol (BDM yok) |

Kural tabanlı şablon, metrikleri sabit cümlelere yerleştirir ve tanım gereği tam sadıktır; BDM'ler için alt referans görevi görür. Gemini (gemini-3.8-flash) de planlanmış, ancak ücretsiz katmandaki günlük istek sınırı nedeniyle yalnızca 6/80 vakada rapor üretebildiği için tüm analizlerden dışlanmıştır.

## 2.6 Değerlendirme

### 2.6.1 Sayısal sadakat

Her vakanın ölçüm dosyasından doğrulanacak "gerçekler" listesi türetilmiştir (en büyük lezyon çapı ve hacmi, toplam tümör hacmi, her damar için mesafe ve sarma açısı). Bir gerçek, raporda aynı birim kategorisinde (uzunluk, hacim, açı) ve ilgili bağlam sözcüğüyle (ör. *diameter, volume, aorta*) aynı cümlede, |fark| ≤ max(0.5, %2 × değer) toleransıyla geçiyorsa "desteklenmiş" sayılmıştır. Birimler normalize edilmiştir (cm → mm; cc, ml → cm³). Sadakat oranı = desteklenen gerçek / toplam gerçek. Doğrulanacak gerçeği olmayan vakalar (lezyon bulunmayan vakalar) ortalamaya katılmamıştır. Bu metrik, raporda verilmeyen bir bulgunun eklenip eklenmediğini ölçmez; bu nedenle lezyon beyanı ayrıca değerlendirilmiştir (2.6.4).

### 2.6.2 Tespit

İşlem hattı, segmentasyon en az bir lezyon bulduğunda vakayı pozitif kabul etmiştir. Referans kohort etiketidir. Duyarlılık, özgüllük, pozitif ve negatif öngörü değerleri hesaplanmış; oranlar için %95 Wilson güven aralıkları verilmiştir. Analiz tüm kohort ve organ bazında, ayrıca en büyük lezyon çapı ≥10 mm eşiğiyle tekrarlanmıştır.

### 2.6.3 Çap uyumu

POS vakalarda işlem hattının hedef lezyon çapı, radyoloğun rapordaki çapıyla karşılaştırılmıştır. Lezyon koordinatı bulunmadığından eşleştirme bir politika olarak tanımlanmıştır (işlem hattının en büyük çaplı lezyonu ↔ radyoloğun ölçtüğü lezyon); bu, aynı lezyonun eşleştirildiğini garanti etmez. Segmentasyonun hiç lezyon bulmadığı POS vakalar "kaçırılan" olarak ayrıca sayılmış ve çap analizine katılmamıştır. Ortalama mutlak hata (MAE), yanlılık (ortalama fark), Pearson korelasyonu ve Bland–Altman uyum sınırları hesaplanmış; hata dağılımı betimleyici olarak (medyan mutlak hata; 5, 10 ve 20 mm içindeki vaka oranı) verilmiştir.

**Keşifsel segmentasyon kalite katmanı.** Vakalar, radyolog referansına bakılmaksızın yalnızca maskenin bir özelliğine göre iki gruba ayrılmıştır: tümör maskesi ≤10 bağlı bileşen içeriyorsa "QC geçti", fazlaysa "QC kaldı" (parçalanma, aşırı segmentasyon göstergesi). Eşik veri incelendikten sonra belirlendiği için bu analiz keşifseldir ve doğrulayıcı olarak yorumlanmamalıdır.

### 2.6.4 Rapor ile radyolog kıyası

Her rapordan kural tabanlı bir ayıklayıcı ile iki bilgi çıkarılmıştır: (i) raporun lezyon varlığı beyanı (olumsuzlanmış, varsayımsal ve damar–mesafe cümleleri hariç tutularak) ve (ii) raporda belirtilen en büyük lezyon boyutu. Ayıklayıcı, modellerin gerçek rapor cümleleri üzerinde elle kontrol edilmiş ve bu cümlelerden regresyon testleri oluşturulmuştur. Lezyon beyanı kohort etiketiyle karşılaştırılarak model başına duyarlılık, özgüllük ve doğruluk; belirtilen boyut radyolog çapıyla karşılaştırılarak MAE hesaplanmıştır.

Metin benzerliği için ROUGE-L (F1, sözcük düzeyinde en uzun ortak alt dizi) kullanılmıştır. Karşılaştırma simetrik tutulmuş; model raporunun organ ve lezyon cümleleri, radyolog raporunun ilgili organa ait cümleleriyle karşılaştırılmıştır. Radyolog raporları tüm karnı anlattığı için tüm metnin karşılaştırılması yapılmamıştır.

**Organ adlandırması.** Her raporda doğru organ adının (karaciğer için *liver/hepatic*, pankreas için *pancreas*) geçip geçmediği kaydedilmiştir. Doğru organ adı hiç geçmediği halde başka bir organ adı (ör. *kidney, spleen*) geçen raporlar "organ karışıklığı" olarak sayılmıştır.

## 2.7 Yazılım, tekrarlanabilirlik ve veri güvenliği

İşlem hattı Python ile yazılmıştır; segmentasyon, ölçüm, raporlama ve değerlendirme tek bir çekirdek paket (`pipeline/`) üzerinden yürütülür ve 112 otomatik test içerir. Ölçüm sonuçları ve raporlar önbelleğe alınır; yarıda kalan koşular kaldığı yerden devam eder. API anahtarları koda gömülmemiş, ortam değişkenlerinden okunmuştur. Kaynak kod GitHub'da paylaşılmaktadır [KONTROL: depo erişimi açık mı olacak]; RUA gereği görüntü, rapor metni ve çalışma kimlikleri paylaşılmamıştır.

---

## Yazarken dikkat (Yöntem'e özgü)

1. **İstemde organ adı Türkçe verildi** (`Organ: karaciger` / `pankreas`), raporlar ise İngilizce istendi. Bu, Bulgular'da bir model için organ karışıklığına yol açtığı için Yöntem'de açıkça belirtilmeli; Sınırlılıklar'da da geçmeli.
2. **"Zero-shot"** ifadesi korunmalı: modeller Merlin'de eğitilmedi. Bu, uçtan uca sonuçların yorumunun merkezinde.
3. **QC katmanı keşifseldir.** Bulgular'da da bu şekilde sunulmalı; ana sonuç tüm vakalar üzerinden verilmeli.
4. **"18 vakada r = 0.96" ya da "12 vakada r = 0.99" kullanılmaz.** O alt kümeler sonuca göre seçildi (bkz. `proje ilerlemesi/ilerleme.md` 15.3).
5. Sadakat metriği **eklemeyi değil, aktarımı** ölçer; bu ayrım hem burada hem Tartışma'da korunmalı.
