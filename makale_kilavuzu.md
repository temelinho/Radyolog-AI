# MAKALE KILAVUZU — Radyoloji AI Sistemi

*Bu dosya makalenin ana referansıdır. Bağlamı, kararları, ilerlemeyi ve açık maddeleri tutar. Yeni bir konuşmada buradan devam edilir. Son güncelleme: yazım aşaması, Yöntem 2.1–2.10 taslak.*

---

## 0. Çalışmanın Künyesi
- **Konu:** Çok organlı BT'de segmentasyondan hekim etkileşimine uçtan uca entegre yapay zeka platformu; ana tez = LLM'i ham görüntü yerine yalnızca deterministik geometrik metriklerle besleyerek (grounded raporlama) halüsinasyonu azaltmak.
- **Tür/hedef:** Türkçe hakemli dergi makalesi; hocayla birlikte yazılıyor. Konumlanma: sistem/platform + uygulanabilirlik (feasibility) makalesi.
- **Kapsam:** 6 organ genişliği vurgulu (karaciğer, akciğer, pankreas, hepatik damar, dalak, kolon); pankreas rezektabilitesi vitrin vaka.
- **Doğrulama:** Public MSD/TCIA; nicel kanıt = segmentasyon Dice (modellerin bildirilen skorları). Geometri/rapor = gösterim düzeyi.
- **Çalışma tarzı:** Kullanıcı kendi cümleleriyle yazıyor; ben bölüm bölüm iskelet/içerik veriyorum, düzeltiyorum. Baştan sona AI'ya yazdırmıyoruz.

## 1. Üç Temel Katkı (novelty)
1. Silo yapısının kırılması — tam entegre ilk platform.
2. Grounded raporlama — sadece metrik verilir, halüsinasyon azaltılır.
3. Hekim-dostu açıklanabilirlik — 3B görselleştirme + chatbot.

---

## 2. Makale Yapısı ve İlerleme Durumu

**Öz** — en son. | **1. Giriş** — bekliyor. | **2. Yöntem** — yazılıyor (aşağıda). | **3. Bulgular** — bekliyor. | **4. Tartışma** — bekliyor. | **5. Sonuç** — bekliyor. | **Kaynakça** — sürüyor.

### 2. YÖNTEM — alt başlık durumu
- 2.1 Genel Bakış — ✅ taslak yazıldı (Şekil 1 atıflı)
- 2.2 Veri Seti — 🔄 kısmen (vaka sayısı [N] eksik; görüntü+etik cümleleri eklenecek)
- 2.3 Segmentasyon (2.3.1 nnU-Net, 2.3.2 TotalSegmentator) — ✅ taslak
- 2.4 Geometrik Analiz — ✅ taslak (eşdeğer küre çapı, kesit-z ekseni, 2B izdüşüm sarma açısı olarak dürüstçe yazıldı)
- 2.5 Rezektabilite ve Evreleme — ✅ taslak (NCCN >180°, basitleştirilmiş)
- 2.6 Grounded LLM Raporlaması — ✅ taslak (Claude Opus 4.8, 0.3 sıcaklık, anonimleştirme)
- 2.7 Boylamsal Takip (RECIST 1.1) — ✅ taslak (kod RECIST 1.1'e düzeltildi)
- 2.8 Üç Boyutlu Görselleştirme — ✅ taslak (Şekil 3 atıflı)
- 2.9 Etkileşimli Chatbot — ✅ taslak (Şekil 5 atıflı)
- 2.10 Uygulama ve Sistem — ✅ taslak
- **2.11 LLM Karşılaştırması (Claude vs Gemini)** — ⏸️ **BEKLİYOR: hocayla görüşülüp bilgi alışverişi sonrası yazılacak.** Altyapı hazır (`compare_llm.py`). Grounded iddiayı güçlendiren opsiyonel katkı.

---

## 3. Kritik Kararlar ve Dürüstlük Noktaları (hakem bunları sorar)
- **LLM:** Makalede birincil model **Claude Opus 4.8**. Kod başta Gemini 2.5 Flash'tı; artık ikisini de destekliyor (arayüzden seçilir, varsayılan Claude Opus 4.8). "Gemini ile geliştirildi, Claude'a taşındı" anlatısı da dürüst bir alternatif.
- **nnU-Net:** Kendi eğitimimiz YOK; Isensee ekibinin Zenodo hazır ağırlıkları (DOI 10.5281/zenodo.4485926, CC-BY-NC 4.0). "Eğittik" DENMEZ.
- **Dice:** Modelin bildirilen MSD performansı alıntılanır; "kendi test setimizde doğruladık" DENMEZ (MSD test etiketleri kapalı). Alternatif: LiTS/NIH-Pancreas ile dış-doğrulama.
- **Geometri yaklaşımları:** çap = eşdeğer küre çapı (en uzun çap değil); lokalizasyon = kesit (z) ekseni; sarma açısı = 2B izdüşüm. Sınırlılıklar'da kabul edilecek.
- **RECIST 1.1:** Kod düzeltildi — çap-temelli, baseline+nadir, CR/PR/SD/PD, PD için ≥5 mm. Kalan sınır: çap eşdeğer küre çapı; yeni-lezyon kriteri kapsam dışı.
- **NCCN:** Basitleştirilmiş (>180°, iki damar, borderline yok). "NCCN'den uyarlanmış" denir.
- **Gizlilik/KVKK:** TC/ad API'ye gönderilmez; sadece metrik.
- **Güvenlik:** Eski Google API anahtarı git geçmişinde açıkta → İPTAL EDİLMELİ. Anahtarlar artık .env'de.

## 4. Açık Maddeler (yapılacaklar)
- [x] **Vaka sayısı = 4** (hocayla artırılacak, her organ için). Bulgular 4 vaka üzerinden pilot/gösterim.
- [ ] Hedef dergi seçimi (atıf formatı: Vancouver/APA).
- [ ] 2.11 için hocayla görüşme.
- [ ] Şekiller: Şekil 1 hazır; Şekil 2 (şema, ben çizerim), Şekil 3/4/5 (ekran görüntüleri).
- [ ] Tablolar: Tablo 1 (literatür), Tablo 2 (organ↔task), Tablo 3 (Dice — MSD/nnU-Net makalelerinden alıntı, ZENODO'DA YOK).
- [ ] Eski API anahtarını iptal et.
- [ ] (Opsiyonel) tümör en-uzun-çap fonksiyonu → RECIST yaklaşımını tamamen doğrular.

### İki katmanlı veri stratejisi
- **1. katman (ŞİMDİ yapılıyor):** Segmentasyon + geometri + rapor gösterimi → MSD (ve gerekirse CRLM). 4 vaka.
- **2. katman (ERTELENDİ, netleşince):** Boylamsal/RECIST gösterimi → gerçek longitudinal veri gerekir (HCC-TACE-Seg gibi TCIA seti ya da hocanın klinik takip vakası). MSD longitudinal DEĞİL.
- **Bütünlük kuralı:** İki farklı hastayı "aynı hasta" gibi gerçek takip verisi diye SUNMA (fabrikasyon). Sadece açıkça "kurgusal demo" etiketiyle ya da gerçek longitudinal veriyle.

## 5. Anahtar Kaynaklar (Yöntem)
- nnU-Net: Isensee F, et al. Nature Methods. 2021;18:203-211.
- nnU-Net hazır ağırlıklar: Isensee F, et al. Zenodo. 2021. DOI 10.5281/zenodo.4485926.
- MSD: Antonelli M, et al. Nature Communications. 2022;13:4128.
- TotalSegmentator: Wasserthal J, et al. Radiology: AI. 2023;5(5):e230024.
- RECIST 1.1: Eisenhauer EA, et al. Eur J Cancer. 2009;45(2):228-247. DOI 10.1016/j.ejca.2008.10.026.
- NCCN Pankreas Kanseri Kılavuzu (sürüm/yıl yazıma göre).
- (İlgili çalışmalar için 23 künye: `makale_hazirlik.md` Bölüm 4.)

## 6. Proje Dosyaları (rehber)
- `makale_hazirlik.md` — genel iskelet, kaynakça (23 çalışma), karşılaştırma tablosu, açık kararlar.
- `yontem_icerik.md` — Yöntem'in projeye özel dolu içeriği.
- `yontem_iskelet.md` — Yöntem yazım kılavuzu (ne/nasıl).
- `sekiller_plani.md` — şekil/tablo planı ve durumu.
- `makale_kilavuzu.md` — (bu dosya) ana referans.
- Kod: `radyoloji_app.py` (Claude+Gemini, RECIST 1.1, .env), `compare_llm.py` (LLM kıyası), `.env.example`.

## 7. Yazım İş Akışı (hatırlatma)
Sıra: Yöntem → Bulgular → İlgili Çalışmalar → Giriş → Tartışma → Sonuç → Öz. Her bölümde: kullanıcı yazar, ben düzeltirim. Tekrar eden dil hataları: (a) özne varsa yüklem etken ("yapar" değil "yapılır"), (b) parantezden önce boşluk, (c) "eğittik" değil "kullanıldı".
