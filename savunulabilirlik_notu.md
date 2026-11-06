# Savunulabilirlik Notu — iddia, tutarlılık ve hakem itirazları

**Tarih:** 6 Kasım 2026
**Amaç:** Makalede elimizi güçlü tutmak = aşırı iddia değil, **her sayının izlenebildiği, her itirazın önceden cevaplandığı tutarlı bir çalışma.** Bu not, hoca ile konuşmaya ve Methods/Limitations yazımına temel sağlar.

---

## 1. Savunabileceğimiz iddia (önerilen çerçeve)

> BT'den otomatik, **grounded** (yalnız ölçülen geometrik metriklere dayalı) radyoloji raporu üreten **tekrarlanabilir bir pipeline** ve bu pipeline'ı radyolog raporuyla üç eksende (tespit / sadakat / içerik) **nicel ve izlenebilir** biçimde değerlendiren bir çerçeve kurduk; aynı girdi üzerinde farklı LLM'leri kontrollü olarak karşılaştırdık.

Bu iddia **tamamen bizim ürettiğimiz ve doğruladığımız** şeylere dayanır. Kodla birebir tutarlıdır (tek ölçüm tanımı, tek manifest, izlenebilir metrikler).

## 2. İddia ETMEYECEĞİMİZ / yumuşatacağımız şeyler

| Satmayalım | Neden |
|---|---|
| "Segmentasyon doğruluğu yüksek" | Merlin'de gerçek tümör maskesi yok → Dice ölçülemez. Zero-shot, betimleyici. |
| "Rezektabiliteyi belirliyoruz" | Kohortta damar tutulumu nadir (1/80), doğrulanmadı → **araştırma ölçümü**. |
| "Lokasyonu doğru veriyoruz" | Baş/gövde/kuyruk kuralı bu turda kapsam dışı; kod artık yanlış üretmek yerine **null** veriyor. |
| "Halüsinasyonu ortadan kaldırdık" | Sadakat ölçümü **otomatik proxy**'dir; nihai söz radyologda. |
| "Bağımsız dış test" | Eski 80 vaka hem geliştirme hem test → **pilot yeniden-değerlendirme** de. |
| Diameter'dan AUC | Boyutu malignite skoru saymak hatalı → sens/özgüllük + GA ver, AUC'yi bırak. |

## 3. Hakem itirazları → önceden cevabımız

| Olası itiraz | Cevabımız | Kodda karşılığı |
|---|---|---|
| "Ölçümleriniz tutarsız; farklı yerde farklı çap" | Tek ölçüm tanımı; LLM girdisi ile değerlendirme **aynı JSON**. | `measurements.py`, `reporting.build_findings` |
| "Çap aslında hacim-eşdeğeri küre çapıydı" | Düzeltildi; **aksiyel en uzun çap**, hacim ayrı alan. | F01/F14, 11 test |
| "Mesafe/açı fiziksel uzayı yok sayıyor" | Dünya-koordinat mesafesi; oblik kapısı; farklı grid resample. | F03/K1/K2, `imaging`, `measure_case` |
| "GT'yi kim, ne zaman, hangi cümleden ölçtü?" | Her GT: kaynak cümle + tarih + transkribe eden + revizyon. | `reference.py` (denetim izi) |
| "MAE farklı lezyonları kıyaslıyor" | Açık eşleştirme politikası; eşleşmeyenler ayrı sayılır. | `matching.py`, F18 |
| "n küçük, iddianız güçlü" | Betimleyici; tüm oranlarda **Wilson GA**; atlananlar ayrı payda. | `evaluation.py` |
| "Sadakat metni yanlış sayıyor" | Bağlam + birim eşleşmeli; '2 cm³'≠20 mm; proxy olarak etiketli. | F15/F16/F17, 13 test |
| "Dosya varsa atlıyorsunuz, eski çıktı" | Hash'li cache anahtarı + atomik yazma; sürüm değişince yeniden. | `runner.py`, F10 |
| "Model karşılaştırması adil mi?" | Aynı girdi/prompt; sağlayıcı farkları (4-bit/token) açıkça loglanır. | `reporting.MODELS`, F19 |

## 4. Yapısal sınırlar (dürüstçe yazılacak, kod çözmez)

1. **Domain shift:** nnU-Net MSD'de eğitildi, Merlin'de zero-shot. Vaka eklemek çözmez. → "farklı merkeze zero-shot genelleme".
2. **GT bağımsızlığı yok:** radyolog raporundan transkripsiyon; bağımsız klinik re-okuma değil.
3. **Kohort bağımsızlığı:** pilot yeniden-değerlendirme; kişi-düzeyi ayrım ve örtüşme denetimi yapılacak.
4. **Pankreas eksik segmentasyon (−14 mm):** model tarafı; betimleyici.

## 5. Hoca ile netleşecek açık kararlar

- [ ] **Merkez iddia cümlesi** — yukarıdaki (§1) çerçeve onaylansın mı?
- [ ] **Claude Opus mu Sonnet mi** (raporlama aşamasında lazım).
- [ ] **Rezektabilite ve lokasyon** makalede yer alacak mı, yoksa "gelecek çalışma" mı?
- [ ] **Kohort:** eski 80 (pilot yeniden-değerlendirme) kesinleşsin mi?

---

*Bu not, kodun sağladıklarıyla tutarlıdır: her satırda bir itiraz ve onu karşılayan modül vardır. Eksik olan, veri/çerçeve kararlarıdır — onlar da açıkça işaretlenmiştir.*
