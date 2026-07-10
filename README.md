# Radyoloji AI Sistemi 🏥

Yapay zeka destekli çoklu organ BT analizi ve raporlama sistemi.

---

## 📌 Proje Nedir?

Bu proje, yapay zeka destekli bir radyoloji analiz sistemidir. Doktor veya radyolog, hastanın BT (Bilgisayarlı Tomografi) görüntüsünü sisteme yükler; sistem otomatik olarak şu adımları gerçekleştirir:

1. **Segmentasyon:** nnU-Net modelleriyle organ ve tümör maskelerini çıkarır
2. **Geometrik Analiz:** Tümör hacmi, damar mesafesi ve sarma açısı (derece) hesaplar
3. **Rezektabilite:** NCCN kriterlerine göre pankreas kanseri rezektabilite değerlendirmesi yapar
4. **3D Görselleştirme:** Plotly ile interaktif 3D rekonstrüksiyon üretir
5. **LLM Raporlama:** Claude 3.5 Sonnet ile uzman-düzeyinde klinik rapor yazar
6. **Chatbot:** Hekim, raporu Claude destekli chatbot ile sorgulayabilir
7. **RECIST Takibi:** Longitudinal boylamsal takip ile tümör gelişimi izlenir

---

## 🏗️ Sistem Mimarisi

```
BT Görüntüsü (.nii.gz)
        ↓
  nnU-Net Segmentasyon (6 organ + tümör)
        ↓
  TotalSegmentator (Aorta + Portal Ven)
        ↓
  Python Geometrik Hesaplama
  (hacim mm³, mesafe mm, sarma açısı °)
        ↓
  Claude 3.5 Sonnet (LLM Raporlama)
        ↓
  Plotly 3D Görselleştirme
        ↓
  Hekim Arayüzü (Streamlit)
```

---

## 📦 Desteklenen Organlar

| Organ | nnU-Net Task | Tümör Tespiti |
|-------|-------------|---------------|
| Akciğer | Task006 | ✅ |
| Karaciğer | Task003 | ✅ |
| Pankreas | Task007 | ✅ |
| Kolon | Task010 | ✅ |
| Dalak | Task009 | ✅ |
| Hepatik Damar | Task008 | ✅ |

---

## 🚀 Kurulum

### Gereksinimler

- Python 3.11
- CUDA destekli GPU (önerilir)
- Anthropic API anahtarı (Claude 3.5 Sonnet için)

### Kurulum Adımları

```bash
# Repo'yu klonla
git clone https://github.com/temelinho/Radyolog-AI.git
cd Radyolog-AI

# Sanal ortam oluştur
python -m venv radyoloji_env_311
# Windows:
radyoloji_env_311\Scripts\activate
# Linux/Mac:
source radyoloji_env_311/bin/activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### Ortam Değişkenleri

`.env` dosyası oluşturun:
```
ANTHROPIC_API_KEY=your_api_key_here
```

### Uygulamayı Başlatın

```bash
streamlit run radyoloji_app.py
```

---

## 📁 Proje Yapısı

```
Radyolog-AI/
├── radyoloji_app.py       # Ana Streamlit uygulaması
├── generate_pdf.py        # Literatür taraması PDF üreticisi
├── Setup.py               # nnU-Net kurulum scripti
├── main.py                # CLI arayüzü
├── .gitignore
├── README.md
└── requirements.txt
```

> **Not:** `nnUNet_results/` (model ağırlıkları, ~GB), `radyoloji_env_311/` (sanal ortam) ve hasta BT verileri gizlilik ve boyut nedeniyle repoya dahil edilmemiştir.

---

## 🔬 Akademik Katkı

Bu sistem, literatürdeki 23 temel çalışmayla karşılaştırmalı olarak analiz edilmiştir:

- **KATEGORI 1:** Grounded LLM Raporlama (CLarGen, MARCH, 3D-CT-GPT vb.)
- **KATEGORI 2:** Tümör-Damar Rezektabilite (PAN-VIQ, Viviers vd., DoDNet vb.)
- **KATEGORI 3:** Longitudinal Takip (LongiSeg, LesionLocator vb.)
- **KATEGORI 4:** Segmentasyon Altyapıları (TotalSegmentator, nnU-Net)
- **KATEGORI 5:** LLM Karşılaştırmalı Analiz (GREEN, CRIMSON, RaDialog vb.)

---

## ⚖️ Lisans

Bu proje akademik araştırma amaçlıdır. Klinik kullanım için ek doğrulama gereklidir.

---

## 👥 İletişim

GitHub: [@temelinho](https://github.com/temelinho)
