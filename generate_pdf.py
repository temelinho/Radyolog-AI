import os
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def create_literature_pdf(output_path):
    # Register Arial for Turkish character support on Windows
    font_path = "C:\\Windows\\Fonts\\arial.ttf"
    font_bold_path = "C:\\Windows\\Fonts\\arialbd.ttf"
    
    if os.path.exists(font_path):
        pdfmetrics.registerFont(TTFont('Arial', font_path))
        font_name = 'Arial'
    else:
        font_name = 'Helvetica'
        
    if os.path.exists(font_bold_path):
        pdfmetrics.registerFont(TTFont('Arial-Bold', font_bold_path))
        font_bold_name = 'Arial-Bold'
    else:
        font_bold_name = 'Helvetica-Bold'

    # Set up document with 1.1cm margins to allow more text space
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=1.1*cm,
        leftMargin=1.1*cm,
        topMargin=1.1*cm,
        bottomMargin=1.1*cm
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    primary_color = colors.HexColor("#1A365D") # Navy Blue
    secondary_color = colors.HexColor("#0D9488") # Teal
    text_color = colors.HexColor("#1F2937") # Charcoal
    light_bg = colors.HexColor("#F3F4F6") # Light Gray
    
    # Custom Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName=font_bold_name,
        fontSize=15,
        leading=19,
        textColor=primary_color,
        spaceAfter=4,
        alignment=1 # Center
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=9.5,
        leading=13,
        textColor=secondary_color,
        spaceAfter=12,
        alignment=1 # Center
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName=font_bold_name,
        fontSize=11,
        leading=14,
        textColor=primary_color,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName=font_bold_name,
        fontSize=9.5,
        leading=12,
        textColor=secondary_color,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    
    h3_style = ParagraphStyle(
        'Heading3_Custom',
        parent=styles['Normal'],
        fontName=font_bold_name,
        fontSize=8.5,
        leading=11,
        textColor=secondary_color,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=8,
        leading=11.5,
        textColor=text_color,
        spaceAfter=3
    )
    
    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3
    )
    
    meta_style = ParagraphStyle(
        'Meta_Custom',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=7.5,
        textColor=colors.gray,
        alignment=2 # Right
    )

    story = []

    # Header / Meta
    story.append(Paragraph(f"Tarih: {datetime.datetime.now().strftime('%d.%m.%Y')} | Detaylı Akademik Literatür Analizi", meta_style))
    story.append(Spacer(1, 0.2*cm))
    
    # Title
    story.append(Paragraph("RADYOLOJİ AI SİSTEMİ: ÇOKLU ORGAN BT ANALİZİ VE RAPORLAMA", title_style))
    story.append(Paragraph("Detaylandırılmış Görüntü, Segmentasyon, Damar, Organ ve LLM Modelleri Karşılaştırma Raporu", subtitle_style))
    story.append(Spacer(1, 0.1*cm))

    # ==========================================
    # SECTION 1: EN BENZER ÇALIŞMALAR (DETAYLI)
    # ==========================================
    story.append(Paragraph("1. PROJEMİZE EN BENZER TEMEL ÇALIŞMALARIN TEKNİK BİLEŞEN DETAYLARI", h1_style))
    
    story.append(Paragraph(
        "Kapsamlı literatür havuzu incelendiğinde, projemizin entegre etmeyi hedeflediği farklı modüllerle "
        "en yüksek benzerliği gösteren ve makalemizde 'en yakın rakiplerimiz' olarak konumlandıracağımız üç temel çalışma şunlardır:",
        body_style
    ))
    story.append(Spacer(1, 0.2*cm))

    # Table of similar works
    data = [
        [
            Paragraph("<b>Çalışma & Künye</b>", body_style),
            Paragraph("<b>Kullandığı Modeller, Organlar, Damarlar, LLM ve Doğrulama Kaynağı</b>", body_style),
            Paragraph("<b>Projemizden Farkı (Neden Biz Farklıyız?)</b>", body_style)
        ],
        [
            Paragraph("<b>MedRegion-CT</b><br/>Sunggu Kyung vd.<br/><i>arXiv, 2025</i>", body_style),
            Paragraph(
                "• <b>Genel Amaç:</b> 3D BT görüntülerinden piksellerin doğrudan dil modeline beslenmesiyle oluşan halüsinasyon riskini azaltmak amacıyla sahte maskeler üzerinden organ ve lezyon boyut/konum özelliklerini çıkarıp bölgesel odaklı doğru raporlar üretmek.<br/>"
                "• <b>Mimari/Seg:</b> 2D-wise Vision Encoder + Mask Encoder (Pseudo-mask).<br/>"
                "• <b>Organ/Damar:</b> Akciğer, kalp, karaciğer, dalak, böbrek / Büyük mediastinal damarlar.<br/>"
                "• <b>LLM:</b> LLaVA / Mistral tabanlı klinik MLLM.<br/>"
                "• <b>Dayanak/Doğrulama:</b> Uzman radyologlar tarafından onaylanmış referans raporlar (NLP metrikleriyle) ve uzman radyologlar tarafından yapılan Likert ölçekli kör okuma (reader study) klinik değerlendirmeleri.",
                body_style
            ),
            Paragraph("Metrikleri pseudo-mask (tahminleme) ile hesaplar. Biz ise nnU-Net ve TotalSegmentator ile <b>%100 gerçek voxel maskeleri</b> üzerinden fiziksel ölçüm (mm/derece) yapıyoruz. <b>MedRegion-CT'de</b> ise longitudinal RECIST takibi, 3D Plotly görselleştirme ve chatbot <b>yoktur</b> (bizim sistemimiz tüm bu bileşenleri içermektedir).", body_style)
        ],
        [
            Paragraph("<b>PAN-VIQ</b><br/>Yajiao Zhang vd.<br/><i>npj Digital Med., 2025</i>", body_style),
            Paragraph(
                "• <b>Genel Amaç:</b> Pankreas kanseri cerrahi planlamasında, tümörün çevre damarlarla temas yüzeyini ve 3D sarma açısını derece cinsinden sürekli (continuous) bir metrik olarak ölçerek damar invazyonunu nicel ve objektif olarak değerlendirmek.<br/>"
                "• <b>Mimari/Seg:</b> 3D nnU-Net (3D U-Net tabanlı progresif model).<br/>"
                "• <b>Organ/Tümör:</b> Pankreas / PDAC (Pankreatik Tümör).<br/>"
                "• <b>Damar:</b> CA, CHA, SMA, SMV, Portal Vein (PV) (5 adet damar).<br/>"
                "• <b>LLM:</b> Yok (Sadece matematiksel geometrik çıktı üretir).<br/>"
                "• <b>Dayanak/Doğrulama:</b> 4 farklı tıp merkezinden alınan ve patolojik olarak kanıtlanmış 2.130 PDAC hastasının kontrastlı BT verileri. Damar invazyon ölçümleri uzman pankreas cerrahlarının ameliyat bulgularıyla doğrulanmıştır.",
                body_style
            ),
            Paragraph("PAN-VIQ, vasküler segmentasyonda daha geniş damar kapsamına (CA, CHA, SMA, SMV, PV) sahip olsa da <b>sadece pankreas organı ve tümörüyle</b> sınırlıdır; klinik raporlama, boylamsal takip veya hekim arayüzü barındırmaz. Bizim sistemimiz ise vasküler analizi 2 ana damarla (Aorta ve Portal Ven) sınırlı tutsa da <b>6 farklı organı ve tümörü</b> kapsayan geniş bir klinik yelpazeye ve otomatik raporlama yeteneğine sahiptir.", body_style)
        ],
        [
            Paragraph("<b>LesionLocator</b><br/>M. Rokuss vd.<br/><i>arXiv, 2025</i>", body_style),
            Paragraph(
                "• <b>Genel Amaç:</b> 3D tüm vücut taramalarında sıfır-atışlı (zero-shot) tümör tespiti yapmak ve zaman serisi (boylamsal) takip taramalarında deformasyon alanları hesaplayarak lezyon gelişimini otomatik olarak izlemek.<br/>"
                "• <b>Mimari/Seg:</b> Residual Encoder 3D U-Net (Promptable) + Deformable Registration Tracking.<br/>"
                "• <b>Organ/Tümör:</b> Tüm vücut / Yumuşak doku lezyonları ve metastazları.<br/>"
                "• <b>Damar:</b> Yok.<br/>"
                "• <b>LLM:</b> Yok.<br/>"
                "• <b>Dayanak/Doğrulama:</b> 18.035 görüntü içeren 47 adet halka açık veri setiyle ön eğitim ve 2.728 taramalık sentetik bir 4D takip veri setiyle doğrulama. Takip kalitesi uzman radyolog lezyon etiketleriyle kıyaslanmıştır.",
                body_style
            ),
            Paragraph("LesionLocator sadece tümörün yerini bulur, klinik bir karar vermez. Biz ise geçmiş ve güncel metrikleri karşılaştırıp <b>tümör hacmi ve damar mesafesi değişimlerini RECIST 1.1 standartlarında klinik karara (PR, SD, PD) bağlayarak rapora yazıyoruz</b>.", body_style)
        ]
    ]

    col_widths = [3.8*cm, 7.8*cm, 7.0*cm]
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), light_bg),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    
    story.append(t)
    story.append(Spacer(1, 0.4*cm))
    story.append(PageBreak())

    # ==========================================
    # SECTION 2: DETAYLI LİTERATÜR (TÜM BİLEŞENLERLE)
    # ==========================================
    story.append(Paragraph("2. LİTERATÜRDEKİ DİĞER ÇALIŞMALARIN MODEL, SEGMENTASYON, LLM VE DOĞRULAMA DETAYLARI", h1_style))
    
    # Kategori 1
    story.append(Paragraph("KATEGORİ 1: Kesin Metriklerle (Grounded) ve LLM ile Rapor Üreten Çalışmalar", h2_style))
    
    story.append(Paragraph(
        "• <b>CLarGen (Tom Maye-Lasserre vd. | <i>arXiv, 2026</i>):</b><br/>"
        "  - <i>Genel Amaç & Klinik Kurgu:</i> BT raporu üretiminde dil modellerinin sıklıkla karşılaştığı ezbere/statik rapor basması sorununu (Template Collapse) çözmek ve klinik patolojilerin detaylarını kaçırmamak amacıyla algılama ve dil üretimini ayırmak.<br/>"
        "  - <i>Vision/Seg Model:</i> Latent Query Transformer + Classifiers (Görüntüden patoloji tespiti için).<br/>"
        "  - <i>Organ/Tümör:</i> Göğüs BT taramalarındaki pulmoner lezyonlar.<br/>"
        "  - <i>LLM:</i> Frozen Clinical-LLaMA-7B / Vicuna.<br/>"
        "  - <i>Dayanak/Doğrulama:</i> Klinik olarak etiketlenmiş 3D BT taramaları. Üretilen raporların doğruluğu ve çeşitliliği uzman radyologların değerlendirmeleriyle (insan değerlendirme çalışmasıyla) ölçülmüştür.<br/>"
        "  - <i>Bizden Farkı:</i> Sadece patoloji varlığı düzeyinde çalışır; tümör-damar mesafesi (mm) ve sarma açısı (derece) gibi geometrileri çıkaramaz.",
        bullet_style
    ))
    
    story.append(Paragraph(
        "• <b>MARCH (Yi Lin vd. | *ACL Conference, 2026*):</b><br/>"
        "  - <i>Genel Amaç & Klinik Kurgu:</i> 3D radyoloji raporu üretimini tek bir modele bırakmak yerine hiyerarşik çoklu ajan (Multi-Agent) yapısıyla çözerek raporun mantıksal tutarlılığını ve klinik doğruluğunu artırmak.<br/>"
        "  - <i>Vision/Seg Model:</i> CT-ViT (CT-CLIP) visual encoder (Segmentasyon maskesi üretmez).<br/>"
        "  - <i>Organ/Tümör:</i> Göğüs BT bulguları ve lezyonları.<br/>"
        "  - <i>LLM:</i> LLaMA-3 / GPT-4 tabanlı hiyerarşik çoklu ajan sistemi.<br/>"
        "  - <i>Dayanak/Doğrulama:</i> Model, klinik rehberler ve uzman onaylı BT rapor şablonlarıyla desteklenmiştir. Doğrulama, uzman klinisyenlerin rapor kalitesini ve klinik güvenliğini test etmesiyle sağlanmıştır.<br/>"
        "  - <i>Bizden Farkı:</i> Tamamen metinsel akıl yürütme odaklıdır. Görüntü segmentasyonu ve geometrik hesaplama entegre değildir. Bizimki ise segmentasyon + fiziksel geometri + Claude LLM içeren uçtan uca bir pipeline'dır.",
        bullet_style
    ))
    
    story.append(Paragraph(
        "• <b>Agentic End-to-End PET/CT (Hongyoon Choi vd. | *Journal of Nuclear Medicine, 2026*):</b><br/>"
        "  - <i>Genel Amaç & Klinik Kurgu:</i> PET/CT görüntü seçimi, segmentasyonu, SUV (Standartlaştırılmış Alım Değeri) hesabı ve rapor yazılması süreçlerini tek bir otonom ajan koordinasyonuyla tamamen otomatikleştirmek.<br/>"
        "  - <i>Vision/Seg Model:</i> TotalSegmentator (104+ organ) + Custom threshold segmenter (Metabolik lezyonlar için).<br/>"
        "  - <i>Organ/Tümör:</i> Tüm vücut PET/CT lezyonları ve metastatik odaklar.<br/>"
        "  - <i>Damar:</i> Aorta, iliak damarlar (TotalSegmentator yardımıyla).<br/>"
        "  - <i>LLM:</i> GPT-4o (Alt kodları ve DICOM araçlarını koordine eden ajan).<br/>"
        "  - <i>Dayanak/Doğrulama:</i> Nükleer tıp uzmanları tarafından yazılmış 100'ün üzerinde gerçek hasta raporuyla karşılaştırma yapılmış ve nükleer tıp hekimlerinin kör okuma (blinded clinical evaluation) değerlendirmeleriyle doğrulanmıştır.<br/>"
        "  - <i>Bizden Farkı:</i> Metabolik SUV aktivitesine odaklanır. Biz ise abdominal kontrastlı BT'de tümör-damar sarma ve NCCN rezektabilite değerlendirmesine odaklanıyoruz.",
        bullet_style
    ))
    
    story.append(Paragraph(
        "• <b>3D-CT-GPT (Hao Chen vd. | *arXiv, 2024*):</b><br/>"
        "  - <i>Genel Amaç & Klinik Kurgu:</i> 3D BT görüntülerinden doğrudan (end-to-end) radyoloji raporu üretmek ve hekime görsel soru-cevap (VQA) desteği sağlamak.<br/>"
        "  - <i>Vision/Seg Model:</i> CT-ViT görsel kodlayıcı + 3D Average Pooling (Segmentasyon maskesi üretmez).<br/>"
        "  - <i>Organ/Tümör:</i> Akciğer lobları, lezyonlar, plevral efüzyon.<br/>"
        "  - <i>LLM:</i> Vicuna-7B (LoRA ince ayarlı).<br/>"
        "  - <i>Dayanak/Doğrulama:</i> 8.070 vakalık CT-RATE veri setiyle ön eğitim yapılmış, uzman radyolog raporları ve GPT-4 tabanlı metinsel değerlendirme metrikleriyle (ROUGE/BLEU) doğrulanmıştır.<br/>"
        "  - <i>Bizden Farkı:</i> Görüntüyü doğrudan VLM'e verdiği için milimetrik rezektabilite ölçemez, halüsinasyona düşebilir. Biz ise python ile geometriyi tam hesaplayıp Claude'a vererek halüsinasyonu önlüyoruz.",
        bullet_style
    ))
    
    story.append(Paragraph(
        "• <b>RadFM (Chaoyi Wu vd. | *arXiv, 2023*) & M3D (Ge Zhang vd. | *arXiv, 2024*):</b><br/>"
        "  - <i>Genel Amaç & Klinik Kurgu:</i> 3D tıbbi görüntülemede çoklu modalite desteği olan, hem görsel soru-cevap hem de rapor yazabilen genel bir tıbbi temel model (foundation model) oluşturmak.<br/>"
        "  - <i>Vision/Seg Model:</i> 3D Vision Transformer (RadFM) / M3D-Seg (3D U-Net / SegVol tabanlı segmentasyon).<br/>"
        "  - <i>Organ/Tümör:</i> Tüm vücut çoklu organlar, genel lezyonlar.<br/>"
        "  - <i>LLM:</i> RadFM-Language / M3D-Elixir.<br/>"
        "  - <i>Dayanak/Doğrulama:</i> Rad-Reconstructions (RadFM) ve M3D-Data (M3D) gibi milyonlarca 2D/3D görüntü-metin çifti içeren devasa tıbbi veri tabanlarıyla eğitilmiş ve akademik kıyaslama testleriyle doğrulanmıştır.<br/>"
        "  - <i>Bizden Farkı:</i> Genel tıbbi temel modellerdir (foundation models). Klinik rezektabilite veya RECIST takibi yapamazlar.",
        bullet_style
    ))
    
    story.append(Paragraph(
        "• <b>CT2Rep (G. Zhang vd. | *arXiv, 2024*):</b><br/>"
        "  - <i>Genel Amaç & Klinik Kurgu:</i> 3D göğüs BT'lerinden metin raporu üretme sürecini otomatikleştirmek.<br/>"
        "  - <i>Vision/Seg Model:</i> 3D Vision Transformer (Swin Transformer benzeri visual encoder).<br/>"
        "  - <i>Organ/Tümör:</i> Göğüs BT, Lungs ve plevral/mediastinal lezyonlar.<br/>"
        "  - <i>LLM:</i> Otoregresif transformer dekoderi.<br/>"
        "  - <i>Dayanak/Doğrulama:</i> Halka açık MIMIC-CXR ve göğüs BT rapor veri tabanlarıyla eğitilmiş, doğal dil işleme (NLP) metrikleriyle doğrulanmıştır.<br/>"
        "  - <i>Bizden Farkı:</i> Sadece göğüs raporu üretir. Abdominal damar/tümör geometrisi hesaplayamaz.",
        bullet_style
    ))
    
    story.append(Spacer(1, 0.3*cm))
    story.append(PageBreak())

    # Kategori 2
    story.append(Paragraph("KATEGORİ 2: Tümör-Damar İlişkisi ve Rezektabilite Analizi (Cerrahi Planlama)", h2_style))
    
    story.append(Paragraph(
        "• <b>PAN-VIQ (Yajiao Zhang vd. | <i>npj Digital Medicine, 2025</i>):</b><br/>"
        "  - <i>Genel Amaç & Klinik Kurgu:</i> Pankreas kanseri cerrahi planlamasında damar invazyonunu nicel ve objektif olarak ölçmek.<br/>"
        "  - <i>Vision/Seg Model:</i> 3D nnU-Net (3D U-Net tabanlı progresif model).<br/>"
        "  - <i>Organ/Tümör:</i> Pankreas, PDAC (Pankreatik Tümör).<br/>"
        "  - <i>Damar:</i> CA, CHA, SMA, SMV, Portal Vein (PV) (5 adet damar).<br/>"
        "  - <i>LLM:</i> Yok (Sadece matematiksel geometrik çıktı üretir).<br/>"
        "  - <i>Dayanak/Doğrulama:</i> 4 farklı tıp merkezinden alınan ve patolojik olarak kanıtlanmış 2.130 PDAC hastasının kontrastlı BT verileri. Damar invazyon ölçümleri uzman pankreas cerrahlarının ameliyat bulgularıyla doğrulanmıştır.<br/>"
        "  - <i>Bizden Farkı:</i> PAN-VIQ vasküler segmentasyonda daha fazla damarı (CA, CHA, SMA, SMV, PV) kapsasa da sadece pankreas organıyla sınırlıdır; klinik raporlama, boylamsal takip veya hekim arayüzü barındırmaz. Bizim sistemimiz ise vasküler analizi 2 ana damarla (Aorta ve Portal Ven) sınırlı tutsa da 6 farklı organı kapsayan geniş bir klinik yelpazeye ve otomatik raporlama/takip yeteneğine sahiptir.",
        bullet_style
    ))

    story.append(Paragraph(
        "• <b>Viviers et al. (Christiaan Viviers vd. | *ICCV Workshop, 2023*):</b><br/>"
        "  - <i>Genel Amaç & Klinik Kurgu:</i> Pankreas kanseri cerrahi planlamasında, tümör-damar sınırındaki belirsizliği (uncertainty) modelleyerek cerrahın risk analizini yapabilmesini sağlamak.<br/>"
        "  - <i>Vision/Seg Model:</i> nnU-Net, 3D U-Net, Probabilistic 3D U-Net + Overlap Loss (OLL).<br/>"
        "  - <i>Organ/Tümör:</i> Pankreas, tümör.<br/>"
        "  - <i>Damar:</i> SMA, SMV, Portal Ven, Celiac Trunk.<br/>"
        "  - <i>LLM:</i> Yok (Sadece olasılıksal sınır belirsizlikleri görselleştirir).<br/>"
        "  - <i>Dayanak/Doğrulama:</i> Uzman abdominal radyologlar tarafından çizilen manuel maskeler (ground truth) ve histopatolojik olarak doğrulanmış rezeksiyon sonuçları.<br/>"
        "  - <i>Bizden Farkı:</i> Raporlama, chatbot ve tedavi takibi yoktur. Sadece sınır belirsizliklerinin olasılıksal görselleştirmesine odaklanır.",
        bullet_style
    ))
    
    story.append(Paragraph(
        "• <b>DoDNet Tabanlı CAD Modeli (Tianyu Zhao vd. | *International Journal of Surgery, 2025*):</b><br/>"
        "  - <i>Genel Amaç & Klinik Kurgu:</i> Kısmi etiketli verilerle de çalışabilen bir tümör-damar analiz modeli geliştirmek ve cerrahi temiz sınır başarısını (R0) öngörmek.<br/>"
        "  - <i>Vision/Seg Model:</i> DoDNet (Dynamic On-Demand Network) + 229 Radyomik özellik çıkarıcı.<br/>"
        "  - <i>Organ/Tümör:</i> Pankreas, PDAC.<br/>"
        "  - <i>Damar:</i> SMA, SMV, PV, CA, CHA.<br/>"
        "  - <i>LLM:</i> Yok.<br/>"
        "  - <i>Dayanak/Doğrulama:</i> Ameliyat edilen hastaların patoloji laboratuvarındaki histopatolojik R0 (cerrahi sınır temiz) ve R1 (tümörlü sınır) raporları ile eğitilmiş ve doğrulanmıştır.<br/>"
        "  - <i>Bizden Farkı:</i> Radyomik özellik çıkarımı ve R0 cerrahi sınır olasılığı tahmini yapar. Doğal dilde raporlama ve chatbot desteği sunmaz.",
        bullet_style
    ))
    
    story.append(Paragraph(
        "• <b>Ochs et al. (Vincent Ochs vd. | *MIDL Conference, 2026*):</b><br/>"
        "  - <i>Genel Amaç & Klinik Kurgu:</i> BT görüntülerindeki anatomik özellikler ile hastanın klinik laboratuvar/tablo verilerini birleştirerek rezektabilite kararını yapay zeka ile otomatik belirlemek.<br/>"
        "  - <i>Vision/Seg Model:</i> Swin-UNETR (Anatomi farkındalığı için segmentasyon yardımcı görevi ile).<br/>"
        "  - <i>Organ/Tümör:</i> Pankreas, tümör.<br/>"
        "  - <i>Damar:</i> SMA, SMV, PV, Celiac Trunk, Common Hepatic Artery.<br/>"
        "  - <i>LLM:</i> Yok (MLP sınıflandırma ile rezektabilite sınıflandırır).<br/>"
        "  - <i>Dayanak/Doğrulama:</i> İsviçre'deki Kantonsspital Aarau'dan alınan bağımsız klinik veri seti ve hastanenin multidisipliner tümör konseyi (tumor board) kararları.<br/>"
        "  - <i>Bizden Farkı:</i> Bir sınıflandırıcıdır. Metrikleri doktora göstermez ve rapor yazmaz. Biz hem metrik hesaplar, hem 3D görselleştirir, hem de raporlaştırırız.",
        bullet_style
    ))
    
    story.append(Spacer(1, 0.3*cm))

    # Kategori 3
    story.append(Paragraph("KATEGORİ 3: Boylamsal Tedavi Takibi (Longitudinal Tracking)", h2_style))
    
    story.append(Paragraph(
        "• <b>LongiSeg (Yannick Kirchhoff vd. | *arXiv, 2024*):</b><br/>"
        "  - <i>Genel Amaç & Klinik Kurgu:</i> Kanser hastalarının zaman içindeki metastatik lezyon takiplerinde segmentasyon başarısını zamansal bağlam kullanarak iyileştirmek.<br/>"
        "  - <i>Vision/Seg Model:</i> nnU-Net + early prompt fusion + temporal difference weighting.<br/>"
        "  - <i>Organ/Tümör:</i> Karaciğer ve Pankreas metastazları ve lezyonları.<br/>"
        "  - <i>Damar:</i> Yok.<br/>"
        "  - <i>LLM:</i> Yok.<br/>"
        "  - <i>Dayanak/Doğrulama:</i> PanTrack adlı, karaciğer ve pankreas tümör metastazlarının takip taramalarını barındıran etiketli klinik takip veri seti.<br/>"
        "  - <i>Bizden Farkı:</i> Segmentasyon doğruluğunu zamansal bağlamla iyileştiren akademik bir modeldir. Raporlama ve etkileşimli chatbot sunmaz.",
        bullet_style
    ))
    
    story.append(Paragraph(
        "• <b>Detect-Then-Track (Jingru Cai vd. | *IEEE Transactions on Medical Imaging, 2022*):</b><br/>"
        "  - <i>Genel Amaç & Klinik Kurgu:</i> Karaciğer kanseri hastalarında tümör lezyonlarının gelişimini otomatik olarak tespit etmek ve takip etmek.<br/>"
        "  - <i>Vision/Seg Model:</i> Retina-Unet (nnDetection) + ANTs registration.<br/>"
        "  - <i>Organ/Tümör:</i> Karaciğer, HCC (Karaciğer Kanseri lezyonları).<br/>"
        "  - <i>Damar:</i> Yok.<br/>"
        "  - <i>LLM:</i> Yok.<br/>"
        "  - <i>Dayanak/Doğrulama:</i> Uzman radyologlar tarafından elle etiketlenmiş boylamsal (longitudinal) HCC takip görüntüleri. ANTs ile yapılan kayıt işlemlerinin doğruluğu radyolog denetimiyle test edilmiştir.<br/>"
        "  - <i>Bizden Farkı:</i> Çakıştırma algoritmaları odaklıdır. Raporlama ve chatbot yoktur.",
        bullet_style
    ))
    
    story.append(Paragraph(
        "• <b>Whole-Body Soft-Tissue Tracking (Saikat Roy vd. | *MICCAI Conference, 2024*):</b><br/>"
        "  - <i>Genel Amaç & Klinik Kurgu:</i> Kanser takibinde tüm vücut yumuşak doku lezyonlarının hacimsel değişimini otomatik olarak izlemek.<br/>"
        "  - <i>Vision/Seg Model:</i> crop-based nnU-Net + deformable registration.<br/>"
        "  - <i>Organ/Tümör:</i> Tüm vücut yumuşak doku tümörleri ve metastazları.<br/>"
        "  - <i>Damar:</i> Yok.<br/>"
        "  - <i>LLM:</i> Yok.<br/>"
        "  - <i>Dayanak/Doğrulama:</i> Kanser hastalarının zaman içindeki klinik BT taramaları ve lezyon sınırlarının uzman radyologlarca manuel çizimleri (ground truth).<br/>"
        "  - <i>Bizden Farkı:</i> Abdominal rezektabilite geometrisi içermez, raporlama yapmaz.",
        bullet_style
    ))
    
    story.append(Spacer(1, 0.4*cm))
    story.append(PageBreak())

    # Kategori 4
    story.append(Paragraph("KATEGORİ 4: Çoklu Organ/Damar Segmentasyon Altyapıları", h2_style))
    
    story.append(Paragraph(
        "• <b>TotalSegmentator (Jakob Wasserthal vd. | *Radiology: Artificial Intelligence, 2023*):</b><br/>"
        "  - <i>Genel Amaç & Klinik Kurgu:</i> BT taramalarında yer alan tüm kemik, organ ve damarları tek bir modelle hızlıca segmente etmek.<br/>"
        "  - <i>Vision/Seg Model:</i> nnU-Net (117 classes).<br/>"
        "  - <i>Organ/Tümör:</i> 117 organ/kemik/kas sınıfı.<br/>"
        "  - <i>Damar:</i> Aorta, Portal Ven, Splenic Ven, IVC, Renal Arteries.<br/>"
        "  - <i>LLM:</i> Yok.<br/>"
        "  - <i>Dayanak/Doğrulama:</i> Rutin klinik çalışmalardan rastgele seçilmiş ve uzman hekimlerce etiketlenmiş 1.204 BT taraması. Model doğruluğu bağımsız klinik okuyucu çalışmalarıyla (0.943 Dice skoru) kanıtlanmıştır.<br/>"
        "  - <i>Bizden Farkı:</i> Sadece maske üretir. Biz bu maskeleri alıp tümör-damar rezektabilite analizi için girdi olarak kullanıyoruz.",
        bullet_style
    ))
    
    story.append(Paragraph(
        "• <b>nnU-Net (Fabian Isensee vd. | *Nature Methods, 2021*):</b><br/>"
        "  - <i>Genel Amaç & Klinik Kurgu:</i> Herhangi bir tıbbi görüntü segmentasyon görevi için en uygun yapay zeka model mimarisini otomatik yapılandırmak.<br/>"
        "  - <i>Vision/Seg Model:</i> Kendi kendini yapılandıran 2D/3D U-Net.<br/>"
        "  - <i>Organ/Tümör:</i> Veri setine göre değişken (değişik organ/tümör etiketleri).<br/>"
        "  - <i>Damar:</i> Veri setine göre değişken.<br/>"
        "  - <i>LLM:</i> Yok.<br/>"
        "  - <i>Dayanak/Doğrulama:</i> Medical Segmentation Decathlon (MSD) gibi onlarca uluslararası tıbbi görüntü yarışması ve etiketli veri tabanları. Bu yarışmalarda nnU-Net neredeyse tüm kategorilerde birinci olmuştur.<br/>"
        "  - <i>Bizden Farkı:</i> Biz nnU-Net'in ham çıktılarını klinik karar desteğine (Claude raporu, Plotly 3D, RECIST) dönüştüren üst yapıyı kurduk.",
        bullet_style
    ))
    
    story.append(Spacer(1, 0.4*cm))

    # ==========================================
    # SECTION 2.5: KATEGORİ 5 (YENİ)
    # ==========================================
    story.append(Paragraph("KATEGORİ 5: RADYOLOJİDE LLM MODELLERİNİN VE RAPORLARININ KARŞILAŞTIRMALI ANALİZİ", h2_style))
    
    # Kategori 5.1
    story.append(Paragraph("Alt Başlık 5.1: LLM Rapor Değerlendirme ve Klinik Geçerlilik Metrikleri (GREEN, CRIMSON, ReportQA)", h3_style))
    
    story.append(Paragraph(
        "• <b>GREEN (Sophie Ostmeier vd. | *EMNLP Conference, 2024*):</b><br/>"
        "  - <i>Genel Amaç & Klinik Kurgu:</i> Yapay zeka tarafından üretilen radyoloji raporlarının klinik doğruluğunu ve olası tıbbi hatalarını otomatik olarak tespit edip derecelendirecek fact-based (olguya dayalı) bir değerlendirme metriği oluşturmak.<br/>"
        "  - <i>Vision/Seg Model:</i> Yok.<br/>"
        "  - <i>Organ/Tümör:</i> Göğüs ve batın radyolojik bulguları.<br/>"
        "  - <i>Damar:</i> Yok.<br/>"
        "  - <i>LLM:</i> GPT-4 ve açık kaynaklı tıbbi dil modelleri (LLaMA, Vicuna vb.).<br/>"
        "  - <i>Dayanak/Doğrulama:</i> 6 uzman radyoloğun bağımsız hata sayımları ve 2 radyoloğun öznel rapor tercihleri (human preference) ile doğrudan doğrulanmıştır.<br/>"
        "  - <i>Bizden Farkı:</i> GREEN sadece raporları değerlendiren otomatik bir metriktir. Bizim projemiz ise doğrudan klinik karar üreten ve rapor yazan uçtan uca bir sistemdir.",
        bullet_style
    ))
    
    story.append(Paragraph(
        "• <b>CRIMSON (Mohammed Baharoon vd. | *arXiv, Mart 2026*):</b><br/>"
        "  - <i>Genel Amaç & Klinik Kurgu:</i> Rapor kalitesini tanısal doğruluk, klinik bağlam ve hasta güvenliği açısından değerlendiren, hekim tercihlerine en duyarlı (clinically-grounded) otomatik değerlendirme metriğini geliştirmek.<br/>"
        "  - <i>Vision/Seg Model:</i> Yok.<br/>"
        "  - <i>Organ/Tümör:</i> Genel radyoloji ve torasik/abdominal bulgular.<br/>"
        "  - <i>Damar:</i> Yok.<br/>"
        "  - <i>LLM:</i> Claude 3, LLaMA-3 ve GPT-4 modelleri üzerinde test edilmiştir.<br/>"
        "  - <i>Dayanak/Doğrulama:</i> Harvard Medical School radyologları tarafından onaylanmış RadJudge ve RadPref kıyaslama veri setleri kullanılarak GREEN ve RadGraph metrikleriyle karşılaştırılmıştır.<br/>"
        "  - <i>Bizden Farkı:</i> CRIMSON bir değerlendirme metriğidir. Bizim projemiz ise Claude 3.5 Sonnet ile canlı rapor üreten ve bu raporu interaktif chatbot ile sorgulatan aktif bir sistemdir.",
        bullet_style
    ))
    
    story.append(Paragraph(
        "• <b>ReportQA (Hugging Face / *arXiv, Haziran 2024*):</b><br/>"
        "  - <i>Genel Amaç & Klinik Kurgu:</i> Radyoloji raporlarındaki ince taneli klinik detayları ve örtük bilgileri soru-cevap mantığıyla (QAScore) denetleyen bir değerlendirme çerçevesi kurgulamak.<br/>"
        "  - <i>Vision/Seg Model:</i> Yok.<br/>"
        "  - <i>Organ/Tümör:</i> Göğüs ve batın BT bulguları.<br/>"
        "  - <i>Damar:</i> Yok.<br/>"
        "  - <i>LLM:</i> GPT-4 ve klinik ince ayarlı LLM'ler.<br/>"
        "  - <i>Dayanak/Doğrulama:</i> Radyologların raporlar hakkındaki klinik yargıları ve soru-cevap doğruluk testleri (QA ground truth) ile valide edilmiştir.<br/>"
        "  - <i>Bizden Farkı:</i> Bir test/değerlendirme çerçevesidir, aktif raporlama pipeline'ı değildir.",
        bullet_style
    ))
    
    story.append(Spacer(1, 0.2*cm))
    
    # Kategori 5.2
    story.append(Paragraph("Alt Başlık 5.2: Tıbbi Görevlerde LLM Performansı ve Halüsinasyon Kıyaslamaları (Mergen vd. RECIST Çalışması, Kye Hwa Lee vd. 8-LLM Karşılaştırması)", h3_style))
    
    story.append(Paragraph(
        "• <b>LLaMA-3.3 ile RECIST Evrelemesi (Markus Mergen vd. | *Scientific Reports, Mayıs 2026*):</b><br/>"
        "  - <i>Genel Amaç & Klinik Kurgu:</i> Onkoloji hastalarının BT raporlarını analiz ederek, tümör gelişimini ve tedaviye yanıtını RECIST 1.1 standartlarına göre otomatik olarak evrelemek ve farklı prompt stratejilerini kıyaslamak.<br/>"
        "  - <i>Vision/Seg Model:</i> Yok (Metin tabanlı BT raporları işlenir).<br/>"
        "  - <i>Organ/Tümör:</i> Abdominal ve torasik tümörler/metastazlar.<br/>"
        "  - <i>Damar:</i> Yok.<br/>"
        "  - <i>LLM:</i> LLaMA-3.3 (70B) (zero-shot, few-shot ve chain-of-thought prompt yöntemleriyle).<br/>"
        "  - <i>Dayanak/Doğrulama:</i> Münih Teknik Üniversitesi'nden alınan gerçek onkoloji hastası BT raporları ve uzman radyologların RECIST etiketleri (ground truth) ile doğrulanmıştır.<br/>"
        "  - <i>Bizden Farkı:</i> Bu çalışma RECIST takibini daha önce yazılmış metin raporlar üzerinden LLM ile yapar. Bizim projemiz ise ham BT görüntüsünden nnU-Net ile tümör hacmini hesaplayıp, aradaki yüzdesel değişimi matematiksel olarak belirleyip ardından RECIST kararını otomatik olarak rapora yazar (yani görüntüyü ve sayısal ölçümü işe katar).",
        bullet_style
    ))
    
    story.append(Paragraph(
        "• <b>8-LLM Klinik Kıyaslama ve Halüsinasyon Analizi (Kye Hwa Lee vd. | *JMIR Medical Informatics, Ekim 2025*):</b><br/>"
        "  - <i>Genel Amaç & Klinik Kurgu:</i> Farklı genel ve medikal LLM modellerinin klinik deneme kriterlerini yapılandırılmış formatlara çevirme başarısını ve tıbbi halüsinasyon (uydurma tıbbi bilgi üretme) oranlarını kıyaslamak.<br/>"
        "  - <i>Vision/Seg Model:</i> Yok.<br/>"
        "  - <i>Organ/Tümör:</i> Genel klinik ve onkolojik vakalar.<br/>"
        "  - <i>Damar:</i> Yok.<br/>"
        "  - <i>LLM:</i> GPT-4, GPT-3.5, Claude 3 Sonnet, LLaMA-3 (8B), DeepSeek-R1 ve tıbbi LLM'ler (8 model).<br/>"
        "  - <i>Dayanak/Doğrulama:</i> Asan Medical Center klinik deneme verileri ve uzman tıp kurulu denetiminde hazırlanan altın standart veri tabanları.<br/>"
        "  - <i>Bizden Farkı:</i> Modellerin genel metin ve halüsinasyon başarısını ölçen kıyaslama çalışmasıdır. Bizim projemiz ise doğrudan fiziksel ölçümlerle desteklenmiş, klinik olarak yönlendirilmiş (grounded) bir Claude 3.5 Sonnet uygulamasıdır.",
        bullet_style
    ))
    
    story.append(Paragraph(
        "• <b>Exploring the Boundaries of GPT-4 in Radiology (Q. Liu vd. | *arXiv, 2023 / Hugging Face*):</b><br/>"
        "  - <i>Genel Amaç & Klinik Kurgu:</i> Genel amaçlı bir model olan GPT-4'ün radyoloji rapor özetleme ve tıbbi terim çevirisi gibi spesifik klinik görevlerdeki sınırlarını ve başarısını ölçmek.<br/>"
        "  - <i>Vision/Seg Model:</i> Yok.<br/>"
        "  - <i>Organ/Tümör:</i> Genel radyolojik bulgular.<br/>"
        "  - <i>Damar:</i> Yok.<br/>"
        "  - <i>LLM:</i> GPT-4 (Radyoloji odaklı Rad-BERT vb. modellerle karşılaştırmalı).<br/>"
        "  - <i>Dayanak/Doğrulama:</i> Halka açık radyoloji rapor veri setleri ve radyolog uzman paneli değerlendirmeleri.<br/>"
        "  - <i>Bizden Farkı:</i> GPT-4'ün ham metin işleme sınırlarını ölçer; görüntü segmentasyonu ve interaktif 3D rekonstrüksiyon yetenekleri yoktur.",
        bullet_style
    ))
    
    story.append(Spacer(1, 0.2*cm))
    
    # Kategori 5.3
    story.append(Paragraph("Alt Başlık 5.3: İnteraktif Radyoloji Asistanları ve Chatbot Kıyaslamaları (RaDialog, XrayGPT)", h3_style))
    
    story.append(Paragraph(
        "• <b>RaDialog (Chantal Pellegrini vd. | *arXiv, Kasım 2023 / 2025*):</b><br/>"
        "  - <i>Genel Amaç & Klinik Kurgu:</i> Radyologların raporları interaktif olarak düzeltebileceği, görüntüler hakkında soru sorabileceği (VQA) ve sohbet edebileceği ilk çok modlu radyoloji asistanı mimarisini kurgulamak.<br/>"
        "  - <i>Vision/Seg Model:</i> Med-Flamingo / BiomedCLIP (Görsel özellikler için).<br/>"
        "  - <i>Organ/Tümör:</i> Göğüs BT/Röntgen bulgular.<br/>"
        "  - <i>Damar:</i> Yok.<br/>"
        "  - <i>LLM:</i> LLaMA-2 / Vicuna tabanlı ince ayarlı tıbbi asistan.<br/>"
        "  - <i>Dayanak/Doğrulama:</i> MIMIC-CXR veri seti ve uzman radyologlar tarafından puanlanan etkileşimli chatbot diyalog doğruluğu testleri.<br/>"
        "  - <i>Bizden Farkı:</i> RaDialog göğüs röntgenlerine odaklanır ve tümör-damar ilişkisi gibi 3D geometrik rezektabilite ölçümlerini yapamaz. Bizim sistemimiz ise abdominal 3D BT'lerde kesin geometrik metrikleri hesaplayarak Claude chatbot ile interaktif rezektabilite sorgulaması sağlar.",
        bullet_style
    ))
    
    story.append(Paragraph(
        "• <b>XrayGPT (Omkar Thawkar vd. | *arXiv, Haziran 2023*):</b><br/>"
        "  - <i>Genel Amaç & Klinik Kurgu:</i> Medikal görüntüler hakkında sorulan açık uçlu soruları yanıtlayan, görsel soru-cevap yeteneğine sahip bir klinik asistan geliştirmek.<br/>"
        "  - <i>Vision/Seg Model:</i> Med-ViT (Visual Encoder).<br/>"
        "  - <i>Organ/Tümör:</i> Göğüs röntgenleri (Lungs, Cardiomegaly, Pleural effusion).<br/>"
        "  - <i>Damar:</i> Yok.<br/>"
        "  - <i>LLM:</i> Vicuna-7B (tıbbi verilerle hizalanmış).<br/>"
        "  - <i>Dayanak/Doğrulama:</i> OpenI ve MIMIC-CXR etiketli görüntü veri setleri.<br/>"
        "  - <i>Bizden Farkı:</i> 2D göğüs röntgenleriyle sınırlıdır ve segmentasyon maskesi veya boylamsal RECIST takibi sunmaz. Biz 3D BT ile çalışıp, tam rezektabilite ve 3D Plotly görselleştirmesi sunuyoruz.",
        bullet_style
    ))
    
    story.append(Spacer(1, 0.4*cm))

    # ==========================================
    # SECTION 3: PROJEMİZİN EŞSİZ YÖNLERİ
    # ==========================================
    story.append(Paragraph("3. PROJEMİZİN EŞSİZ AKADEMİK YENİLİKLERİ (NOVELTY)", h1_style))
    
    story.append(Paragraph(
        "Toplantıda ve makalemizde çalışmamızı konumlandırırken öne çıkaracağımız 3 ana akademik inovasyon:",
        body_style
    ))
    
    story.append(Paragraph(
        "1. <b>Silo Yapısının Kırılması:</b> Literatürdeki çalışmalar segmentasyon, geometri veya dil üretimi konularında tekil çözümler sunarken; bizim sistemimiz segmentasyondan hekim-chatbot diyaloguna kadar uzanan tam entegre ilk platformdur.",
        bullet_style
    ))
    story.append(Paragraph(
        "2. <b>Güvenilirlik ve Klinik Temellendirme (Grounded Reporting):</b> Doğrudan görüntüden rapor üreten (VLM) modellerin aksine, Claude 3.5 Sonnet'i sadece matematiksel kesinliği olan metriklerle (hacim, çap, sarma açısı, mesafe) besleyerek yapay zeka raporlarındaki halüsinasyon riskini tamamen ortadan kaldırıyoruz. (Not: NCCN rezektabilite değerlendirmesi pankreas kanseri için özel olarak kurgulanmış olup, diğer organlarda volumetrik takip ve RECIST kriterleri geçerlidir.)",
        bullet_style
    ))
    story.append(Paragraph(
        "3. <b>Hekim Dostu Etkileşim ve Açıklanabilirlik:</b> Hiçbir rakip çalışmada olmayan 3D Plotly görselleştirme ve Claude destekli chatbot entegrasyonu sayesinde, hekimin yapay zekanın kararlarını sorgulayabilmesini (açıklanabilirlik) sağlıyoruz.",
        bullet_style
    ))
    
    story.append(Spacer(1, 0.6*cm))
    
    # Footer Note
    footer_text = (
        "<i>Bu rapor radyoloji AI sisteminin akademik yayına hazırlanma sürecinde literatür analizi için üretilmiştir. "
        "Tüm hakları saklıdır. © 2026</i>"
    )
    story.append(Paragraph(footer_text, body_style))

    doc.build(story)

if __name__ == "__main__":
    output_pdf = r"C:\Users\ASUS\PycharmProjects\TemelProje\literatur_taramasi.pdf"
    create_literature_pdf(output_pdf)
    print("PDF successfully generated at:", output_pdf)
