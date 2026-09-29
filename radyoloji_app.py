import streamlit as st
import subprocess
import os
import json
import tempfile
import shutil
import copy
import SimpleITK as sitk
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, center_of_mass
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO
import datetime

# ============================================================
# KONFIGURASYON
# ============================================================
RESULTS_FOLDER = r"C:\Users\ASUS\PycharmProjects\TemelProje\nnUNet_results"
NNUNET_RAW = r"C:\Users\ASUS\PycharmProjects\TemelProje\nnUNet_raw"
NNUNET_PREPROCESSED = r"C:\Users\ASUS\PycharmProjects\TemelProje\nnUNet_preprocessed"
# API anahtarlari ve LLM ayarlari (.env / ortam degiskeninden okunur, koda gomulmez)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
PATIENTS_DB_DIR = os.environ.get("PATIENTS_DB_DIR", r"C:\Users\ASUS\PycharmProjects\TemelProje\patients_db")

# Kullanilabilir LLM modelleri (arayuzden secilir)
LLM_MODELS = {
    "Claude Opus 4.8": {"provider": "anthropic", "model": "claude-opus-4-8"},
    "Gemini 2.5 Flash": {"provider": "google", "model": "gemini-2.5-flash"},
}
DEFAULT_LLM = "Claude Opus 4.8"
# Metrikler API'ye gonderilmeden once hasta kimligini (TC/ad) cikar (KVKK/gizlilik)
ANONYMIZE_FOR_API = True

os.makedirs(PATIENTS_DB_DIR, exist_ok=True)


def call_llm(system_prompt, user_prompt, model_label=DEFAULT_LLM, temperature=0.3, max_tokens=2500):
    """Secilen saglayiciya gore (Anthropic Claude / Google Gemini) LLM cagrisi yapar."""
    cfg = LLM_MODELS.get(model_label, LLM_MODELS[DEFAULT_LLM])
    provider, model = cfg["provider"], cfg["model"]
    if provider == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY tanimli degil. .env dosyasina ekleyin.")
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            system=system_prompt, messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(b.text for b in message.content if getattr(b, "type", None) == "text")
    elif provider == "google":
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY tanimli degil. .env dosyasina ekleyin.")
        from google import genai
        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model=model, contents=user_prompt,
            config=genai.types.GenerateContentConfig(system_instruction=system_prompt, temperature=temperature),
        )
        return response.text
    raise ValueError("Bilinmeyen saglayici: " + str(provider))

ORGAN_TASK = {
    "Pankreas": "007",
    "Karaciğer": "003",
    "Dalak": "009",
    "Hepatik Damar": "008",
    "Akciğer": "006",
    "Kolon": "010"
}

ORGAN_LABELS = {
    "007": {"organ": 1, "tumor": 2, "organ_adi": "pankreas"},
    "003": {"organ": 1, "tumor": 2, "organ_adi": "karaciğer"},
    "009": {"organ": 1, "tumor": None, "organ_adi": "dalak"},
    "008": {"organ": 1, "tumor": 2, "organ_adi": "hepatik damar"},
    "006": {"organ": None, "tumor": 1, "organ_adi": "akciğer"},
    "010": {"organ": None, "tumor": 1, "organ_adi": "kolon"}
}

# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def run_nnunet(input_dir, output_dir, task_id):
    env = os.environ.copy()
    env['RESULTS_FOLDER'] = RESULTS_FOLDER
    env['nnUNet_raw_data_base'] = NNUNET_RAW
    env['nnUNet_preprocessed'] = NNUNET_PREPROCESSED

    # Tam yolu ver
    nnunet_exe = r"C:\Users\ASUS\PycharmProjects\TemelProje\radyoloji_env_311\Scripts\nnUNet_predict.exe"

    cmd = [
        nnunet_exe,
        '-i', input_dir,
        '-o', output_dir,
        '-t', task_id,
        '-m', '3d_fullres',
        '-f', '0',
        '--num_threads_preprocessing', '1',
        '--num_threads_nifti_save', '1',
        '--overwrite_existing'
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    # Windows/CUDA IPC kapanış uyarısı kaynaklı sahte exit-code hatalarını bypass etmek için dosya varlığını kontrol ediyoruz
    expected_output = os.path.join(output_dir, "hasta_001.nii.gz")
    return os.path.exists(expected_output) and os.path.getsize(expected_output) > 0

def run_totalsegmentator(input_path, output_dir):
    ts_exe = r"C:\Users\ASUS\PycharmProjects\TemelProje\radyoloji_env_311\Scripts\TotalSegmentator.exe"
    cmd = [
        ts_exe,
        '-i', input_path,
        '-o', output_dir,
        '--roi_subset', 'aorta', 'portal_vein_and_splenic_vein'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

def extract_metrics(mask_path, task_id, damar_dir=None):
    mask = sitk.ReadImage(mask_path)
    arr = sitk.GetArrayFromImage(mask)
    spacing = mask.GetSpacing()
    voxel_cm3 = (spacing[0] * spacing[1] * spacing[2]) / 1000

    labels = ORGAN_LABELS[task_id]
    bulgular = {
        "organ": labels["organ_adi"],
        "goruntuleme": "Portal venöz faz BT",
        "slice_thickness_mm": round(spacing[2], 1),
        "pixel_spacing_mm": round(spacing[0], 2)
    }

    if labels["organ"]:
        organ_arr = (arr == labels["organ"])
        bulgular["organ_hacmi_cm3"] = round(float(np.sum(organ_arr) * voxel_cm3), 2)

    if labels["tumor"]:
        tumor_arr = (arr == labels["tumor"])
        tumor_vol = float(np.sum(tumor_arr) * voxel_cm3)
        bulgular["tumor_hacmi_cm3"] = round(tumor_vol, 2)
        bulgular["tumor_cap_mm"] = round((tumor_vol * 6 / 3.14159) ** (1/3) * 10, 1)

        if labels["organ"] and np.any(organ_arr) and np.any(tumor_arr):
            pan_z = np.where(organ_arr)[0]
            tum_z_center = center_of_mass(tumor_arr)[0]
            relative_pos = (tum_z_center - pan_z.min()) / (pan_z.max() - pan_z.min() + 1e-6)
            bulgular["tumor_lokasyon"] = "baş" if relative_pos < 0.4 else "gövde" if relative_pos < 0.7 else "kuyruk"

        # Damar mesafesi
        if damar_dir:
            aorta_path = os.path.join(damar_dir, 'aorta.nii.gz')
            portal_path = os.path.join(damar_dir, 'portal_vein_and_splenic_vein.nii.gz')

            if os.path.exists(aorta_path):
                aorta = sitk.GetArrayFromImage(sitk.ReadImage(aorta_path))
                aorta_dist = distance_transform_edt(aorta == 0, sampling=spacing[::-1])
                bulgular["aorta_mesafe_mm"] = round(float(aorta_dist[tumor_arr].min()), 2)
                bulgular["aorta_sarma_derecesi"] = calculate_encasement(tumor_arr, aorta.astype(bool), spacing)

            if os.path.exists(portal_path):
                portal = sitk.GetArrayFromImage(sitk.ReadImage(portal_path))
                portal_dist = distance_transform_edt(portal == 0, sampling=spacing[::-1])
                bulgular["portal_ven_mesafe_mm"] = round(float(portal_dist[tumor_arr].min()), 2)
                bulgular["portal_ven_sarma_derecesi"] = calculate_encasement(tumor_arr, portal.astype(bool), spacing)

    bulgular["model"] = f"nnU-Net Task{task_id} (Isensee et al., Nature Methods 2021)"
    bulgular["model_agirlik"] = "Zenodo DOI: 10.5281/zenodo.4485926"
    return bulgular

def calculate_encasement(tumor_mask, vessel_mask, spacing):
    if not (np.any(tumor_mask) and np.any(vessel_mask)):
        return 0.0
    vessel_center = center_of_mass(vessel_mask)
    vessel_dist = distance_transform_edt(vessel_mask == 0, sampling=spacing[::-1])
    contact_voxels = np.argwhere((tumor_mask) & (vessel_dist < 2.0))
    if len(contact_voxels) == 0:
        return 0.0
    angles = []
    for voxel in contact_voxels:
        dy = voxel[1] - vessel_center[1]
        dx = voxel[2] - vessel_center[2]
        angles.append(np.degrees(np.arctan2(dy, dx)))
    angles = sorted(set([round(a, 0) for a in angles]))
    if len(angles) < 2:
        return 0.0
    gaps = [angles[i+1] - angles[i] for i in range(len(angles)-1)]
    gaps.append(360 - angles[-1] + angles[0])
    return round(360 - max(gaps), 1)

def recist_yanit(caplar_mm):
    """RECIST 1.1'e gore tedavi yanitini dondurur.
    caplar_mm: kronolojik sirali tumor caplari (mm); ilk eleman baslangic (baseline).
    Referans: Eisenhauer et al., Eur J Cancer 2009;45:228-247 (RECIST 1.1).
    Not: Cap, esdeger kure capidir (RECIST'in en uzun cap olcumunun yaklasigidir).
    Doner: (etiket, kisaltma, stil)  stil: 'success'|'warning'|'info'
    """
    if not caplar_mm or len(caplar_mm) < 2:
        return None
    baseline = caplar_mm[0]
    guncel = caplar_mm[-1]
    nadir = min(caplar_mm)  # calismadaki en kucuk olcum (baseline dahil)
    # Tam Yanit (CR): tumor tamamen kaybolmus
    if guncel <= 0:
        return ("Tam Yanıt (Complete Response, CR)", "CR", "success")
    # Progresif Hastalik (PD): nadire gore >=%20 VE mutlak >=5 mm artis
    if nadir > 0 and (guncel - nadir) >= 5 and ((guncel - nadir) / nadir * 100) >= 20:
        return ("Progresif Hastalık (Progressive Disease, PD)", "PD", "warning")
    # Kismi Yanit (PR): baseline'a gore >=%30 azalma
    if baseline > 0 and ((guncel - baseline) / baseline * 100) <= -30:
        return ("Kısmi Yanıt (Partial Response, PR)", "PR", "success")
    # Stabil Hastalik (SD)
    return ("Stabil Hastalık (Stable Disease, SD)", "SD", "info")

def generate_3d_visualization(mask_path, task_id, damar_dir=None):
    import SimpleITK as sitk
    import numpy as np
    from skimage import measure
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Read mask
    if not os.path.exists(mask_path):
        return None
        
    mask = sitk.ReadImage(mask_path)
    arr = sitk.GetArrayFromImage(mask) # shape: (z, y, x)
    spacing = mask.GetSpacing() # (x_spacing, y_spacing, z_spacing)
    
    labels = ORGAN_LABELS[task_id]
    
    # Performans için downsample adımı (hızlı 3D oluşturma ve akıcı 60 FPS döndürme için)
    ds = 2
    
    # 1. Organ Mesh
    if labels["organ"] is not None:
        organ_arr = (arr == labels["organ"])
        if np.any(organ_arr):
            organ_ds = organ_arr[::ds, ::ds, ::ds]
            if np.any(organ_ds):
                try:
                    # Daha pürüzsüz organik geçişler için hafif Gaussian yumuşatma uyguluyoruz
                    organ_smooth = ndimage.gaussian_filter(organ_ds.astype(float), sigma=1.0)
                    verts, faces, _, _ = measure.marching_cubes(organ_smooth, level=0.5)
                    # Milimetrik gerçek boyutlara esnet
                    x = verts[:, 2] * spacing[0] * ds
                    y = verts[:, 1] * spacing[1] * ds
                    z = verts[:, 0] * spacing[2] * ds
                    
                    fig.add_trace(go.Mesh3d(
                        x=x, y=y, z=z,
                        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                        name="Organ",
                        color="rgba(46, 204, 113, 0.15)", # Şeffaf yeşil
                        opacity=0.15,
                        showlegend=True
                    ))
                except Exception:
                    pass
                
    # 2. Tümör Mesh
    if labels["tumor"] is not None:
        tumor_arr = (arr == labels["tumor"])
        if np.any(tumor_arr):
            tumor_ds = tumor_arr[::ds, ::ds, ::ds]
            if np.any(tumor_ds):
                try:
                    # Tümör sınırlarında pürüzsüzleştirme
                    tumor_smooth = ndimage.gaussian_filter(tumor_ds.astype(float), sigma=0.8)
                    verts, faces, _, _ = measure.marching_cubes(tumor_smooth, level=0.5)
                    x = verts[:, 2] * spacing[0] * ds
                    y = verts[:, 1] * spacing[1] * ds
                    z = verts[:, 0] * spacing[2] * ds
                    
                    fig.add_trace(go.Mesh3d(
                        x=x, y=y, z=z,
                        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                        name="Tümör",
                        color="rgba(231, 76, 60, 0.85)", # Parlak kırmızı
                        opacity=0.85,
                        showlegend=True
                    ))
                except Exception:
                    pass
                
    # 3. Damarlar (Aorta ve Portal Ven)
    if damar_dir and os.path.exists(damar_dir):
        # Dosya yolları kalıcı klasörde farklı adlandırılmış olabilir
        aorta_path = os.path.join(damar_dir, 'aorta.nii.gz')
        portal_path = os.path.join(damar_dir, 'portal_vein.nii.gz')
        
        # Eğer temp klasöründeysek (Analiz anı)
        if not os.path.exists(aorta_path):
            aorta_path = os.path.join(damar_dir, 'aorta.nii.gz')
        if not os.path.exists(portal_path):
            portal_path = os.path.join(damar_dir, 'portal_vein_and_splenic_vein.nii.gz')
        
        if os.path.exists(aorta_path):
            try:
                aorta_img = sitk.ReadImage(aorta_path)
                aorta_arr = sitk.GetArrayFromImage(aorta_img)
                a_spacing = aorta_img.GetSpacing()
                if np.any(aorta_arr):
                    a_ds = aorta_arr[::ds, ::ds, ::ds]
                    if np.any(a_ds):
                        # Damarları yumuşat
                        aorta_smooth = ndimage.gaussian_filter(a_ds.astype(float), sigma=1.0)
                        verts, faces, _, _ = measure.marching_cubes(aorta_smooth, level=0.5)
                        x = verts[:, 2] * a_spacing[0] * ds
                        y = verts[:, 1] * a_spacing[1] * ds
                        z = verts[:, 0] * a_spacing[2] * ds
                        fig.add_trace(go.Mesh3d(
                            x=x, y=y, z=z,
                            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                            name="Aorta",
                            color="rgba(241, 196, 15, 0.4)", # Yarı-şeffaf altın sarısı (ayırt edilebilirlik için)
                            opacity=0.4,
                            showlegend=True
                        ))
            except Exception:
                pass
                    
        if os.path.exists(portal_path):
            try:
                portal_img = sitk.ReadImage(portal_path)
                portal_arr = sitk.GetArrayFromImage(portal_img)
                p_spacing = portal_img.GetSpacing()
                if np.any(portal_arr):
                    p_ds = portal_arr[::ds, ::ds, ::ds]
                    if np.any(p_ds):
                        # Portal ven için yumuşatma
                        portal_smooth = ndimage.gaussian_filter(p_ds.astype(float), sigma=1.0)
                        verts, faces, _, _ = measure.marching_cubes(portal_smooth, level=0.5)
                        x = verts[:, 2] * p_spacing[0] * ds
                        y = verts[:, 1] * p_spacing[1] * ds
                        z = verts[:, 0] * p_spacing[2] * ds
                        fig.add_trace(go.Mesh3d(
                            x=x, y=y, z=z,
                            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                            name="Portal Ven",
                            color="rgba(52, 152, 219, 0.5)", # Mavi
                            opacity=0.5,
                            showlegend=True
                        ))
            except Exception:
                pass
                    
    # Premium Tıbbi Siyah Tema
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Genişlik (X - mm)', backgroundcolor="rgb(15, 17, 22)", gridcolor="rgba(128,128,128,0.2)", showbackground=True),
            yaxis=dict(title='Derinlik (Y - mm)', backgroundcolor="rgb(15, 17, 22)", gridcolor="rgba(128,128,128,0.2)", showbackground=True),
            zaxis=dict(title='Yükseklik (Z - mm)', backgroundcolor="rgb(15, 17, 22)", gridcolor="rgba(128,128,128,0.2)", showbackground=True),
            aspectmode='data'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(color="white"))
    )
    return fig

def generate_report(bulgular, hasta_bilgi, gecmis_bulgular=None, model_label=DEFAULT_LLM):
    system_prompt = """Sen 15 yıllık deneyimli bir abdominal radyologsun.
Sana verilen BT segmentasyon bulgularını ACR ve ESR standartlarına uygun şekilde raporla.
ÖNEMLİ KURALLAR:
- Sadece verilen metriklere dayanarak rapor yaz.
- Karaciğer ve Pankreas gibi tümörlü vakalarda tümör boyutuna göre T evrelemesi yap (T1: ≤2cm, T2: 2-4cm, T3: >4cm).
- Vasküler temas varsa NCCN kriterlerine göre rezektabilite değerlendir (180° üzeri sarma lokal ileri/irresektabl anlamına gelir).
- Eğer incelenen organ DALAK ise: Dalak hacmini klinik olarak derecelendir (Erişkin normal aralığı: 100-250 cm³). 250-500 cm³ arasını "hafif splenomegali", 500-1000 cm³ arasını "orta dereceli splenomegali", 1000 cm³ üzerini ise "masif splenomegali" olarak yorumla. Splenomegali saptanması durumunda portal hipertansiyon, karaciğer parankim hastalığı (siroz), lenfoproliferatif hastalıklar ve enfeksiyonlar yönünden klinik/laboratuvar korelasyonu öner.
- Eğer incelenen organ AKCİĞER veya KOLON ise: Bulunan lezyon/tümör hacmi ve çapını klinik önemiyle yorumla (malignite şüphesi, takip veya histopatolojik korelasyon gereksinimi vb.).
- EĞER GEÇMİŞ İNCELEME BULGULARI VERİLMİŞSE: Güncel bulgularla karşılaştır. Tümör hacminin yüzdesel değişimini hesapla (küçüldü mü, büyüdü mü). Tümör regresyonunu (küçülme/iyileşme), progresyonunu (büyüme/kötüleşme) veya stabil durumu belirt. Damar mesafelerindeki ve sarma derecelerindeki iyileşmeleri (örneğin "damarı artık sarmıyor", "portal ven ile mesafe açıldı" gibi) analiz et. Bunu "KARŞILAŞTIRMALI DEĞERLENDİRME" başlığı altında detaylandır ve "SONUÇ VE ÖNERİ" kısmında tedavi yanıtını (kısmi yanıt, progresyon, stabil) klinik dille yorumla.
- Kesin tanı koyma, histopatolojik doğrulama önerilir de.
- Raporu Türkçe yaz, kısa ve öz tut.
- Markdown formatı KULLANMA, düz metin yaz.
RAPOR YAPISI:
1. TEKNİK
2. BULGULAR
3. KARŞILAŞTIRMALI DEĞERLENDİRME (Eğer geçmiş inceleme varsa)
4. VASKÜLER DEĞERLENDİRME (varsa)
5. EVRELEMESİ VE REZEKTABİLİTE (varsa)
6. SONUÇ VE ÖNERİ"""

    # KVKK/gizlilik: API'ye kimlik gonderme, sadece tarih ve klinik metrikler
    hasta_str = ""
    if not ANONYMIZE_FOR_API:
        if hasta_bilgi.get("ad_soyad"):
            hasta_str += f"Hasta: {hasta_bilgi['ad_soyad']}\n"
        if hasta_bilgi.get("tc"):
            hasta_str += f"TC: {hasta_bilgi['tc']}\n"
    if hasta_bilgi.get("tarih"):
        hasta_str += f"Tarih: {hasta_bilgi['tarih']}\n"

    gecmis_str = ""
    if gecmis_bulgular:
        gecmis_str = "\n\nHastanın Geçmiş BT İnceleme Bulguları (Kronolojik Sırayla):\n" + json.dumps(gecmis_bulgular, ensure_ascii=False, indent=2)

    prompt = f"{hasta_str}\nBT Segmentasyon Bulguları:\n{json.dumps(bulgular, ensure_ascii=False, indent=2)}{gecmis_str}\n\nLütfen standart ve karşılaştırmalı radyoloji raporu oluştur."

    return call_llm(system_prompt, prompt, model_label=model_label, temperature=0.3)

def create_pdf(rapor_text, hasta_bilgi):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    story = []

    # Başlık
    title_style = ParagraphStyle('title', parent=styles['Heading1'],
                                  fontSize=16, spaceAfter=12)
    story.append(Paragraph("RADYOLOJİ RAPORU", title_style))
    story.append(Spacer(1, 0.3*cm))

    # Hasta bilgileri
    if any(hasta_bilgi.values()):
        info_style = ParagraphStyle('info', parent=styles['Normal'], fontSize=10)
        if hasta_bilgi.get("ad_soyad"):
            story.append(Paragraph(f"<b>Hasta:</b> {hasta_bilgi['ad_soyad']}", info_style))
        if hasta_bilgi.get("tc"):
            story.append(Paragraph(f"<b>TC:</b> {hasta_bilgi['tc']}", info_style))
        if hasta_bilgi.get("tarih"):
            story.append(Paragraph(f"<b>Tarih:</b> {hasta_bilgi['tarih']}", info_style))
        story.append(Spacer(1, 0.5*cm))

    # Rapor içeriği
    rapor_style = ParagraphStyle('rapor', parent=styles['Normal'],
                                  fontSize=11, leading=16)
    for line in rapor_text.split('\n'):
        if line.strip():
            story.append(Paragraph(line, rapor_style))
            story.append(Spacer(1, 0.2*cm))

    # Footer
    story.append(Spacer(1, 1*cm))
    footer_style = ParagraphStyle('footer', parent=styles['Normal'],
                                   fontSize=9, textColor='grey')
    story.append(Paragraph(f"Oluşturulma tarihi: {datetime.datetime.now().strftime('%d.%m.%Y %H:%M')}", footer_style))
    story.append(Paragraph("Bu rapor yapay zeka destekli otomatik analiz sistemi tarafından üretilmiştir. Klinik karar için radyolog onayı gereklidir.", footer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer

def slugify(text):
    translation_table = str.maketrans({
        'ğ': 'g', 'Ğ': 'G',
        'ü': 'u', 'Ü': 'U',
        'ş': 's', 'Ş': 'S',
        'ı': 'i', 'İ': 'I',
        'ö': 'o', 'Ö': 'O',
        'ç': 'c', 'Ç': 'C'
    })
    return text.translate(translation_table)

def save_patient(hasta_bilgi, bulgular, rapor, chat_history, current_file=None):
    os.makedirs(PATIENTS_DB_DIR, exist_ok=True)
    if current_file and os.path.exists(os.path.join(PATIENTS_DB_DIR, current_file)):
        file_name = current_file
    else:
        tc = hasta_bilgi.get("tc") or "BilinmeyenTC"
        ad_soyad = slugify(hasta_bilgi.get("ad_soyad") or "Adsiz")
        tarih_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{tc}_{ad_soyad}_{tarih_str}".replace(" ", "_").replace("/", "-") + ".json"
        
    file_path = os.path.join(PATIENTS_DB_DIR, file_name)
    data = {
        "hasta_bilgi": hasta_bilgi,
        "bulgular": bulgular,
        "rapor": rapor,
        "chat_history": chat_history,
        "file_name": file_name
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return file_name

def load_patient(file_name):
    file_path = os.path.join(PATIENTS_DB_DIR, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_saved_patients():
    os.makedirs(PATIENTS_DB_DIR, exist_ok=True)
    patients = []
    for f in os.listdir(PATIENTS_DB_DIR):
        if f.endswith(".json"):
            patients.append(f)
    return patients

def get_patient_history(tc, current_tarih=None):
    if not tc:
        return []
    history = []
    os.makedirs(PATIENTS_DB_DIR, exist_ok=True)
    for f in os.listdir(PATIENTS_DB_DIR):
        if f.endswith(".json"):
            try:
                data = load_patient(f)
                patient_tc = data.get("hasta_bilgi", {}).get("tc")
                patient_tarih = data.get("hasta_bilgi", {}).get("tarih")
                # Aynı TC ve farklı tarih (veya tarih yoksa da al)
                if str(patient_tc) == str(tc):
                    if current_tarih and patient_tarih == current_tarih:
                        continue
                    history.append({
                        "tarih": patient_tarih or "Bilinmeyen Tarih",
                        "bulgular": data.get("bulgular", {}),
                        "rapor": data.get("rapor", "")
                    })
            except Exception:
                continue
    # Tarihe göre sırala
    try:
        def parse_date(d_str):
            try:
                return datetime.datetime.strptime(d_str, "%d.%m.%Y")
            except Exception:
                return datetime.datetime.min
        history.sort(key=lambda x: parse_date(x["tarih"]))
    except Exception:
        pass
    return history

# ============================================================
# STREAMLIT ARAYÜZÜ
# ============================================================

st.set_page_config(page_title="Radyoloji AI", layout="wide")
st.title("🏥 Radyoloji AI Sistemi")
st.caption("nnU-Net + TotalSegmentator + LLM (Claude / Gemini)")

# Session state
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = DEFAULT_LLM
if 'rapor' not in st.session_state:
    st.session_state.rapor = None
if 'bulgular' not in st.session_state:
    st.session_state.bulgular = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_patient_file' not in st.session_state:
    st.session_state.current_patient_file = None
if 'hasta_bilgi' not in st.session_state:
    st.session_state.hasta_bilgi = {"ad_soyad": "", "tc": "", "tarih": datetime.date.today().strftime("%d.%m.%Y")}

# ---- SOL PANEL ----
with st.sidebar:
    st.header("⚙️ Ayarlar")

    st.subheader("🤖 LLM Modeli")
    st.session_state.llm_model = st.selectbox(
        "Rapor/chatbot için model",
        list(LLM_MODELS.keys()),
        index=list(LLM_MODELS.keys()).index(st.session_state.llm_model),
    )
    _sel = LLM_MODELS[st.session_state.llm_model]
    _key_ok = (ANTHROPIC_API_KEY if _sel["provider"] == "anthropic" else GOOGLE_API_KEY)
    if not _key_ok:
        st.warning(f"{_sel['provider'].upper()} API anahtarı .env'de tanımlı değil.")

    st.subheader("📁 Geçmiş Hastalar")
    saved_patients = get_saved_patients()
    if saved_patients:
        selected_patient_file = st.selectbox("Kayıtlı Hastalardan Seç", ["Seçiniz..."] + saved_patients)
        if selected_patient_file != "Seçiniz...":
            if st.button("📂 Seçili Hastayı Yükle", use_container_width=True):
                data = load_patient(selected_patient_file)
                st.session_state.hasta_bilgi = data.get("hasta_bilgi", {})
                st.session_state.bulgular = data.get("bulgular")
                st.session_state.rapor = data.get("rapor")
                st.session_state.chat_history = data.get("chat_history", [])
                st.session_state.current_patient_file = data.get("file_name", selected_patient_file)
                st.success("Hasta başarıyla yüklendi!")
    else:
        st.info("Kayıtlı hasta bulunamadı.")
        
    st.divider()

    st.subheader("👤 Yeni Hasta Bilgileri")
    ad_soyad = st.text_input("Ad Soyad", value=st.session_state.hasta_bilgi.get("ad_soyad", ""))
    tc = st.text_input("TC Kimlik No", value=st.session_state.hasta_bilgi.get("tc", ""))
    tarih = st.text_input("Tarih", value=st.session_state.hasta_bilgi.get("tarih", datetime.date.today().strftime("%d.%m.%Y")))
    
    st.session_state.hasta_bilgi["ad_soyad"] = ad_soyad
    st.session_state.hasta_bilgi["tc"] = tc
    st.session_state.hasta_bilgi["tarih"] = tarih

    st.subheader("🫁 Görüntü Yükle (Yeni Analiz)")
    organ = st.selectbox("Organ Seç", list(ORGAN_TASK.keys()))
    uploaded_file = st.file_uploader("CT Dosyası Yükle (.nii veya .nii.gz)", type=['nii', 'gz'])

    analiz_btn = st.button("🔬 Analiz Et", type="primary", use_container_width=True)

# ---- ANA PANEL ----
col1, col2 = st.columns([3, 2])

with col1:
    if analiz_btn:
        if uploaded_file is None:
            st.error("Lütfen CT dosyası yükleyin!")
        else:
            task_id = ORGAN_TASK[organ]
            hasta_bilgi = {"ad_soyad": ad_soyad, "tc": tc, "tarih": tarih}

            # Geçici klasörler
            tmp_dir = tempfile.mkdtemp()
            input_dir = os.path.join(tmp_dir, "input")
            output_dir = os.path.join(tmp_dir, "output")
            damar_dir = os.path.join(tmp_dir, "damar")
            os.makedirs(input_dir)
            os.makedirs(output_dir)
            os.makedirs(damar_dir)

            # CT dosyasını kaydet
            ct_path = os.path.join(input_dir, "hasta_001_0000.nii.gz")
            plain_path = os.path.join(input_dir, "hasta_001_plain.nii")

            uploaded_bytes = uploaded_file.read()

            import gzip
            # Eğer yüklenen dosya .gz (sıkıştırılmış) ise doğrudan kaydet ve TotalSegmentator için açıp kaydet
            if uploaded_file.name.endswith('.gz'):
                with open(ct_path, 'wb') as f:
                    f.write(uploaded_bytes)
                try:
                    decompressed_bytes = gzip.decompress(uploaded_bytes)
                    with open(plain_path, 'wb') as f:
                        f.write(decompressed_bytes)
                except Exception:
                    with open(plain_path, 'wb') as f:
                        f.write(uploaded_bytes)
            # Eğer sıkıştırılmamış .nii ise nnU-Net için gzip ile sıkıştırıp kaydet, TotalSegmentator için doğrudan kaydet
            else:
                with gzip.open(ct_path, 'wb') as f:
                    f.write(uploaded_bytes)
                with open(plain_path, 'wb') as f:
                    f.write(uploaded_bytes)

            try:
                # 1. nnU-Net
                with st.status("🔬 Segmentasyon yapılıyor...", expanded=True) as status:
                    st.write("nnU-Net modeli çalışıyor...")
                    success = run_nnunet(input_dir, output_dir, task_id)
                    
                    if not success:
                        st.error("nnU-Net hatası!")
                        st.stop()

                    # 2. TotalSegmentator
                    st.write("Damar segmentasyonu yapılıyor...")
                    run_totalsegmentator(plain_path, damar_dir)

                    # 3. Metrik çıkarımı
                    st.write("Metrikler hesaplanıyor...")
                    mask_path = os.path.join(output_dir, "hasta_001.nii.gz")
                    bulgular = extract_metrics(mask_path, task_id, damar_dir)
                    st.session_state.bulgular = bulgular

                    # 4. Rapor üretimi
                    st.write("Geçmiş incelemeler sorgulanıyor...")
                    gecmis_bulgular = get_patient_history(hasta_bilgi.get("tc"), hasta_bilgi.get("tarih"))
                    
                    st.write(f"Radyoloji raporu üretiliyor ({st.session_state.llm_model})...")
                    rapor = generate_report(bulgular, hasta_bilgi, gecmis_bulgular, model_label=st.session_state.llm_model)
                    st.session_state.rapor = rapor
                    st.session_state.chat_history = []
                    
                    # Sonucu Veritabanına Kaydet
                    current_file = save_patient(hasta_bilgi, bulgular, rapor, [])
                    st.session_state.current_patient_file = current_file
                    
                    # Maskeleri Kalıcı Olarak Kaydet (3D Rekonstrüksiyon için)
                    patient_mask_dir = os.path.join(PATIENTS_DB_DIR, current_file.replace(".json", "_masks"))
                    os.makedirs(patient_mask_dir, exist_ok=True)
                    shutil.copy(mask_path, os.path.join(patient_mask_dir, "segmentation.nii.gz"))
                    
                    # Damarları kopyala
                    aorta_src = os.path.join(damar_dir, "aorta.nii.gz")
                    portal_src = os.path.join(damar_dir, "portal_vein_and_splenic_vein.nii.gz")
                    if os.path.exists(aorta_src):
                        shutil.copy(aorta_src, os.path.join(patient_mask_dir, "aorta.nii.gz"))
                    if os.path.exists(portal_src):
                        shutil.copy(portal_src, os.path.join(patient_mask_dir, "portal_vein.nii.gz"))

                    status.update(label="✅ Analiz tamamlandı!", state="complete")

            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    # Rapor göster
    if st.session_state.rapor:
        tab1, tab2, tab3 = st.tabs(["📋 Güncel Analiz & Rapor", "📈 Tedavi Yanıt Takibi", "🌐 İnteraktif 3D Anatomi"])

        with tab1:
            st.subheader("📋 Radyoloji Raporu")
            st.text_area("", st.session_state.rapor, height=400)

            # PDF indir
            pdf_buffer = create_pdf(st.session_state.rapor, st.session_state.hasta_bilgi)
            st.download_button(
                label="📄 PDF İndir",
                data=pdf_buffer,
                file_name=f"radyoloji_raporu_{datetime.date.today()}.pdf",
                mime="application/pdf"
            )

        with tab2:
            st.subheader("📈 Boylamsal Tedavi Yanıt Analizi")
            
            # Tüm geçmiş taramaları çek (şu anki dahil)
            patient_history = get_patient_history(st.session_state.hasta_bilgi.get("tc"))
            
            # Şu anki taramayı geçmişe dahil değilse ekle
            current_date = st.session_state.hasta_bilgi.get("tarih") or "Güncel"
            has_current = any(h["tarih"] == current_date for h in patient_history)
            
            all_scans = list(patient_history)
            if not has_current:
                all_scans.append({
                    "tarih": current_date,
                    "bulgular": st.session_state.bulgular,
                    "rapor": st.session_state.rapor
                })
            
            # Tarihe göre sırala
            try:
                def parse_date(d_str):
                    try:
                        return datetime.datetime.strptime(d_str, "%d.%m.%Y")
                    except Exception:
                        return datetime.datetime.min
                all_scans.sort(key=lambda x: parse_date(x["tarih"]))
            except Exception:
                pass
            
            # Tümör olan taramaları süz
            tumor_scans = [s for s in all_scans if s["bulgular"].get("tumor_hacmi_cm3") is not None]
            
            if len(tumor_scans) > 1:
                st.info("💡 Hastanın sistemde kayıtlı birden fazla taraması tespit edildi. Tedavi yanıt grafikleri ve karşılaştırma metrikleri aşağıdadır:")
                
                # Karşılaştırma kartları
                prev_scan = tumor_scans[-2]
                curr_scan = tumor_scans[-1]
                
                p_vol = prev_scan["bulgular"].get("tumor_hacmi_cm3", 0)
                c_vol = curr_scan["bulgular"].get("tumor_hacmi_cm3", 0)
                p_cap = prev_scan["bulgular"].get("tumor_cap_mm", 0)
                c_cap = curr_scan["bulgular"].get("tumor_cap_mm", 0)
                
                vol_change = ((c_vol - p_vol) / p_vol * 100) if p_vol > 0 else 0
                cap_change = ((c_cap - p_cap) / p_cap * 100) if p_cap > 0 else 0
                
                # RECIST 1.1'e gore tedavi yaniti (cap-temelli; baseline & nadir referansli)
                caplar = [float(s2["bulgular"].get("tumor_cap_mm", 0) or 0) for s2 in tumor_scans]
                baseline_cap = caplar[0]
                nadir_cap = min(caplar)
                yanit = recist_yanit(caplar)
                if yanit:
                    etiket, _kisalt, stil = yanit
                    baseline_degisim = ((c_cap - baseline_cap) / baseline_cap * 100) if baseline_cap > 0 else 0
                    mesaj = (f"**{etiket}** — Başlangıca göre çap değişimi: %{baseline_degisim:+.1f} "
                             f"(başlangıç {baseline_cap:.1f} mm, nadir {nadir_cap:.1f} mm, güncel {c_cap:.1f} mm).")
                    getattr(st, stil)(mesaj)
                
                # Metrik sütunları
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.metric(
                        label="Tümör Hacmi (cm³)",
                        value=f"{c_vol:.2f} cm³",
                        delta=f"{vol_change:+.1f}%",
                        delta_color="inverse"
                    )
                with m_col2:
                    st.metric(
                        label="Tümör Çapı (mm)",
                        value=f"{c_cap:.1f} mm",
                        delta=f"{cap_change:+.1f}%",
                        delta_color="inverse"
                    )
                with m_col3:
                    p_port = prev_scan["bulgular"].get("portal_ven_sarma_derecesi", 0)
                    c_port = curr_scan["bulgular"].get("portal_ven_sarma_derecesi", 0)
                    port_diff = c_port - p_port
                    st.metric(
                        label="Portal Ven Sarma Derecesi",
                        value=f"{c_port}°",
                        delta=f"{port_diff:+.1f}°",
                        delta_color="inverse"
                    )
                
                # Çizgi Grafik
                st.subheader("📊 Boylamsal Tümör Hacim Trendi")
                import pandas as pd
                chart_data = pd.DataFrame({
                    "Tarih": [s["tarih"] for s in tumor_scans],
                    "Tümör Hacmi (cm³)": [s["bulgular"].get("tumor_hacmi_cm3", 0) for s in tumor_scans]
                }).set_index("Tarih")
                
                st.line_chart(chart_data["Tümör Hacmi (cm³)"])
                
                # Detaylı Tarihçe Tablosu
                st.subheader("📋 Tetkik Geçmişi")
                table_data = []
                for s in reversed(all_scans):
                    b = s["bulgular"]
                    table_data.append({
                        "Tarih": s["tarih"],
                        "Organ": b.get("organ", "Bilinmiyor").capitalize(),
                        "Organ Hacmi (cm³)": b.get("organ_hacmi_cm3", "-"),
                        "Tümör Hacmi (cm³)": b.get("tumor_hacmi_cm3", "-"),
                        "Tümör Çapı (mm)": b.get("tumor_cap_mm", "-"),
                        "Portal Ven Mesafe (mm)": b.get("portal_ven_mesafe_mm", "-"),
                        "Portal Ven Sarma": f"{b.get('portal_ven_sarma_derecesi', 0)}°" if b.get('portal_ven_sarma_derecesi') is not None else "-"
                    })
                st.table(table_data)
            else:
                st.info("💡 Hastanın boylamsal takibini yapabilmek için sistemde en az 2 farklı tarihte tümör segmentasyon analizi bulunmalıdır.")
                st.write("Şu anki analize ait metrikler:")
                st.json(st.session_state.bulgular)

        with tab3:
            st.subheader("🌐 İnteraktif 3D Rekonstrüksiyon")
            if st.session_state.current_patient_file:
                patient_mask_dir = os.path.join(PATIENTS_DB_DIR, st.session_state.current_patient_file.replace(".json", "_masks"))
                seg_path = os.path.join(patient_mask_dir, "segmentation.nii.gz")
                
                if os.path.exists(seg_path):
                    st.info("💡 Farenizin sol tuşu ile modeli döndürebilir, sağ tuşu ile kaydırabilir ve tekerlek ile yakınlaştırabilirsiniz.")
                    with st.spinner("3D model rekonstrükte ediliyor, lütfen bekleyin..."):
                        organ_adi = st.session_state.bulgular.get("organ", "karaciğer")
                        organ_key = next((k for k, v in ORGAN_TASK.items() if k.lower() == organ_adi.lower()), "Karaciğer")
                        task_id = ORGAN_TASK[organ_key]
                        
                        fig = generate_3d_visualization(seg_path, task_id, patient_mask_dir)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("3D model oluşturulamadı.")
                else:
                    st.warning("Bu hasta için kayıtlı 3D segmentasyon maskesi bulunamadı.")
            else:
                st.info("3D rekonstrüksiyonu görüntülemek için lütfen bir analiz yapın veya kayıtlı hasta yükleyin.")

with col2:
    if st.session_state.rapor:
        st.subheader("💬 Radyolog ile Soru-Cevap")

        # Chat geçmişi için kaydırılabilir pencere (Scrollable Container)
        chat_container = st.container(height=500, border=True)
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        # Soru gir (Sabit alt kısımda kalır)
        soru = st.chat_input("Rapor hakkında soru sorun...")

        if soru:
            st.session_state.chat_history.append({"role": "user", "content": soru})

            with chat_container:
                with st.chat_message("user"):
                    st.write(soru)

            # Cevap üret
            patient_history = get_patient_history(st.session_state.hasta_bilgi.get("tc"))
            gecmis_str = ""
            if patient_history:
                gecmis_str = "\n\nHastanın Veritabanındaki Tüm Tetkik Geçmişi ve Bulguları:\n" + json.dumps(patient_history, ensure_ascii=False, indent=2)

            system = f"""Sen 15 yıllık deneyimli bir abdominal radyologsun.
Aşağıdaki raporu sen yazdın. Sorulara kısa, net ve klinik dilde cevap ver.
Hastanın geçmişteki diğer taramaları ve bulguları da sana sunulmuştur. Eğer kullanıcı farklı tarihlerdeki taramalar arasındaki farkları, tümörün küçülüp küçülmediğini, tedaviye yanıtı sorarsa bu verileri karşılaştırarak tıbbi dilde yanıt ver.
Bilmediğin şeyleri uydurmadan 'bu veri mevcut değil' de. Türkçe cevap ver.
RAPOR:
{st.session_state.rapor}{gecmis_str}"""

            cevap = call_llm(system, soru, model_label=st.session_state.llm_model, temperature=0.3)

            st.session_state.chat_history.append({"role": "assistant", "content": cevap})

            with chat_container:
                with st.chat_message("assistant"):
                    st.write(cevap)

            # Chat geçmişi güncellendi, kaydet
            if st.session_state.current_patient_file:
                save_patient(
                    st.session_state.hasta_bilgi, 
                    st.session_state.bulgular, 
                    st.session_state.rapor, 
                    st.session_state.chat_history,
                    st.session_state.current_patient_file
                )

    elif not analiz_btn:
        st.info("Sol panelden CT dosyası yükleyip 'Analiz Et' butonuna basın.")