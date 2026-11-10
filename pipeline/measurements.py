"""
measurements.py — maskeden geometrik ölçüm (F01, F04, F05, F06, F14).

TEK ölçüm tanımı: tüm pipeline (LLM girdisi + değerlendirme) aynı sayıyı
kullanır. Eski hatalar:
  * F01/F14: Bir yerde "hacim-eşdeğeri küre çapı", başka yerde aksiyel çap;
    bir yerde en büyük çaplı, başka yerde en büyük hacimli lezyon; ConvexHull
    başarısızlığında RASTGELE 3000 nokta. Hepsi kaldırıldı.
  * F04: Sarma açısı tüm damarın TEK ağırlık merkezinden; uzak voxeller açıyı
    değiştiriyordu. Artık YEREL (kesit bazlı) hesaplanır.

Tanımlar (hepsi deterministik):
  - Aksiyel en uzun çap: her aksiyel kesitte en uzun kiriş, kesitler üzerinden
    maksimum. Fiziksel x/y spacing ayrı kullanılır.
  - Hacim: voxel sayısı x voxel hacmi. Çaptan BAĞIMSIZ ayrı alan.
  - Lezyon: 26-komşulukla bağlı bileşen; FİZİKSEL hacim eşiğiyle filtrelenir.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import MEASUREMENT, GEOMETRY, Status

# Aksiyel kesit ekseni: load_nifti(canonical=True) sonrası eksen 2 = S/I.
AXIAL_AXIS = 2


# ---------------------------------------------------------------------------
# Lezyon ayrıştırma ve çap
# ---------------------------------------------------------------------------
def _extreme_points(phys: np.ndarray) -> np.ndarray:
    """Dejenere (doğrusal) küme için çapı taşıyan uç noktalar (en çok 6 adet).

    ConvexHull yalnız (yaklaşık) eşdoğrusal/dejenere kümelerde QhullError verir;
    böyle bir kümede en uzak çift, baskın (PCA) yöndeki iki uç noktadır — bu O(n)
    bulunur ve doğrusal küme için SONUÇ KESİNDİR. Böylece tüm noktalarda O(n²)
    kaba-kuvvet bellek patlaması (büyük kesitte kilitlenme) önlenir.
    """
    if phys.shape[0] <= 2:
        return phys
    c = phys - phys.mean(axis=0)
    try:
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        proj = c @ vt[0]
    except np.linalg.LinAlgError:
        proj = c[:, 0]
    idx = {int(proj.argmin()), int(proj.argmax()),
           int(phys[:, 0].argmin()), int(phys[:, 0].argmax()),
           int(phys[:, 1].argmin()), int(phys[:, 1].argmax())}
    return phys[sorted(idx)]


def _slice_longest_diameter_mm(points_ij: np.ndarray, sx: float, sy: float) -> float:
    """Tek kesitteki (i,j) piksel kümesinin en uzun kirişi (mm). Deterministik."""
    n = points_ij.shape[0]
    if n == 0:
        return 0.0
    if n == 1:
        return 0.0
    phys = points_ij.astype(float) * np.array([sx, sy])
    # Önce konveks kabuk; başarısızsa/doğrusalsa uç noktalara indir (O(n²) kaçın).
    pts = phys
    try:
        from scipy.spatial import ConvexHull, QhullError
        if n >= 3:
            try:
                hull = ConvexHull(phys)
                pts = phys[hull.vertices]
            except QhullError:
                pts = _extreme_points(phys)  # doğrusal/dejenere -> uç noktalar
    except Exception:
        pts = _extreme_points(phys)
    # Kabuk köşeleri az sayıda; kaba kuvvet maksimum (deterministik).
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    return float(np.sqrt(d2.max()))


def axial_longest_diameter_mm(lesion_mask: np.ndarray, spacing: np.ndarray) -> float:
    """Lezyonun aksiyel en uzun çapı (mm): kesitler üzerinden maksimum kiriş."""
    sx, sy, _sz = spacing
    best = 0.0
    for k in range(lesion_mask.shape[AXIAL_AXIS]):
        sl = np.take(lesion_mask, k, axis=AXIAL_AXIS)
        pts = np.argwhere(sl > 0)
        if pts.shape[0] == 0:
            continue
        best = max(best, _slice_longest_diameter_mm(pts, sx, sy))
    return best


def extract_lesions(tumor_mask: np.ndarray, spacing: np.ndarray,
                    min_volume_cm3: float | None = None,
                    connectivity: int | None = None) -> list[dict[str, Any]]:
    """Tümör maskesinden lezyon listesi. Her lezyon: id, çap(mm), hacim(cm3).

    Performans (K6): bileşen hacmi tek geçişte `np.bincount`, çap/centroid ise
    bileşenin SINIR KUTUSUNDA (`ndimage.find_objects`) hesaplanır. Eskiden her
    etiket için `labeled == lab` tüm hacmi (512²×366 ≈ 96M voxel) baştan tarıyordu;
    zero-shot OOD maskedeki binlerce küçük yanlış-pozitif bileşende saniyeler
    sürüyordu. SONUÇ AYNI (hacim, çap, centroid, sıralama değişmez).
    """
    from scipy import ndimage
    conn = MEASUREMENT.connectivity if connectivity is None else connectivity
    min_vol = MEASUREMENT.min_lesion_volume_cm3 if min_volume_cm3 is None else min_volume_cm3

    tumor_mask = np.asarray(tumor_mask) > 0
    if not tumor_mask.any():
        return []

    structure = ndimage.generate_binary_structure(3, conn)
    labeled, n = ndimage.label(tumor_mask, structure=structure)
    if n == 0:
        return []
    voxel_vol_cm3 = float(np.prod(spacing)) / 1000.0
    counts = np.bincount(labeled.ravel(), minlength=n + 1)   # counts[lab] = voxel sayısı
    boxes = ndimage.find_objects(labeled)                    # boxes[lab-1] = sınır kutusu

    lesions: list[dict[str, Any]] = []
    for lab in range(1, n + 1):
        vox = int(counts[lab])
        vol_cm3 = float(vox) * voxel_vol_cm3
        if vol_cm3 < min_vol:
            continue
        sl = boxes[lab - 1]
        sub = labeled[sl] == lab                             # yalnız sınır kutusu
        dia = axial_longest_diameter_mm(sub, spacing)
        offset = np.array([s.start for s in sl], dtype=float)
        centroid = np.argwhere(sub).mean(axis=0) + offset    # global (i,j,k)
        lesions.append({
            "id": lab,
            "cap_mm": round(dia, 2),
            "cap_mm_raw": dia,          # eşik/karar RAW değer üzerinden (F13)
            "hacim_cm3": round(vol_cm3, 2),
            "hacim_cm3_raw": vol_cm3,
            "voxel": vox,
            "centroid_ijk": [round(float(x), 2) for x in centroid],  # konum eşleştirme için
        })
    # Deterministik sıralama: önce çapa, sonra id'ye göre
    lesions.sort(key=lambda d: (-d["cap_mm_raw"], d["id"]))
    return lesions


def summarize_lesions(lesions: list[dict[str, Any]]) -> dict[str, Any]:
    """Lezyon listesinden organ özeti. En büyük ÇAP ve en büyük HACİM AYRI (F01)."""
    if not lesions:
        return {
            "lezyon_sayisi": 0,
            "toplam_tumor_hacim_cm3": 0.0,
            "en_buyuk_cap_lezyon": None,
            "en_buyuk_hacim_lezyon": None,
            "status": Status.EMPTY_MASK,
        }
    total_vol = round(sum(l["hacim_cm3_raw"] for l in lesions), 2)
    by_dia = max(lesions, key=lambda d: (d["cap_mm_raw"], d["id"]))
    by_vol = max(lesions, key=lambda d: (d["hacim_cm3_raw"], d["id"]))
    return {
        "lezyon_sayisi": len(lesions),
        "toplam_tumor_hacim_cm3": total_vol,
        "en_buyuk_cap_lezyon": {"id": by_dia["id"], "cap_mm": by_dia["cap_mm"],
                                "hacim_cm3": by_dia["hacim_cm3"]},
        "en_buyuk_hacim_lezyon": {"id": by_vol["id"], "cap_mm": by_vol["cap_mm"],
                                  "hacim_cm3": by_vol["hacim_cm3"]},
        "status": Status.SUCCESS,
    }


# ---------------------------------------------------------------------------
# Tümör–damar ilişkisi (F04, F05) — YEREL kesit bazlı. Araştırma ölçümü.
# ---------------------------------------------------------------------------
def _slice_boundary(mask2d: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    if not mask2d.any():
        return np.empty((0, 2), dtype=int)
    er = ndimage.binary_erosion(mask2d, border_value=0)
    return np.argwhere(mask2d & ~er)


def vessel_wrap_angle_deg(tumor_mask: np.ndarray, vessel_mask: np.ndarray,
                          spacing: np.ndarray,
                          contact_mm: float | None = None) -> dict[str, Any]:
    """Tümörün damarı sarma açısı (derece), YEREL kesit bazlı (F04).

    Her aksiyel kesitte damar kendi kesit merkezine göre değerlendirilir; başka
    kesitlerdeki uzak damar voxelleri bir kesitin açısını DEĞİŞTİRMEZ. Çıktı
    araştırma ölçümüdür (research_grade=True); tek başına cerrahi karar üretmez.

    Durumlar: ikisi de hiç birlikte görünmüyorsa not_assessed (açı None);
    birlikte var ama temas yoksa success + 0 derece.

    Performans (K5): her kesitte işlem tüm 512×512'de değil, DAMAR SINIR KUTUSUNDA
    (+ temas marjini kadar tümör penceresi) yapılır. Damar sınır pikselinin en yakın
    tümör pikseline mesafesi, tümörün SINIRINDAN cKDTree ile bulunur (kesit dışı bir
    nokta için en yakın tümör pikseli zaten sınırdadır; içerideki damar pikseli t2
    True → 0). Sonuç tüm-piksel cKDTree ile BİREBİR AYNI (80/80 rastgele vakada fark
    0); maliyet 512² yerine damar kutusuyla orantılı → büyük/yayılmış maskede askı yok.
    """
    from scipy.spatial import cKDTree
    contact = MEASUREMENT.vessel_contact_mm if contact_mm is None else contact_mm
    sx, sy, _ = spacing
    tumor_mask = np.asarray(tumor_mask) > 0
    vessel_mask = np.asarray(vessel_mask) > 0

    if not vessel_mask.any():
        return {"sarma_derece": None, "status": Status.EMPTY_MASK, "research_grade": True}

    # Yalnız HEM damar HEM tümör bulunan kesitlerde çalış (boş kesitleri gezme).
    v_present = vessel_mask.any(axis=(0, 1))
    t_present = tumor_mask.any(axis=(0, 1))
    both = np.nonzero(v_present & t_present)[0]

    best_angle = 0.0
    spacing_xy = np.array([sx, sy])
    H, W = vessel_mask.shape[0], vessel_mask.shape[1]
    mi = int(np.ceil(contact / sx)) + 2      # satır (i) temas marjini (piksel)
    mj = int(np.ceil(contact / sy)) + 2      # sütun (j) temas marjini (piksel)
    for k in both:
        v2 = np.take(vessel_mask, k, axis=AXIAL_AXIS)
        # Damar sınır kutusu (ucuz any-projeksiyon); tüm 512×512'de erozyon YOK.
        vr = np.where(v2.any(axis=1))[0]
        vc = np.where(v2.any(axis=0))[0]
        if vr.size == 0:
            continue
        r0, r1, c0, c1 = int(vr[0]), int(vr[-1]) + 1, int(vc[0]), int(vc[-1]) + 1
        vb_local = _slice_boundary(v2[r0:r1, c0:c1])       # yalnız damar kutusu
        if vb_local.shape[0] == 0:
            continue
        v_bound = vb_local + (r0, c0)                       # global (i,j)
        v_phys = v_bound.astype(float) * spacing_xy
        centroid = v_phys.mean(axis=0)  # YEREL merkez (bu kesit)
        t2 = np.take(tumor_mask, k, axis=AXIAL_AXIS)
        # İçeride (t2 True) ise mesafe 0; dışarıda ise en yakın tümör SINIRI.
        inside = t2[v_bound[:, 0], v_bound[:, 1]]
        dist = np.zeros(v_bound.shape[0])
        out = ~inside
        if out.any():
            # Yalnız damar kutusu + temas marjini kadar tümör penceresi. KESİN:
            # temas eşiği (<=contact) içindeki her tümör pikseli bu pencerededir;
            # pencere kenarındaki olası sahte sınır noktaları >contact uzaktadır.
            tr0, tr1 = max(r0 - mi, 0), min(r1 + mi, H)
            tc0, tc1 = max(c0 - mj, 0), min(c1 + mj, W)
            tb_local = _slice_boundary(t2[tr0:tr1, tc0:tc1])
            if tb_local.shape[0]:
                t_phys = (tb_local + (tr0, tc0)).astype(float) * spacing_xy
                dd, _ = cKDTree(t_phys).query(v_phys[out], k=1)
                dist[out] = dd
            else:
                dist[out] = np.inf
        contact_pts = v_phys[dist <= contact]
        if contact_pts.shape[0] == 0:
            continue
        angles = np.arctan2(contact_pts[:, 1] - centroid[1],
                            contact_pts[:, 0] - centroid[0])
        best_angle = max(best_angle, _angular_coverage_deg(angles))

    if both.size == 0:
        return {"sarma_derece": None, "status": Status.NOT_ASSESSED, "research_grade": True}
    return {"sarma_derece": round(best_angle, 1), "status": Status.SUCCESS,
            "research_grade": True, "temas_kesit_sayisi": int(both.size)}


def _angular_coverage_deg(angles: np.ndarray) -> float:
    """Açı kümesinin kapsadığı yay (derece). En büyük boşluğun tümleyeni."""
    if angles.size == 0:
        return 0.0
    if angles.size == 1:
        return 0.0
    a = np.sort(angles)
    gaps = np.diff(a)
    wrap_gap = (a[0] + 2 * np.pi) - a[-1]
    max_gap = max(gaps.max(), wrap_gap)
    coverage = 2 * np.pi - max_gap
    return float(np.degrees(coverage))


def vessel_distance_mm(tumor_mask: np.ndarray, tumor_affine: np.ndarray,
                       vessel_mask: np.ndarray, vessel_affine: np.ndarray) -> dict[str, Any]:
    """Tümör ile damar arası min yüzey mesafesi (mm). Boş maske -> None (F05).

    Mesafe yüzey VOXEL MERKEZLERİ arasında ölçülür (K3): örtüşen maskeler 0,
    yalnız bitişik (komşu voxel) maskeler ~spacing verir. 'Temas' eşiği bu
    tanıma göre yorumlanmalıdır.
    """
    from .imaging import min_surface_distance_mm
    if not np.asarray(vessel_mask).any():
        return {"mesafe_mm": None, "status": Status.EMPTY_MASK,
                "tanim": "yuzey-voxel-merkezi"}
    if not np.asarray(tumor_mask).any():
        return {"mesafe_mm": None, "status": Status.NOT_ASSESSED,
                "tanim": "yuzey-voxel-merkezi"}
    d = min_surface_distance_mm(tumor_mask, tumor_affine, vessel_mask, vessel_affine)
    return {"mesafe_mm": None if d is None else round(d, 1),
            "status": Status.SUCCESS, "tanim": "yuzey-voxel-merkezi"}


# ---------------------------------------------------------------------------
# K4 — lezyon bağlantılılığı sağlamlık kontrolü (tanım farkını görünür kılar)
# ---------------------------------------------------------------------------
def lesion_count_sensitivity(tumor_mask: np.ndarray, spacing: np.ndarray,
                             min_volume_cm3: float | None = None) -> dict[str, int]:
    """Lezyon sayısını 6- ve 26-komşulukla ayrı verir (K4).

    26 vs 6 komşuluk seçimi lezyon sayısını ve 'en büyük bağlı kütle'yi
    değiştirir (pilotta karaciğer düşük korelasyonun tanım-farkı kaynağı).
    Bu fonksiyon seçimi SAKLAMAZ; her ikisini de raporlar.
    """
    min_vol = MEASUREMENT.min_lesion_volume_cm3 if min_volume_cm3 is None else min_volume_cm3
    return {
        "n_6komsu": _count_lesions_above(tumor_mask, spacing, min_vol, 1),
        "n_26komsu": _count_lesions_above(tumor_mask, spacing, min_vol, 3),
        "secilen": MEASUREMENT.connectivity,
    }


def _count_lesions_above(tumor_mask: np.ndarray, spacing: np.ndarray,
                         min_vol: float, conn: int) -> int:
    """Hacim eşiğini geçen bağlı bileşen SAYISI (çap hesabı YOK → hızlı).

    len(extract_lesions(...)) ile birebir aynı sayıyı verir ama pahalı çap/kabuk
    adımını atlar; yalnız etiketleme + bincount yapar.
    """
    from scipy import ndimage
    m = np.asarray(tumor_mask) > 0
    if not m.any():
        return 0
    structure = ndimage.generate_binary_structure(3, conn)
    labeled, n = ndimage.label(m, structure=structure)
    if n == 0:
        return 0
    counts = np.bincount(labeled.ravel(), minlength=n + 1)[1:]   # arka planı at
    vvol = float(np.prod(spacing)) / 1000.0
    return int((counts.astype(float) * vvol >= min_vol).sum())


# ---------------------------------------------------------------------------
# measure_case — BT→ölçüm TEK giriş (K2 oblik kapısı + F03 fiziksel uzay)
# ---------------------------------------------------------------------------
def measure_case(tumor_vol, organ: str, vessel_vols: dict | None = None,
                 tumor_label: int = 2) -> dict[str, Any]:
    """Bir vakanın tüm geometrik ölçümü. tumor_vol/vessel_vols: imaging.Volume.

    - K2: Güçlü oblik hacimde aksiyel çap güvenilmez => diameter not_assessed,
      quality_flags'a 'oblik' eklenir (yanlış sayı üretmek yerine eksik bırak).
    - F03: Damar maskesi farklı grid'de olsa bile mesafe DÜNYA koordinatında;
      sarma açısı için damar tümör grid'ine resample edilir.
    """
    from .imaging import oblique_angle_deg, geometry_close, resample_mask_to_reference, Volume

    spacing = tumor_vol.spacing
    tumor_bin = (np.asarray(tumor_vol.data) == tumor_label).astype(np.uint8)
    quality_flags: list[str] = []

    oblique = oblique_angle_deg(tumor_vol.affine)
    measurable = oblique <= GEOMETRY.max_oblique_deg
    if not measurable:
        quality_flags.append(f"oblik_{oblique:.1f}deg")

    if measurable:
        lesions = extract_lesions(tumor_bin, spacing)
        summary = summarize_lesions(lesions)
    else:
        # Oblik: yanlış aksiyel çap üretme (K2)
        summary = {"lezyon_sayisi": None, "toplam_tumor_hacim_cm3": None,
                   "en_buyuk_cap_lezyon": None, "en_buyuk_hacim_lezyon": None,
                   "status": Status.NOT_ASSESSED}
        lesions = []

    vessels_out: dict[str, Any] = {}
    for vname, vvol in (vessel_vols or {}).items():
        dist = vessel_distance_mm(tumor_bin, tumor_vol.affine,
                                  np.asarray(vvol.data) > 0, vvol.affine)
        # Sarma için aynı grid gerekir
        if geometry_close(tumor_vol, vvol):
            v_same = np.asarray(vvol.data) > 0
        else:
            v_same = resample_mask_to_reference(
                Volume((np.asarray(vvol.data) > 0).astype(np.uint8), vvol.affine),
                tumor_vol) > 0
        wrap = (vessel_wrap_angle_deg(tumor_bin, v_same, spacing)
                if measurable else {"sarma_derece": None, "status": Status.NOT_ASSESSED})
        vessels_out[vname] = {"mesafe_mm": dist["mesafe_mm"],
                              "sarma_derece": wrap["sarma_derece"],
                              "status": dist["status"]}

    return {
        "organ": organ,
        "summary": summary,
        "lesions": lesions,
        "vessels": vessels_out,
        "quality_flags": quality_flags,
        "lezyon_sayisi_saglamlik": (lesion_count_sensitivity(tumor_bin, spacing)
                                    if measurable else None),
        "geometry": {"spacing_mm": [round(float(s), 3) for s in spacing],
                     "oblique_deg": round(oblique, 3), "measurable": measurable},
    }
