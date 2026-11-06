"""
imaging.py — görüntü geometrisi, yükleme ve yeniden örnekleme (F03, F06).

Temel ilke: İKİ MASKENİN ARRAY ŞEKLİNİN AYNI OLMASI, FİZİKSEL OLARAK AYNI
UZAYDA OLDUKLARI ANLAMINA GELMEZ. Mesafe/temas hesapları DÜNYA koordinatında
(mm) yapılır; affine (origin + spacing + yön) dikkate alınır.

Eski hata (F03): mevcut kod, farklı origin'li iki maske arasındaki mesafeyi
0 mm veriyordu çünkü sadece voxel indekslerini karşılaştırıyordu.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import GEOMETRY


@dataclass
class Volume:
    """Bir 3B hacim: veri + affine (voxel->dünya mm) + metadata."""
    data: np.ndarray
    affine: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data)
        self.affine = np.asarray(self.affine, dtype=float)
        if self.affine.shape != (4, 4):
            raise ValueError(f"affine 4x4 olmalı, geldi: {self.affine.shape}")
        if self.data.ndim != 3:
            raise ValueError(f"data 3B olmalı, geldi: {self.data.ndim}B")

    @property
    def spacing(self) -> np.ndarray:
        """(sx, sy, sz) mm — affine sütun normlarından."""
        return spacing_from_affine(self.affine)

    @property
    def voxel_volume_mm3(self) -> float:
        return float(np.prod(self.spacing))


def spacing_from_affine(affine: np.ndarray) -> np.ndarray:
    R = np.asarray(affine, dtype=float)[:3, :3]
    return np.linalg.norm(R, axis=0)


def voxel_indices_to_world(indices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """(N,3) voxel indeks -> (N,3) dünya mm. indices (i,j,k) sırasında."""
    indices = np.asarray(indices, dtype=float)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("indices (N,3) olmalı")
    R = affine[:3, :3]
    t = affine[:3, 3]
    return indices @ R.T + t


def oblique_angle_deg(affine: np.ndarray) -> float:
    """Affine'in en büyük eksen sapması (derece). 0 = tam eksen hizalı."""
    R = np.asarray(affine, dtype=float)[:3, :3]
    norms = np.linalg.norm(R, axis=0)
    worst = 0.0
    for c, n in zip(R.T, norms):
        if n == 0:
            continue
        # En yakın ana eksene olan açı (birim vektörde en büyük bileşen = cos)
        cos_to_axis = np.clip(np.max(np.abs(c)) / n, -1.0, 1.0)
        worst = max(worst, np.degrees(np.arccos(cos_to_axis)))
    return float(worst)


def geometry_close(a: Volume, b: Volume, tol_mm: float | None = None) -> bool:
    """İki hacim AYNI fiziksel grid mi? Şekil + affine karşılaştırılır."""
    tol = GEOMETRY.affine_tol_mm if tol_mm is None else tol_mm
    if tuple(a.data.shape) != tuple(b.data.shape):
        return False
    return bool(np.allclose(a.affine, b.affine, atol=tol))


def surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Yüzey voxel indeksleri (N,3): en az bir arka-plan komşusu olan ön-plan."""
    from scipy import ndimage
    mask = np.asarray(mask) > 0
    if not mask.any():
        return np.empty((0, 3), dtype=int)
    eroded = ndimage.binary_erosion(mask, border_value=0)
    surf = mask & ~eroded
    return np.argwhere(surf)


def min_surface_distance_mm(a_mask: np.ndarray, a_affine: np.ndarray,
                            b_mask: np.ndarray, b_affine: np.ndarray) -> float | None:
    """İki maskenin yüzeyleri arasındaki min mesafe (mm), DÜNYA koordinatında.

    Her iki maske kendi affine'i ile dünya koordinatına taşınır; böylece
    farklı grid/origin'li maskeler doğru karşılaştırılır (F03). Maskelerden
    biri boşsa None (ölçülemez — 0 DEĞİL) (F05).
    """
    from scipy.spatial import cKDTree
    sa = surface_voxels(a_mask)
    sb = surface_voxels(b_mask)
    if sa.shape[0] == 0 or sb.shape[0] == 0:
        return None
    wa = voxel_indices_to_world(sa, a_affine)
    wb = voxel_indices_to_world(sb, b_affine)
    tree = cKDTree(wb)
    d, _ = tree.query(wa, k=1)
    return float(d.min())


def resample_mask_to_reference(src: Volume, ref: Volume) -> np.ndarray:
    """src maskesini ref'in grid'ine nearest-neighbor ile taşır (F03).

    Dönüş: ref.data.shape biçiminde maske array'i. Etiket değerleri korunur
    (nearest => ara değer üretilmez).

    K1 düzeltmesi: Eskiden tüm ref grid'i açıkça üretiliyordu (512²×424 ≈ 5 GB).
    Artık `affine_transform` ile doğrudan afin bileşimi kullanılır; büyük
    hacimlerde bellek güvenli. Haritalama: ref_idx -> src_idx lineer olduğundan
    matrix = (inv(src.affine) @ ref.affine)[:3,:3], offset = [:3,3].
    """
    from scipy import ndimage
    combined = np.linalg.inv(src.affine) @ ref.affine   # ref voxel -> src voxel
    matrix = combined[:3, :3]
    offset = combined[:3, 3]
    return ndimage.affine_transform(
        src.data, matrix, offset=offset, output_shape=ref.data.shape,
        order=0, mode="constant", cval=0).astype(src.data.dtype)


def load_nifti(path: str | Path, canonical: bool = True) -> Volume:
    """NIfTI yükle. canonical=True ise en yakın RAS eksen sırasına çevirir.

    `as_closest_canonical` eksen SIRASINI düzeltir; farklı gridleri aynı
    fiziksel uzaya TAŞIMAZ (F03 notu). Oblik görüntü için kalite bayrağı
    meta['oblique_deg'] altında bırakılır.
    """
    import nibabel as nib
    img = nib.load(str(path))
    if canonical:
        img = nib.as_closest_canonical(img)
    data = np.asanyarray(img.dataobj)
    affine = img.affine
    meta = {
        "path": str(path),
        "oblique_deg": oblique_angle_deg(affine),
        "zooms": tuple(float(z) for z in img.header.get_zooms()[:3]),
    }
    return Volume(data=data, affine=affine, meta=meta)
