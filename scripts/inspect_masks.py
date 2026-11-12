#!/usr/bin/env python3
"""Çap uyumsuzluğu segmentasyondan mı geliyor? Maskeleri OBJEKTİF incele.

nnU-Net'i YENİDEN KOŞMAZ, GPU gerektirmez: Drive'daki kayıtlı maskeleri
(LAYOUT.masks/<study_id>.nii.gz) okuyup her outlier vaka için şunları basar:

  - maskedeki etiket dağılımı (tumor_label=2 doğru mu? olası etiket hatası)
  - toplam tümör hacmi (cm3) ve maske sınır-kutusu boyutu (mm, x/y/z)
  - bağlı-bileşen sayısı farklı hacim eşiklerinde (0.1 .. 5 cm3)
  - en büyük 5 bileşenin çap/hacmi
  - radyolog çapı (cap_uyumu_vaka.csv'den) ile yan yana

Yorum: toplam hacim/boyut DEVASA => aşırı-segmentasyon (organ tümör sayılmış);
TÜİCİK ama radyolog büyük => eksik-segmentasyon; çok bileşen + büyük toplam
=> parçalanma (morfolojik birleştirme yardımcı olabilir).

Kullanım (Colab):
    python scripts/inspect_masks.py            # cap_uyumu_vaka.csv'den en kötü 10
    python scripts/inspect_masks.py AC423e1ef AC4243af8   # belirli study_id'ler
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import imaging, measurements  # noqa: E402
from pipeline.config import LAYOUT  # noqa: E402

TUMOR_LABEL = 2
THRESHOLDS_CM3 = (0.1, 0.5, 1.0, 2.0, 5.0)


def _worst_from_csv(n: int) -> list[tuple[str, float, float]]:
    """cap_uyumu_vaka.csv'den |fark| en büyük 'eslesti' vakaları."""
    p = LAYOUT.outputs / "cap_uyumu_vaka.csv"
    if not p.exists():
        return []
    rows = []
    with open(p, encoding="utf-8-sig") as f:  # cap_uyumu_vaka.csv BOM'lu yazılır
        for row in csv.DictReader(f):
            if row.get("durum") != "eslesti":
                continue
            try:
                fark = abs(float(row["fark_mm"]))
                rad = float(row["radyolog_mm"])
            except (ValueError, KeyError):
                continue
            rows.append((row["study_id"], fark, rad))
    rows.sort(key=lambda t: -t[1])
    return rows[:n]


def _bbox_extent_mm(bin_mask: np.ndarray, spacing: np.ndarray) -> list[float]:
    idx = np.where(bin_mask)
    if len(idx[0]) == 0:
        return [0.0, 0.0, 0.0]
    return [round(float((ax.max() - ax.min() + 1) * sp), 1)
            for ax, sp in zip(idx, spacing)]


def inspect(study_id: str, rad_mm: float | None = None) -> None:
    mask_path = LAYOUT.masks / f"{study_id}.nii.gz"
    print(f"\n==================== {study_id} ====================")
    if rad_mm is not None:
        print(f"  radyolog çapı: {rad_mm} mm")
    if not mask_path.exists():
        print(f"  !! maske yok: {mask_path}")
        return
    vol = imaging.load_nifti(mask_path)
    data = np.asarray(vol.data)
    spacing = vol.spacing
    vox_cm3 = float(np.prod(spacing)) / 1000.0
    print(f"  şekil: {data.shape} · spacing(mm): "
          f"[{spacing[0]:.2f}, {spacing[1]:.2f}, {spacing[2]:.2f}] · "
          f"voxel: {vox_cm3*1000:.3f} mm3")

    uniq, counts = np.unique(data, return_counts=True)
    print("  etiketler (değer: voxel):",
          {int(u): int(c) for u, c in zip(uniq, counts)})

    bin_mask = (data == TUMOR_LABEL).astype(np.uint8)
    n_tumor = int(bin_mask.sum())
    if n_tumor == 0:
        print(f"  !! etiket=={TUMOR_LABEL} hiç voxel yok "
              f"(etiket hatası olabilir: yukarıdaki etiketlere bak)")
        return

    total_cm3 = round(n_tumor * vox_cm3, 2)
    print(f"  toplam tümör: {n_tumor} voxel = {total_cm3} cm3 · "
          f"sınır kutusu (mm) x/y/z: {_bbox_extent_mm(bin_mask, spacing)}")

    lez = measurements.extract_lesions(bin_mask, spacing, min_volume_cm3=0.0)
    print(f"  bileşen sayısı eşiklere göre:", end=" ")
    print(" ".join(f"≥{t}cm3:{sum(1 for l in lez if l['hacim_cm3_raw'] >= t)}"
                    for t in THRESHOLDS_CM3))
    top_vol = sorted(lez, key=lambda l: -l["hacim_cm3_raw"])[:5]
    print("  en büyük 5 bileşen (hacme göre): "
          + ", ".join(f"{l['cap_mm']:.0f}mm/{l['hacim_cm3']:.1f}cm3" for l in top_vol))
    top_dia = sorted(lez, key=lambda l: -l["cap_mm_raw"])[:5]
    print("  en büyük 5 bileşen (çapa göre) : "
          + ", ".join(f"{l['cap_mm']:.0f}mm/{l['hacim_cm3']:.1f}cm3" for l in top_dia))


def main(argv: list[str]) -> int:
    if argv:
        for sid in argv:
            inspect(sid)
    else:
        worst = _worst_from_csv(10)
        if not worst:
            print("cap_uyumu_vaka.csv bulunamadı; study_id'leri argüman olarak ver.")
            return 1
        print(f"En kötü {len(worst)} çap sapması incelenecek "
              f"(RADYOLOG_BASE={LAYOUT.base}):")
        for sid, _fark, rad in worst:
            inspect(sid, rad)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
