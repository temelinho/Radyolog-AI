"""
matching.py — lezyon eşleştirme (F18).

Eski hata: Ölçüm uyumu (C ekseni) pipeline'ın EN BÜYÜK lezyonunu radyoloğun
HEDEF lezyonuyla kıyaslıyordu; bunlar farklı lezyon olabilir (pilotta karaciğer
korelasyon 0.19'un gerçek sebebi). Düzeltme: eşleştirme AÇIK bir politikayla
yapılır; eşleşmeyen/atlanan vakalar AYRI sayılır — payda gizlenmez.

Önemli dürüstlük notu: Merlin raporlarında çoğu zaman yalnız tek hedef lezyonun
ÇAPI vardır, KOORDİNATI yoktur. Koordinat yoksa konum-bazlı eşleştirme
yapılamaz; o durumda açık bir "hedef seçim politikası" (ör. en büyük çap)
kullanılır ve bu, garantili aynı-lezyon eşleşmesi DEĞİL, beyan edilen bir
politikadır. Koordinat varsa `match_by_centroid` tercih edilir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TARGET_POLICIES = ("largest_diameter", "largest_volume")


def select_target(lesions: list[dict[str, Any]],
                  policy: str = "largest_diameter") -> dict[str, Any] | None:
    """Pipeline lezyonları arasından kıyaslanacak hedefi AÇIK politikayla seç."""
    if not lesions:
        return None
    if policy == "largest_diameter":
        return max(lesions, key=lambda d: (d.get("cap_mm_raw", d.get("cap_mm", 0)), d["id"]))
    if policy == "largest_volume":
        return max(lesions, key=lambda d: (d.get("hacim_cm3_raw", d.get("hacim_cm3", 0)), d["id"]))
    raise ValueError(f"bilinmeyen politika: {policy}")


def match_by_centroid(lesions: list[dict[str, Any]], ref_centroid_ijk,
                      max_dist_voxel: float = 20.0) -> dict[str, Any] | None:
    """Referans lezyon KOORDİNATI varsa en yakın pipeline lezyonunu eşle."""
    import math
    best, best_d = None, float("inf")
    for l in lesions:
        c = l.get("centroid_ijk")
        if c is None:
            continue
        d = math.dist(c, ref_centroid_ijk)
        if d < best_d:
            best, best_d = l, d
    if best is None or best_d > max_dist_voxel:
        return None
    return best


@dataclass
class PairingResult:
    pairs: list[tuple[float, float]] = field(default_factory=list)  # (pipeline, referans)
    matched_ids: list[str] = field(default_factory=list)
    skipped_no_pipeline: list[str] = field(default_factory=list)    # pipeline hedef yok
    skipped_no_reference: list[str] = field(default_factory=list)   # referans yok
    policy: str = "largest_diameter"

    def summary(self) -> dict[str, Any]:
        return {
            "n_eslesen": len(self.pairs),
            "n_pipeline_yok": len(self.skipped_no_pipeline),
            "n_referans_yok": len(self.skipped_no_reference),
            "politika": self.policy,
        }


def build_pairs(records: list[dict[str, Any]],
                policy: str = "largest_diameter") -> PairingResult:
    """Vaka kayıtlarından (pipeline, referans) çap çiftleri üret (F18).

    Her record: {study_id, lesions:[...], reference_mm: float|None,
                 reference_centroid_ijk: [...]|None}
    Eşleşmeyenler PairingResult içinde AYRI listelenir; paydaya karışmaz.
    """
    res = PairingResult(policy=policy)
    for rec in records:
        sid = rec.get("study_id", "?")
        ref = rec.get("reference_mm")
        lesions = rec.get("lesions") or []
        if ref is None:
            res.skipped_no_reference.append(sid)
            continue
        ref_c = rec.get("reference_centroid_ijk")
        target = (match_by_centroid(lesions, ref_c) if ref_c is not None
                  else select_target(lesions, policy))
        if target is None:
            res.skipped_no_pipeline.append(sid)
            continue
        pipe_mm = target.get("cap_mm_raw", target.get("cap_mm"))
        res.pairs.append((float(pipe_mm), float(ref)))
        res.matched_ids.append(sid)
    return res
