"""
clinical_rules.py — organ/hastalık özel, sürümlü klinik kurallar (F07).

Eski hata: Pankreas boyut eşikleri karaciğere de uygulanıyordu; her damarda
>180° sarma "irresektabl" sayılıyordu; eksik girdide bile kesin kategori
isteniyordu. Düzeltme:
  * Boyut kategorisi AÇIKÇA "boyut" olarak adlandırılır; tam klinik evre (T)
    DEĞİLDİR (histoloji/diğer girdiler gerekir).
  * Organ bazlı eşikler; karaciğere pankreas eşiği uygulanmaz.
  * Eksik girdide "değerlendirilemez".
  * Damar sarması araştırma ölçümüdür; tek başına rezektabilite KARARI vermez.

Not: Bizim kurallarımızdan türeyen "derived_truth" klinik referans DEĞİLDİR;
yalnızca kural tutarlılığı referansıdır (F07).
"""

from __future__ import annotations

from typing import Any

RULES_VERSION = "2.0.0"

# Boyut kategorisi eşikleri (mm) — YALNIZ boyut; klinik T-evresi değil.
_SIZE_THRESHOLDS = {
    "pankreas": [(20.0, "≤2 cm"), (40.0, "2–4 cm"), (float("inf"), ">4 cm")],
    "karaciger": [(20.0, "≤2 cm"), (50.0, "2–5 cm"), (float("inf"), ">5 cm")],
}

_STAGE_NOTE = {
    "pankreas": "Boyut kategorisidir; AJCC T-evresi için histopatoloji ve "
                "ek bulgular gerekir.",
    "karaciger": "Boyut kategorisidir; karaciğer T-sınıflaması yalnız boyuta "
                 "dayanmaz (lezyon sayısı, vasküler invazyon vb.).",
}


def size_category(diameter_mm: float | None, organ: str) -> dict[str, Any]:
    """Aksiyel çaptan BOYUT kategorisi (klinik evre DEĞİL) (F07)."""
    organ = organ.lower()
    if organ not in _SIZE_THRESHOLDS:
        return {"kategori": None, "durum": "değerlendirilemez",
                "neden": f"'{organ}' için tanımlı boyut kuralı yok"}
    if diameter_mm is None:
        return {"kategori": None, "durum": "değerlendirilemez",
                "neden": "çap ölçülemedi"}
    for limit, label in _SIZE_THRESHOLDS[organ]:
        if diameter_mm <= limit:
            return {"kategori": label, "taban": "yalnız boyut",
                    "uyari": _STAGE_NOTE[organ], "durum": "ok"}
    return {"kategori": None, "durum": "değerlendirilemez", "neden": "sınıflanamadı"}


def vascular_assessment(vessels: dict[str, Any], organ: str) -> dict[str, Any]:
    """Damar temas/sarma özeti. KESİN rezektabilite kararı vermez (F07, F04).

    Sarma açısı araştırma ölçümüdür; >180° tek başına "irresektabl" demek
    değildir (arter/ven ayrımı ve klinik bağlam gerekir).
    """
    bulgular = []
    for vname, vinfo in (vessels or {}).items():
        if not isinstance(vinfo, dict):
            continue
        derece = vinfo.get("sarma_derece")
        mesafe = vinfo.get("mesafe_mm")
        if derece is None and mesafe is None:
            bulgular.append({"damar": vname, "durum": "değerlendirilemez"})
            continue
        temas = (mesafe is not None and mesafe <= 2.0)
        not_ = None
        if derece is not None and derece > 180:
            not_ = (">180° temas saptandı (araştırma ölçümü); rezektabilite "
                    "kararı için arter/ven ayrımı ve uzman değerlendirmesi gerekir.")
        bulgular.append({
            "damar": vname, "mesafe_mm": mesafe, "sarma_derece": derece,
            "temas": temas, "not": not_, "research_grade": True,
        })
    return {
        "damarlar": bulgular,
        "uyari": "Vasküler sarma araştırma ölçümüdür; kesin cerrahi karar "
                 "(NCCN rezektabilite) uzman + tam klinik veri gerektirir.",
        "rules_version": RULES_VERSION,
    }
