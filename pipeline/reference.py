"""
reference.py — ground truth (GT) referans tablosu, DENETİM İZİ ile (F18, F03'in GT notu).

Eski hata: Referans boyutlar sürümler arası ciddi değişmiş (ör. 4.3->6.1,
3.5->8.7 cm) ama KAYNAK CÜMLE / TARİH / GEREKÇE kaydı yoktu. Hakem "GT'yi kim,
ne zaman, hangi cümleden ölçtü?" dediğinde cevap yoktu.

Bu modül her referans ölçümünü izlenebilir kılar: kaynak cümle, rapor tarihi,
transkribe eden, revizyon no ve belirsizlik notu. GT = radyolog raporundan
transkripsiyon (bağımsız klinik re-okuma DEĞİL) — bu açıkça saklanır.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any


@dataclass
class ReferenceMeasurement:
    study_id: str
    organ: str
    group: str                      # POS / NEG
    target_diameter_mm: str = ""    # boş = ölçü yok (NEG veya ölçülemez)
    source_sentence: str = ""       # GT'nin alındığı rapor cümlesi (DENETİM İZİ)
    report_date: str = ""           # rapor tarihi
    transcriber: str = ""           # transkribe eden
    revision: str = "0"             # revizyon no (değişiklikler izlenir)
    uncertainty_note: str = ""      # belirsizlik / birden çok lezyon notu
    reference_centroid_ijk: str = ""  # varsa konum (konum-bazlı eşleştirme için)


def validate_references(refs: list[ReferenceMeasurement]) -> dict[str, Any]:
    """Denetim-izi eksiklerini görünür kılar (F18)."""
    missing_source = [r.study_id for r in refs
                      if r.target_diameter_mm.strip() and not r.source_sentence.strip()]
    pos_without_dia = [r.study_id for r in refs
                       if r.group.upper() == "POS" and not r.target_diameter_mm.strip()]
    revised = [r.study_id for r in refs
               if r.revision.strip() not in ("", "0") and not r.uncertainty_note.strip()]
    dup = _duplicates([r.study_id for r in refs])
    return {
        "toplam": len(refs),
        "kaynak_cumlesi_eksik": missing_source,   # ölçü var ama cümle yok -> denetim açığı
        "POS_olcusu_eksik": pos_without_dia,
        "gerekcesiz_revizyon": revised,           # değişmiş ama not yok
        "tekrar_study_id": dup,
    }


def _duplicates(xs: list[str]) -> list[str]:
    return sorted({x for x in xs if xs.count(x) > 1})


def index_by_study(refs: list[ReferenceMeasurement]) -> dict[str, ReferenceMeasurement]:
    return {r.study_id: r for r in refs}


def reference_mm(ref: ReferenceMeasurement) -> float | None:
    s = ref.target_diameter_mm.strip()
    if not s:
        return None
    return float(s.replace(",", "."))


def save_csv(refs: list[ReferenceMeasurement], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [f.name for f in fields(ReferenceMeasurement)]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in refs:
            w.writerow(asdict(r))
    tmp.replace(path)


def load_csv(path: str | Path) -> list[ReferenceMeasurement]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return [ReferenceMeasurement(**{k: (v or "") for k, v in row.items()})
                for row in csv.DictReader(f)]
