"""
manifest.py — tek kaynak vaka listesi (F12, F18).

Eski hata: bazı adımlar sabit liste, bazıları JSON, bazıları `glob` ile tüm
klasörü kullanıyordu (80 iş vs 87 tarama). Artık HER aşama aynı manifestten
çalışır. Yerel POS/NEG pankreas JSON'u ortak şemaya çevrilir; orijinal
korunur. Aynı kişinin iki organı iki BAĞIMSIZ hasta sayılmaz (çakışma denetimi).
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any

from .config import Status


@dataclass
class Case:
    patient_id: str            # kişi (pseudonymized). Bilinmiyorsa study_id ile eşit.
    study_id: str              # inceleme
    task: str                  # Task003 / Task007 ...
    organ: str                 # karaciger / pankreas
    group: str                 # POS / NEG
    role: str = "dev"          # dev / test
    input_path: str = ""       # BT NIfTI yolu
    input_hash: str = ""       # sha256 (varsa)
    reference_source: str = "" # GT kaynağı (radyolog raporu referansı)
    status: str = ""           # boş=beklemede; sonra success/failed/...

    def key(self) -> str:
        return f"{self.study_id}:{self.task}"


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def from_pos_neg_json(path: str | Path, organ: str, task: str,
                      data_dir: str | Path | None = None,
                      role: str = "dev") -> list[Case]:
    """{"POS":[...],"NEG":[...]} biçimini Case listesine çevirir (F12)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases: list[Case] = []
    for group in ("POS", "NEG"):
        for sid in data.get(group, []):
            input_path = ""
            input_hash = ""
            if data_dir is not None:
                p = Path(data_dir) / f"{sid}.nii.gz"
                input_path = str(p)
                if p.exists():
                    input_hash = file_sha256(p)
            cases.append(Case(
                patient_id=sid, study_id=sid, task=task, organ=organ,
                group=group, role=role, input_path=input_path,
                input_hash=input_hash, reference_source=str(path),
            ))
    return cases


def find_patient_collisions(cases: list[Case]) -> dict[str, list[str]]:
    """Aynı patient_id birden çok task/study'de => bağımsızlık ihlali adayı (F12)."""
    by_patient: dict[str, list[str]] = {}
    for c in cases:
        by_patient.setdefault(c.patient_id, []).append(c.key())
    return {pid: keys for pid, keys in by_patient.items() if len(set(keys)) > 1}


def validate(cases: list[Case]) -> dict[str, Any]:
    """Manifest tutarlılık raporu: tekrar key, durum dağılımı, çakışma."""
    keys = [c.key() for c in cases]
    dup_keys = sorted({k for k in keys if keys.count(k) > 1})
    status_counts: dict[str, int] = {}
    for c in cases:
        status_counts[c.status or "beklemede"] = status_counts.get(c.status or "beklemede", 0) + 1
    group_counts: dict[str, int] = {}
    for c in cases:
        gk = f"{c.organ}/{c.group}"
        group_counts[gk] = group_counts.get(gk, 0) + 1
    return {
        "toplam": len(cases),
        "tekrar_key": dup_keys,
        "patient_cakisma": find_patient_collisions(cases),
        "durum": status_counts,
        "grup": group_counts,
    }


def save_csv(cases: list[Case], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [f.name for f in fields(Case)]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for c in cases:
            w.writerow(asdict(c))
    tmp.replace(path)


def load_csv(path: str | Path) -> list[Case]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return [Case(**{k: (v or "") for k, v in row.items()}) for row in csv.DictReader(f)]
