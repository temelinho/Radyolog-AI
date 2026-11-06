"""
runner.py — atomik, devam ettirilebilir toplu işlem (F10).

Eski hata: dosyanın VARLIĞI "tamamlandı" sayılıyordu; kod/prompt/model/girdi
değişse bile atlanıyordu. Başarısız API çağrısı boş dosya bırakabiliyordu.
Düzeltme:
  * Cache anahtarı = girdi hash + ağırlık hash + kod/şema/prompt sürümü + config.
  * Önce .tmp'ye yaz, doğrula, sonra atomik rename. Yanındaki .meta.json
    cache anahtarını tutar; anahtar uyuşmazsa YENİDEN hesaplanır.
  * Başarısız denemeler ayrı; boş/kesik çıktı başarı sayılmaz.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config
from .config import Status


def cache_key(*, input_hash: str, weight_hash: str = "",
              config_fingerprint: str = "", extra: dict[str, Any] | None = None) -> str:
    """Girdi + model + sürümlerden deterministik cache anahtarı (F10)."""
    payload = {
        "input_hash": input_hash,
        "weight_hash": weight_hash,
        "schema_version": config.SCHEMA_VERSION,
        "pipeline_version": config.PIPELINE_VERSION,
        "prompt_version": config.PROMPT_VERSION,
        "config_fingerprint": config_fingerprint,
        "extra": extra or {},
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def atomic_write_json(path: str | Path, obj: Any, key: str | None = None) -> None:
    """Önce .tmp'ye yaz, doğrula, atomik rename. key verilirse .meta.json yaz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(obj, ensure_ascii=False, indent=2)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    json.loads(data)  # yazılanın geçerli JSON olduğunu doğrula
    tmp.replace(path)
    if key is not None:
        meta = path.with_suffix(path.suffix + ".meta.json")
        with open(meta, "w", encoding="utf-8") as f:
            json.dump({"cache_key": key}, f)


def is_complete(path: str | Path, key: str) -> bool:
    """Dosya VAR ve .meta.json cache anahtarı EŞLEŞİYOR mu? (F10)"""
    path = Path(path)
    meta = path.with_suffix(path.suffix + ".meta.json")
    if not path.exists() or not meta.exists():
        return False
    if path.stat().st_size == 0:
        return False
    try:
        with open(meta, "r", encoding="utf-8") as f:
            return json.load(f).get("cache_key") == key
    except Exception:
        return False


@dataclass
class JobResult:
    job_id: str
    status: str                       # config.Status.*
    output_path: str | None = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in Status.ALL:
            raise ValueError(f"geçersiz status: {self.status}")


def run_jobs(jobs, process, *, resume: bool = True) -> list[JobResult]:
    """Her iş bağımsız; bir işin hatası diğerlerini DÜŞÜRMEZ (F19).

    jobs: iterable of (job_id, output_path, cache_key_str)
    process: fn(job_id) -> (obj_to_write, result_status, meta)
    """
    results: list[JobResult] = []
    for job_id, out_path, key in jobs:
        if resume and is_complete(out_path, key):
            results.append(JobResult(job_id, Status.SUCCESS, str(out_path),
                                     meta={"cached": True}))
            continue
        try:
            obj, status, meta = process(job_id)
            wrote = False
            if status == Status.SUCCESS and obj is not None:
                atomic_write_json(out_path, obj, key=key)
                wrote = True
            results.append(JobResult(job_id, status, str(out_path) if wrote else None,
                                     meta=meta or {}))
        except Exception as e:  # tek iş hatası kalanları düşürmez
            results.append(JobResult(job_id, Status.FAILED, None, error=repr(e)))
    return results
