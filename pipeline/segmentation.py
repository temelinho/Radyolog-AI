"""
segmentation.py — nnU-Net + TotalSegmentator adaptörü (C8–C13, F10, F11, F20).

Eski hatalar:
  * C8/F20: app `-f 0` (tek fold), main.py 5-fold → iki farklı protokol.
  * C9/F11: sabit `C:\\Users\\ASUS\\...` yolları.
  * C10/F10: başarı = sabit `hasta_001.nii.gz` var + boyut>0; çıkış kodu yok sayılıyor.
  * C11/F05: TotalSegmentator dönüş değeri kullanılmıyor; damar seg başarısızsa sessiz.
  * C12: main.py kurulu nnU-Net dosyasını import'ta yamалıyor.

Düzeltme: TEK protokol (config'ten 5-fold), 5 fold'un VARLIK kontrolü (yoksa
sessizce tek-fold'a düşmez → FAILED), çıkış kodu + çıktı doğrulaması, ve
TotalSegmentator dönüşünün AÇIKÇA kullanılması. Yollar config'ten (Colab/Drive).

Not: Gerçek çıkarım GPU'da (Colab) çalışır; bu modül komutu KURAR ve SONUCU
DOĞRULAR. Komut kurma / fold kontrolü / çıktı doğrulaması binary olmadan test
edilebilir (alt fonksiyonlar saf).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import SEGMENTATION, LAYOUT, Status


# ---------------------------------------------------------------------------
# Saf (test edilebilir) yardımcılar
# ---------------------------------------------------------------------------
def model_dir(task: str, results_dir: Path | None = None) -> Path:
    base = Path(results_dir) if results_dir else LAYOUT.nnunet_results
    return base / task


def check_folds_present(task: str, results_dir: Path | None = None,
                        folds: tuple[int, ...] | None = None) -> dict[str, Any]:
    """5 fold ağırlığının VARLIĞINI doğrular (C8). Eksikse sessizce düşmez."""
    folds = SEGMENTATION.folds if folds is None else folds
    md = model_dir(task, results_dir)
    missing = []
    for f in folds:
        fdir = md / f"fold_{f}"
        has_ckpt = fdir.is_dir() and any(fdir.glob("*checkpoint*"))
        if not has_ckpt:
            missing.append(f)
    return {"ok": len(missing) == 0, "eksik_foldlar": missing,
            "beklenen": list(folds), "model_dir": str(md)}


def build_nnunet_command(input_dir: str, output_dir: str, task: str,
                         cfg=SEGMENTATION) -> list[str]:
    """nnU-Net çıkarım komutunu KURAR (5-fold). Tek fold bayrağı YOK (C8)."""
    cmd = [cfg.nnunet_cmd, "-i", str(input_dir), "-o", str(output_dir),
           "-d", task, "-c", cfg.configuration,
           "-f", *[str(f) for f in cfg.folds]]
    if not cfg.tta:
        cmd.append("--disable_tta")
    return cmd


def build_totalseg_command(input_path: str, output_dir: str,
                           cfg=SEGMENTATION) -> list[str]:
    return [cfg.totalseg_cmd, "-i", str(input_path), "-o", str(output_dir),
            "--roi_subset", *cfg.totalseg_roi]


def validate_nnunet_output(output_dir: str) -> dict[str, Any]:
    """Çıktı klasöründe en az bir .nii.gz üretilmiş mi? (sabit ad YOK) (C10)."""
    out = Path(output_dir)
    masks = sorted(str(p) for p in out.glob("*.nii.gz")) if out.exists() else []
    return {"ok": len(masks) > 0, "maskeler": masks}


# ---------------------------------------------------------------------------
# Çalıştırıcılar (GPU gerektirir) — durum AÇIKÇA döner (F05, F10)
# ---------------------------------------------------------------------------
@dataclass
class SegResult:
    status: str                       # Status.*
    output_dir: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


def run_nnunet(input_dir: str, output_dir: str, task: str,
               results_dir: Path | None = None, _runner=subprocess.run) -> SegResult:
    """5-fold nnU-Net çıkarımı. Fold eksikse veya çıkış kodu≠0 ise FAILED (C8/C10)."""
    folds = check_folds_present(task, results_dir)
    if not folds["ok"]:
        return SegResult(Status.FAILED, None,
                         {"neden": "eksik_fold", **folds})
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cmd = build_nnunet_command(input_dir, output_dir, task)
    proc = _runner(cmd, capture_output=True, text=True)
    if getattr(proc, "returncode", 1) != 0:
        return SegResult(Status.FAILED, None,
                         {"neden": "cikis_kodu", "returncode": proc.returncode,
                          "stderr": (proc.stderr or "")[-500:]})
    val = validate_nnunet_output(output_dir)
    if not val["ok"]:
        return SegResult(Status.FAILED, None, {"neden": "cikti_yok", **val})
    return SegResult(Status.SUCCESS, output_dir, {"maskeler": val["maskeler"]})


def run_totalsegmentator(input_path: str, output_dir: str,
                         _runner=subprocess.run) -> SegResult:
    """Damar segmentasyonu. Dönüş değeri AÇIKÇA kullanılır (C11)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cmd = build_totalseg_command(input_path, output_dir)
    proc = _runner(cmd, capture_output=True, text=True)
    if getattr(proc, "returncode", 1) != 0:
        return SegResult(Status.FAILED, None,
                         {"neden": "cikis_kodu", "returncode": proc.returncode,
                          "stderr": (proc.stderr or "")[-500:]})
    produced = [r for r in SEGMENTATION.totalseg_roi
                if (Path(output_dir) / f"{r}.nii.gz").exists()]
    if not produced:
        return SegResult(Status.EMPTY_MASK, output_dir,
                         {"neden": "roi_uretilmedi", "beklenen": list(SEGMENTATION.totalseg_roi)})
    return SegResult(Status.SUCCESS, output_dir, {"uretilen_roi": produced})
