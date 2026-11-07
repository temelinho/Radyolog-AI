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

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import SEGMENTATION, LAYOUT, Status


# ---------------------------------------------------------------------------
# Saf (test edilebilir) yardımcılar
# ---------------------------------------------------------------------------
def _results_env_key() -> str:
    return "RESULTS_FOLDER" if SEGMENTATION.nnunet_version == "v1" else "nnUNet_results"


def _results_root(results_dir: Path | None) -> Path:
    return Path(results_dir or os.environ.get("NNUNET_RESULTS")
                or os.environ.get(_results_env_key()) or LAYOUT.nnunet_results).resolve()


def model_dir(task: str, results_dir: Path | None = None) -> Path:
    root = _results_root(results_dir)
    configuration = SEGMENTATION.configuration
    if SEGMENTATION.nnunet_version == "v1":
        datasets = root / "nnUNet" / configuration
        prefix = "Task"
        model = "nnUNetTrainerV2__nnUNetPlansv2.1"
    else:
        datasets = root
        prefix = "Dataset"
        model = f"nnUNetTrainer__nnUNetPlans__{configuration}"
        if task.startswith("Task") and task[4:7].isdigit():
            task = task[4:7]
    if task.isdigit():
        matches = sorted(p for p in datasets.glob(f"{prefix}{int(task):03d}_*") if p.is_dir())
        if len(matches) != 1:
            raise ValueError(f"Model görevi {task}: {len(matches)} eşleşme ({datasets})")
        task = matches[0].name
    return datasets / task / model


def check_folds_present(task: str, results_dir: Path | None = None,
                        folds: tuple[int, ...] | None = None) -> dict[str, Any]:
    """Çıkarımın kullanacağı tek model dizinindeki checkpoint'leri doğrula."""
    folds = SEGMENTATION.folds if folds is None else folds
    try:
        md = model_dir(task, results_dir)
    except ValueError as exc:
        return {"ok": False, "eksik_foldlar": list(folds), "model_hatasi": str(exc)}

    def has_checkpoint(fold: int) -> bool:
        folder = md / f"fold_{fold}"
        if SEGMENTATION.nnunet_version == "v1":
            return all((folder / f"model_final_checkpoint.{ext}").is_file()
                       for ext in ("model", "model.pkl"))
        return any((folder / f"checkpoint_final.{ext}").is_file()
                   for ext in ("pth", "safetensors"))

    missing = [fold for fold in folds if not has_checkpoint(fold)]
    return {"ok": not missing, "eksik_foldlar": missing, "beklenen": list(folds),
            "model_dir": str(md), "task": md.parent.name}


def is_readable_mask(path: str | Path) -> bool:
    from nibabel import load
    from nibabel.filebasedimages import ImageFileError

    try:
        img = load(str(path))
        return len(img.shape) == 3 and all(img.shape)
    except (ImageFileError, OSError, ValueError):
        return False


def build_nnunet_command(input_dir: str, output_dir: str, task: str,
                         cfg=None) -> list[str]:
    """nnU-Net çıkarım komutunu KURAR (5-fold). Tek fold bayrağı YOK (C8).

    v1: nnUNet_predict -t <task> -m <cfg>   ·   v2: nnUNetv2_predict -d <id> -c <cfg>
    (Zenodo MSD ağırlıkları v1; task adı 'Task003_Liver' gibi. v2 için dataset
    kimliği gerekir — NNUNET_VERSION=v2 ve task yerine Dataset id verilmeli.)
    """
    cfg = SEGMENTATION if cfg is None else cfg
    folds = [str(f) for f in cfg.folds]
    if cfg.nnunet_version == "v1":
        cmd = [cfg.nnunet_cmd, "-i", str(input_dir), "-o", str(output_dir),
               "-t", task, "-m", cfg.configuration, "-f", *folds,
               "--overwrite_existing"]
    else:
        cmd = [cfg.nnunet_cmd, "-i", str(input_dir), "-o", str(output_dir),
               "-d", task, "-c", cfg.configuration, "-f", *folds]
    if not cfg.tta:
        cmd.append("--disable_tta")
    return cmd


def build_totalseg_command(input_path: str, output_dir: str,
                           cfg=SEGMENTATION) -> list[str]:
    return [cfg.totalseg_cmd, "-i", str(input_path), "-o", str(output_dir),
            "--roi_subset", *cfg.totalseg_roi]


def validate_nnunet_output(output_dir: str, min_bytes: int = 1) -> dict[str, Any]:
    """Çıktıda BOYUTU>0 en az bir .nii.gz var mı? 0-byte dosya geçersiz (C10/#4)."""
    out = Path(output_dir)
    masks = (sorted(str(p) for p in out.glob("*.nii.gz")
                    if p.stat().st_size >= min_bytes) if out.exists() else [])
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
    root = _results_root(results_dir)
    folds = check_folds_present(task, root)
    if not folds["ok"]:
        return SegResult(Status.FAILED, None,
                         {"neden": "eksik_fold", **folds})
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cmd = build_nnunet_command(input_dir, output_dir, folds["task"])
    env = os.environ.copy()
    env[_results_env_key()] = str(root)
    raw_key = "nnUNet_raw_data_base" if SEGMENTATION.nnunet_version == "v1" else "nnUNet_raw"
    env.setdefault(raw_key, env.get("NNUNET_RAW") or str(LAYOUT.base / "nnUNet_raw"))
    env.setdefault("nnUNet_preprocessed", env.get("NNUNET_PREPROCESSED")
                   or str(LAYOUT.base / "nnUNet_preprocessed"))
    proc = _runner(cmd, capture_output=True, text=True, env=env)
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
