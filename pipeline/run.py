"""
run.py — uçtan uca orkestratör (C33/C34).

Zincir: manifest -> (segmentasyon) -> measure_case -> bulgular.json -> (rapor).
TEK manifestten çalışır (F12); her iş hash'li cache + atomik yazma ile
devam ettirilebilir (F10). Segmentasyon GPU (Colab) gerektirir; "maskeden
bulgular" adımı GPU'suz çalışır ve test edilir.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import imaging, measurements, reporting, segmentation, runner
from . import manifest as man
from .config import LAYOUT, SEGMENTATION, Status

VESSEL_FILES = {
    "aorta": "aorta.nii.gz",
    "portal_ven": "portal_vein_and_splenic_vein.nii.gz",
    "vci": "inferior_vena_cava.nii.gz",
}


def findings_for_masks(tumor_mask_path: str, organ: str,
                       vessel_dir: str | None = None) -> dict[str, Any]:
    """Maskelerden schema v2 bulgular.json (GPU gerektirmez) — test edilebilir çekirdek."""
    tvol = imaging.load_nifti(tumor_mask_path)
    vessel_vols = {}
    if vessel_dir:
        for vname, fn in VESSEL_FILES.items():
            p = Path(vessel_dir) / fn
            if p.exists():
                vessel_vols[vname] = imaging.load_nifti(str(p))
    res = measurements.measure_case(tvol, organ, vessel_vols,
                                    tumor_label=SEGMENTATION.tumor_label)
    findings = reporting.build_findings(
        organ=organ, summary=res["summary"], vessels=res["vessels"],
        provenance={"mask": str(tumor_mask_path),
                    "quality_flags": res["quality_flags"],
                    "geometry": res["geometry"]})
    return findings


def segment_case(case: man.Case, work_dir: Path) -> dict[str, Any]:
    """Bir vaka için nnU-Net (5-fold) + TotalSegmentator (GPU gerekir)."""
    task = SEGMENTATION.organ_task.get(case.organ, case.task)
    in_dir = Path(case.input_path).parent if case.input_path else LAYOUT.data
    mask_dir = work_dir / "masks"
    vessel_dir = work_dir / "vessels"
    seg = segmentation.run_nnunet(str(in_dir), str(mask_dir), task)
    if seg.status != Status.SUCCESS:
        return {"status": Status.FAILED, "detail": seg.detail}
    masks = seg.detail.get("maskeler") or []
    vseg = segmentation.run_totalsegmentator(case.input_path, str(vessel_dir))
    return {"status": Status.SUCCESS, "mask_path": masks[0] if masks else None,
            "vessel_dir": str(vessel_dir) if vseg.status == Status.SUCCESS else None,
            "vessel_status": vseg.status}


def process_case(case: man.Case, do_segment: bool = True,
                 work_dir: Path | None = None) -> tuple[dict | None, str]:
    """Bir vakanın bulgularını üretir. do_segment=False ise maskeler hazır varsayılır."""
    work_dir = work_dir or (LAYOUT.base / "work" / case.study_id)
    if do_segment:
        seg = segment_case(case, Path(work_dir))
        if seg["status"] != Status.SUCCESS:
            return None, Status.FAILED
        mask_path, vessel_dir = seg["mask_path"], seg["vessel_dir"]
    else:
        mask_path = str(LAYOUT.masks / f"{case.study_id}.nii.gz")
        vessel_dir = str(LAYOUT.vessels / case.study_id)
    if not mask_path or not Path(mask_path).exists():
        return None, Status.FAILED
    return findings_for_masks(mask_path, case.organ, vessel_dir), Status.SUCCESS


def run_all(manifest_path: str | None = None, do_segment: bool = True) -> dict[str, Any]:
    """Tüm kohortu işle; her iş cache'li + atomik; biri hata verirse diğerleri sürer."""
    cases = man.load_csv(manifest_path or LAYOUT.manifest)
    LAYOUT.findings.mkdir(parents=True, exist_ok=True)
    ok = fail = cached = 0
    for case in cases:
        key = runner.cache_key(input_hash=case.input_hash or case.study_id)
        out = LAYOUT.findings / f"{case.study_id}.json"
        if runner.is_complete(out, key):
            cached += 1
            continue
        try:
            findings, status = process_case(case, do_segment)
        except Exception as e:  # tek vaka hatası kalanı düşürmez
            print(f"❌ {case.study_id}: {e!r}"); fail += 1; continue
        if status == Status.SUCCESS and findings is not None:
            runner.atomic_write_json(out, findings, key=key)
            ok += 1
            print(f"✅ {case.study_id} -> {out.name}")
        else:
            fail += 1
            print(f"❌ {case.study_id}: {status}")
    return {"basarili": ok, "basarisiz": fail, "cache": cached, "toplam": len(cases)}
