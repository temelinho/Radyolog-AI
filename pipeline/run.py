"""
run.py — uçtan uca orkestratör (C33/C34).

Zincir: manifest -> (segmentasyon) -> measure_case -> bulgular.json -> (rapor).
TEK manifestten çalışır (F12); her iş hash'li cache + atomik yazma ile
devam ettirilebilir (F10). Segmentasyon GPU (Colab) gerektirir; "maskeden
bulgular" adımı GPU'suz çalışır ve test edilir.
"""

from __future__ import annotations

import shutil
import tempfile
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
    """Bir vaka için nnU-Net (5-fold) + TotalSegmentator (GPU gerekir).

    #1 düzeltmesi: nnU-Net'e vakanın BULUNDUĞU KLASÖR değil, SADECE bu vakayı
    içeren izole bir dizin verilir ('<study_id>_0000.nii.gz'). Çıktı maskesi
    'masks[0]' ile DEĞİL, beklenen '<study_id>.nii.gz' adıyla seçilir — böylece
    başka vakanın maskesi seçilemez. Damar da AYNI vakanın BT'sinden üretilir.
    """
    task = case.task or SEGMENTATION.organ_task.get(case.organ, "")
    if not case.input_path or not Path(case.input_path).exists():
        return {"status": Status.FAILED, "detail": {"neden": "girdi_yok", "path": case.input_path}}
    work_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir = Path(tempfile.mkdtemp(prefix="attempt-", dir=work_dir))
    in_dir = attempt_dir / "in"
    in_dir.mkdir()
    mask_dir = attempt_dir / "masks"
    vessel_dir = attempt_dir / "vessels"
    staged = in_dir / f"{case.study_id}_0000.nii.gz"   # tek vaka, nnU-Net adlandırması
    shutil.copy(case.input_path, staged)

    try:
        seg = segmentation.run_nnunet(str(in_dir), str(mask_dir), task)
    finally:
        staged.unlink(missing_ok=True)
    if seg.status != Status.SUCCESS:
        return {"status": Status.FAILED, "detail": seg.detail}
    mask_path = mask_dir / f"{case.study_id}.nii.gz"   # beklenen çıktı; masks[0] DEĞİL
    if not segmentation.is_readable_mask(mask_path):
        return {"status": Status.FAILED,
                "detail": {"neden": "beklenen_maske_gecersiz", "beklenen": str(mask_path)}}
    vseg = segmentation.run_totalsegmentator(case.input_path, str(vessel_dir))
    return {"status": Status.SUCCESS, "mask_path": str(mask_path),
            "vessel_dir": str(vessel_dir) if vseg.status == Status.SUCCESS else None,
            "vessel_status": vseg.status}


def process_case(case: man.Case, do_segment: bool = True,
                 work_dir: Path | None = None) -> tuple[dict | None, str]:
    """Bir vakanın bulgularını üretir. do_segment=False ise maskeler hazır varsayılır."""
    if do_segment:
        task = case.task or SEGMENTATION.organ_task.get(case.organ, "")
        work_dir = work_dir or (LAYOUT.base / "work" / f"{case.study_id}__{task}")
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
    """Tüm kohortu işle; her iş cache'li + atomik; biri hata verirse diğerleri sürer.

    #4: Çıktı adı '<study_id>__<task>.json' (aynı tetkik birden çok task'ta ise
    JSON çakışmaz). Cache anahtarı task + segmentasyon modunu içerir.
    """
    cases = man.load_csv(manifest_path or LAYOUT.manifest)
    LAYOUT.findings.mkdir(parents=True, exist_ok=True)
    ok = fail = cached = 0
    for case in cases:
        task = case.task or SEGMENTATION.organ_task.get(case.organ, "")
        key = runner.cache_key(input_hash=case.input_hash or case.study_id,
                               extra={"task": task, "mod": "seg" if do_segment else "premask"})
        out = LAYOUT.findings / f"{case.study_id}__{task}.json"
        if runner.is_complete(out, key):
            cached += 1
            continue
        try:
            findings, status = process_case(case, do_segment)
        except Exception as e:  # tek vaka hatası kalanı düşürmez
            print(f"[FAIL] {case.study_id}: {e!r}"); fail += 1; continue
        if status == Status.SUCCESS and findings is not None:
            runner.atomic_write_json(out, findings, key=key)
            ok += 1
            print(f"[OK] {case.study_id} -> {out.name}")
        else:
            fail += 1
            print(f"[FAIL] {case.study_id}: {status}")
    return {"basarili": ok, "basarisiz": fail, "cache": cached, "toplam": len(cases)}


def _cli(argv: list[str]) -> int:
    """python -m pipeline.run [--manifest PATH] [--no-segment]"""
    import argparse
    p = argparse.ArgumentParser(description="Kohortu işle: BT/maske -> bulgular.json")
    p.add_argument("--manifest", default=None)
    p.add_argument("--no-segment", action="store_true",
                   help="Maskeler hazır varsayılır (segmentasyon çalıştırılmaz)")
    args = p.parse_args(argv)
    rapor = run_all(args.manifest, do_segment=not args.no_segment)
    print(rapor)
    return 0 if rapor["basarisiz"] == 0 else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(_cli(sys.argv[1:]))
