"""
finalize_cohort.py — insan onayından NİHAİ 20+20 kohortu KİLİTLER (tekrarlanabilir).

Zincir: select_cohort (aday) -> [insan onay_<organ>.csv doldurur] -> finalize (bu).
Deterministik: onaylı satırları draw_order'a göre sıralar, her sınıftan ilk `target`
GEÇERLİ onaylıyı alır. Aynı onaylı CSV = aynı kohort.

Çıktı:
  <organ>_cases.json  : {"POS":[study_id...], "NEG":[study_id...]}  (build_cohort.py'ye girer)
  reference_gt.csv    : GT denetim izi (study_id, organ, group, çap, kaynak cümle)

onay_<organ>.csv beklenen kolonlar: draw_order, study id, onay_pos_neg (POS/NEG),
cap_mm (POS için), kaynak_cumle.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def finalize_from_rows(rows: list[dict], organ: str, target: int = 20):
    """Onaylı satırlardan kohort + GT üret. rows: onay CSV satırları (dict).

    Döner: (cases {POS,NEG}, gt_rows list, uyarilar list).
    """
    def okey(r):
        try:
            return int(r.get("draw_order", 0))
        except (TypeError, ValueError):
            return 0
    rows = sorted(rows, key=okey)
    pos, neg, gt, uyari = [], [], [], []
    for r in rows:
        lab = str(r.get("onay_pos_neg", "")).strip().upper()
        sid = str(r.get("study id") or r.get("study_id") or "").strip()
        if not sid or lab not in ("POS", "NEG"):
            continue
        if lab == "POS" and len(pos) < target:
            pos.append(sid)
            cap = str(r.get("cap_mm", "")).strip()
            if not cap:
                uyari.append(f"{sid}: POS ama cap_mm boş")
            gt.append({"study_id": sid, "organ": organ, "group": "POS",
                       "target_diameter_mm": cap,
                       "source_sentence": str(r.get("kaynak_cumle", "")).strip(),
                       "transcriber": str(r.get("transcriber", "")).strip(), "revision": "0"})
        elif lab == "NEG" and len(neg) < target:
            neg.append(sid)
            gt.append({"study_id": sid, "organ": organ, "group": "NEG",
                       "target_diameter_mm": "", "source_sentence": "",
                       "transcriber": str(r.get("transcriber", "")).strip(), "revision": "0"})
    if len(pos) < target:
        uyari.append(f"POS {len(pos)}/{target} (havuz yetmedi — belgele)")
    if len(neg) < target:
        uyari.append(f"NEG {len(neg)}/{target} (havuz yetmedi — belgele)")
    return {"POS": pos, "NEG": neg}, gt, uyari


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("organ")
    ap.add_argument("--onay", required=True, help="doldurulmuş onay_<organ>.csv")
    ap.add_argument("--out", default=".")
    ap.add_argument("--target", type=int, default=20)
    a = ap.parse_args(argv)

    with open(a.onay, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    cases, gt, uyari = finalize_from_rows(rows, a.organ, a.target)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.organ}_cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")

    gt_path = out / "reference_gt.csv"
    cols = ["study_id", "organ", "group", "target_diameter_mm", "source_sentence",
            "transcriber", "revision"]
    exists = gt_path.exists()
    with open(gt_path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not exists:
            w.writeheader()
        w.writerows(gt)

    print(f"[{a.organ}] KİLİT: POS={len(cases['POS'])} NEG={len(cases['NEG'])}")
    for u in uyari:
        print("  [UYARI]", u)
    print(f"  -> {a.organ}_cases.json · reference_gt.csv (eklendi)")


if __name__ == "__main__":
    main(sys.argv[1:])
