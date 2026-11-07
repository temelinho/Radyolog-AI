"""
extract_diameters.py — POS satırlarında rapordan ÇAP TASLAĞI çıkarır (insan DOĞRULAR).

onay_<organ>.csv'de onay_pos_neg=POS olan ve cap_mm BOŞ satırlar için, aynı
satırdaki Findings metninden en uzun eksen çapını (mm) ve kaynak cümleyi TASLAK
olarak doldurur. 'kontrol' kolonuna işaret koyar — sen göz atıp onaylarsın/düzeltirsin.

Bu bir YARDIMCI; nihai ground truth insan onayıdır. Zaten sen doldurduysan
(cap_mm dolu) o satıra dokunmaz.

Kullanım:
    python scripts/extract_diameters.py liver --onay onay_liver.csv --out onay_liver_taslak.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys

ORGAN = {"liver": r"hepatic|liver", "pancreas": r"pancrea"}
MASS = r"mass|lesion|metastas|tumou?r|carcinoma|hcc|neoplasm|adenocarcinoma|nodule"
# "2.7 x 1.9 cm", "27 mm", "3.5 cm", "2.7 x 1.9 x 3.0 cm"
_MEAS = re.compile(r"(\d+(?:\.\d+)?)((?:\s*[x×]\s*\d+(?:\.\d+)?)*)\s*(mm|cm)", re.IGNORECASE)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<!\d)[.\n](?!\d)", str(text)) if s.strip()]


def _measures_mm(sentence: str) -> list[float]:
    out = []
    for m in _MEAS.finditer(sentence):
        vals = [float(m.group(1))]
        for extra in re.findall(r"\d+(?:\.\d+)?", m.group(2) or ""):
            vals.append(float(extra))
        mult = 10.0 if m.group(3).lower() == "cm" else 1.0
        out.append(max(vals) * mult)   # en uzun eksen
    return out


def extract_diameter(findings: str, organ: str) -> tuple[float | None, str]:
    """(cap_mm, kaynak_cumle). Önce organ+kitle cümlesi, yoksa organ cümlesi."""
    org = ORGAN.get(organ, organ)
    best, best_s = None, ""
    # 1) organ + kitle terimi geçen cümleler (öncelik)
    for want_mass in (True, False):
        for s in _sentences(findings):
            sl = s.lower()
            if not re.search(org, sl):
                continue
            if want_mass and not re.search(MASS, sl):
                continue
            for mm in _measures_mm(sl):
                if best is None or mm > best:
                    best, best_s = mm, s
        if best is not None:
            return round(best, 1), best_s
    return None, ""


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("organ", choices=list(ORGAN))
    ap.add_argument("--onay", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    with open(a.onay, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
        cols = list(rows[0].keys()) if rows else []
    for c in ("cap_mm", "kaynak_cumle", "kontrol"):
        if c not in cols:
            cols.append(c)

    dolduruldu = 0
    for r in rows:
        lab = str(r.get("onay_pos_neg", "")).strip().upper()
        if lab != "POS":
            continue
        if str(r.get("cap_mm", "")).strip():   # zaten sen doldurduysan dokunma
            continue
        cap, sent = extract_diameter(r.get("Findings", ""), a.organ)
        if cap is not None:
            r["cap_mm"] = cap
            if not str(r.get("kaynak_cumle", "")).strip():
                r["kaynak_cumle"] = sent
            r["kontrol"] = "TASLAK-DOGRULA"
            dolduruldu += 1
        else:
            r["kontrol"] = "CAP BULUNAMADI-ELLE"

    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[{a.organ}] {dolduruldu} POS satırına çap taslağı yazıldı -> {a.out}")
    print("  'kontrol' kolonu: TASLAK-DOGRULA (göz at) / CAP BULUNAMADI-ELLE (sen gir)")


if __name__ == "__main__":
    main(sys.argv[1:])
