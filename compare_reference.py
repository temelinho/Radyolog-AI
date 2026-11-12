"""
compare_reference.py — boru hattı çıktılarının GERÇEK radyolog referansıyla kıyası.

Üç katman (hepsi API'siz, GPU'suz; yalnız Drive'daki dosyaları okur):
  1. TESPİT   : vaka düzeyi tümör var/yok (GT: reference_gt.csv 'group' POS/NEG).
  2. ÇAP UYUMU: POS vakalarda pipeline çapı vs radyolog çapı (MAE, yanlılık, r,
                Bland-Altman) + vaka bazında döküm (eşleşti / kaçırıldı).
  3. RAPOR    : her modelin raporu vs radyolog raporu
                  - lezyon var/yok beyanı (GT'ye göre),
                  - raporda yazan çap vs radyolog çapı,
                  - ROUGE-L (yalnız ilgili ORGAN cümleleri üzerinde; ops. xlsx).

GİZLİLİK: reports_final.xlsx ve study_id'ler RUA kapsamındadır. Bu betik onları
yalnız Drive'dan okur; çıktılar LAYOUT.outputs (Drive) altına yazılır, repoya değil.

Kullanım:
    python compare_reference.py                       # katman 1-2 + rapor beyanı
    python compare_reference.py --reports-xlsx <yol>  # + ROUGE-L (organ cümleleri)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from pipeline import config, evaluation, matching, reference

ORGAN_RE = {
    "liver": r"\b(liver|hepat\w*|intrahepatic)\b",
    "pancreas": r"\bpancrea\w*\b",
}
LESION_RE = r"\b(lesions?|mass(?:es)?|tumou?rs?|nodules?|metasta\w+|neoplasm\w*|carcinoma\w*)\b"
NEG_RE = (r"\bno\b[^.]{0,60}\b(lesions?|mass(?:es)?|tumou?rs?|nodules?|metasta\w+)\b"
          r"|\bwithout\b[^.]{0,40}\b(lesions?|mass(?:es)?|tumou?rs?)\b"
          r"|\b(not|none)\b[^.]{0,30}\b(identified|detected|seen|present)\b")
DIA_RE = r"\b(diameter|measur\w+|largest|size|in size|dimension\w*)\b"
DIST_RE = r"\b(distance|away|from the|separat\w+|contact|abut\w*|encase\w*)\b"


def organ_key(s: str) -> str | None:
    s = (s or "").lower()
    if "liver" in s or "karaci" in s:
        return "liver"
    if "pancrea" in s or "pankrea" in s:
        return "pancreas"
    return None


def load_findings(findings_dir: str) -> dict[str, dict[str, Any]]:
    """study_id -> {ad, organ, findings}. '<ad>.json.meta.json' elenir."""
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(glob.glob(os.path.join(findings_dir, "*.json"))):
        if p.endswith(".meta.json"):
            continue
        ad = os.path.splitext(os.path.basename(p))[0]
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        fnd = data.get("bulgular", data)
        if not isinstance(fnd, dict) or "organ" not in fnd:
            continue
        sid = ad.split("__")[0]
        out[sid] = {"ad": ad, "organ": organ_key(fnd.get("organ", "")) or organ_key(ad),
                    "findings": fnd}
    return out


def pipeline_diameter(fnd: dict[str, Any]) -> float:
    """Pipeline'ın hedef çapı (RAW tercih). Lezyon yoksa 0.0 (= negatif karar)."""
    enb = fnd.get("en_buyuk_cap_lezyon")
    if not isinstance(enb, dict) or not fnd.get("lezyon_sayisi"):
        return 0.0
    v = enb.get("cap_mm_raw", enb.get("cap_mm"))
    return float(v) if v is not None else 0.0


# ---------------------------------------------------------------------------
# Rapor metni yardımcıları
# ---------------------------------------------------------------------------
def organ_sentences(text: str, organ: str) -> list[str]:
    pat = re.compile(ORGAN_RE[organ], re.I)
    return [s for s in evaluation.split_sentences(text or "") if pat.search(s)]


def report_claims(text: str) -> tuple[bool, float | None]:
    """Rapor metninden (lezyon var beyanı, yazılan en büyük çap mm)."""
    positive, best = False, None
    for s in evaluation.split_sentences(text or ""):
        if not re.search(LESION_RE, s, re.I):
            continue
        negated = bool(re.search(NEG_RE, s, re.I))
        lens = [m.value for m in evaluation.extract_measurements(s)
                if m.category == "length"]
        if lens and not negated:
            positive = True
            if re.search(DIA_RE, s, re.I) and not re.search(DIST_RE, s, re.I):
                best = max(lens) if best is None else max(best, max(lens))
        elif not negated and not lens:
            # ölçüsüz ama olumlu lezyon cümlesi ("a hepatic mass is present")
            if re.search(r"\b(is|are|was|were)\b[^.]{0,20}\b(present|seen|identified|noted)\b",
                         s, re.I):
                positive = True
    return positive, best


def rouge_l(cand: str, ref: str) -> float | None:
    """ROUGE-L F1 (LCS, sözcük düzeyi). Bağımlılıksız."""
    a = re.findall(r"[a-z0-9.]+", (cand or "").lower())
    b = re.findall(r"[a-z0-9.]+", (ref or "").lower())
    if not a or not b:
        return None
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return round(2 * p * r / (p + r), 4)


def load_radiologist_text(xlsx: str | None, col_id: str = "study id",
                          col_text: str = "Findings") -> dict[str, str]:
    if not xlsx or not os.path.exists(xlsx):
        return {}
    import pandas as pd
    df = pd.read_excel(xlsx)[[col_id, col_text]]
    return {str(r[col_id]).strip(): str(r[col_text] or "") for _, r in df.iterrows()}


def _fmt_ci(ci) -> str:
    return "-" if not ci else f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--findings", default=str(config.LAYOUT.findings))
    ap.add_argument("--reports", default=str(config.LAYOUT.reports))
    ap.add_argument("--reference", default=str(config.LAYOUT.reference))
    ap.add_argument("--reports-xlsx", default=os.environ.get("RADYOLOG_REPORTS_XLSX"))
    ap.add_argument("--out", default=str(config.LAYOUT.outputs))
    ap.add_argument("--threshold", type=float, default=10.0)
    a = ap.parse_args(argv)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    refs = reference.index_by_study(reference.load_csv(a.reference))
    cases = load_findings(a.findings)
    ortak = sorted(set(refs) & set(cases))
    print(f"referans: {len(refs)} · bulgu: {len(cases)} · ortak: {len(ortak)}")
    eksik = sorted(set(refs) - set(cases))
    if eksik:
        print(f"⚠️  bulgusu olmayan referans vaka: {len(eksik)}")
    denetim = reference.validate_references(list(refs.values()))
    print("referans denetimi:", {k: (len(v) if isinstance(v, list) else v)
                                 for k, v in denetim.items()})

    # ---------------- 1) TESPİT ----------------
    sonuc: dict[str, Any] = {"tespit": {}, "cap_uyumu": {}, "rapor": {}}
    print("\n=== 1) TESPİT (vaka düzeyi) ===")
    for esik_ad, esik in (("herhangi_lezyon", 1e-9), (f">={a.threshold:g}mm", a.threshold)):
        for grup in ("hepsi", "liver", "pancreas"):
            recs = [evaluation.DetectionRecord(
                        gt_positive=refs[s].group.upper() == "POS",
                        pred_diameter_raw=pipeline_diameter(cases[s]["findings"]))
                    for s in ortak if grup == "hepsi" or cases[s]["organ"] == grup]
            m = evaluation.detection_metrics(recs, threshold=esik)
            ppv = m["TP"] / (m["TP"] + m["FP"]) if (m["TP"] + m["FP"]) else None
            npv = m["TN"] / (m["TN"] + m["FN"]) if (m["TN"] + m["FN"]) else None
            m["PPV"], m["NPV"] = ppv, npv
            sonuc["tespit"][f"{esik_ad}|{grup}"] = m
            f2 = lambda v: "-" if v is None else f"{v:.2f}"
            print(f"  [{esik_ad:>15} | {grup:8}] TP={m['TP']} FP={m['FP']} TN={m['TN']} "
                  f"FN={m['FN']} · duy={f2(m['duyarlilik'])} {_fmt_ci(m['duyarlilik_GA'])} "
                  f"· özg={f2(m['ozgulluk'])} {_fmt_ci(m['ozgulluk_GA'])} "
                  f"· PPV={f2(ppv)} NPV={f2(npv)}")

    # ---------------- 2) ÇAP UYUMU ----------------
    print("\n=== 2) ÇAP UYUMU (POS vakalar, hedef = en büyük çap politikası) ===")
    satirlar, ciftler = [], {"hepsi": [], "liver": [], "pancreas": []}
    for s in ortak:
        r = refs[s]
        if r.group.upper() != "POS":
            continue
        ref_mm = reference.reference_mm(r)
        pipe = pipeline_diameter(cases[s]["findings"])
        if ref_mm is None:
            durum = "referans_olcusu_yok"
        elif pipe <= 0:
            durum = "kacirildi"
        else:
            durum = "eslesti"
            ciftler["hepsi"].append((pipe, ref_mm))
            ciftler[cases[s]["organ"]].append((pipe, ref_mm))
        satirlar.append({"study_id": s, "organ": cases[s]["organ"], "durum": durum,
                         "pipeline_mm": round(pipe, 2) if pipe else "",
                         "radyolog_mm": "" if ref_mm is None else ref_mm,
                         "fark_mm": round(pipe - ref_mm, 2) if durum == "eslesti" else "",
                         "lezyon_sayisi": cases[s]["findings"].get("lezyon_sayisi", 0)})
    _write_csv(out / "cap_uyumu_vaka.csv", satirlar)
    for grup, pairs in ciftler.items():
        kacan = sum(1 for x in satirlar if x["durum"] == "kacirildi"
                    and (grup == "hepsi" or x["organ"] == grup))
        m = evaluation.measurement_agreement(pairs, n_skipped=kacan)
        # ek: |fark| <= 10 mm olan alt küme (segmentasyonun "tuttuğu" vakalar)
        yakin = [(p, r) for p, r in pairs if abs(p - r) <= 10.0]
        m["yakin_10mm"] = evaluation.measurement_agreement(yakin)
        if pairs:
            d = [p - r for p, r in pairs]
            mu = sum(d) / len(d)
            sd = (sum((x - mu) ** 2 for x in d) / max(len(d) - 1, 1)) ** 0.5
            m["LoA"] = [round(mu - 1.96 * sd, 2), round(mu + 1.96 * sd, 2)]
        sonuc["cap_uyumu"][grup] = m
        print(f"  [{grup:8}] n={m['n_eslesen']} kaçan={m['n_atlanan']} · MAE={m['MAE']} "
              f"· yanlılık={m['yanlilik']} · r={m['korelasyon']} · LoA={m.get('LoA')}"
              f" · |fark|<=10mm: n={m['yakin_10mm']['n_eslesen']} "
              f"MAE={m['yakin_10mm']['MAE']} r={m['yakin_10mm']['korelasyon']}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pairs = ciftler["hepsi"]
        if pairs:
            ort = [(p + r) / 2 for p, r in pairs]; d = [p - r for p, r in pairs]
            mu = sum(d) / len(d); lo, hi = sonuc["cap_uyumu"]["hepsi"]["LoA"]
            fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
            ax[0].scatter([r for _, r in pairs], [p for p, _ in pairs])
            mx = max(max(p, r) for p, r in pairs)
            ax[0].plot([0, mx], [0, mx], "--", color="gray")
            ax[0].set_xlabel("Radiologist diameter (mm)"); ax[0].set_ylabel("Pipeline diameter (mm)")
            ax[1].scatter(ort, d)
            for y, ls in ((mu, "-"), (lo, "--"), (hi, "--")):
                ax[1].axhline(y, linestyle=ls, color="gray")
            ax[1].set_xlabel("Mean of two (mm)"); ax[1].set_ylabel("Pipeline - radiologist (mm)")
            ax[1].set_title("Bland-Altman")
            fig.tight_layout(); fig.savefig(out / "cap_uyumu.png", dpi=150); plt.close(fig)
            print(f"  grafik: {out / 'cap_uyumu.png'}")
    except ImportError:
        print("  (matplotlib yok, grafik atlandı)")

    # ---------------- 3) RAPOR KIYASI ----------------
    print("\n=== 3) RAPOR KIYASI (model raporu vs radyolog) ===")
    rad_text = load_radiologist_text(a.reports_xlsx)
    if not rad_text:
        print("  (reports_final.xlsx verilmedi/bulunamadı -> ROUGE-L atlanır; "
              "lezyon beyanı ve çap kıyası yine yapılır)")
    rapor_satir: list[dict[str, Any]] = []
    for s in ortak:
        ad, organ = cases[s]["ad"], cases[s]["organ"]
        r = refs[s]
        gt_pos, ref_mm = r.group.upper() == "POS", reference.reference_mm(r)
        ref_org = " . ".join(organ_sentences(rad_text.get(s, ""), organ)) if organ else ""
        for p in sorted(glob.glob(os.path.join(a.reports, f"{ad}__*.txt"))):
            model = os.path.splitext(os.path.basename(p))[0].split("__")[-1]
            with open(p, "r", encoding="utf-8") as f:
                text = f.read()
            pos, mm = report_claims(text)
            rapor_satir.append({
                "study_id": s, "organ": organ, "model": model,
                "gt_pozitif": int(gt_pos), "rapor_pozitif": int(pos),
                "radyolog_mm": "" if ref_mm is None else ref_mm,
                "rapor_mm": "" if mm is None else round(mm, 2),
                "abs_fark_mm": (round(abs(mm - ref_mm), 2)
                                if (mm is not None and ref_mm is not None) else ""),
                "rouge_l": (rouge_l(text, ref_org) if ref_org else ""),
                "radyolog_organ_cumle": len(organ_sentences(rad_text.get(s, ""), organ))
                                        if organ else 0,
            })
    _write_csv(out / "rapor_kiyas_vaka.csv", rapor_satir)

    ozet = []
    for model in sorted({x["model"] for x in rapor_satir}):
        rs = [x for x in rapor_satir if x["model"] == model]
        tp = sum(1 for x in rs if x["gt_pozitif"] and x["rapor_pozitif"])
        fn = sum(1 for x in rs if x["gt_pozitif"] and not x["rapor_pozitif"])
        fp = sum(1 for x in rs if not x["gt_pozitif"] and x["rapor_pozitif"])
        tn = sum(1 for x in rs if not x["gt_pozitif"] and not x["rapor_pozitif"])
        farklar = [x["abs_fark_mm"] for x in rs if x["abs_fark_mm"] != ""]
        rouges = [x["rouge_l"] for x in rs if x["rouge_l"] not in ("", None)]
        ozet.append({
            "model": model, "n": len(rs), "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "duyarlilik": round(tp / (tp + fn), 3) if (tp + fn) else "",
            "ozgulluk": round(tn / (tn + fp), 3) if (tn + fp) else "",
            "dogruluk": round((tp + tn) / len(rs), 3) if rs else "",
            "cap_n": len(farklar),
            "cap_MAE_mm": round(sum(farklar) / len(farklar), 2) if farklar else "",
            "cap_10mm_icinde": sum(1 for d in farklar if d <= 10.0),
            "rouge_l_ort": round(sum(rouges) / len(rouges), 4) if rouges else "",
            "rouge_n": len(rouges),
        })
    _write_csv(out / "rapor_kiyas_ozet.csv", ozet)
    sonuc["rapor"] = ozet
    for o in ozet:
        print(f"  {o['model']:9} n={o['n']:3} · duy={o['duyarlilik']} özg={o['ozgulluk']} "
              f"doğ={o['dogruluk']} · çap MAE={o['cap_MAE_mm']} (n={o['cap_n']}, "
              f"<=10mm: {o['cap_10mm_icinde']}) · ROUGE-L={o['rouge_l_ort']} (n={o['rouge_n']})")

    with open(out / "referans_kiyas.json", "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n📁 Çıktılar: {out}")
    return sonuc


if __name__ == "__main__":
    main(sys.argv[1:])
