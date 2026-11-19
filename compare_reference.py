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
NEG_RE = (r"\bno\b[^.]{0,60}\b(lesions?|mass(?:es)?|tumou?rs?|nodules?|metasta\w+|neoplas\w+)\b"
          r"|\bwithout\b[^.]{0,40}\b(lesions?|mass(?:es)?|tumou?rs?)\b"
          r"|\b(not|none)\b[^.]{0,30}\b(identified|detected|seen|present)\b"
          # sıfır sayısı: "0 hepatic lesions", "number of lesions: 0", "zero lesions"
          r"|\b(0|zero)\s+(?:\w+\s+){0,2}(lesions?|mass(?:es)?|tumou?rs?|nodules?)\b"
          r"|\b(lesions?|nodules?|tumou?rs?)\b[^.\d]{0,25}:\s*0\b"
          r"|\babsen(ce|t)\b[^.]{0,40}\b(lesions?|mass(?:es)?|tumou?rs?)\b")
DIA_RE = r"\b(diameter|measur\w+|largest|size|in size|dimension\w*)\b"
DIST_RE = r"\b(distance|away|from the|separat\w+|contact|abut\w*|encase\w*)\b"
# Damar cümleleri ("distance of the VCI to the tumor is 47 mm") lezyon beyanı DEĞİL.
# Varsayımsal/koşullu cümleler beyan değildir ("If a lesion is later identified ...",
# "does not exclude a subtle lesion", "lesions that may not have been segmented").
HEDGE_RE = (r"\b(if|whether|unless|should|may|might|could|would|cannot|can't|"
            r"does not exclude|do not exclude|not exclude|not replace|later)\b")
VESSEL_RE = (r"\b(aort\w*|vein|venous|vena|vci|ivc|smv|sma|artery|arter\w*|celiac|"
             r"portal|mesenteric|vascular|vessels?)\b")
# Rapor bir organ hakkında mı? Yanlış organ adı (ör. 'karaciger'i böbrek sanma) kontrolü.
OTHER_ORGAN_RE = r"\b(kidneys?|renal|spleen|splenic|lungs?|pulmonary|stomach|gastric|colon\w*|bladder)\b"


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


def _kohort_to_refs(path: str) -> list[reference.ReferenceMeasurement]:
    """kohort_80_liste.csv (no, organ, grup, study_id, cap_mm) -> referans kayıtları.

    grup: Tumor -> POS, Normal -> NEG. cap_mm radyolog raporundan transkribe çap.
    """
    refs = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            sid = row.get("study_id") or row.get("study id")
            if not sid:
                continue
            grup = row.get("grup", row.get("group", "")).lower()
            group = "POS" if grup in ("tumor", "tümör", "pos", "1") else "NEG"
            refs.append(reference.ReferenceMeasurement(
                study_id=sid, organ=organ_key(row.get("organ", "")) or row.get("organ", ""),
                group=group,
                target_diameter_mm=row.get("cap_mm", "") if group == "POS" else "",
                source_sentence="kohort_80_liste.csv", transcriber="kohort"))
    return refs


def load_references(path: str) -> list[reference.ReferenceMeasurement]:
    """reference_gt.csv varsa onu, yoksa aynı klasördeki kohort_80_liste.csv'yi oku."""
    p = Path(path)
    if p.exists() and p.name != "kohort_80_liste.csv":
        return reference.load_csv(p)
    alt = p if p.name == "kohort_80_liste.csv" else p.parent / "kohort_80_liste.csv"
    if not alt.exists():
        raise FileNotFoundError(f"referans yok: {p} ve {alt}")
    print(f"ℹ️  referans: {alt.name}")
    return _kohort_to_refs(str(alt))


# Rapor kıyasından dışlanan modeller. Gemini günlük kota duvarı nedeniyle yalnız
# birkaç vakada koştu (n=6/80) -> ortalamalar yanıltıcı. Env ile değiştirilebilir:
# RADYOLOG_EXCLUDE_MODELS="" verilirse hiçbiri dışlanmaz.
EXCLUDE_MODELS = {m.strip() for m in os.environ.get(
    "RADYOLOG_EXCLUDE_MODELS", "Gemini").split(",") if m.strip()}


def pipeline_diameter(fnd: dict[str, Any]) -> float:
    """Pipeline'ın hedef çapı (RAW tercih). Lezyon yoksa 0.0 (= negatif karar)."""
    enb = fnd.get("en_buyuk_cap_lezyon")
    if not isinstance(enb, dict) or not fnd.get("lezyon_sayisi"):
        return 0.0
    v = enb.get("cap_mm_raw", enb.get("cap_mm"))
    return float(v) if v is not None else 0.0


def pipeline_diameter_by_volume(fnd: dict[str, Any]) -> float:
    """Alternatif hedef: en büyük HACİMLİ lezyonun çapı (ince damar/çizgi
    artefaktlarını eler; RECIST'te hedef lezyon damar değil kütledir).
    Lezyon yoksa 0.0. en_buyuk_hacim_lezyon ham çap tutmaz -> cap_mm kullanılır."""
    env = fnd.get("en_buyuk_hacim_lezyon")
    if not isinstance(env, dict) or not fnd.get("lezyon_sayisi"):
        return 0.0
    v = env.get("cap_mm_raw", env.get("cap_mm"))
    return float(v) if v is not None else 0.0


# Segmentasyon QC bayrağı: RADYOLOĞA BAKMADAN, yalnız maskenin kendi özelliğinden.
# Parçalanma (çok sayıda bileşen) aşırı-segmentasyonun işaretidir. Eşik veri
# görüldükten sonra seçildiği için analiz KEŞİFSELDİR (makalede böyle beyan edilir).
QC_MAX_LESIONS = int(os.environ.get("RADYOLOG_QC_MAX_LESIONS", "10"))


def qc_pass(fnd: dict[str, Any]) -> bool:
    return int(fnd.get("lezyon_sayisi") or 0) <= QC_MAX_LESIONS


def error_distribution(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    """Betimleyici hata dağılımı (alt küme SEÇMEZ; tüm çiftler üzerinde)."""
    if not pairs:
        return {}
    ae = sorted(abs(p - r) for p, r in pairs)
    n = len(ae)
    med = ae[n // 2] if n % 2 else (ae[n // 2 - 1] + ae[n // 2]) / 2
    out = {"medyan_AE": round(med, 2)}
    for t in (5, 10, 20):
        out[f"<={t}mm"] = f"{sum(1 for x in ae if x <= t)}/{n}"
    return out


def _loa(pairs):
    d = [p - r for p, r in pairs]
    mu = sum(d) / len(d)
    sd = (sum((x - mu) ** 2 for x in d) / max(len(d) - 1, 1)) ** 0.5
    return [round(mu - 1.96 * sd, 2), round(mu + 1.96 * sd, 2)]


# ---------------------------------------------------------------------------
# Rapor metni yardımcıları
# ---------------------------------------------------------------------------
def sentences(text: str) -> list[str]:
    """Cümle böl: ondalığı (41.2) korur ama rakamla biten cümleyi de ayırır
    ("segment 7. The pancreas ..."). evaluation.split_sentences ikincisini
    birleştirdiği için burada ayrı bir bölücü kullanılır (sadakat skoru değişmez)."""
    parts = re.split(r"(?<!\d)\.(?!\d)|(?<=\d)\.(?=\s+[A-Z])|\.\s*$|\n+", text or "")
    return [p.strip() for p in parts if p and p.strip()]


def organ_sentences(text: str, organ: str) -> list[str]:
    pat = re.compile(ORGAN_RE[organ], re.I)
    return [s for s in sentences(text) if pat.search(s)]


def candidate_sentences(text: str, organ: str) -> list[str]:
    """Model raporundan kıyaslanacak cümleler: organ ya da lezyon cümleleri.
    (Şablon/istem organ adını Türkçe 'karaciger/pankreas' verdiği için onlar da
    yakalanır.) Böylece ROUGE organ-cümlesi vs organ-cümlesi olur, simetrik."""
    pat = re.compile(ORGAN_RE[organ] + r"|karaci|pankrea|" + LESION_RE, re.I)
    return [s for s in sentences(text) if pat.search(s)]


def report_claims(text: str) -> tuple[bool, float | None]:
    """Rapor metninden (lezyon var beyanı, yazılan en büyük çap mm)."""
    positive, best = False, None
    for s in sentences(text):
        if not re.search(LESION_RE, s, re.I):
            continue
        negated = bool(re.search(NEG_RE, s, re.I))
        lens = [m.value for m in evaluation.extract_measurements(s)
                if m.category == "length"]
        has_dia = bool(re.search(DIA_RE, s, re.I))
        if re.search(HEDGE_RE, s, re.I) and not (lens and has_dia):
            continue
        # damar/mesafe cümlesi, lezyon boyutu söylemiyorsa beyan sayılmaz
        if (re.search(VESSEL_RE, s, re.I) or re.search(DIST_RE, s, re.I)) and not has_dia:
            continue
        if lens and not negated:
            positive = True
            if has_dia:  # boyut cümlesi (mesafe cümleleri yukarıda elendi)
                best = max(lens) if best is None else max(best, max(lens))
        elif not negated and not lens:
            # ölçüsüz ama olumlu lezyon cümlesi ("a hepatic mass is present")
            if re.search(r"\b(is|are|was|were)\b[^.]{0,20}\b(present|seen|identified|noted)\b",
                         s, re.I):
                positive = True
    return positive, best


def organ_naming(text: str, organ: str | None) -> tuple[int, int]:
    """(doğru organ adı geçiyor mu, ORGAN KARIŞIKLIĞI var mı) — 0/1.

    Karışıklık = doğru organ adı HİÇ geçmiyor ve başka bir organ adı geçiyor
    (ör. 'karaciger' -> 'right kidney'). Doğru organı anlatıp komşu organdan
    söz etmek ("no gastric invasion") karışıklık sayılmaz."""
    t = text or ""
    dogru = int(bool(organ) and bool(re.search(ORGAN_RE[organ], t, re.I)))
    baska = bool(re.search(OTHER_ORGAN_RE, t, re.I))
    return dogru, int(baska and not dogru)


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
    ap.add_argument("--out", default=None,
                    help="varsayılan: outputs/<rapor klasörünün adı>")
    ap.add_argument("--threshold", type=float, default=10.0)
    a = ap.parse_args(argv)

    out = Path(a.out) if a.out else config.LAYOUT.outputs / Path(a.reports).name
    out.mkdir(parents=True, exist_ok=True)
    print(f"raporlar: {a.reports}\nçıktı: {out}")
    refs = reference.index_by_study(load_references(a.reference))
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
    print("\n=== 2) ÇAP UYUMU (POS vakalar) ===")
    print("  eşleştirme politikası: pipeline'ın EN BÜYÜK ÇAPLI lezyonu vs radyoloğun "
          "ölçtüğü lezyon (koordinat yok -> garantili aynı-lezyon eşleşmesi DEĞİL)")
    ciftler_qc = {"gecti": [], "kaldi": []}
    satirlar = []
    ciftler = {"hepsi": [], "liver": [], "pancreas": []}
    ciftler_vol = {"hepsi": [], "liver": [], "pancreas": []}   # alternatif: hacim politikası
    for s in ortak:
        r = refs[s]
        if r.group.upper() != "POS":
            continue
        ref_mm = reference.reference_mm(r)
        pipe = pipeline_diameter(cases[s]["findings"])
        pipe_vol = pipeline_diameter_by_volume(cases[s]["findings"])
        if ref_mm is None:
            durum = "referans_olcusu_yok"
        elif pipe <= 0:
            durum = "kacirildi"
        else:
            durum = "eslesti"
            ciftler["hepsi"].append((pipe, ref_mm))
            ciftler[cases[s]["organ"]].append((pipe, ref_mm))
            ciftler_qc["gecti" if qc_pass(cases[s]["findings"]) else "kaldi"].append(
                (pipe, ref_mm))
        # hacim politikası: aynı "pozitif" vakalarda (lezyon bulunan) kıyasla
        if durum == "eslesti" and pipe_vol > 0:
            ciftler_vol["hepsi"].append((pipe_vol, ref_mm))
            ciftler_vol[cases[s]["organ"]].append((pipe_vol, ref_mm))
        satirlar.append({"study_id": s, "organ": cases[s]["organ"], "durum": durum,
                         "pipeline_mm": round(pipe, 2) if pipe else "",
                         "pipeline_vol_mm": round(pipe_vol, 2) if pipe_vol else "",
                         "radyolog_mm": "" if ref_mm is None else ref_mm,
                         "fark_mm": round(pipe - ref_mm, 2) if durum == "eslesti" else "",
                         "fark_vol_mm": (round(pipe_vol - ref_mm, 2)
                                         if (durum == "eslesti" and pipe_vol > 0) else ""),
                         "lezyon_sayisi": cases[s]["findings"].get("lezyon_sayisi", 0),
                         "qc": "gecti" if qc_pass(cases[s]["findings"]) else "kaldi"})
    _write_csv(out / "cap_uyumu_vaka.csv", satirlar)
    for grup, pairs in ciftler.items():
        kacan = sum(1 for x in satirlar if x["durum"] == "kacirildi"
                    and (grup == "hepsi" or x["organ"] == grup))
        m = evaluation.measurement_agreement(pairs, n_skipped=kacan)
        if pairs:
            m["LoA"] = _loa(pairs)
        m["hata_dagilimi"] = error_distribution(pairs)
        sonuc["cap_uyumu"][grup] = m
        print(f"  [{grup:8}] n={m['n_eslesen']} kaçan={m['n_atlanan']} · MAE={m['MAE']} "
              f"· yanlılık={m['yanlilik']} · r={m['korelasyon']} · LoA={m.get('LoA')} "
              f"· {m['hata_dagilimi']}")

    print(f"\n  --- QC katmanı (keşifsel; radyoloğa bakmadan: lezyon_sayisi <= "
          f"{QC_MAX_LESIONS} ise 'geçti') ---")
    sonuc["cap_uyumu_qc"] = {}
    for k, pairs in ciftler_qc.items():
        mq = evaluation.measurement_agreement(pairs)
        if pairs:
            mq["LoA"] = _loa(pairs)
        mq["hata_dagilimi"] = error_distribution(pairs)
        sonuc["cap_uyumu_qc"][k] = mq
        print(f"  [QC {k:5}] n={mq['n_eslesen']} · MAE={mq['MAE']} · yanlılık={mq['yanlilik']} "
              f"· r={mq['korelasyon']} · {mq['hata_dagilimi']}")

    # --- Alternatif politika: hedef = en büyük HACİMLİ lezyon (artefakt elenir) ---
    print("\n  --- alternatif hedef: en büyük HACİMLİ lezyon ---")
    sonuc["cap_uyumu_hacim"] = {}
    for grup, pairs in ciftler_vol.items():
        mv = evaluation.measurement_agreement(pairs)
        if pairs:
            mv["LoA"] = _loa(pairs)
        mv["hata_dagilimi"] = error_distribution(pairs)
        sonuc["cap_uyumu_hacim"][grup] = mv
        print(f"  [{grup:8}] n={mv['n_eslesen']} · MAE={mv['MAE']} "
              f"· yanlılık={mv['yanlilik']} · r={mv['korelasyon']} · LoA={mv.get('LoA')} "
              f"· {mv['hata_dagilimi']}")
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
    gecersiz: dict[str, list[str]] = {}   # model -> geçersiz (kesik/boş) rapor vakaları
    for s in ortak:
        ad, organ = cases[s]["ad"], cases[s]["organ"]
        r = refs[s]
        gt_pos, ref_mm = r.group.upper() == "POS", reference.reference_mm(r)
        ref_org = " . ".join(organ_sentences(rad_text.get(s, ""), organ)) if organ else ""
        for p in sorted(glob.glob(os.path.join(a.reports, f"{ad}__*.txt"))):
            if p.endswith(".gecersiz.txt"):
                # Geçersiz çıktı RAPOR DEĞİLDİR: puanlanmaz, ayrı sayılır.
                m_bad = os.path.basename(p)[:-len(".gecersiz.txt")].split("__")[-1]
                if m_bad not in EXCLUDE_MODELS:
                    gecersiz.setdefault(m_bad, []).append(s)
                continue
            model = os.path.splitext(os.path.basename(p))[0].split("__")[-1]
            if model in EXCLUDE_MODELS:
                continue
            with open(p, "r", encoding="utf-8") as f:
                text = f.read()
            pos, mm = report_claims(text)
            org_ok, org_bad = organ_naming(text, organ)
            rapor_satir.append({
                "study_id": s, "organ": organ, "model": model,
                "gt_pozitif": int(gt_pos), "rapor_pozitif": int(pos),
                "radyolog_mm": "" if ref_mm is None else ref_mm,
                "rapor_mm": "" if mm is None else round(mm, 2),
                "abs_fark_mm": (round(abs(mm - ref_mm), 2)
                                if (mm is not None and ref_mm is not None) else ""),
                "rouge_l": (rouge_l(" . ".join(candidate_sentences(text, organ)), ref_org)
                            if (ref_org and organ) else ""),
                "radyolog_organ_cumle": len(organ_sentences(rad_text.get(s, ""), organ))
                                        if organ else 0,
                "organ_adi_dogru": org_ok, "organ_karisikligi": org_bad,
            })
    _write_csv(out / "rapor_kiyas_vaka.csv", rapor_satir)

    ozet = []
    for model in sorted({x["model"] for x in rapor_satir} | set(gecersiz)):
        rs = [x for x in rapor_satir if x["model"] == model]
        tp = sum(1 for x in rs if x["gt_pozitif"] and x["rapor_pozitif"])
        fn = sum(1 for x in rs if x["gt_pozitif"] and not x["rapor_pozitif"])
        fp = sum(1 for x in rs if not x["gt_pozitif"] and x["rapor_pozitif"])
        tn = sum(1 for x in rs if not x["gt_pozitif"] and not x["rapor_pozitif"])
        farklar = [x["abs_fark_mm"] for x in rs if x["abs_fark_mm"] != ""]
        rouges = [x["rouge_l"] for x in rs if x["rouge_l"] not in ("", None)]
        ozet.append({
            "model": model, "n": len(rs), "n_gecersiz": len(gecersiz.get(model, [])),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "duyarlilik": round(tp / (tp + fn), 3) if (tp + fn) else "",
            "ozgulluk": round(tn / (tn + fp), 3) if (tn + fp) else "",
            "dogruluk": round((tp + tn) / len(rs), 3) if rs else "",
            "cap_n": len(farklar),
            "cap_MAE_mm": round(sum(farklar) / len(farklar), 2) if farklar else "",
            "cap_medyan_AE": (sorted(farklar)[len(farklar) // 2] if farklar else ""),
            "rouge_l_ort": round(sum(rouges) / len(rouges), 4) if rouges else "",
            "rouge_n": len(rouges),
            "organ_adi_dogru": f"{sum(x['organ_adi_dogru'] for x in rs)}/{len(rs)}",
            "organ_karisikligi": f"{sum(x['organ_karisikligi'] for x in rs)}/{len(rs)}",
        })
    _write_csv(out / "rapor_kiyas_ozet.csv", ozet)
    sonuc["rapor"] = ozet
    for o in ozet:
        print(f"  {o['model']:9} n={o['n']:3} (+{o['n_gecersiz']} geçersiz) · duy={o['duyarlilik']} özg={o['ozgulluk']} "
              f"doğ={o['dogruluk']} · çap MAE={o['cap_MAE_mm']} (n={o['cap_n']}, "
              f"medyan={o['cap_medyan_AE']}) · ROUGE-L={o['rouge_l_ort']} (n={o['rouge_n']}) · organ adı doğru={o['organ_adi_dogru']} organ karışıklığı={o['organ_karisikligi']}")

    with open(out / "referans_kiyas.json", "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n📁 Çıktılar: {out}")
    return sonuc


if __name__ == "__main__":
    main(sys.argv[1:])
