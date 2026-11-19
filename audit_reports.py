"""
audit_reports.py — dayanaksız ekleme ve eksik bulgu denetimi (API/GPU kullanmaz).

Her raporu, o vakada modele GERÇEKTEN verilen İngilizce klinik JSON'la
(reporting.model_findings) karşılaştırır.

DAYANAKSIZ EKLEME (raporda var, JSON'da yok):
  sayi            birimli değer (uzunluk/hacim/açı) JSON'daki hiçbir değere uymuyor
  tani            tanı/histoloji terimi (HCC, metastaz, kist ...). JSON'da tanı yoktur.
  konum_morfoloji lob/segment/baş-gövde-kuyruk, hipodens/kontrastlanan ...
  faz             JSON'da faz null iken faz/kontrast iddiası
  rezektabilite   rezektabl/irrezektabl, T evresi, evre
EKSİK BULGU (JSON'da var, raporda yok):
  sayisal         sadakat ölçümünde eksik kalan gerçekler (numeric_fidelity)
  boyut_kategorisi, lezyon_sayisi, organ_adi

Olumsuzlanmış cümleler (no/not/cannot/without/...) ekleme sayılmaz.
Kural tabanlı ve KEŞİFSEL bir denetimdir; örnek cümleler elle kontrol için basılır.
Şablon (Template) tanımı gereği hiçbir şey eklemez: şablonda ekleme bulunursa kural hatalıdır.

Kullanım:
    RADYOLOG_BASE=... RADYOLOG_REPORTS_DIR=... python audit_reports.py
Çıktı: outputs/<deney>/ekleme_eksik_vaka.csv, ekleme_eksik_ozet.csv
"""

from __future__ import annotations

import collections
import csv
import glob
import os
import re
import sys
from pathlib import Path
from typing import Any

import compare_llm
import compare_reference as C
from pipeline import config, evaluation, reporting

NEG_RE = (r"\b(no|not|cannot|can't|without|absence|absent|exclude[sd]?|rule[sd]?\s+out|"
          r"unable|unknown|unspecified|none|neither|nor)\b|n't\b")

TANI_RE = (r"\b(hepatocellular|hcc|cholangiocarcinoma|adenocarcinoma|carcinoma|metasta\w*|"
           r"lymphoma|neuroendocrine|sarcoma|cancer|malignan\w*|benign|h[ae]mangioma|"
           r"cysts?|abscess|adenoma|focal\s+nodular\s+hyperplasia|fnh|ipmn|cystadenoma)\b")

KONUM_RE = {
    "liver": (r"\b(left|right)\s+(?:hepatic\s+)?lobe\b|\b(left|right)\s+liver\b|\bcaudate\b|"
              r"\bsegment(?:s)?\s*(?:[ivx]+|\d+)\b|\bdome\b|\bporta\s+hepatis\b|\bhilum\b"),
    "pancreas": (r"\b(head|body|tail|uncinate|neck)\s+of\s+the\s+pancreas\b|"
                 r"\bpancreatic\s+(head|body|tail|neck)\b|\buncinate\b"),
}
MORFOLOJI_RE = (r"\b(hypodens\w*|hyperdens\w*|hypoattenuat\w*|hyperattenuat\w*|isoattenuat\w*|"
                r"hypoenhanc\w*|hyperenhanc\w*|enhancing|enhancement|washout|calcif\w*|cystic|"
                r"necro\w*|heterogeneous\w*|homogeneous\w*|ill[\s-]defined|well[\s-]defined|"
                r"lobulated|infiltrat\w*|spiculat\w*|rim)\b")
# "portal venous" tek başına damar ifadesi de olabilir ("portal venous encasement"): yalnız faz bağlamı
FAZ_RE = (r"\b(portal[\s-]+venous\s+(?:phase|contrast|imaging|acquisition)|arterial\s+phase|"
          r"late\s+arterial|delayed\s+phase|"
          r"contrast[\s-]+enhanced|non[\s-]?contrast|unenhanced|post[\s-]?contrast|"
          r"with\s+(?:iv\s+|intravenous\s+)?contrast)\b")
# Yalnız DURUM bildirimi: "resectable", "unresectable", "borderline resectable", T/evre.
# "rezektabilite değerlendirilmeli" (resectability ...) sayılmaz.
REZ_RE = (r"\b(?:un|non[\s-]?)?resectable\b|\bborderline\s+resectable\b|\binoperable\b|"
          r"\boperable\b|\bT[1-4][abc]?\b|\bstage\s+(?:i{1,3}v?|iv|[1-4])\b")

# Öneri / koşul / yönlendirme cümleleri bir BULGU İDDİASI değildir
# ("further contrast-enhanced imaging is recommended", "to assess for malignancy").
ONERI_RE = (r"\b(recommend\w*|consider\w*|advis\w*|suggest\w*\s+(?:further|additional|that)|"
            r"further\s+(?:imaging|evaluation|assessment|characteri\w*|work[\s-]?up)|"
            r"additional\s+(?:imaging|evaluation|assessment|techniques?)|follow[\s-]?up|should|"
            r"if\s+(?:clinical|there)|to\s+(?:assess|evaluate|determine|characteri[sz]e|exclude|"
            r"rule\s+out|confirm|clarify)|assess(?:ed|ment)?\s+for|correlat\w*|tumou?r\s+board|"
            r"multidisciplinary|required|necessary|needed|warrant\w*)\b")

SAYI_KELIME = {1: "one|single|solitary", 2: "two", 3: "three", 4: "four", 5: "five",
               6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
# Organa özgü boyut kategorisi eşikleri (mm): modeller kategorileri açıklarken anabilir
ESIK_MM = {"liver": [20.0, 50.0], "pancreas": [20.0, 40.0]}


def _cumleler(text: str) -> list[str]:
    t = re.sub(r"\b(e\.g|i\.e|etc|vs|approx)\.", lambda m: m.group(1).replace(".", ""),
               text or "", flags=re.I)
    out = []
    for s in C.sentences(t):
        out += [x.strip() for x in s.split(";") if x.strip()]
    return out


def _olumsuz(s: str) -> bool:
    return bool(re.search(NEG_RE, s, re.I))


def _oneri(s: str) -> bool:
    return bool(re.search(ONERI_RE, s, re.I))


def izinli_degerler(mf: dict[str, Any]) -> dict[str, list[float]]:
    """JSON'dan kategori bazında izinli sayısal değerler (kanonik birimde)."""
    d = {"length": [], "volume": [], "angle": []}
    for k in ("largest_lesion_by_diameter", "largest_lesion_by_volume"):
        les = mf.get(k) or {}
        if les.get("axial_diameter_mm") is not None:
            d["length"].append(float(les["axial_diameter_mm"]))
        if les.get("volume_cm3") is not None:
            d["volume"].append(float(les["volume_cm3"]))
    if mf.get("total_tumor_volume_cm3") is not None:
        d["volume"].append(float(mf["total_tumor_volume_cm3"]))
    for v in (mf.get("vessels") or {}).values():
        if v.get("distance_mm") is not None:
            d["length"].append(float(v["distance_mm"]))
        if v.get("encasement_angle_deg") is not None:
            d["angle"].append(float(v["encasement_angle_deg"]))
    d["length"] += ESIK_MM.get(mf.get("organ"), [])
    d["length"].append(0.0); d["angle"].append(0.0)   # "> 0 mm", "0 degrees" karşılaştırma
    cat = mf.get("size_category") or ""
    for n in re.findall(r"\d+(?:\.\d+)?", cat):
        d["length"].append(float(n) * (10.0 if "cm" in cat else 1.0))
    return d


def _yakin(a: float, b: float) -> bool:
    # Ekleme tespiti için yuvarlamaya hoşgörülü (3.1 cm ~ 31.4 mm): max(0.6, %5)
    return abs(a - b) <= max(0.6, 0.05 * max(abs(a), abs(b)))


def turetilmis_degerler(mf: dict[str, Any]) -> list[float]:
    """Anlamlı türetilmiş HACİMLER: toplam tümör hacmi ile lezyon hacimleri arasındaki
    fark (ör. "kalan lezyonlar 3.25 cm3"). JSON'da yazılı değildir ama verilen sayılardan
    hesaplanmıştır: UYDURMA DEĞİL. Uzunluk/açı kombinasyonları anlamsız olduğu için yok."""
    tot = mf.get("total_tumor_volume_cm3")
    if not tot:
        return []
    out = []
    for k in ("largest_lesion_by_diameter", "largest_lesion_by_volume"):
        v = (mf.get(k) or {}).get("volume_cm3")
        if v is not None and tot - v > 0:
            out.append(round(tot - v, 4))
    return out


def _tam(a: float, b: float) -> bool:
    # Hesaplanan değer için dar tolerans (yuvarlama payı): max(0.05, %3)
    return abs(a - b) <= max(0.05, 0.03 * max(abs(a), abs(b)))


def dayanaksiz_sayilar(text: str, mf: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(uydurma sayılar, türetilmiş sayılar). Türetilmiş = JSON değerlerinin farkı/toplamı."""
    izin = izinli_degerler(mf)
    turev = turetilmis_degerler(mf)
    uydurma, turetilmis = [], []
    for s in _cumleler(text):
        for m in evaluation.extract_measurements(s):
            if m.category not in izin:
                continue
            if any(_yakin(m.value, v) for v in izin[m.category]):
                continue
            hedef = (turetilmis if m.category == "volume" and any(_tam(m.value, v) for v in turev)
                     else uydurma)
            hedef.append(f"{m.raw_text} ⟨{s[:160]}⟩")
    return uydurma, turetilmis


def _terim_cumleleri(text: str, pattern: str) -> list[str]:
    return [s[:200] for s in _cumleler(text)
            if re.search(pattern, s, re.I) and not _olumsuz(s) and not _oneri(s)]


def eksik_boyut_kategorisi(text: str, mf: dict[str, Any]) -> bool:
    cat = mf.get("size_category")
    if not cat:
        return False
    nums = re.findall(r"\d+(?:\.\d+)?", cat)
    for s in C.sentences(text):
        if "categor" not in s.lower() and "cm" not in s.lower():
            continue
        if all(re.search(rf"(?<![\d.]){re.escape(n)}(?![\d.])", s) or
               re.search(rf"(?<![\d.]){int(float(n) * 10)}(?![\d.])\s*mm", s) for n in nums):
            return False
    return True


def eksik_lezyon_sayisi(text: str, mf: dict[str, Any]) -> bool:
    n = mf.get("lesion_count")
    if not n:
        return False
    alt = rf"\b{n}\b" + (rf"|\b(?:{SAYI_KELIME[n]})\b" if n in SAYI_KELIME else "")
    if n > 1:
        alt += r"|\bmultiple\b|\bmultifocal\b|\bnumerous\b|\bseveral\b"
    for s in C.sentences(text):
        if re.search(C.LESION_RE, s, re.I) and re.search(alt, s, re.I):
            return False
    return True


def denetle(text: str, mf: dict[str, Any], findings: dict[str, Any]) -> dict[str, Any]:
    organ = mf.get("organ")
    sayi, turetilmis = dayanaksiz_sayilar(text, mf)
    tani = _terim_cumleleri(text, TANI_RE)
    konum = _terim_cumleleri(text, KONUM_RE.get(organ, r"$^") + "|" + MORFOLOJI_RE)
    faz = _terim_cumleleri(text, FAZ_RE) if mf.get("imaging_phase") is None else []
    rez = _terim_cumleleri(text, REZ_RE)
    fid = evaluation.numeric_fidelity(text, findings)
    return {
        "ek_sayi": len(sayi), "ek_tani": int(bool(tani)), "ek_konum_morfoloji": int(bool(konum)),
        "ek_faz": int(bool(faz)), "ek_rezektabilite": int(bool(rez)),
        "turetilmis_sayi": len(turetilmis),   # bilgi amaçlı; ekleme SAYILMAZ
        "eksik_sayisal": len(fid.missing),
        "eksik_boyut_kategorisi": int(eksik_boyut_kategorisi(text, mf)),
        "eksik_lezyon_sayisi": int(eksik_lezyon_sayisi(text, mf)),
        "eksik_organ_adi": int(not C.organ_naming(text, organ)[0]),
        "_ornek": {"sayi": sayi[:2], "turetilmis_sayi": turetilmis[:2],
                   "tani": tani[:2], "konum_morfoloji": konum[:2],
                   "faz": faz[:2], "rezektabilite": rez[:2]},
    }


def kor_ornek(tum, out: Path, k: int = 5, seed: int = 42) -> None:
    """Elle doğrulama için KÖR örneklem: her (kategori, model) için en fazla k işaretli
    cümle, sabit seed ile rastgele. Model adı ayrı anahtar dosyasında; değerlendirici
    yalnız cümleyi ve kategoriyi görür, 'karar' sütununu doldurur (dogru / yanlis_alarm)."""
    import random
    rnd = random.Random(seed)
    grup = collections.defaultdict(list)
    for model, ad, kat, x in tum:
        if model != "Template" and kat != "turetilmis_sayi":   # türetilmiş ekleme değil
            grup[(kat, model)].append((ad, x))
    secim = []
    for key in sorted(grup):
        xs = grup[key]
        secim += [(key[0], key[1], ad, x) for ad, x in rnd.sample(xs, min(k, len(xs)))]
    rnd.shuffle(secim)
    with open(out / "elle_dogrulama_kor.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "kategori", "cumle", "karar"])
        for i, (kat, _m, _ad, x) in enumerate(secim, 1):
            w.writerow([i, kat, x, ""])
    with open(out / "elle_dogrulama_anahtar.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "model", "vaka"])
        for i, (_k, m, ad, _x) in enumerate(secim, 1):
            w.writerow([i, m, ad])
    print(f"elle doğrulama örneklemi: {len(secim)} cümle -> elle_dogrulama_kor.csv (+ anahtar)")


EK = ["ek_sayi", "ek_tani", "ek_konum_morfoloji", "ek_faz", "ek_rezektabilite"]
EKSIK = ["eksik_sayisal", "eksik_boyut_kategorisi", "eksik_lezyon_sayisi", "eksik_organ_adi"]


def main() -> int:
    exp = Path(config.LAYOUT.reports)
    out = config.LAYOUT.outputs / exp.name
    out.mkdir(parents=True, exist_ok=True)
    satirlar, ornekler, tum = [], collections.defaultdict(list), []
    for p in sorted(glob.glob(os.path.join(str(config.LAYOUT.findings), "*.json"))):
        if p.endswith(".meta.json"):
            continue
        findings = compare_llm._load_findings(p)
        if findings is None:
            continue
        ad = os.path.splitext(os.path.basename(p))[0]
        mf = reporting.model_findings(findings)
        for t in sorted(glob.glob(str(exp / f"{ad}__*.txt"))):
            if t.endswith(".gecersiz.txt"):
                continue
            model = os.path.basename(t)[:-4].split("__")[-1]
            if model in C.EXCLUDE_MODELS:
                continue
            r = denetle(open(t, encoding="utf-8").read(), mf, findings)
            for kat, xs in r.pop("_ornek").items():
                for x in xs:
                    ornekler[kat].append((model, x))
                    tum.append((model, ad, kat, x))
            satirlar.append({"vaka": ad, "organ": mf["organ"], "model": model, **r})
    if not satirlar:
        print("rapor bulunamadı:", exp)
        return 1
    with open(out / "ekleme_eksik_vaka.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0].keys()))
        w.writeheader(); w.writerows(satirlar)

    ozet = []
    for m in sorted({x["model"] for x in satirlar}):
        rs = [x for x in satirlar if x["model"] == m]
        n = len(rs)
        row = {"model": m, "n": n,
               "herhangi_ekleme_%": round(100 * sum(1 for x in rs if any(x[k] for k in EK)) / n, 1)}
        for k in EK:
            row[f"{k}_%"] = round(100 * sum(1 for x in rs if x[k]) / n, 1)
        row["turetilmis_sayi_%"] = round(100 * sum(1 for x in rs if x["turetilmis_sayi"]) / n, 1)
        row["herhangi_eksik_%"] = round(100 * sum(1 for x in rs if any(x[k] for k in EKSIK)) / n, 1)
        for k in EKSIK:
            row[f"{k}_%"] = round(100 * sum(1 for x in rs if x[k]) / n, 1)
        ozet.append(row)
    with open(out / "ekleme_eksik_ozet.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(ozet[0].keys()))
        w.writeheader(); w.writerows(ozet)

    kor_ornek(tum, out)
    print(f"rapor: {len(satirlar)} | çıktı: {out}\n")
    print("DAYANAKSIZ EKLEME (% rapor)")
    print(f"  {'model':16}{'herhangi':>9}{'sayı':>7}{'tanı':>7}{'konum/morf':>11}{'faz':>6}{'rezek.':>8}{'| türetilmiş':>13}")
    for o in ozet:
        print(f"  {o['model']:16}{o['herhangi_ekleme_%']:>9}{o['ek_sayi_%']:>7}{o['ek_tani_%']:>7}"
              f"{o['ek_konum_morfoloji_%']:>11}{o['ek_faz_%']:>6}{o['ek_rezektabilite_%']:>8}{o['turetilmis_sayi_%']:>13}")
    print("\nEKSİK BULGU (% rapor)")
    print(f"  {'model':16}{'herhangi':>9}{'sayısal':>9}{'boyut kat.':>11}{'lez. sayısı':>12}{'organ adı':>10}")
    for o in ozet:
        print(f"  {o['model']:16}{o['herhangi_eksik_%']:>9}{o['eksik_sayisal_%']:>9}"
              f"{o['eksik_boyut_kategorisi_%']:>11}{o['eksik_lezyon_sayisi_%']:>12}{o['eksik_organ_adi_%']:>10}")
    print("\nELLE KONTROL İÇİN ÖRNEKLER (kategori başına en fazla 4)")
    for kat, xs in ornekler.items():
        if not xs:
            continue
        print(f"\n[{kat}]")
        goruldu = set()
        for model, x in xs:
            if model in goruldu and len(goruldu) < 4:
                continue
            goruldu.add(model)
            print(f"  {model:10} {x[:230]}")
            if len(goruldu) >= 4:
                break
    tmpl = next((o for o in ozet if o["model"] == "Template"), None)
    if tmpl and tmpl["herhangi_ekleme_%"] > 0:
        print("\n⚠️  ŞABLONDA EKLEME BULUNDU — kural hatalı, sonuçları kullanma.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
