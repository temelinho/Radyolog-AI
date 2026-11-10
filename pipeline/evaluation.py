"""
evaluation.py — tespit, sadakat ve ölçüm uyumu (F13, F15, F17, F18).

Eski hatalar:
  * F13: `df.gt` pandas metoduyla sütun karışması; eşik ÖNCESİ yuvarlama
    (9.96 mm -> 10.0 -> yanlış pozitif). Artık karar RAW değerle.
  * F15: Metindeki tüm sayılar ortak havuzda aranıyor; yanlış bulgudaki /
    yanlış birimdeki sayı doğru sayılıyordu ("Aorta distance 20 mm" tümör
    çapı 20'yi onaylıyordu; "2 cm^3" -> 20 mm). Artık BAĞLAM + BİRİM eşleşmeli.
  * F17: `re.split(r'[.\n]')` 41.2'yi bölüyordu. Artık ondalık korunur.
  * F18: Payda sabit 20; atlanan/eşleşmeyen vakalar gizli. Artık ayrı sayılır.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Metin ayrıştırma — ondalık koruyan (F17)
# ---------------------------------------------------------------------------
# Birim normalizasyonu: kategori + mm'ye/cm3'e/dereceye çevrim.
_UNIT_PATTERNS = [
    # (regex, kategori, çarpan -> kanonik birim)
    (r"cm\s*(?:\^?3|³)|cc\b|ml\b", "volume", 1.0),   # hacim: cm3 ~ ml ~ cc
    (r"mm\b", "length", 1.0),                         # mm
    (r"cm\b", "length", 10.0),                        # cm -> mm
    (r"°|derece|degrees?\b|deg\b", "angle", 1.0),     # açı
]

_NUM = r"\d+(?:[.,]\d+)?"


@dataclass
class Measurement:
    value: float          # kanonik (length->mm, volume->cm3, angle->derece)
    category: str         # length | volume | angle | dimensionless
    raw_text: str
    unit: str | None


def _to_float(s: str) -> float:
    return float(s.replace(",", "."))


def extract_measurements(text: str) -> list[Measurement]:
    """Metinden (sayı + birim) ölçümleri çıkarır. Ondalıklar KORUNUR (F17)."""
    out: list[Measurement] = []
    # Sayı + hemen ardından (ops.) birim
    pattern = re.compile(_NUM + r"\s*(cm\s*\^?3|cm³|cc|ml|mm|cm|°|derece|degrees?|deg)?",
                         re.IGNORECASE)
    for m in pattern.finditer(text):
        num = m.group(0)
        unit_raw = (m.group(1) or "").strip().lower()
        val = _to_float(re.match(_NUM, num).group(0))
        category = "dimensionless"
        unit = unit_raw or None
        if unit_raw:
            for upat, cat, mult in _UNIT_PATTERNS:
                if re.fullmatch(upat, unit_raw, re.IGNORECASE):
                    category = cat
                    val = val * mult
                    break
        out.append(Measurement(value=val, category=category,
                               raw_text=num.strip(), unit=unit))
    return out


def split_sentences(text: str) -> list[str]:
    """Cümlelere böl; sayı İÇİNDEKİ noktayı (41.2) BÖLME (F17)."""
    parts = re.split(r"(?<!\d)\.(?!\d)|\n+", text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Sadakat — bağlam + birim eşleşmeli (F15)
# ---------------------------------------------------------------------------
@dataclass
class Fact:
    name: str
    category: str            # length | volume | angle
    value: float             # kanonik
    keywords: list[str]      # cümlede aranacak bağlam sözcükleri


def findings_to_facts(findings: dict[str, Any]) -> list[Fact]:
    """bulgular.json -> doğrulanacak tipli gerçekler listesi."""
    facts: list[Fact] = []
    enb = findings.get("en_buyuk_cap_lezyon") or findings.get("en_buyuk_lezyon")
    if isinstance(enb, dict):
        if enb.get("cap_mm") is not None:
            facts.append(Fact("en_buyuk_cap", "length", float(enb["cap_mm"]),
                              ["çap", "cap", "diameter", "boyut", "lezyon", "kitle",
                               "lesion", "mass", "measures", "measuring"]))
        if enb.get("hacim_cm3") is not None:
            facts.append(Fact("en_buyuk_hacim", "volume", float(enb["hacim_cm3"]),
                              ["hacm", "volume", "cm3", "cm³"]))  # "hacm": hacim/hacmi kökü
    if findings.get("toplam_tumor_hacim_cm3"):
        facts.append(Fact("toplam_hacim", "volume",
                          float(findings["toplam_tumor_hacim_cm3"]),
                          ["toplam", "hacm", "volume"]))
    damar = findings.get("damar") or {}
    for vname, vinfo in damar.items() if isinstance(damar, dict) else []:
        if not isinstance(vinfo, dict):
            continue
        kw = [vname.replace("_", " "), *vname.split("_")]
        if vinfo.get("mesafe_mm") is not None:
            facts.append(Fact(f"{vname}_mesafe", "length", float(vinfo["mesafe_mm"]),
                              kw + ["mesafe", "distance", "temas", "contact", "abut"]))
        if vinfo.get("sarma_derece") is not None:
            facts.append(Fact(f"{vname}_sarma", "angle", float(vinfo["sarma_derece"]),
                              kw + ["sarma", "derece", "angle", "temas",
                                    "encasement", "encases", "contact", "abut"]))
    return facts


def _close(a: float, b: float, rel: float = 0.02, abs_tol: float = 0.5) -> bool:
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


def fact_supported(fact: Fact, sentences: list[str]) -> bool:
    """Gerçek, DOĞRU bağlamdaki bir cümlede DOĞRU birim/değerle geçiyor mu?"""
    kws = [k.lower() for k in fact.keywords if k]
    for s in sentences:
        sl = s.lower()
        if not any(k in sl for k in kws):
            continue
        for meas in extract_measurements(s):
            if meas.category != fact.category:
                continue
            if _close(meas.value, fact.value):
                return True
    return False


@dataclass
class FidelityResult:
    supported: int
    total: int
    ratio: float | None
    missing: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        # F16: Bu bir OTOMATİK proxy'dir (sayısal grounding göstergesi);
        # klinik doğruluk/halüsinasyon kararı değildir, radyolog teyidi gerekir.
        return {"gecen": self.supported, "toplam": self.total,
                "oran": self.ratio, "eksik": self.missing,
                "yontem": "otomatik-proxy", "uyari": "nihai karar radyolog teyidi gerektirir"}


def numeric_fidelity(report: str, findings: dict[str, Any]) -> FidelityResult:
    """Raporun, bulgulardaki her sayıyı DOĞRU bağlamda aktarıp aktarmadığı (F15)."""
    facts = findings_to_facts(findings)
    sentences = split_sentences(report)
    if not facts:
        return FidelityResult(0, 0, None)
    supported, missing = 0, []
    for f in facts:
        if fact_supported(f, sentences):
            supported += 1
        else:
            missing.append(f.name)
    return FidelityResult(supported, len(facts),
                          round(supported / len(facts), 3), missing)


# ---------------------------------------------------------------------------
# Tespit — RAW eşik + Wilson GA (F13)
# ---------------------------------------------------------------------------
def is_positive(diameter_mm_raw: float | None, threshold: float = 10.0) -> bool:
    """RAW çapla karar. 9.96 -> False (yuvarlamadan etkilenmez) (F13)."""
    if diameter_mm_raw is None:
        raise ValueError("ölçülemeyen vaka tespit paydasına girmez; ayrı say (F18)")
    return float(diameter_mm_raw) >= threshold


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass
class DetectionRecord:
    gt_positive: bool
    pred_diameter_raw: float | None   # None => ölçülemedi/atlandı


def detection_metrics(records: list[DetectionRecord],
                      threshold: float = 10.0) -> dict[str, Any]:
    """Duyarlılık/özgüllük + Wilson GA. Ölçülemeyenler AYRI sayılır (F18)."""
    tp = fp = tn = fn = 0
    skipped = 0
    for r in records:
        if r.pred_diameter_raw is None:
            skipped += 1
            continue
        pred = r.pred_diameter_raw >= threshold
        if r.gt_positive and pred:
            tp += 1
        elif r.gt_positive and not pred:
            fn += 1
        elif not r.gt_positive and pred:
            fp += 1
        else:
            tn += 1
    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "degerlendirilen": tp + fp + tn + fn,
        "atlanan": skipped,
        "duyarlilik": sens,
        "duyarlilik_GA": wilson_ci(tp, tp + fn) if (tp + fn) else None,
        "ozgulluk": spec,
        "ozgulluk_GA": wilson_ci(tn, tn + fp) if (tn + fp) else None,
    }


# ---------------------------------------------------------------------------
# Ölçüm uyumu — eşleşmiş alt kümede, ayrı paydalarla (F18)
# ---------------------------------------------------------------------------
def measurement_agreement(pairs: list[tuple[float, float]],
                          n_skipped: int = 0) -> dict[str, Any]:
    """Eşleşmiş (pipeline, referans) çiftlerinde MAE / yanlılık / korelasyon.

    pairs YALNIZ aynı lezyonla eşleşmiş vakaları içermeli. Eşleşmeyen/atlanan
    vakalar n_skipped ile AYRI raporlanır; paydaya gizlice katılmaz (F18).
    """
    n = len(pairs)
    if n == 0:
        return {"n_eslesen": 0, "n_atlanan": n_skipped, "MAE": None,
                "yanlilik": None, "korelasyon": None}
    diffs = [p - r for p, r in pairs]
    mae = sum(abs(d) for d in diffs) / n
    bias = sum(diffs) / n
    corr = _pearson([p for p, _ in pairs], [r for _, r in pairs])
    return {
        "n_eslesen": n,
        "n_atlanan": n_skipped,
        "MAE": round(mae, 2),
        "yanlilik": round(bias, 2),
        "korelasyon": None if corr is None else round(corr, 3),
    }


def _pearson(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)
