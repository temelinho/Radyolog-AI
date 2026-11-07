"""
select_cohort.py — Merlin'den SIFIRDAN, seed'li, TEKRARLANABİLİR kohort ADAYI seçimi.

Eski seçimler KULLANILMAZ. Zincir: select (aday) -> insan onayı -> finalize (kilit).
Tekrarlanabilirlik: sabit seed + girdi dosyası SHA256 + terim sürümü + tüm parametreler
`provenance_<organ>.json`'a yazılır. Aynı girdi + aynı seed = aynı adaylar.

Aday üretimi (bu script):
  1. reports_final.xlsx (Findings) üzerinde organ-özel regex -> POS/NEG/BELİRSİZ.
  2. Seed'li karıştır; her sınıftan `buffer` kadar ADAY topla (draw_order ile).
     (buffer > 20: insan onayında eleme olursa 20 yine dolar.)
  3. BELİRSİZ kohorta girmez ama study_id + draw_order + sebep ile LOGLANIR (STARD).
  4. Çıktı: onay_<organ>.csv (adaylar + Findings -> sen onay/çap girersin),
     akis_<organ>.csv (tüm walk logu), provenance_<organ>.json (tekrar için).

Nihai 20+20 KİLİDİ için: onay_<organ>.csv'yi doldur -> finalize_cohort.py çalıştır.
Gerçek raporları script açar (senin makinende); terminale BASMAZ.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

TERMS_VERSION = "1.1.0"   # kullanıcı geri bildirimiyle precision iyileştirildi
TERMS = {
    "liver": {
        "organ": r"hepatic|liver|karaciğer",
        "malign": r"mass|metastas|hcc|hepatocellular|tumou?r|neoplasm|malignan|carcinoma",
        "benign": r"cyst|h[ae]mangioma|hamartoma|fnh|focal nodular hyperplasia",
    },
    "pancreas": {
        "organ": r"pancrea|pankreas",
        "malign": r"mass|adenocarcinoma|tumou?r|neoplasm|carcinoma|cancer|malignan",
        "benign": r"cyst|ipmn|pseudocyst|cystadenoma|serous",
    },
}
# Malign terim geçse de bunlar varsa KANSER DEĞİL -> POS sayma (kullanıcı geri bildirimi):
NONMALIG = r"abscess|steatos|fatty|atroph|post[- ]?op|post[- ]?surg|postsurg|resect|lipoma|inflammat|pancreatitis"
HEDGE = r"cannot exclude|indeterminate|too small to characteri|follow[- ]?up|nonspecific|equivocal|possible|cannot be excluded|not characteri"
NORMAL = r"normal|unremarkable|no focal|no suspicious|within normal limits|no evidence of (mass|lesion|tumou?r)"
NEG_BEFORE = r"(no|without|negative for|absence of|no evidence of|free of)\s+\w*\s*$"


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<!\d)[.\n](?!\d)", str(text)) if s.strip()]


def _negated(sentence: str, m: re.Match) -> bool:
    return re.search(NEG_BEFORE, sentence[:m.start()].lower() + " ") is not None


def classify_finding(text: str, organ: str) -> tuple[str, str]:
    """Rapordan organ etiketi. Döner: (POS|NEG|INDETERMINATE|SKIP, sebep)."""
    t = TERMS[organ]
    has_pos = has_benign = has_normal = has_hedge = False
    for s in _sentences(text):
        sl = s.lower()
        if not re.search(t["organ"], sl):
            continue
        hedged = re.search(HEDGE, sl) is not None
        m = re.search(t["malign"], sl)
        if m:
            if _negated(s, m):
                has_normal = True
            elif re.search(NONMALIG, sl) or re.search(t["benign"], sl):
                has_benign = True   # kitle var ama açıklaması benign (apse/yağ/atrofi/kist) → NEG
            elif hedged:
                has_hedge = True
            else:
                has_pos = True
        if re.search(t["benign"], sl):
            has_benign = True
        if re.search(NORMAL, sl):
            has_normal = True
    if has_pos:
        return "POS", "organ+malign (olumsuz değil)"
    if has_hedge:
        return "INDETERMINATE", "lezyon + belirsizlik ifadesi"
    if has_benign:
        return "NEG", "yalnız benign (kist) → NEG"
    if has_normal:
        return "NEG", "organ normal / no focal lesion"
    return "SKIP", "organ hakkında net ifade yok"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def select(reports_path, organ, seed=42, buffer=30, metadata_path=None,
           require_contrast=False, col_id="study id", col_text="Findings",
           col_contrast="contrast"):
    import pandas as pd
    df = pd.read_excel(reports_path)[[col_id, col_text]].copy()
    if metadata_path and require_contrast:
        meta = pd.read_csv(metadata_path)[[col_id, col_contrast]]
        df = df.merge(meta, on=col_id, how="left")
        df = df[df[col_contrast].astype(str).str.lower().isin(
            ["1", "true", "with", "yes", "contrast"])]

    order = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # seed'li
    pos, neg, flow = [], [], []
    for i, row in order.iterrows():
        sid, text = row[col_id], row[col_text]
        label, reason = classify_finding(text, organ)
        rec = {"draw_order": int(i), "study_id": sid, "regex_label": label, "sebep": reason}
        if label == "POS" and len(pos) < buffer:
            pos.append((i, sid, text)); rec["aday"] = "POS"
        elif label == "NEG" and len(neg) < buffer:
            neg.append((i, sid, text)); rec["aday"] = "NEG"
        else:
            rec["aday"] = ""   # INDETERMINATE/SKIP/kota-dolu: kohorta girmez, loglanır
        flow.append(rec)
        if len(pos) >= buffer and len(neg) >= buffer:
            break
    return pos, neg, flow, len(order)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("organ", choices=list(TERMS))
    ap.add_argument("--reports", required=True)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--out", default=".")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--buffer", type=int, default=30)     # 20 hedef + eleme payı
    ap.add_argument("--target", type=int, default=20)
    ap.add_argument("--require-contrast", action="store_true")
    a = ap.parse_args(argv)

    import pandas as pd
    pos, neg, flow, n_screened = select(a.reports, a.organ, a.seed, a.buffer,
                                        a.metadata, a.require_contrast)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    cand = [(o, s, t, "POS") for (o, s, t) in pos] + [(o, s, t, "NEG") for (o, s, t) in neg]
    cand.sort(key=lambda x: x[0])   # draw_order
    onay = pd.DataFrame({
        "draw_order": [c[0] for c in cand], "study id": [c[1] for c in cand],
        "regex_label": [c[3] for c in cand], "Findings": [c[2] for c in cand],
        "onay_pos_neg": "", "cap_mm": "", "kaynak_cumle": "",
    })
    onay.to_csv(out / f"onay_{a.organ}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(flow).to_csv(out / f"akis_{a.organ}.csv", index=False, encoding="utf-8-sig")

    prov = {
        "organ": a.organ, "seed": a.seed, "buffer": a.buffer, "target": a.target,
        "require_contrast": a.require_contrast, "terms_version": TERMS_VERSION,
        "reports_sha256": _sha256(a.reports), "n_screened": n_screened,
        "n_pos_aday": len(pos), "n_neg_aday": len(neg),
        "tarih": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (out / f"provenance_{a.organ}.json").write_text(
        json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[{a.organ}] aday POS={len(pos)} NEG={len(neg)} (buffer={a.buffer}) | "
          f"tarandı={n_screened}")
    print(f"  -> onay_{a.organ}.csv (doldur) · akis_{a.organ}.csv (STARD) · "
          f"provenance_{a.organ}.json")
    print(f"  Sonra: onay dosyasını doldur -> python scripts/finalize_cohort.py {a.organ} "
          f"--onay onay_{a.organ}.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
