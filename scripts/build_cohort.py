"""
build_cohort.py — yerel POS/NEG JSON'larından TEK cohort.csv üretir (C33, F12).

Her aşamanın aynı manifestten çalışması için (glob yerine) tek kaynak vaka
listesi. Veri repoya dahil değil (RUA); bu script Drive/Colab'daki JSON'ları
işler ve manifest'i config.LAYOUT.manifest konumuna yazar.

Kullanım:
    python scripts/build_cohort.py \
        karaciger=Task003_Liver=/content/drive/.../liver_cases.json \
        pankreas=Task007_Pancreas=/content/drive/.../pancreas_cases.json

JSON biçimi: {"POS": [...sid...], "NEG": [...sid...]}
"""

import sys

from pipeline import manifest as man
from pipeline.config import LAYOUT


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    all_cases: list[man.Case] = []
    for spec in argv:
        try:
            organ, task, path = spec.split("=", 2)
        except ValueError:
            print(f"Hatalı argüman: {spec} (organ=task=jsonpath bekleniyor)")
            return 2
        cases = man.from_pos_neg_json(path, organ=organ, task=task,
                                      data_dir=LAYOUT.data)
        print(f"  {organ}: {len(cases)} vaka ({path})")
        all_cases.extend(cases)

    rapor = man.validate(all_cases)
    if rapor["patient_cakisma"]:
        print("⚠️ Aynı kişi birden çok task'ta (bağımsızlık notu):",
              list(rapor["patient_cakisma"].keys()))
    if rapor["tekrar_key"]:
        print("⚠️ Tekrar eden study:task:", rapor["tekrar_key"])

    out = LAYOUT.manifest
    man.save_csv(all_cases, out)
    print(f"✅ Manifest yazıldı: {out} | toplam {rapor['toplam']} | grup {rapor['grup']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
