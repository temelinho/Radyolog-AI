"""
main.py — nnU-Net 5-fold çıkarım CLI (pipeline.segmentation üzerinden).

Eski main.py kurulu nnU-Net dosyasını import ANINDA metinle yamалıyor ve yalnız
Task007'yi sabit çalıştırıyordu (C12/C13). Artık:
  * Yama YOK (kurulu paket değiştirilmez).
  * Tek protokol config'ten (5-fold).
  * Task parametreyle; herhangi bir organ.

Kullanım:
    python main.py <task> <input_dir> <output_dir>
    ör: python main.py Task003_Liver ./data ./masks
"""

import sys

from pipeline import segmentation
from pipeline.config import Status


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("Kullanım: python main.py <task> <input_dir> <output_dir>")
        print("task örn: Task003_Liver, Task007_Pancreas")
        return 2
    task, in_dir, out_dir = argv[0], argv[1], argv[2]
    res = segmentation.run_nnunet(in_dir, out_dir, task)
    print(f"[{res.status}] {task} -> {out_dir}")
    if res.status != Status.SUCCESS:
        print("Detay:", res.detail)
        return 1
    print("Maskeler:", res.detail.get("maskeler"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
