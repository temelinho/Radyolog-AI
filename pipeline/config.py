"""
config.py — merkezî yapılandırma (F11, F19, F20).

Tüm yollar, sürümler ve çalışma parametreleri TEK yerde. Sabit kodlanmış
`C:\\Users\\ASUS\\...` yolları kaldırıldı. Hedef ortam: **Google Colab + Drive.**

Yol çözümü (öncelik sırası):
  1) RADYOLOG_BASE ortam değişkeni (ör. /content/drive/MyDrive/radyolog_ai)
  2) Colab'da Drive bağlıysa otomatik: /content/drive/MyDrive/radyolog_ai
  3) Aksi halde geçerli dizin altında ./radyolog_ai

Tüm alt klasörler bu TEK kökten türetilir; böylece Colab, yerel ve başka
makinede aynı kod çalışır. Import yan etkisizdir (dizin OLUŞTURMAZ).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Sürümler — şema/kod/prompt değişince ARTIRILIR (cache + provenance) (F10).
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "2.0.0"
PIPELINE_VERSION = "2.0.0"
PROMPT_VERSION = "2.0.0"


# ---------------------------------------------------------------------------
# Kök dizin çözümü — Colab/Drive öncelikli (C9, F11)
# ---------------------------------------------------------------------------
_DRIVE_DEFAULT = "/content/drive/MyDrive/radyolog_ai"


def base_dir() -> Path:
    """Projenin TEK kök dizini. Env > Colab Drive > yerel ./radyolog_ai."""
    env = os.environ.get("RADYOLOG_BASE")
    if env:
        return Path(env)
    # Colab'da Drive bağlıysa onu kullan
    drive = Path("/content/drive/MyDrive")
    if drive.exists():
        return Path(_DRIVE_DEFAULT)
    return Path.cwd() / "radyolog_ai"


@dataclass(frozen=True)
class Layout:
    """Tek kökten türeyen klasör yapısı. 'Yeni klasör' = bu kök."""
    base: Path = field(default_factory=base_dir)

    @property
    def data(self) -> Path: return self.base / "data"                 # ham BT (.nii.gz)
    @property
    def masks(self) -> Path: return self.base / "masks"               # segmentasyon çıktısı
    @property
    def vessels(self) -> Path: return self.base / "vessels"           # TotalSegmentator
    @property
    def findings(self) -> Path: return self.base / "findings"         # bulgular.json
    @property
    def reports(self) -> Path: return self.base / "reports"           # LLM raporları
    @property
    def outputs(self) -> Path: return self.base / "outputs"           # değerlendirme/tablolar
    @property
    def nnunet_results(self) -> Path:
        return Path(os.environ.get("NNUNET_RESULTS", self.base / "nnUNet_results"))
    @property
    def manifest(self) -> Path:
        return Path(os.environ.get("RADYOLOG_MANIFEST", self.base / "protocol" / "cohort.csv"))
    @property
    def reference(self) -> Path:
        return self.base / "protocol" / "reference_gt.csv"

    def all_dirs(self) -> list[Path]:
        return [self.data, self.masks, self.vessels, self.findings,
                self.reports, self.outputs, self.base / "protocol"]


# ---------------------------------------------------------------------------
# Segmentasyon ayarı (C8, C10, C13, F20) — tek protokol, 5-fold SABİT.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SegmentationConfig:
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)   # 5-fold ensemble; tek fold DEĞİL
    configuration: str = "3d_fullres"
    tta: bool = True
    # nnU-Net yürütülebiliri: env > PATH'teki komut. Sabit Windows yolu YOK (F11).
    nnunet_cmd: str = os.environ.get("NNUNET_PREDICT", "nnUNetv2_predict")
    totalseg_cmd: str = os.environ.get("TOTALSEG_CMD", "TotalSegmentator")
    totalseg_roi: tuple[str, ...] = ("aorta", "portal_vein_and_splenic_vein",
                                     "inferior_vena_cava")
    # Organ -> nnU-Net task. Tek kaynak (app ve runner aynısını kullanır).
    organ_task: dict = field(default_factory=lambda: {
        "karaciger": "Task003_Liver",
        "pankreas": "Task007_Pancreas",
    })
    # Maske etiketleri: 0=arka plan, 1=organ, 2=tümör
    tumor_label: int = 2
    organ_label: int = 1


# ---------------------------------------------------------------------------
# Ölçüm / geometri sabitleri (F13, F14)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MeasurementConfig:
    connectivity: int = 3               # 26-komşuluk (scipy 3); 6 DEĞİL (F14)
    min_lesion_volume_cm3: float = 0.1  # fiziksel hacim eşiği (voxel sayısı değil)
    detection_diameter_mm: float = 10.0  # RAW değere uygulanır (F13)
    vessel_contact_mm: float = 2.0


@dataclass(frozen=True)
class GeometryConfig:
    affine_tol_mm: float = 1e-3
    max_oblique_deg: float = 1.0


LAYOUT = Layout()
SEGMENTATION = SegmentationConfig()
MEASUREMENT = MeasurementConfig()
GEOMETRY = GeometryConfig()


# ---------------------------------------------------------------------------
# Durum sözlüğü (F05, F10) — "ölçülemedi" ile "0" ASLA karışmasın.
# ---------------------------------------------------------------------------
class Status:
    SUCCESS = "success"
    EMPTY_MASK = "empty_mask"
    NOT_ASSESSED = "not_assessed"
    FAILED = "failed"
    ALL = (SUCCESS, EMPTY_MASK, NOT_ASSESSED, FAILED)


def get_api_key(provider: str) -> str | None:
    """Yalnız raporlama aşamasında çağrılır; import zamanında DEĞİL."""
    env = {"anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY"}.get(provider)
    return os.environ.get(env) if env else None
