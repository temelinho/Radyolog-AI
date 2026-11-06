"""
config.py — merkezî yapılandırma (F11, F19, F20).

Tüm yollar, sürümler ve çalışma parametreleri TEK yerde. Sabit kodlanmış
`C:\\Users\\ASUS\\...` yolları kaldırıldı; her şey ortam değişkeni + güvenli
varsayılan üzerinden gelir. Import yan etkisizdir (dizin oluşturmaz, model
indirmez, API çağırmaz).

Ortam değişkenleri (hepsi opsiyonel; yoksa varsayılan kullanılır):
  RADYOLOG_DATA_DIR        Ham BT / maske verisinin kök dizini
  RADYOLOG_OUTPUT_DIR      run_<id> çıktılarının yazılacağı kök
  RADYOLOG_MANIFEST        Vaka manifest dosyası (CSV)
  NNUNET_RESULTS           nnU-Net ağırlık dizini
  ANTHROPIC_API_KEY        (yalnız raporlama aşamasında okunur)
  GOOGLE_API_KEY           (yalnız raporlama aşamasında okunur)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Sürümler — şema/kod/prompt değişince ARTIRILIR. Cache ve provenance bunları
# kullanır (F10): aynı girdi + farklı sürüm => yeniden hesap.
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "2.0.0"      # bulgular.json şeması
PIPELINE_VERSION = "2.0.0"    # ölçüm/değerlendirme kodu
PROMPT_VERSION = "2.0.0"      # ortak rapor promptu


# ---------------------------------------------------------------------------
# Ölçüm / değerlendirme sabitleri — "sihirli sayı" dağılmasın (F13, F14).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MeasurementConfig:
    # Lezyon bağlantılılığı: 3B tam komşuluk (26). scipy varsayılan (6) DEĞİL (F14).
    connectivity: int = 3  # scipy.ndimage: 1=6-komşu, 2=18, 3=26-komşu
    # Lezyon filtresi FİZİKSEL hacme göre (voxel sayısına göre değil) (F14).
    min_lesion_volume_cm3: float = 0.1
    # Tespit eşiği: aksiyel en uzun çap (mm). RAW değere uygulanır (F13).
    detection_diameter_mm: float = 10.0
    # Tümör–damar "temas" eşiği (mm): bu mesafenin altı temas sayılır.
    vessel_contact_mm: float = 2.0


@dataclass(frozen=True)
class GeometryConfig:
    # İki hacmin fiziksel olarak "aynı grid" sayılması için tolerans (mm).
    affine_tol_mm: float = 1e-3
    # Affine'in ne kadar "oblik" olursa kalite bayrağı atılacağı (derece).
    max_oblique_deg: float = 1.0


@dataclass(frozen=True)
class Paths:
    data_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("RADYOLOG_DATA_DIR", Path.cwd() / "data")))
    output_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("RADYOLOG_OUTPUT_DIR", Path.cwd() / "outputs")))
    manifest: Path = field(default_factory=lambda: Path(
        os.environ.get("RADYOLOG_MANIFEST", Path.cwd() / "protocol" / "cohort.csv")))
    nnunet_results: Path = field(default_factory=lambda: Path(
        os.environ.get("NNUNET_RESULTS", Path.cwd() / "nnUNet_results")))


MEASUREMENT = MeasurementConfig()
GEOMETRY = GeometryConfig()
PATHS = Paths()


# ---------------------------------------------------------------------------
# Durum sözlüğü (F05, F10) — "ölçülemedi" ile "0" ASLA karışmasın.
# ---------------------------------------------------------------------------
class Status:
    SUCCESS = "success"
    EMPTY_MASK = "empty_mask"      # maske var ama hedef yapı yok
    NOT_ASSESSED = "not_assessed"  # girdi/yöntem yetersiz; ölçüm yapılmadı
    FAILED = "failed"              # süreç hata ile bitti

    ALL = (SUCCESS, EMPTY_MASK, NOT_ASSESSED, FAILED)


def get_api_key(provider: str) -> str | None:
    """Yalnız raporlama aşamasında çağrılır; import zamanında DEĞİL."""
    env = {"anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY"}.get(provider)
    return os.environ.get(env) if env else None
