"""
Radyolog AI — ortak çekirdek (pipeline).

Bu paket, UI'dan (Streamlit) ve notebook'lardan BAĞIMSIZ, import edilebilir
çekirdektir. Amaç: ölçüm / klinik kural / rapor / değerlendirme mantığının
TEK kaynaktan gelmesi (29 Ekim 2026 incelemesi F01–F23).

Önemli: Bu paketi import etmek GPU çıkarımı, model indirme veya API çağrısı
YAPMAZ. Yan etkisiz import (F11).
"""

from . import config  # noqa: F401

SCHEMA_VERSION = config.SCHEMA_VERSION
PIPELINE_VERSION = config.PIPELINE_VERSION
