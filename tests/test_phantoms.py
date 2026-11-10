"""
test_phantoms.py — ANALİTİK fantom doğrulaması (ölçüm kodunun KESİN kanıtı).

Cevabı kapalı-form formülle bilinen yapay maskeler üretir ve pipeline'ın aynı
değeri (voxel ayrıklaştırma hatası içinde) verdiğini gösterir. Böylece "maske
verildiğinde çap/hacim/sayı doğru mu?" sorusu matematiksel olarak yanıtlanır.
Maskenin gerçeği yansıtıp yansıtmadığı AYRI bir soru (radyolog cap_mm kıyası).
"""

from __future__ import annotations

import numpy as np

from pipeline import measurements as M


def _grid(shape, spacing):
    sx, sy, sz = spacing
    i, j, k = np.indices(shape)
    return i * sx, j * sy, k * sz


def _rel_err(a, b):
    return abs(a - b) / b


def test_kure_cap_ve_hacim():
    """Küre R: aksiyel en uzun çap = 2R (tam); hacim ≈ 4/3·π·R³ (ayrıklaştırma)."""
    sp = np.array([0.8, 0.8, 3.0])
    shape = (120, 120, 40)
    cx, cy, cz = 60 * 0.8, 60 * 0.8, 20 * 3.0
    R = 20.0
    X, Y, Z = _grid(shape, sp)
    mask = (((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) <= R ** 2).astype(np.uint8)

    lesions = M.extract_lesions(mask, sp)
    assert len(lesions) == 1
    les = lesions[0]
    # Çap: tek voxel içinde (anizotropik x/y spacing 0.8 mm)
    assert abs(les["cap_mm"] - 2 * R) <= 0.8
    # Hacim: voxel ayrıklaştırma hatası < %3
    assert _rel_err(les["hacim_cm3"], (4 / 3) * np.pi * R ** 3 / 1000) < 0.03


def test_elipsoid_en_uzun_aksiyel_cap():
    """Elipsoid yarı-eksen a>b (xy düzleminde): en uzun aksiyel çap = 2a."""
    sp = np.array([1.0, 1.0, 2.0])
    shape = (140, 80, 40)
    cx, cy, cz = 70.0, 40.0, 40.0
    a, b, c = 30.0, 15.0, 10.0
    X, Y, Z = _grid(shape, sp)
    mask = ((((X - cx) / a) ** 2 + ((Y - cy) / b) ** 2 + ((Z - cz) / c) ** 2) <= 1
            ).astype(np.uint8)

    les = M.extract_lesions(mask, sp)[0]
    assert abs(les["cap_mm"] - 2 * a) <= 1.0            # 2a, 2b/2c DEĞİL
    assert _rel_err(les["hacim_cm3"], (4 / 3) * np.pi * a * b * c / 1000) < 0.03


def test_iki_ayri_kure_sayisi_ve_caplari():
    """İki ayrık küre: lezyon sayısı = 2; çaplar = 2R1, 2R2 (büyükten küçüğe)."""
    sp = np.array([1.0, 1.0, 1.0])
    shape = (120, 60, 60)
    X, Y, Z = _grid(shape, sp)
    s1 = ((X - 30) ** 2 + (Y - 30) ** 2 + (Z - 30) ** 2) <= 12 ** 2
    s2 = ((X - 90) ** 2 + (Y - 30) ** 2 + (Z - 30) ** 2) <= 8 ** 2
    mask = (s1 | s2).astype(np.uint8)

    lesions = M.extract_lesions(mask, sp)
    assert len(lesions) == 2
    assert abs(lesions[0]["cap_mm"] - 24.0) <= 1.0     # en büyük çap önce (sıralama)
    assert abs(lesions[1]["cap_mm"] - 16.0) <= 1.0


def test_anizotropik_spacing_fiziksel_cap():
    """Çap FİZİKSEL mm'dir: voxel sayısı aynı, spacing farklıysa çap spacing ölçekli."""
    shape = (80, 80, 20)
    # Tek kesitte 20 voxel çaplı disk; x spacing 2 mm => fiziksel çap ~40 mm
    i, j, k = np.indices(shape)
    disk = ((i - 40) ** 2 + (j - 40) ** 2 <= 10 ** 2) & (k == 10)
    mask = disk.astype(np.uint8)
    les = M.extract_lesions(mask, np.array([2.0, 2.0, 3.0]))[0]
    # 10 voxel yarıçap * 2 mm = 20 mm yarıçap => 40 mm çap (±1 voxel=2mm)
    assert abs(les["cap_mm"] - 40.0) <= 2.0
