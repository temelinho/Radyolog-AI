"""measurements.py testleri — F01, F04, F14."""

import unittest

import numpy as np

from pipeline import measurements as M


def disk(shape2d, center, radius):
    ii, jj = np.indices(shape2d)
    return ((ii - center[0]) ** 2 + (jj - center[1]) ** 2) <= radius ** 2


class TestAxialDiameter(unittest.TestCase):
    def test_collinear_fallback(self):
        """Doğrusal noktalarda ConvexHull başarısız; kaba kuvvet uç noktaları bulur (F14)."""
        m = np.zeros((10, 10, 1), dtype=np.uint8)
        m[0:4, 0, 0] = 1  # (0,0),(1,0),(2,0),(3,0)
        d = M.axial_longest_diameter_mm(m, np.array([1.0, 1.0, 1.0]))
        self.assertAlmostEqual(d, 3.0, places=6)

    def test_spacing_anisotropic(self):
        m = np.zeros((10, 10, 1), dtype=np.uint8)
        m[0:4, 0, 0] = 1
        d = M.axial_longest_diameter_mm(m, np.array([2.0, 1.0, 1.0]))
        self.assertAlmostEqual(d, 6.0, places=6)

    def test_sphere_diameter(self):
        r = 5
        n = 2 * r + 3
        ii, jj, kk = np.indices((n, n, n))
        c = n // 2
        sph = ((ii - c) ** 2 + (jj - c) ** 2 + (kk - c) ** 2) <= r ** 2
        d = M.axial_longest_diameter_mm(sph.astype(np.uint8), np.array([1.0, 1.0, 1.0]))
        self.assertTrue(9.0 <= d <= 11.0, f"çap {d}")


class TestLesionsF01(unittest.TestCase):
    """En büyük ÇAPLI lezyon ile en büyük HACİMLİ lezyon farklı olabilir (F01)."""

    def _two_lesions(self):
        m = np.zeros((40, 40, 10), dtype=np.uint8)
        m[2, 5:25, 2] = 1          # ince uzun çubuk: çap ~19, hacim 20 voxel
        m[30:35, 30:35, 4:9] = 1   # kompakt küp: çap ~5.66, hacim 125 voxel
        return m

    def test_count_and_split(self):
        m = self._two_lesions()
        lez = M.extract_lesions(m, np.array([1.0, 1.0, 1.0]), min_volume_cm3=0.0)
        self.assertEqual(len(lez), 2)
        s = M.summarize_lesions(lez)
        self.assertEqual(s["lezyon_sayisi"], 2)
        # En büyük çaplı != en büyük hacimli
        self.assertNotEqual(s["en_buyuk_cap_lezyon"]["id"],
                            s["en_buyuk_hacim_lezyon"]["id"])
        self.assertGreater(s["en_buyuk_cap_lezyon"]["cap_mm"], 15)
        self.assertGreater(s["en_buyuk_hacim_lezyon"]["hacim_cm3"],
                           s["en_buyuk_cap_lezyon"]["hacim_cm3"])

    def test_deterministic(self):
        m = self._two_lesions()
        a = M.extract_lesions(m, np.array([1.0, 1.0, 1.0]), min_volume_cm3=0.0)
        b = M.extract_lesions(m, np.array([1.0, 1.0, 1.0]), min_volume_cm3=0.0)
        self.assertEqual(a, b)

    def test_volume_filter(self):
        m = np.zeros((20, 20, 5), dtype=np.uint8)
        m[0, 0, 0] = 1  # 1 voxel = 0.001 cm3 < varsayılan 0.1 eşiği
        lez = M.extract_lesions(m, np.array([1.0, 1.0, 1.0]))
        self.assertEqual(lez, [])

    def test_empty_summary(self):
        s = M.summarize_lesions([])
        self.assertEqual(s["lezyon_sayisi"], 0)
        self.assertIsNone(s["en_buyuk_cap_lezyon"])


class TestVesselWrapF04(unittest.TestCase):
    """Uzak kesitteki damar voxelleri bir kesitin yerel açısını değiştirmemeli (F04)."""

    def test_remote_slice_does_not_change_angle(self):
        shape = (41, 41, 3)
        sp = np.array([1.0, 1.0, 1.0])
        vessel1 = np.zeros(shape, dtype=np.uint8)
        vessel1[:, :, 0] = disk((41, 41), (20, 20), 6)
        tumor = np.zeros(shape, dtype=np.uint8)
        tumor[27:33, 14:27, 0] = 1  # diskin +x tarafına bitişik blok

        a1 = M.vessel_wrap_angle_deg(tumor, vessel1, sp)["sarma_derece"]

        vessel2 = vessel1.copy()
        vessel2[:, :, 1] = disk((41, 41), (20, 20), 6)  # UZAK kesit (tümörsüz)
        a2 = M.vessel_wrap_angle_deg(tumor, vessel2, sp)["sarma_derece"]

        self.assertIsNotNone(a1)
        self.assertEqual(a1, a2)

    def test_full_surround_vs_one_side(self):
        shape = (41, 41, 1)
        sp = np.array([1.0, 1.0, 1.0])
        vessel = np.zeros(shape, dtype=np.uint8)
        vessel[:, :, 0] = disk((41, 41), (20, 20), 6)

        # Tam çevreleme: diski saran halka
        ring = disk((41, 41), (20, 20), 9) & ~disk((41, 41), (20, 20), 6)
        tumor_full = np.zeros(shape, dtype=np.uint8)
        tumor_full[:, :, 0] = ring
        ang_full = M.vessel_wrap_angle_deg(tumor_full, vessel, sp)["sarma_derece"]
        self.assertGreater(ang_full, 300)

        # Tek taraf
        tumor_side = np.zeros(shape, dtype=np.uint8)
        tumor_side[27:31, 16:25, 0] = 1
        ang_side = M.vessel_wrap_angle_deg(tumor_side, vessel, sp)["sarma_derece"]
        self.assertGreater(ang_side, 0)
        self.assertLess(ang_side, 190)

    def test_empty_vessel_not_assessed(self):
        shape = (10, 10, 2)
        sp = np.array([1.0, 1.0, 1.0])
        tumor = np.ones(shape, dtype=np.uint8)
        empty = np.zeros(shape, dtype=np.uint8)
        r = M.vessel_wrap_angle_deg(tumor, empty, sp)
        self.assertIsNone(r["sarma_derece"])


class TestVesselDistance(unittest.TestCase):
    def test_empty_vessel_none(self):
        t = np.ones((5, 5, 5), dtype=np.uint8)
        e = np.zeros((5, 5, 5), dtype=np.uint8)
        aff = np.eye(4)
        self.assertIsNone(M.vessel_distance_mm(t, aff, e, aff)["mesafe_mm"])


if __name__ == "__main__":
    unittest.main()
