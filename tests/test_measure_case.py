"""measure_case + K2/K4 testleri — oblik kapısı, fiziksel uzay, bağlantılılık."""

import unittest

import numpy as np

from pipeline import measurements as M
from pipeline.imaging import Volume


def _aff(spacing=(1, 1, 1), origin=(0, 0, 0)):
    a = np.eye(4)
    a[0, 0], a[1, 1], a[2, 2] = spacing
    a[:3, 3] = origin
    return a


def _two_cubes():
    d = np.zeros((40, 40, 12), dtype=np.uint8)
    d[2:7, 2:7, 2:7] = 2      # kup 1 (etiket 2)
    d[30:35, 30:35, 5:10] = 2  # kup 2
    return d


class TestMeasureCaseK2(unittest.TestCase):
    def test_normal_case(self):
        tv = Volume(_two_cubes(), _aff())
        r = M.measure_case(tv, organ="karaciger")
        self.assertTrue(r["geometry"]["measurable"])
        self.assertEqual(r["quality_flags"], [])
        self.assertEqual(r["summary"]["lezyon_sayisi"], 2)
        self.assertIsNotNone(r["lezyon_sayisi_saglamlik"])

    def test_oblique_gate_blocks_diameter(self):
        """Güçlü oblik hacimde aksiyel çap üretilmez (K2)."""
        a = _aff()
        c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
        a[:2, :2] = [[c, -s], [s, c]]  # 30° in-plane rotasyon
        tv = Volume(_two_cubes(), a)
        r = M.measure_case(tv, organ="karaciger")
        self.assertFalse(r["geometry"]["measurable"])
        self.assertTrue(any("oblik" in f for f in r["quality_flags"]))
        self.assertIsNone(r["summary"]["en_buyuk_cap_lezyon"])

    def test_vessel_different_grid_world_distance(self):
        """Damar farklı origin'de olsa da mesafe dünya koordinatında (F03)."""
        tumor = np.zeros((20, 20, 6), dtype=np.uint8)
        tumor[5:9, 5:9, 2:4] = 2
        tv = Volume(tumor, _aff())
        vessel = np.zeros((20, 20, 6), dtype=np.uint8)
        vessel[5:9, 5:9, 2:4] = 1
        vv = Volume(vessel, _aff(origin=(2, 0, 0)))  # 2 mm ötelenmiş
        r = M.measure_case(tv, organ="pankreas", vessel_vols={"aorta": vv})
        self.assertIn("aorta", r["vessels"])
        self.assertIsNotNone(r["vessels"]["aorta"]["mesafe_mm"])


class TestConnectivityK4(unittest.TestCase):
    def test_sensitivity_reports_both(self):
        d = _two_cubes()
        sens = M.lesion_count_sensitivity(d, np.array([1.0, 1.0, 1.0]))
        self.assertIn("n_6komsu", sens)
        self.assertIn("n_26komsu", sens)
        self.assertEqual(sens["n_6komsu"], 2)
        self.assertEqual(sens["n_26komsu"], 2)


if __name__ == "__main__":
    unittest.main()
