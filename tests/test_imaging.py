"""imaging.py testleri — fiziksel uzay doğruluğu (F03)."""

import unittest

import numpy as np

from pipeline import imaging
from pipeline.imaging import Volume


def _affine(spacing=(1, 1, 1), origin=(0, 0, 0)):
    a = np.eye(4)
    a[0, 0], a[1, 1], a[2, 2] = spacing
    a[:3, 3] = origin
    return a


class TestGeometry(unittest.TestCase):
    def test_spacing_from_affine(self):
        a = _affine(spacing=(1.25, 0.66, 0.66))
        np.testing.assert_allclose(imaging.spacing_from_affine(a), [1.25, 0.66, 0.66])

    def test_oblique_angle_identity_zero(self):
        self.assertAlmostEqual(imaging.oblique_angle_deg(np.eye(4)), 0.0, places=6)

    def test_oblique_angle_45(self):
        a = np.eye(4)
        c, s = np.cos(np.radians(45)), np.sin(np.radians(45))
        a[:2, :2] = [[c, -s], [s, c]]
        self.assertAlmostEqual(imaging.oblique_angle_deg(a), 45.0, places=4)

    def test_geometry_close_true_false(self):
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        va = Volume(arr, _affine(origin=(0, 0, 0)))
        vb = Volume(arr, _affine(origin=(0, 0, 0)))
        vc = Volume(arr, _affine(origin=(100, 0, 0)))
        self.assertTrue(imaging.geometry_close(va, vb))
        self.assertFalse(imaging.geometry_close(va, vc))


class TestDistanceF03(unittest.TestCase):
    """F03 tekrar üretimi: 100 mm ötelenmiş maske mesafesi 0 OLMAMALI."""

    def test_translated_mask_distance_is_not_zero(self):
        a = np.zeros((3, 3, 3), dtype=np.uint8)
        a[0, 0, 0] = 1
        b = np.zeros((3, 3, 3), dtype=np.uint8)
        b[0, 0, 0] = 1
        aff_a = _affine(origin=(0, 0, 0))
        aff_b = _affine(origin=(100, 0, 0))  # dünyada 100 mm ötede

        d = imaging.min_surface_distance_mm(a, aff_a, b, aff_b)
        self.assertIsNotNone(d)
        self.assertAlmostEqual(d, 100.0, places=6)

    def test_same_grid_overlap_zero(self):
        a = np.zeros((3, 3, 3), dtype=np.uint8)
        a[1, 1, 1] = 1
        d = imaging.min_surface_distance_mm(a, _affine(), a, _affine())
        self.assertEqual(d, 0.0)

    def test_empty_mask_returns_none_not_zero(self):
        a = np.zeros((3, 3, 3), dtype=np.uint8)
        a[0, 0, 0] = 1
        empty = np.zeros((3, 3, 3), dtype=np.uint8)
        self.assertIsNone(imaging.min_surface_distance_mm(a, _affine(), empty, _affine()))


class TestResample(unittest.TestCase):
    def test_identity_resample_unchanged(self):
        src = np.zeros((4, 4, 4), dtype=np.uint8)
        src[1, 2, 3] = 2
        v = Volume(src, _affine())
        out = imaging.resample_mask_to_reference(v, v)
        np.testing.assert_array_equal(out, src)

    def test_translated_reference_grid(self):
        src = np.zeros((4, 4, 4), dtype=np.uint8)
        src[2, 0, 0] = 2  # src world (2,0,0)
        v_src = Volume(src, _affine(origin=(0, 0, 0)))
        # ref origin +1 in x: ref voxel (1,0,0) -> world (1,0,0) -> src idx (1,0,0)
        # src etiketi (2,0,0) dünyada (2,0,0); ref'te voxel (1,0,0) dünyada (1,0,0)=0,
        # ref voxel (1,0,0)->world(1,0,0); src world(2,0,0) ref voxel (1,0,0)? world2 -> ref idx 1. Evet.
        v_ref = Volume(np.zeros((4, 4, 4), np.uint8), _affine(origin=(1, 0, 0)))
        out = imaging.resample_mask_to_reference(v_src, v_ref)
        # src etiketi dünya (2,0,0); ref'te bu voxel idx (2-1,0,0)=(1,0,0)
        self.assertEqual(out[1, 0, 0], 2)
        self.assertEqual(out.sum(), 2)


if __name__ == "__main__":
    unittest.main()
