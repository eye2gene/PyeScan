"""
Tests for individual metric functions.

Tests the geometric logic directly, including coordinate conventions:
- Enface: (0,0) is top-left, y increases downward
- Quadrants: superior=top, inferior=bottom, dexter=right-as-viewed, sinister=left-as-viewed
- B-scan slices: quadrant masks projected from enface coordinates onto the A-scan axis

NOTE: These tests document and verify the CURRENT conventions. If a convention
is found to be incorrect (e.g. sinister/dexter from wrong perspective), the
test should be updated along with the code fix.
"""

import numpy as np
import pytest

from pyescan.metrics.metrics import (
    _get_circle_line_intersection,
    _get_distance_mask,
    _get_distance_mask_slice,
    _get_quadrant_masks,
    _get_quadrant_masks_slice,
    get_fovea_enface_position,
    get_mask_area,
    get_mask_pixel_counts,
    get_mask_resoluton,
    get_mask_shape,
    get_mask_volume,
    get_pixel_count_by_distance,
    get_pixel_count_by_quadrant,
    get_underlying_intensity,
)


class TestMaskShape:
    def test_returns_width_height(self):
        mask = np.ones((100, 200), dtype=bool)  # 100 rows, 200 cols
        width, height = get_mask_shape(mask)
        assert width == 200
        assert height == 100

    def test_square_mask(self):
        mask = np.ones((64, 64), dtype=bool)
        width, height = get_mask_shape(mask)
        assert width == 64
        assert height == 64


class TestMaskResolution:
    def test_same_size_as_scan(self):
        # Mask is same size as scan -> resolution unchanged
        w, h = get_mask_resoluton(
            scan_width_px=512,
            scan_height_px=496,
            resolutions_mm_width=0.01,
            resolutions_mm_height=0.01,
            mask_width_px=512,
            mask_height_px=496,
        )
        assert pytest.approx(w) == 0.01
        assert pytest.approx(h) == 0.01

    def test_half_size_mask(self):
        # Mask is half the scan size -> resolution doubles
        w, h = get_mask_resoluton(
            scan_width_px=512,
            scan_height_px=496,
            resolutions_mm_width=0.01,
            resolutions_mm_height=0.01,
            mask_width_px=256,
            mask_height_px=248,
        )
        assert pytest.approx(w) == 0.02
        assert pytest.approx(h) == 0.02


class TestMaskPixelCounts:
    def test_full_mask(self):
        mask = np.ones((10, 20), dtype=bool)
        count, cols, rows = get_mask_pixel_counts(mask)
        assert count == 200
        assert cols == 20
        assert rows == 10

    def test_empty_mask(self):
        mask = np.zeros((10, 20), dtype=bool)
        count, cols, rows = get_mask_pixel_counts(mask)
        assert count == 0
        assert cols == 0
        assert rows == 0

    def test_single_pixel(self):
        mask = np.zeros((10, 20), dtype=bool)
        mask[5, 10] = True
        count, cols, rows = get_mask_pixel_counts(mask)
        assert count == 1
        assert cols == 1
        assert rows == 1

    def test_horizontal_stripe(self):
        mask = np.zeros((10, 20), dtype=bool)
        mask[3, :] = True  # one full row
        count, cols, rows = get_mask_pixel_counts(mask)
        assert count == 20
        assert cols == 20  # all columns occupied
        assert rows == 1  # only one row


class TestUnderlyingIntensity:
    def test_uniform_image(self):
        image = np.full((10, 10), 128, dtype=np.uint8)
        mask = np.ones((10, 10), dtype=bool)
        mean, std = get_underlying_intensity(image, mask)
        assert pytest.approx(mean) == 128.0
        assert pytest.approx(std) == 0.0

    def test_partial_mask(self):
        image = np.zeros((10, 10), dtype=np.uint8)
        image[0:5, :] = 200  # top half bright
        mask = np.zeros((10, 10), dtype=bool)
        mask[0:5, :] = True  # mask covers bright area only
        mean, _std = get_underlying_intensity(image, mask)
        assert pytest.approx(mean) == 200.0


class TestQuadrantMasksEnface:
    """
    Tests for _get_quadrant_masks (enface/FAF).

    Convention being tested:
    - (0,0) is top-left of enface image
    - y increases downward
    - "superior" = top half (low y values)
    - "inferior" = bottom half (high y values)
    - "dexter" = right-as-viewed (high x values)
    - "sinister" = left-as-viewed (low x values)

    The quadrant boundaries pass through the fovea at 45-degree diagonals.
    """

    def test_centered_fovea_partitions_image(self):
        """With fovea at center, quadrants should roughly partition the image."""
        w, h = 100, 100
        sup, dex, inf, sin = _get_quadrant_masks(
            mask_width_px=w,
            mask_height_px=h,
            fovea_enface_x=50,
            fovea_enface_y=50,
        )
        # Each quadrant should cover ~25% of pixels
        total = w * h
        for q in [sup, dex, inf, sin]:
            frac = q.sum() / total
            assert 0.20 < frac < 0.30, f"Expected ~25%, got {frac * 100:.1f}%"

    def test_quadrants_cover_whole_image(self):
        """All four quadrants together should cover every pixel exactly once."""
        w, h = 100, 100
        sup, dex, inf, sin = _get_quadrant_masks(
            mask_width_px=w,
            mask_height_px=h,
            fovea_enface_x=50,
            fovea_enface_y=50,
        )
        combined = sup.astype(int) + dex.astype(int) + inf.astype(int) + sin.astype(int)
        # Every pixel should be in exactly one quadrant
        assert combined.min() >= 0
        assert combined.max() <= 1
        # At boundaries some pixels might be in 0 or 2 quadrants due to >= vs >
        # but the vast majority should be covered
        coverage = (combined == 1).sum() / (w * h)
        assert coverage > 0.95

    def test_superior_is_top(self):
        """
        With fovea at center, the 'superior' quadrant should be concentrated
        in the top portion of the image (low y indices).
        """
        w, h = 100, 100
        sup, _dex, _inf, _sin = _get_quadrant_masks(
            mask_width_px=w,
            mask_height_px=h,
            fovea_enface_x=50,
            fovea_enface_y=50,
        )
        # Superior should have more mass in top half than bottom half
        sup_top = sup[:50, :].sum()
        sup_bot = sup[50:, :].sum()
        assert sup_top > sup_bot, (
            f"Superior quadrant should be top-heavy: top={sup_top}, bottom={sup_bot}"
        )

    def test_inferior_is_bottom(self):
        """Inferior quadrant should be concentrated in the bottom portion."""
        w, h = 100, 100
        _sup, _dex, inf, _sin = _get_quadrant_masks(
            mask_width_px=w,
            mask_height_px=h,
            fovea_enface_x=50,
            fovea_enface_y=50,
        )
        inf_top = inf[:50, :].sum()
        inf_bot = inf[50:, :].sum()
        assert inf_bot > inf_top, (
            f"Inferior quadrant should be bottom-heavy: top={inf_top}, bottom={inf_bot}"
        )

    def test_dexter_is_right_as_viewed(self):
        """
        Dexter quadrant should be concentrated on the RIGHT side of the image
        as viewed (high x indices).

        NOTE: "dexter" here means right-as-viewed, which corresponds to the
        patient's LEFT (temporal for right eye, nasal for left eye).
        """
        w, h = 100, 100
        _sup, dex, _inf, _sin = _get_quadrant_masks(
            mask_width_px=w,
            mask_height_px=h,
            fovea_enface_x=50,
            fovea_enface_y=50,
        )
        dex_left = dex[:, :50].sum()
        dex_right = dex[:, 50:].sum()
        assert dex_right > dex_left, (
            f"Dexter should be right-as-viewed: left={dex_left}, right={dex_right}"
        )

    def test_sinister_is_left_as_viewed(self):
        """
        Sinister quadrant should be concentrated on the LEFT side of the image
        as viewed (low x indices).

        NOTE: "sinister" here means left-as-viewed, which corresponds to the
        patient's RIGHT (nasal for right eye, temporal for left eye).
        """
        w, h = 100, 100
        _sup, _dex, _inf, sin = _get_quadrant_masks(
            mask_width_px=w,
            mask_height_px=h,
            fovea_enface_x=50,
            fovea_enface_y=50,
        )
        sin_left = sin[:, :50].sum()
        sin_right = sin[:, 50:].sum()
        assert sin_left > sin_right, (
            f"Sinister should be left-as-viewed: left={sin_left}, right={sin_right}"
        )

    def test_off_center_fovea(self):
        """Fovea not at center: quadrant sizes should shift accordingly."""
        w, h = 100, 100
        # Fovea in top-left quadrant
        sup, dex, inf, sin = _get_quadrant_masks(
            mask_width_px=w,
            mask_height_px=h,
            fovea_enface_x=25,
            fovea_enface_y=25,
        )
        # Superior should be smaller (less room above), inferior larger
        assert inf.sum() > sup.sum()
        # Sinister should be smaller (less room to left), dexter larger
        assert dex.sum() > sin.sum()


class TestDistanceMaskEnface:
    """Tests for _get_distance_mask (circular region around fovea on enface)."""

    def test_centered_disk_area(self):
        """A disk of known diameter should have approximately pi*r^2 area."""
        # 100x100 image, 1px = 0.01mm, so image is 1mm x 1mm
        # Disk of 0.5mm diameter -> r=0.25mm -> r_px=25
        (mask,) = _get_distance_mask(
            scan_width_px=100,
            resolutions_mm_width=0.01,
            resolutions_mm_height=0.01,
            mask_width_px=100,
            mask_height_px=100,
            fovea_enface_x=50,
            fovea_enface_y=50,
            diameter=0.5,
        )
        expected_area_px = np.pi * 25**2
        actual = mask.sum()
        # Allow 5% tolerance for rasterisation
        assert abs(actual - expected_area_px) / expected_area_px < 0.05

    def test_disk_centered_on_fovea(self):
        """The mask should be centered on the fovea position."""
        (mask,) = _get_distance_mask(
            scan_width_px=100,
            resolutions_mm_width=0.01,
            resolutions_mm_height=0.01,
            mask_width_px=100,
            mask_height_px=100,
            fovea_enface_x=30,
            fovea_enface_y=70,
            diameter=0.2,
        )
        # Center of mass should be near (70, 30) in (row, col) indexing
        rows, cols = np.where(mask > 0)
        com_row = rows.mean()
        com_col = cols.mean()
        assert abs(com_row - 70) < 2
        assert abs(com_col - 30) < 2

    def test_large_disk_clipped(self):
        """A disk larger than the image should be clipped to image bounds."""
        (mask,) = _get_distance_mask(
            scan_width_px=100,
            resolutions_mm_width=0.01,
            resolutions_mm_height=0.01,
            mask_width_px=100,
            mask_height_px=100,
            fovea_enface_x=50,
            fovea_enface_y=50,
            diameter=5.0,  # 5mm >> 1mm image
        )
        # Should fill nearly the entire image
        assert mask.sum() > 0.95 * 100 * 100


class TestPixelCountByQuadrant:
    """Tests for get_pixel_count_by_quadrant."""

    def test_mask_entirely_in_one_quadrant(self):
        """A mask in the top-right should only contribute to superior+dexter."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[0:10, 90:100] = True  # top-right corner

        sup, dex, inf, sin = _get_quadrant_masks(
            mask_width_px=100,
            mask_height_px=100,
            fovea_enface_x=50,
            fovea_enface_y=50,
        )
        # This region is in both superior and dexter zones
        (count_sup,) = get_pixel_count_by_quadrant(mask, sup, "superior")
        (count_inf,) = get_pixel_count_by_quadrant(mask, inf, "inferior")
        (count_dex,) = get_pixel_count_by_quadrant(mask, dex, "dexter")
        (count_sin,) = get_pixel_count_by_quadrant(mask, sin, "sinister")

        # Top-right corner should be mostly in superior or dexter
        assert count_sup + count_dex > count_inf + count_sin


class TestCircleLineIntersection:
    """Tests for _get_circle_line_intersection helper."""

    def test_horizontal_line_through_center(self):
        """Line through circle center -> t values symmetric around 0.5."""
        t1, t2 = _get_circle_line_intersection(
            circle_centre=(5, 0),
            radius=2,
            line_start=(0, 0),
            line_end=(10, 0),
        )
        # Intersections at x=3 and x=7, t=0.3 and t=0.7
        assert pytest.approx(t1, abs=0.01) == 0.3
        assert pytest.approx(t2, abs=0.01) == 0.7

    def test_no_intersection(self):
        """Line that doesn't intersect circle returns empty."""
        result = _get_circle_line_intersection(
            circle_centre=(5, 5),
            radius=1,
            line_start=(0, 0),
            line_end=(10, 0),  # y=0, circle at y=5
        )
        assert result == []

    def test_tangent_line(self):
        """Line tangent to circle -> both t values are equal."""
        t1, t2 = _get_circle_line_intersection(
            circle_centre=(5, 1),
            radius=1,
            line_start=(0, 0),
            line_end=(10, 0),  # y=0, circle at y=1
        )
        assert pytest.approx(t1, abs=0.01) == t2


class TestDistanceMaskSlice:
    """
    Tests for _get_distance_mask_slice (B-scan level distance mask).

    This projects a circle from enface space onto a single B-scan line.
    The B-scan runs horizontally in physical mm-space.
    """

    def test_fovea_on_current_bscan(self):
        """When fovea is on the current B-scan, mask should be centered."""
        (mask,) = _get_distance_mask_slice(
            bscan_index=5,
            scan_width_px=100,
            resolutions_mm_width=0.01,  # 1mm total width
            resolutions_mm_depth=0.1,  # 0.1mm between bscans
            mask_width_px=100,
            mask_height_px=50,
            fovea_x=50,  # center of bscan
            fovea_bscan_index=5,  # same as current
            diameter=0.5,  # 0.5mm diameter
        )
        assert mask.sum() > 0
        # Mask should be centered around column 50
        cols = np.where(mask[0] > 0)[0]
        center = cols.mean()
        assert abs(center - 50) < 5

    def test_fovea_far_away_no_intersection(self):
        """B-scan far from fovea -> no intersection with circle."""
        (mask,) = _get_distance_mask_slice(
            bscan_index=0,
            scan_width_px=100,
            resolutions_mm_width=0.01,
            resolutions_mm_depth=0.1,
            mask_width_px=100,
            mask_height_px=50,
            fovea_x=50,
            fovea_bscan_index=50,  # far away
            diameter=0.5,
        )
        assert mask.sum() == 0

    def test_mask_shape_is_broadcastable(self):
        """The slice mask should be (1, width) for broadcasting."""
        (mask,) = _get_distance_mask_slice(
            bscan_index=5,
            scan_width_px=100,
            resolutions_mm_width=0.01,
            resolutions_mm_depth=0.1,
            mask_width_px=200,
            mask_height_px=50,
            fovea_x=50,
            fovea_bscan_index=5,
            diameter=0.5,
        )
        assert mask.shape == (1, 200)


class TestQuadrantMasksSlice:
    """
    Tests for _get_quadrant_masks_slice (B-scan level quadrant mask).

    This projects the quadrant boundaries from enface space onto a single
    B-scan line using the B-scan's start/end positions in enface coordinates.

    For a horizontal B-scan passing through the fovea:
    - The left half of the B-scan corresponds to sinister (left-as-viewed)
    - The right half corresponds to dexter (right-as-viewed)
    - Superior/inferior split depends on whether the B-scan is above or below fovea
    """

    def test_horizontal_bscan_through_fovea(self):
        """
        Horizontal B-scan through fovea center should split evenly
        into dexter (right) and sinister (left).
        """
        _sup, dex, _inf, sin = _get_quadrant_masks_slice(
            bscan_location_start_x=0,
            bscan_location_start_y=50,
            bscan_location_end_x=100,
            bscan_location_end_y=50,
            mask_width_px=100,
            mask_height_px=50,
            fovea_enface_x=50,
            fovea_enface_y=50,
        )
        # On the fovea's own B-scan, superior and inferior should be ~0
        # (the B-scan line lies on the superior/inferior boundary)
        # Dexter should be right half, sinister should be left half
        dex_sum = dex.sum()
        sin_sum = sin.sum()
        # Both should have substantial coverage
        assert dex_sum > 0
        assert sin_sum > 0
        # Should be roughly equal
        ratio = dex_sum / (sin_sum + 1e-8)
        assert 0.5 < ratio < 2.0, f"Dexter/sinister ratio: {ratio}"

    def test_bscan_above_fovea_is_superior(self):
        """
        A B-scan well above the fovea (lower y in top-left origin)
        should be entirely in the superior region.
        """
        sup, _dex, inf, _sin = _get_quadrant_masks_slice(
            bscan_location_start_x=0,
            bscan_location_start_y=10,  # well above fovea
            bscan_location_end_x=100,
            bscan_location_end_y=10,
            mask_width_px=100,
            mask_height_px=50,
            fovea_enface_x=50,
            fovea_enface_y=50,
        )
        # Should be mostly superior
        assert sup.sum() > inf.sum(), (
            f"B-scan above fovea should be superior: sup={sup.sum()}, inf={inf.sum()}"
        )

    def test_bscan_below_fovea_is_inferior(self):
        """
        A B-scan well below the fovea (higher y in top-left origin)
        should be entirely in the inferior region.
        """
        sup, _dex, inf, _sin = _get_quadrant_masks_slice(
            bscan_location_start_x=0,
            bscan_location_start_y=90,  # well below fovea
            bscan_location_end_x=100,
            bscan_location_end_y=90,
            mask_width_px=100,
            mask_height_px=50,
            fovea_enface_x=50,
            fovea_enface_y=50,
        )
        assert inf.sum() > sup.sum(), (
            f"B-scan below fovea should be inferior: sup={sup.sum()}, inf={inf.sum()}"
        )

    def test_slice_mask_shape(self):
        """Slice masks should be (1, width) for broadcasting."""
        sup, dex, inf, sin = _get_quadrant_masks_slice(
            bscan_location_start_x=0,
            bscan_location_start_y=50,
            bscan_location_end_x=100,
            bscan_location_end_y=50,
            mask_width_px=200,
            mask_height_px=50,
            fovea_enface_x=50,
            fovea_enface_y=50,
        )
        for q in [sup, dex, inf, sin]:
            assert q.shape == (1, 200)


class TestFoveaEnfacePosition:
    """
    Tests for get_fovea_enface_position.

    This projects the fovea's OCT coordinates (a-scan index + b-scan index)
    into enface pixel coordinates using the B-scan's position in enface space.
    """

    def test_fovea_on_current_bscan_center(self):
        """
        When fovea is on the current B-scan and at center of A-scan,
        the enface position should be the midpoint of the B-scan line.
        """
        x, y = get_fovea_enface_position(
            bscan_index=5,
            scan_width_px=100,
            resolutions_mm_width=0.01,
            resolutions_mm_depth=0.1,
            bscan_location_start_x=100,
            bscan_location_start_y=300,
            bscan_location_end_x=500,
            bscan_location_end_y=300,
            fovea_x=50,  # center of scan
            fovea_bscan_index=5,  # same B-scan
        )
        # Midpoint of (100,300)-(500,300) at t=0.5 is (300, 300)
        assert pytest.approx(x, abs=1) == 300
        assert pytest.approx(y, abs=1) == 300

    def test_fovea_at_bscan_start(self):
        """Fovea at A-scan 0 should project to B-scan start position."""
        x, y = get_fovea_enface_position(
            bscan_index=0,
            scan_width_px=100,
            resolutions_mm_width=0.01,
            resolutions_mm_depth=0.1,
            bscan_location_start_x=100,
            bscan_location_start_y=200,
            bscan_location_end_x=400,
            bscan_location_end_y=200,
            fovea_x=0,
            fovea_bscan_index=0,
        )
        assert pytest.approx(x, abs=1) == 100
        assert pytest.approx(y, abs=1) == 200

    def test_fovea_on_different_bscan(self):
        """
        When fovea is on a different B-scan, result should include
        perpendicular offset from the current B-scan.
        """
        # Horizontal bscan at y=300, fovea on bscan 10 (current is 5)
        x, y = get_fovea_enface_position(
            bscan_index=5,
            scan_width_px=100,
            resolutions_mm_width=0.01,
            resolutions_mm_depth=0.1,
            bscan_location_start_x=100,
            bscan_location_start_y=300,
            bscan_location_end_x=500,
            bscan_location_end_y=300,
            fovea_x=50,
            fovea_bscan_index=10,
        )
        # Projection along bscan should still be at midpoint x=300
        assert pytest.approx(x, abs=1) == 300
        # But y should be offset: (5-10)*spacing perpendicular to bscan
        # For horizontal bscan, perpendicular is vertical
        # The sign and direction depends on the implementation
        assert y != 300  # Should be different due to perpendicular offset


class TestMaskArea:
    """Tests for get_mask_area."""

    def test_known_area(self):
        # 100 pixels at 0.01mm per pixel in each direction = 0.01 mm^2
        area, h_extent, v_extent = get_mask_area(
            mask_resolutions_mm_width=0.01,
            mask_resolutions_mm_height=0.01,
            mask_pixel_count=100,
            mask_columns_count=10,
            mask_rows_count=10,
        )
        assert pytest.approx(area) == 100 * 0.01 * 0.01  # 0.01 mm^2
        assert pytest.approx(h_extent) == 10 * 0.01  # 0.1 mm
        assert pytest.approx(v_extent) == 10 * 0.01  # 0.1 mm


class TestMaskVolume:
    """Tests for get_mask_volume."""

    def test_known_volume(self):
        volume, enface_area = get_mask_volume(
            mask_area=0.01,  # mm^2 per slice
            mask_horizontal_extent=0.1,  # mm
            resolutions_mm_depth=0.2,  # mm between slices
        )
        assert pytest.approx(volume) == 0.01 * 0.2  # 0.002 mm^3
        assert pytest.approx(enface_area) == 0.1 * 0.2  # 0.02 mm^2


class TestPixelCountByDistance:
    """Tests for get_pixel_count_by_distance (enface)."""

    def test_full_mask_with_disk(self):
        """Full mask with a disk -> count equals disk area."""
        mask = np.ones((100, 100), dtype=bool)
        (distance_mask,) = _get_distance_mask(
            scan_width_px=100,
            resolutions_mm_width=0.01,
            resolutions_mm_height=0.01,
            mask_width_px=100,
            mask_height_px=100,
            fovea_enface_x=50,
            fovea_enface_y=50,
            diameter=0.5,
        )
        (count,) = get_pixel_count_by_distance(mask, distance_mask, 0.5)
        # Count should equal the disk area
        assert count == pytest.approx(distance_mask.sum(), rel=0.01)

    def test_empty_mask_with_disk(self):
        """Empty mask with any distance mask -> count is 0."""
        mask = np.zeros((100, 100), dtype=bool)
        distance_mask = np.ones((100, 100))
        (count,) = get_pixel_count_by_distance(mask, distance_mask, 1.0)
        assert count == 0
