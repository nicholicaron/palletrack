"""Unit tests for frame quality assessment."""

import numpy as np
import pytest

from app.cv_models import BoundingBox
from app.cv_processing import (
    BestFrameSelector,
    FrameQualityScorer,
    calculate_angle_score,
    calculate_sharpness,
    calculate_size_score,
)


class TestSharpnessCalculation:
    """Tests for sharpness calculation."""

    def test_sharp_image(self):
        """Test that a sharp image has high sharpness score."""
        # Create a sharp test pattern (checkerboard)
        img = np.zeros((100, 100), dtype=np.uint8)
        img[::2, ::2] = 255
        img[1::2, 1::2] = 255

        sharpness = calculate_sharpness(img)
        assert sharpness > 100, "Sharp image should have sharpness > 100"

    def test_blurry_image(self):
        """Test that a blurry image has low sharpness score."""
        # Create a uniform gray image (very blurry)
        img = np.ones((100, 100), dtype=np.uint8) * 128

        sharpness = calculate_sharpness(img)
        assert sharpness < 10, "Uniform image should have very low sharpness"

    def test_gradient_image(self):
        """Test sharpness of gradient image."""
        # Create an image with both sharp edges and gradients
        img = np.zeros((100, 100), dtype=np.uint8)
        # Add sharp vertical lines every 10 pixels
        img[:, ::10] = 255

        sharpness = calculate_sharpness(img)
        assert sharpness > 100, "Image with sharp edges should have high sharpness"

    def test_color_image_conversion(self):
        """Test that color images are converted to grayscale."""
        # Create a color checkerboard
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[::2, ::2, :] = 255
        img[1::2, 1::2, :] = 255

        sharpness = calculate_sharpness(img)
        assert sharpness > 0, "Color image should be processed successfully"


class TestSizeScore:
    """Tests for size score calculation."""

    def test_large_bbox(self):
        """Test that large bboxes get high scores."""
        # 40% of frame area
        bbox = BoundingBox(x1=0, y1=0, x2=800, y2=600, confidence=0.9)
        score = calculate_size_score(bbox, (1000, 1000))

        assert score == 1.0, "Large bbox (>30% of frame) should score 1.0"

    def test_small_bbox(self):
        """Test that small bboxes get low scores."""
        # < 1% of frame area
        bbox = BoundingBox(x1=0, y1=0, x2=50, y2=50, confidence=0.9)
        score = calculate_size_score(bbox, (1000, 1000))

        assert score < 0.1, "Small bbox (<1% of frame) should score near 0.0"

    def test_medium_bbox(self):
        """Test that medium bboxes get intermediate scores."""
        # ~10% of frame area
        bbox = BoundingBox(x1=0, y1=0, x2=316, y2=316, confidence=0.9)
        score = calculate_size_score(bbox, (1000, 1000))

        assert 0.3 < score < 0.5, "Medium bbox (~10%) should score around 0.3-0.5"

    def test_edge_case_30_percent(self):
        """Test bbox at exactly 30% of frame."""
        # Exactly 30% of frame
        bbox = BoundingBox(x1=0, y1=0, x2=547, y2=547, confidence=0.9)
        score = calculate_size_score(bbox, (1000, 1000))

        assert score >= 0.99, "Bbox at 30% should score close to 1.0"

    def test_edge_case_1_percent(self):
        """Test bbox at exactly 1% of frame."""
        # Exactly 1% of frame
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=0.9)
        score = calculate_size_score(bbox, (1000, 1000))

        assert score == 0.0, "Bbox at 1% should score 0.0"


class TestAngleScore:
    """Tests for angle score calculation."""

    def test_portrait_document_ratio(self):
        """Test typical portrait document aspect ratio."""
        # US Letter: 8.5 x 11 (~1.29)
        bbox = BoundingBox(x1=0, y1=0, x2=850, y2=1100, confidence=0.9)
        score = calculate_angle_score(bbox)

        assert score == 1.0, "Portrait document ratio should score 1.0"

    def test_landscape_document_ratio(self):
        """Test typical landscape document aspect ratio."""
        # Landscape: 11 x 8.5
        bbox = BoundingBox(x1=0, y1=0, x2=1100, y2=850, confidence=0.9)
        score = calculate_angle_score(bbox)

        assert score == 1.0, "Landscape document ratio should score 1.0"

    def test_square_document(self):
        """Test square shipping label."""
        # Square label
        bbox = BoundingBox(x1=0, y1=0, x2=500, y2=500, confidence=0.9)
        score = calculate_angle_score(bbox)

        assert score >= 0.9, "Square document should score >= 0.9"

    def test_very_distorted(self):
        """Test extremely distorted aspect ratio."""
        # Very wide and short (extreme angle)
        bbox = BoundingBox(x1=0, y1=0, x2=1000, y2=100, confidence=0.9)
        score = calculate_angle_score(bbox)

        assert score < 0.3, "Extremely distorted bbox should score low"

    def test_zero_height(self):
        """Test edge case with zero height."""
        # Invalid bbox with zero height (should be caught by validation)
        # But test the angle score function directly
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=0.001, confidence=0.9)
        score = calculate_angle_score(bbox)

        assert score >= 0.0, "Zero height should return valid score"


class TestFrameQualityScorer:
    """Tests for FrameQualityScorer class."""

    def test_scorer_initialization(self):
        """Test scorer initialization with config."""
        config = {
            "sharpness_weight": 0.5,
            "size_weight": 0.3,
            "angle_weight": 0.2,
            "min_sharpness": 100.0,
        }
        scorer = FrameQualityScorer(config)

        assert scorer.sharpness_weight == 0.5
        assert scorer.size_weight == 0.3
        assert scorer.angle_weight == 0.2
        assert scorer.min_sharpness == 100.0

    def test_scorer_default_config(self):
        """Test scorer with default config values."""
        config = {}
        scorer = FrameQualityScorer(config)

        assert scorer.sharpness_weight == 0.5
        assert scorer.size_weight == 0.3
        assert scorer.angle_weight == 0.2

    def test_score_high_quality_frame(self):
        """Test scoring a high quality frame."""
        config = {"min_sharpness": 100.0}
        scorer = FrameQualityScorer(config)

        # Create sharp checkerboard
        frame = np.zeros((1000, 1000), dtype=np.uint8)
        frame[::2, ::2] = 255
        frame[1::2, 1::2] = 255

        # Large, well-proportioned bbox
        bbox = BoundingBox(x1=100, y1=100, x2=600, y2=700, confidence=0.9)

        scores = scorer.score_frame(frame, bbox)

        assert scores["acceptable"] is True, "High quality frame should be acceptable"
        assert scores["composite_score"] > 0.5, "Should have high composite score"
        assert "sharpness_raw" in scores
        assert "sharpness_score" in scores
        assert "size_score" in scores
        assert "angle_score" in scores

    def test_score_blurry_frame(self):
        """Test scoring a blurry frame."""
        config = {"min_sharpness": 100.0}
        scorer = FrameQualityScorer(config)

        # Create uniform (blurry) image
        frame = np.ones((1000, 1000), dtype=np.uint8) * 128

        bbox = BoundingBox(x1=100, y1=100, x2=600, y2=700, confidence=0.9)

        scores = scorer.score_frame(frame, bbox)

        assert scores["acceptable"] is False, "Blurry frame should not be acceptable"
        assert scores["sharpness_raw"] < 100.0

    def test_score_small_region(self):
        """Test scoring a small region."""
        config = {"min_sharpness": 50.0}  # Lower threshold for this test
        scorer = FrameQualityScorer(config)

        # Sharp image
        frame = np.zeros((1000, 1000), dtype=np.uint8)
        frame[::2, ::2] = 255
        frame[1::2, 1::2] = 255

        # Very small bbox
        bbox = BoundingBox(x1=100, y1=100, x2=150, y2=150, confidence=0.9)

        scores = scorer.score_frame(frame, bbox)

        assert scores["size_score"] < 0.1, "Small region should have low size score"

    def test_bbox_clipping(self):
        """Test that bbox coordinates are clipped to frame bounds."""
        config = {"min_sharpness": 100.0}
        scorer = FrameQualityScorer(config)

        frame = np.zeros((1000, 1000), dtype=np.uint8)
        frame[::2, ::2] = 255
        frame[1::2, 1::2] = 255

        # Bbox extending beyond frame
        bbox = BoundingBox(x1=900, y1=900, x2=1500, y2=1500, confidence=0.9)

        # Should not raise error
        scores = scorer.score_frame(frame, bbox)
        assert "composite_score" in scores


class TestBestFrameSelector:
    """Tests for BestFrameSelector class."""

    def test_selector_initialization(self):
        """Test selector initialization with config."""
        config = {
            "frames_to_select": 5,
            "min_sharpness": 100.0,
            "min_size_score": 0.3,
        }
        selector = BestFrameSelector(config)

        assert selector.n_frames == 5
        assert selector.min_size_score == 0.3

    def test_select_fewer_frames_than_available(self):
        """Test selecting when fewer frames available than requested."""
        config = {"frames_to_select": 5, "min_sharpness": 50.0}
        selector = BestFrameSelector(config)

        # Create 3 high-quality frames
        frames = []
        for i in range(3):
            frame = np.zeros((1000, 1000), dtype=np.uint8)
            frame[::2, ::2] = 255
            frame[1::2, 1::2] = 255
            bbox = BoundingBox(x1=100, y1=100, x2=600, y2=700, confidence=0.9)
            frames.append((i, frame, bbox))

        selected = selector.select_best_frames(frames)

        assert len(selected) == 3, "Should return all acceptable frames when fewer than requested"
        # Note: order might vary based on scores, so just check all are present
        assert set(selected) == {0, 1, 2}

    def test_select_best_from_mixed_quality(self):
        """Test selecting best frames from mixed quality frames."""
        config = {"frames_to_select": 2, "min_sharpness": 50.0}
        selector = BestFrameSelector(config)

        frames = []

        # Frame 0: Sharp
        frame0 = np.zeros((1000, 1000), dtype=np.uint8)
        frame0[::2, ::2] = 255
        frame0[1::2, 1::2] = 255
        bbox0 = BoundingBox(x1=100, y1=100, x2=600, y2=700, confidence=0.9)
        frames.append((0, frame0, bbox0))

        # Frame 1: Blurry
        frame1 = np.ones((1000, 1000), dtype=np.uint8) * 128
        bbox1 = BoundingBox(x1=100, y1=100, x2=600, y2=700, confidence=0.9)
        frames.append((1, frame1, bbox1))

        # Frame 2: Sharp
        frame2 = np.zeros((1000, 1000), dtype=np.uint8)
        frame2[::2, ::2] = 255
        frame2[1::2, 1::2] = 255
        bbox2 = BoundingBox(x1=100, y1=100, x2=600, y2=700, confidence=0.9)
        frames.append((2, frame2, bbox2))

        selected = selector.select_best_frames(frames)

        # Should select the two sharp frames (0 and 2)
        assert len(selected) == 2
        assert 0 in selected
        assert 2 in selected
        assert 1 not in selected

    def test_select_with_size_threshold(self):
        """Test that frames below size threshold are filtered out."""
        config = {
            "frames_to_select": 3,
            "min_sharpness": 50.0,
            "min_size_score": 0.5,
        }
        selector = BestFrameSelector(config)

        frames = []

        # Frame 0: Sharp but small
        frame0 = np.zeros((1000, 1000), dtype=np.uint8)
        frame0[::2, ::2] = 255
        frame0[1::2, 1::2] = 255
        bbox0 = BoundingBox(x1=100, y1=100, x2=150, y2=150, confidence=0.9)
        frames.append((0, frame0, bbox0))

        # Frame 1: Sharp and large
        frame1 = np.zeros((1000, 1000), dtype=np.uint8)
        frame1[::2, ::2] = 255
        frame1[1::2, 1::2] = 255
        bbox1 = BoundingBox(x1=100, y1=100, x2=800, y2=800, confidence=0.9)
        frames.append((1, frame1, bbox1))

        selected = selector.select_best_frames(frames)

        # Only frame 1 should be selected (meets size threshold)
        assert len(selected) == 1
        assert selected == [1]

    def test_select_with_diversity(self):
        """Test frame selection with diversity constraint."""
        config = {"frames_to_select": 3, "min_sharpness": 50.0}
        selector = BestFrameSelector(config)

        # Create 10 sharp frames
        frames = []
        for i in range(10):
            frame = np.zeros((1000, 1000), dtype=np.uint8)
            frame[::2, ::2] = 255
            frame[1::2, 1::2] = 255
            bbox = BoundingBox(x1=100, y1=100, x2=600, y2=700, confidence=0.9)
            frames.append((i * 10, frame, bbox))  # Frame IDs: 0, 10, 20, ..., 90

        # Select with minimum 15 frame gap
        selected = selector.get_frame_diversity_selection(frames, n_frames=3, min_frame_gap=15)

        assert len(selected) == 3
        # Check that selected frames have at least 15 frame gap
        for i in range(len(selected) - 1):
            gap = selected[i + 1] - selected[i]
            assert gap >= 15, f"Gap between frames should be >= 15, got {gap}"

    def test_select_frames_with_scores(self):
        """Test selecting frames with detailed scores returned."""
        config = {"frames_to_select": 2, "min_sharpness": 50.0}
        selector = BestFrameSelector(config)

        frames = []

        # Create 3 sharp frames
        for i in range(3):
            frame = np.zeros((1000, 1000), dtype=np.uint8)
            frame[::2, ::2] = 255
            frame[1::2, 1::2] = 255
            bbox = BoundingBox(x1=100, y1=100, x2=600, y2=700, confidence=0.9)
            frames.append((i, frame, bbox))

        results = selector.select_best_frames_with_scores(frames)

        assert len(results) == 2
        for frame_id, scores in results:
            assert isinstance(frame_id, int)
            assert "composite_score" in scores
            assert "sharpness_score" in scores
            assert "size_score" in scores
            assert "angle_score" in scores

    def test_no_acceptable_frames(self):
        """Test when no frames meet quality thresholds."""
        config = {"frames_to_select": 2, "min_sharpness": 500.0}  # Very high threshold
        selector = BestFrameSelector(config)

        frames = []

        # Create blurry frames
        for i in range(3):
            frame = np.ones((1000, 1000), dtype=np.uint8) * 128
            bbox = BoundingBox(x1=100, y1=100, x2=600, y2=700, confidence=0.9)
            frames.append((i, frame, bbox))

        selected = selector.select_best_frames(frames)

        assert len(selected) == 0, "Should return empty list when no frames are acceptable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
