"""PaddleOCR wrapper for document text extraction.

This module provides a clean interface to PaddleOCR with custom configuration,
preprocessing integration, and result parsing into OCRResult objects.
"""

from typing import Dict, List, Tuple

import numpy as np
from paddleocr import PaddleOCR

from app.cv_models import BoundingBox, OCRResult
from app.cv_processing.ocr.preprocessor import OCRPreprocessor


class DocumentOCR:
    """Wrapper for PaddleOCR with preprocessing and custom configuration.

    Provides a high-level interface for extracting text from document images
    with optional preprocessing and confidence filtering.

    Attributes:
        ocr: PaddleOCR instance
        min_confidence: Minimum confidence threshold for results
        preprocessor: OCRPreprocessor instance
        config: Configuration dictionary

    Example:
        >>> config = {'ocr': {'language': 'en', 'use_gpu': True}}
        >>> ocr_engine = DocumentOCR(config)
        >>> results = ocr_engine.extract_text(document_image)
    """

    def __init__(self, config: Dict):
        """Initialize PaddleOCR with configuration.

        Args:
            config: Configuration dict with ocr settings:
                - ocr.language: Language code (en, es, fr, de, etc.)
                - ocr.use_gpu: Whether to use GPU acceleration
                - ocr.min_text_confidence: Minimum confidence threshold

        Example:
            >>> config = {
            ...     'ocr': {
            ...         'language': 'en',
            ...         'use_gpu': True,
            ...         'min_text_confidence': 0.6
            ...     }
            ... }
            >>> ocr_engine = DocumentOCR(config)
        """
        self.config = config
        ocr_config = config.get('ocr', {})

        # Initialize PaddleOCR
        self.ocr = PaddleOCR(
            use_angle_cls=True,  # Enable text angle classification
            lang=ocr_config.get('language', 'en'),
            use_gpu=ocr_config.get('use_gpu', True),
            show_log=False,  # Suppress verbose logs
            use_space_char=True,  # Recognize spaces between words
            det_db_thresh=0.3,  # Detection threshold
            det_db_box_thresh=0.5,  # Box threshold
            rec_batch_num=6  # Recognition batch size
        )

        self.min_confidence = ocr_config.get('min_text_confidence', 0.6)
        self.preprocessor = OCRPreprocessor()

    def extract_text(
        self,
        image: np.ndarray,
        preprocess: bool = True,
        frame_number: int = -1
    ) -> List[OCRResult]:
        """Extract text from document image with optional preprocessing.

        Process:
        1. Optionally preprocess image for better OCR
        2. Run PaddleOCR detection and recognition
        3. Parse results into OCRResult objects
        4. Filter by confidence threshold

        Args:
            image: Input document image (BGR or grayscale)
            preprocess: Whether to apply preprocessing (default: True)
            frame_number: Frame number for tracking (default: -1)

        Returns:
            List of OCRResult objects with detected text

        Example:
            >>> results = ocr_engine.extract_text(document_image)
            >>> for result in results:
            ...     print(f"Text: {result.text}, Confidence: {result.confidence}")
        """
        # Preprocess if requested
        if preprocess:
            processed_image = self.preprocessor.preprocess(image, self.config)
        else:
            processed_image = image

        # Ensure image is in correct format for PaddleOCR
        if len(processed_image.shape) == 2:
            # Convert grayscale to BGR
            processed_image = np.stack([processed_image] * 3, axis=-1)

        # Run PaddleOCR
        try:
            paddle_results = self.ocr.ocr(processed_image, cls=True)
        except Exception as e:
            print(f"PaddleOCR error: {e}")
            return []

        # Parse results
        if not paddle_results or paddle_results[0] is None:
            return []

        ocr_results = []
        for line in paddle_results[0]:
            if line is None:
                continue

            # Parse PaddleOCR output format
            bbox_points, (text, confidence) = line

            # Filter by confidence
            if confidence < self.min_confidence:
                continue

            # Convert bbox points to BoundingBox
            bbox = self._points_to_bbox(bbox_points)

            # Create OCRResult
            ocr_result = OCRResult(
                text=text,
                confidence=confidence,
                bbox=bbox,
                frame_number=frame_number
            )
            ocr_results.append(ocr_result)

        return ocr_results

    def extract_full_text(
        self,
        image: np.ndarray,
        preprocess: bool = True,
        join_char: str = ' '
    ) -> str:
        """Extract all text as single string in reading order.

        Args:
            image: Input document image
            preprocess: Whether to apply preprocessing (default: True)
            join_char: Character to join text regions (default: space)

        Returns:
            Concatenated text string

        Example:
            >>> full_text = ocr_engine.extract_full_text(document_image)
            >>> print(f"Document contains: {full_text}")
        """
        ocr_results = self.extract_text(image, preprocess=preprocess)

        if not ocr_results:
            return ""

        # Sort by y-coordinate (top to bottom), then x-coordinate (left to right)
        sorted_results = sorted(
            ocr_results,
            key=lambda r: (r.bbox.y_min, r.bbox.x_min)
        )

        # Join text with specified character
        text_parts = [result.text for result in sorted_results]
        return join_char.join(text_parts)

    def extract_text_with_structure(
        self,
        image: np.ndarray,
        preprocess: bool = True,
        line_threshold: float = 10.0
    ) -> List[str]:
        """Extract text preserving line structure.

        Groups text regions into lines based on y-coordinate proximity.

        Args:
            image: Input document image
            preprocess: Whether to apply preprocessing (default: True)
            line_threshold: Maximum y-distance to consider same line (pixels)

        Returns:
            List of text lines

        Example:
            >>> lines = ocr_engine.extract_text_with_structure(document_image)
            >>> for i, line in enumerate(lines):
            ...     print(f"Line {i}: {line}")
        """
        ocr_results = self.extract_text(image, preprocess=preprocess)

        if not ocr_results:
            return []

        # Sort by y-coordinate first
        sorted_results = sorted(ocr_results, key=lambda r: r.bbox.y_min)

        # Group into lines
        lines = []
        current_line = []
        last_y = None

        for result in sorted_results:
            current_y = result.bbox.y_min

            if last_y is None or abs(current_y - last_y) < line_threshold:
                # Same line
                current_line.append(result)
            else:
                # New line
                if current_line:
                    # Sort current line by x-coordinate
                    current_line.sort(key=lambda r: r.bbox.x_min)
                    line_text = ' '.join(r.text for r in current_line)
                    lines.append(line_text)
                current_line = [result]

            last_y = current_y

        # Add last line
        if current_line:
            current_line.sort(key=lambda r: r.bbox.x_min)
            line_text = ' '.join(r.text for r in current_line)
            lines.append(line_text)

        return lines

    def extract_text_regions(
        self,
        image: np.ndarray,
        preprocess: bool = True
    ) -> List[Tuple[str, float, BoundingBox, np.ndarray]]:
        """Extract text with regions for further processing.

        Returns both text and cropped image regions for each detection.

        Args:
            image: Input document image
            preprocess: Whether to apply preprocessing (default: True)

        Returns:
            List of tuples: (text, confidence, bbox, cropped_region)

        Example:
            >>> regions = ocr_engine.extract_text_regions(document_image)
            >>> for text, conf, bbox, region in regions:
            ...     cv2.imshow(f"Text: {text}", region)
        """
        ocr_results = self.extract_text(image, preprocess=preprocess)

        regions = []
        for result in ocr_results:
            # Crop region from original image
            bbox = result.bbox
            x1, y1, x2, y2 = int(bbox.x_min), int(bbox.y_min), int(bbox.x_max), int(bbox.y_max)

            # Validate bounds
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            cropped = image[y1:y2, x1:x2]

            regions.append((result.text, result.confidence, result.bbox, cropped))

        return regions

    @staticmethod
    def _points_to_bbox(points: List[List[float]]) -> BoundingBox:
        """Convert PaddleOCR bbox points to BoundingBox.

        PaddleOCR returns 4 corner points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        We convert to axis-aligned bounding box.

        Args:
            points: List of 4 [x, y] corner points

        Returns:
            BoundingBox object

        Example:
            >>> points = [[10, 20], [100, 20], [100, 50], [10, 50]]
            >>> bbox = DocumentOCR._points_to_bbox(points)
        """
        # Extract x and y coordinates
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        # Get min/max to form axis-aligned box
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        return BoundingBox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max
        )

    @staticmethod
    def parse_paddle_output(
        paddle_result: List
    ) -> List[Tuple[str, float, BoundingBox]]:
        """Parse PaddleOCR's output format into cleaner structure.

        Converts PaddleOCR's format into simple tuples.

        Args:
            paddle_result: Raw output from PaddleOCR.ocr()

        Returns:
            List of tuples: (text, confidence, BoundingBox)

        Example:
            >>> parsed = DocumentOCR.parse_paddle_output(paddle_results)
            >>> for text, conf, bbox in parsed:
            ...     print(f"{text}: {conf:.2f}")
        """
        if not paddle_result or paddle_result[0] is None:
            return []

        results = []
        for line in paddle_result[0]:
            if line is None:
                continue

            bbox_points, (text, confidence) = line
            bbox = DocumentOCR._points_to_bbox(bbox_points)
            results.append((text, confidence, bbox))

        return results

    def calculate_average_confidence(self, ocr_results: List[OCRResult]) -> float:
        """Calculate average confidence across all OCR results.

        Args:
            ocr_results: List of OCRResult objects

        Returns:
            Average confidence score (0.0-1.0)

        Example:
            >>> results = ocr_engine.extract_text(image)
            >>> avg_conf = ocr_engine.calculate_average_confidence(results)
            >>> print(f"Average confidence: {avg_conf:.2%}")
        """
        if not ocr_results:
            return 0.0

        total_confidence = sum(r.confidence for r in ocr_results)
        return total_confidence / len(ocr_results)

    def filter_by_confidence(
        self,
        ocr_results: List[OCRResult],
        min_confidence: float
    ) -> List[OCRResult]:
        """Filter OCR results by confidence threshold.

        Args:
            ocr_results: List of OCRResult objects
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered list of OCRResult objects

        Example:
            >>> all_results = ocr_engine.extract_text(image)
            >>> high_conf = ocr_engine.filter_by_confidence(all_results, 0.8)
        """
        return [r for r in ocr_results if r.confidence >= min_confidence]

    def get_statistics(self, ocr_results: List[OCRResult]) -> Dict:
        """Get statistics about OCR results.

        Args:
            ocr_results: List of OCRResult objects

        Returns:
            Dictionary with statistics:
                - total_detections: Number of text detections
                - avg_confidence: Average confidence
                - min_confidence: Minimum confidence
                - max_confidence: Maximum confidence
                - total_characters: Total character count

        Example:
            >>> results = ocr_engine.extract_text(image)
            >>> stats = ocr_engine.get_statistics(results)
            >>> print(f"Detected {stats['total_detections']} text regions")
        """
        if not ocr_results:
            return {
                'total_detections': 0,
                'avg_confidence': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0,
                'total_characters': 0
            }

        confidences = [r.confidence for r in ocr_results]
        total_chars = sum(len(r.text) for r in ocr_results)

        return {
            'total_detections': len(ocr_results),
            'avg_confidence': sum(confidences) / len(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'total_characters': total_chars
        }
