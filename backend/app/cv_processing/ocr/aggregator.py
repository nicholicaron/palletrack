"""Multi-frame OCR result aggregation for improved accuracy.

This module combines OCR results from multiple frames of the same document
to achieve more reliable text extraction through voting and consensus methods.
"""

from collections import Counter
from typing import Dict, List

import numpy as np
from difflib import SequenceMatcher

from app.cv_models import OCRResult


class MultiFrameOCRAggregator:
    """Combine OCR results from multiple frames for reliability.

    Uses various aggregation strategies to merge text extracted from
    multiple frames of the same document, reducing single-frame errors
    and improving overall accuracy.

    Example:
        >>> frame1_results = [OCRResult(...), ...]
        >>> frame2_results = [OCRResult(...), ...]
        >>> all_results = [frame1_results, frame2_results]
        >>> consensus = MultiFrameOCRAggregator.aggregate_ocr_results(
        ...     all_results, method='voting'
        ... )
    """

    @staticmethod
    def aggregate_ocr_results(
        ocr_results: List[List[OCRResult]],
        method: str = 'voting'
    ) -> str:
        """Combine text from multiple frames using specified method.

        Args:
            ocr_results: List of OCR results per frame (outer list = frames)
            method: Aggregation method:
                - 'voting': Character-level voting (most robust)
                - 'highest_confidence': Use frame with highest confidence
                - 'longest': Use longest extracted text
                - 'confidence_weighted': Merge weighted by confidence

        Returns:
            Aggregated text string

        Example:
            >>> results = [frame1_ocr, frame2_ocr, frame3_ocr]
            >>> consensus_text = MultiFrameOCRAggregator.aggregate_ocr_results(
            ...     results, method='voting'
            ... )
        """
        if not ocr_results:
            return ""

        # Filter out empty frames
        ocr_results = [frame for frame in ocr_results if frame]

        if not ocr_results:
            return ""

        # Single frame - no aggregation needed
        if len(ocr_results) == 1:
            return MultiFrameOCRAggregator._results_to_text(ocr_results[0])

        # Apply selected method
        if method == 'voting':
            return MultiFrameOCRAggregator.voting_aggregation(ocr_results)
        elif method == 'highest_confidence':
            return MultiFrameOCRAggregator.highest_confidence_method(ocr_results)
        elif method == 'longest':
            return MultiFrameOCRAggregator.longest_text_method(ocr_results)
        elif method == 'confidence_weighted':
            return MultiFrameOCRAggregator.confidence_weighted_merge(ocr_results)
        else:
            # Default to voting
            return MultiFrameOCRAggregator.voting_aggregation(ocr_results)

    @staticmethod
    def voting_aggregation(ocr_results: List[List[OCRResult]]) -> str:
        """Character-level voting across frames.

        Aligns text strings from multiple frames and votes on each
        character position. Most common character wins.

        Args:
            ocr_results: List of OCR results per frame

        Returns:
            Consensus text based on voting

        Example:
            >>> # Frame 1: "TRACK123", Frame 2: "TRACK1Z3", Frame 3: "TRACK123"
            >>> # Result: "TRACK123" (123 wins over 1Z3)
            >>> consensus = MultiFrameOCRAggregator.voting_aggregation(results)
        """
        # Convert each frame to text
        texts = [MultiFrameOCRAggregator._results_to_text(frame) for frame in ocr_results]

        if not texts:
            return ""

        # Filter empty texts
        texts = [t for t in texts if t]

        if not texts:
            return ""

        if len(texts) == 1:
            return texts[0]

        # Find longest text as reference
        reference = max(texts, key=len)

        # Align all texts to reference using sequence matching
        aligned_texts = []
        for text in texts:
            aligned = MultiFrameOCRAggregator._align_texts(reference, text)
            aligned_texts.append(aligned)

        # Vote on each character position
        consensus = []
        for i in range(len(reference)):
            # Collect characters at position i
            chars = []
            for aligned_text in aligned_texts:
                if i < len(aligned_text) and aligned_text[i] != ' ':
                    chars.append(aligned_text[i])

            if chars:
                # Vote - most common character wins
                char_counts = Counter(chars)
                most_common_char, _ = char_counts.most_common(1)[0]
                consensus.append(most_common_char)
            else:
                # No consensus - use reference
                consensus.append(reference[i])

        return ''.join(consensus)

    @staticmethod
    def _align_texts(reference: str, text: str) -> str:
        """Align text to reference using sequence matching.

        Args:
            reference: Reference string (typically longest)
            text: Text to align

        Returns:
            Aligned text with padding

        Example:
            >>> ref = "TRACKING123"
            >>> text = "TRACK123"
            >>> aligned = MultiFrameOCRAggregator._align_texts(ref, text)
            >>> # Returns "TRACK   123" (padded to align)
        """
        matcher = SequenceMatcher(None, reference, text)
        aligned = list(' ' * len(reference))

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal' or tag == 'replace':
                # Copy matched characters
                for k in range(i2 - i1):
                    if i1 + k < len(reference) and j1 + k < len(text):
                        aligned[i1 + k] = text[j1 + k]

        return ''.join(aligned)

    @staticmethod
    def highest_confidence_method(ocr_results: List[List[OCRResult]]) -> str:
        """Use text from frame with highest average confidence.

        Args:
            ocr_results: List of OCR results per frame

        Returns:
            Text from highest-confidence frame

        Example:
            >>> # Frame 1 avg conf: 0.85, Frame 2: 0.92, Frame 3: 0.78
            >>> # Returns text from Frame 2
            >>> text = MultiFrameOCRAggregator.highest_confidence_method(results)
        """
        if not ocr_results:
            return ""

        # Calculate average confidence per frame
        frame_confidences = []
        for frame_results in ocr_results:
            if not frame_results:
                frame_confidences.append(0.0)
            else:
                avg_conf = sum(r.confidence for r in frame_results) / len(frame_results)
                frame_confidences.append(avg_conf)

        # Select frame with highest confidence
        best_frame_idx = np.argmax(frame_confidences)
        best_frame = ocr_results[best_frame_idx]

        return MultiFrameOCRAggregator._results_to_text(best_frame)

    @staticmethod
    def longest_text_method(ocr_results: List[List[OCRResult]]) -> str:
        """Use longest extracted text.

        Assumption: Longer text likely means better OCR quality.

        Args:
            ocr_results: List of OCR results per frame

        Returns:
            Longest text string

        Example:
            >>> # Frame 1: "TRACK", Frame 2: "TRACKING123", Frame 3: "TRACK12"
            >>> # Returns "TRACKING123"
            >>> text = MultiFrameOCRAggregator.longest_text_method(results)
        """
        if not ocr_results:
            return ""

        texts = [MultiFrameOCRAggregator._results_to_text(frame) for frame in ocr_results]
        texts = [t for t in texts if t]

        if not texts:
            return ""

        return max(texts, key=len)

    @staticmethod
    def confidence_weighted_merge(ocr_results: List[List[OCRResult]]) -> str:
        """Merge results weighted by confidence scores.

        Similar to voting but gives more weight to high-confidence detections.

        Args:
            ocr_results: List of OCR results per frame

        Returns:
            Confidence-weighted consensus text

        Example:
            >>> # High-confidence "TRACK123" gets more weight than low-conf "TRACK1Z3"
            >>> text = MultiFrameOCRAggregator.confidence_weighted_merge(results)
        """
        # Convert to texts with confidence weights
        weighted_texts = []
        for frame_results in ocr_results:
            if not frame_results:
                continue

            text = MultiFrameOCRAggregator._results_to_text(frame_results)
            avg_confidence = sum(r.confidence for r in frame_results) / len(frame_results)
            weighted_texts.append((text, avg_confidence))

        if not weighted_texts:
            return ""

        if len(weighted_texts) == 1:
            return weighted_texts[0][0]

        # Find reference (longest text)
        reference = max(weighted_texts, key=lambda x: len(x[0]))[0]

        # Align and weight
        consensus = []
        for i in range(len(reference)):
            char_weights = {}

            for text, weight in weighted_texts:
                aligned = MultiFrameOCRAggregator._align_texts(reference, text)
                if i < len(aligned) and aligned[i] != ' ':
                    char = aligned[i]
                    char_weights[char] = char_weights.get(char, 0.0) + weight

            if char_weights:
                # Character with highest weight wins
                best_char = max(char_weights.items(), key=lambda x: x[1])[0]
                consensus.append(best_char)
            else:
                consensus.append(reference[i])

        return ''.join(consensus)

    @staticmethod
    def calculate_consensus_confidence(ocr_results: List[List[OCRResult]]) -> float:
        """Calculate overall confidence based on agreement across frames.

        High agreement between frames → high confidence.
        Low agreement → low confidence.

        Args:
            ocr_results: List of OCR results per frame

        Returns:
            Consensus confidence score (0.0-1.0)

        Example:
            >>> conf = MultiFrameOCRAggregator.calculate_consensus_confidence(results)
            >>> if conf > 0.85:
            ...     print("High confidence - auto-accept")
        """
        if not ocr_results or len(ocr_results) < 2:
            # Single frame - return average confidence
            if ocr_results and ocr_results[0]:
                return sum(r.confidence for r in ocr_results[0]) / len(ocr_results[0])
            return 0.0

        # Get texts from all frames
        texts = [MultiFrameOCRAggregator._results_to_text(frame) for frame in ocr_results]
        texts = [t for t in texts if t]

        if len(texts) < 2:
            return 0.0

        # Calculate pairwise similarity
        similarities = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = SequenceMatcher(None, texts[i], texts[j]).ratio()
                similarities.append(similarity)

        if not similarities:
            return 0.0

        # Average similarity as consensus confidence
        avg_similarity = sum(similarities) / len(similarities)

        # Combine with OCR confidences
        all_confidences = []
        for frame_results in ocr_results:
            if frame_results:
                all_confidences.extend([r.confidence for r in frame_results])

        if all_confidences:
            avg_ocr_confidence = sum(all_confidences) / len(all_confidences)
        else:
            avg_ocr_confidence = 0.0

        # Weighted combination: 60% agreement, 40% OCR confidence
        consensus_confidence = 0.6 * avg_similarity + 0.4 * avg_ocr_confidence

        return consensus_confidence

    @staticmethod
    def _results_to_text(ocr_results: List[OCRResult]) -> str:
        """Convert OCRResult list to text string.

        Sorts by position (top-to-bottom, left-to-right) and joins.

        Args:
            ocr_results: List of OCRResult objects

        Returns:
            Concatenated text string

        Example:
            >>> text = MultiFrameOCRAggregator._results_to_text(results)
        """
        if not ocr_results:
            return ""

        # Sort by y (top to bottom), then x (left to right)
        sorted_results = sorted(
            ocr_results,
            key=lambda r: (r.bbox.y_min, r.bbox.x_min)
        )

        # Join text
        return ' '.join(r.text for r in sorted_results)

    @staticmethod
    def get_aggregation_statistics(
        ocr_results: List[List[OCRResult]]
    ) -> Dict:
        """Get statistics about multi-frame OCR results.

        Args:
            ocr_results: List of OCR results per frame

        Returns:
            Dictionary with statistics:
                - num_frames: Number of frames processed
                - avg_detections_per_frame: Average text detections per frame
                - avg_confidence_per_frame: Confidence per frame
                - text_length_variance: Variance in text lengths
                - consensus_confidence: Overall consensus confidence

        Example:
            >>> stats = MultiFrameOCRAggregator.get_aggregation_statistics(results)
            >>> print(f"Processed {stats['num_frames']} frames")
        """
        if not ocr_results:
            return {
                'num_frames': 0,
                'avg_detections_per_frame': 0.0,
                'avg_confidence_per_frame': 0.0,
                'text_length_variance': 0.0,
                'consensus_confidence': 0.0
            }

        num_frames = len(ocr_results)

        # Detections per frame
        detections_per_frame = [len(frame) for frame in ocr_results]
        avg_detections = sum(detections_per_frame) / num_frames if num_frames > 0 else 0

        # Confidence per frame
        confidences_per_frame = []
        for frame in ocr_results:
            if frame:
                avg_conf = sum(r.confidence for r in frame) / len(frame)
                confidences_per_frame.append(avg_conf)

        avg_confidence = (
            sum(confidences_per_frame) / len(confidences_per_frame)
            if confidences_per_frame else 0.0
        )

        # Text length variance
        text_lengths = [
            len(MultiFrameOCRAggregator._results_to_text(frame))
            for frame in ocr_results
        ]
        text_length_variance = np.var(text_lengths) if text_lengths else 0.0

        # Consensus confidence
        consensus_conf = MultiFrameOCRAggregator.calculate_consensus_confidence(ocr_results)

        return {
            'num_frames': num_frames,
            'avg_detections_per_frame': avg_detections,
            'avg_confidence_per_frame': avg_confidence,
            'text_length_variance': float(text_length_variance),
            'consensus_confidence': consensus_conf
        }
