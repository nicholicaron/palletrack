"""OCR preprocessing pipeline for document images.

This module provides image preprocessing techniques to improve OCR accuracy
on document images captured from warehouse pallets. Handles common issues
like poor contrast, noise, glare, and perspective distortion.
"""

from typing import Dict, Tuple

import cv2
import numpy as np


class OCRPreprocessor:
    """Prepare document images for OCR with adaptive preprocessing.

    Applies a series of image enhancement techniques to improve OCR accuracy:
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Bilateral filtering (denoise while preserving edges)
    - Glare detection and removal
    - Optional perspective correction
    - Adaptive sharpening

    Example:
        >>> preprocessor = OCRPreprocessor()
        >>> config = {'ocr': {'preprocessing': {...}}}
        >>> enhanced = preprocessor.preprocess(document_image, config)
    """

    @staticmethod
    def preprocess(image: np.ndarray, config: Dict) -> np.ndarray:
        """Apply complete preprocessing pipeline to document image.

        Applies preprocessing steps in optimal order to maximize OCR accuracy.
        Each step can be enabled/disabled via configuration.

        Args:
            image: Input document image (BGR or grayscale)
            config: Configuration dict with preprocessing settings

        Returns:
            Preprocessed image ready for OCR

        Example:
            >>> config = {
            ...     'ocr': {
            ...         'preprocessing': {
            ...             'apply_clahe': True,
            ...             'bilateral_filter': True,
            ...             'remove_glare': True
            ...         }
            ...     }
            ... }
            >>> enhanced = OCRPreprocessor.preprocess(image, config)
        """
        preprocess_config = config.get('ocr', {}).get('preprocessing', {})

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Step 1: CLAHE - improve contrast
        if preprocess_config.get('apply_clahe', True):
            gray = OCRPreprocessor.apply_clahe(
                gray,
                clip_limit=preprocess_config.get('clahe_clip_limit', 2.0),
                grid_size=preprocess_config.get('clahe_grid_size', 8)
            )

        # Step 2: Bilateral filter - denoise while preserving edges
        if preprocess_config.get('bilateral_filter', True):
            gray = OCRPreprocessor.bilateral_filter(
                gray,
                d=preprocess_config.get('bilateral_d', 9),
                sigma_color=preprocess_config.get('bilateral_sigma_color', 75),
                sigma_space=preprocess_config.get('bilateral_sigma_space', 75)
            )

        # Step 3: Remove glare (saturated regions from plastic pouch)
        if preprocess_config.get('remove_glare', True):
            gray = OCRPreprocessor.remove_glare(
                gray,
                threshold=preprocess_config.get('glare_threshold', 240)
            )

        # Step 4: Perspective correction (optional - experimental)
        if preprocess_config.get('apply_perspective_correction', False):
            gray = OCRPreprocessor.correct_perspective(gray)

        # Step 5: Adaptive sharpening (only if image is reasonably sharp)
        sharpen_threshold = preprocess_config.get('sharpen_threshold', 100)
        if OCRPreprocessor.calculate_sharpness(gray) > sharpen_threshold:
            gray = OCRPreprocessor.sharpen_image(gray)

        return gray

    @staticmethod
    def apply_clahe(
        image: np.ndarray,
        clip_limit: float = 2.0,
        grid_size: int = 8
    ) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization.

        CLAHE improves local contrast while avoiding over-amplification of noise.
        Particularly effective for documents with uneven lighting.

        Args:
            image: Grayscale input image
            clip_limit: Contrast limit (higher = more contrast)
            grid_size: Size of grid for local histogram equalization

        Returns:
            Contrast-enhanced image

        Example:
            >>> enhanced = OCRPreprocessor.apply_clahe(gray_image)
        """
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(grid_size, grid_size)
        )
        return clahe.apply(image)

    @staticmethod
    def bilateral_filter(
        image: np.ndarray,
        d: int = 9,
        sigma_color: float = 75,
        sigma_space: float = 75
    ) -> np.ndarray:
        """Apply bilateral filter for edge-preserving denoising.

        Reduces noise while maintaining sharp edges - critical for text clarity.

        Args:
            image: Grayscale input image
            d: Diameter of pixel neighborhood
            sigma_color: Filter sigma in color space (larger = more colors mixed)
            sigma_space: Filter sigma in coordinate space (larger = farther pixels)

        Returns:
            Denoised image with preserved edges

        Example:
            >>> denoised = OCRPreprocessor.bilateral_filter(noisy_image)
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    @staticmethod
    def remove_glare(
        image: np.ndarray,
        threshold: int = 240
    ) -> np.ndarray:
        """Detect and mitigate glare from plastic document pouches.

        Identifies over-saturated regions (likely glare) and applies inpainting
        to reconstruct the underlying content.

        Args:
            image: Grayscale input image
            threshold: Intensity threshold for glare detection (0-255)

        Returns:
            Image with glare reduced

        Example:
            >>> no_glare = OCRPreprocessor.remove_glare(image, threshold=240)
        """
        # Create mask of glare regions (very bright pixels)
        _, glare_mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

        # Dilate mask slightly to capture glare edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        glare_mask = cv2.dilate(glare_mask, kernel, iterations=1)

        # If significant glare detected, use inpainting
        glare_percentage = np.sum(glare_mask > 0) / glare_mask.size
        if glare_percentage > 0.01:  # More than 1% of image is glare
            # Inpaint to reconstruct glare regions
            result = cv2.inpaint(image, glare_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            return result

        return image

    @staticmethod
    def correct_perspective(image: np.ndarray) -> np.ndarray:
        """Attempt basic perspective correction for skewed documents.

        Tries to detect document edges and apply perspective transform
        to create a frontal view. This is experimental and may not always
        improve results.

        Args:
            image: Grayscale input image

        Returns:
            Perspective-corrected image, or original if correction fails

        Example:
            >>> corrected = OCRPreprocessor.correct_perspective(skewed_doc)
        """
        # Edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image

        # Find largest contour (likely document boundary)
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # If we found a quadrilateral, apply perspective transform
        if len(approx) == 4:
            # Order points: top-left, top-right, bottom-right, bottom-left
            pts = approx.reshape(4, 2)
            rect = OCRPreprocessor._order_points(pts)

            # Calculate destination points
            (tl, tr, br, bl) = rect
            width_a = np.linalg.norm(br - bl)
            width_b = np.linalg.norm(tr - tl)
            max_width = int(max(width_a, width_b))

            height_a = np.linalg.norm(tr - br)
            height_b = np.linalg.norm(tl - bl)
            max_height = int(max(height_a, height_b))

            dst = np.array([
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1]
            ], dtype=np.float32)

            # Apply perspective transform
            matrix = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)
            warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

            return warped

        # If perspective correction failed, return original
        return image

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """Order points in consistent order: TL, TR, BR, BL.

        Args:
            pts: Array of 4 points (x, y)

        Returns:
            Ordered points array
        """
        # Sort by y-coordinate
        sorted_pts = pts[np.argsort(pts[:, 1])]

        # Top two points
        top_pts = sorted_pts[:2]
        top_pts = top_pts[np.argsort(top_pts[:, 0])]  # Sort by x
        tl, tr = top_pts

        # Bottom two points
        bottom_pts = sorted_pts[2:]
        bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]  # Sort by x
        bl, br = bottom_pts

        return np.array([tl, tr, br, bl])

    @staticmethod
    def sharpen_image(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Apply adaptive sharpening to enhance text edges.

        Only applies sharpening if the image is already reasonably sharp
        (to avoid amplifying noise in blurry images).

        Args:
            image: Grayscale input image
            strength: Sharpening strength multiplier (default: 1.0)

        Returns:
            Sharpened image

        Example:
            >>> sharp = OCRPreprocessor.sharpen_image(image, strength=1.5)
        """
        # Unsharp mask: original - blurred = edges, then add back to original
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)

        return sharpened

    @staticmethod
    def calculate_sharpness(image: np.ndarray) -> float:
        """Calculate sharpness score using Laplacian variance.

        Higher values indicate sharper images. Used to decide whether
        to apply sharpening.

        Args:
            image: Grayscale input image

        Returns:
            Sharpness score (higher = sharper)

        Example:
            >>> sharpness = OCRPreprocessor.calculate_sharpness(image)
            >>> if sharpness > 100:
            ...     print("Image is sharp enough for OCR")
        """
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return laplacian.var()

    @staticmethod
    def adaptive_binarization(
        image: np.ndarray,
        block_size: int = 11,
        c: int = 2
    ) -> np.ndarray:
        """Apply adaptive thresholding for binarization.

        Converts grayscale to binary (black/white) using local thresholding.
        Useful for documents with varying lighting conditions.

        Args:
            image: Grayscale input image
            block_size: Size of pixel neighborhood (must be odd)
            c: Constant subtracted from weighted mean

        Returns:
            Binary image (0 or 255)

        Example:
            >>> binary = OCRPreprocessor.adaptive_binarization(gray_image)
        """
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c
        )

    @staticmethod
    def preprocess_for_visualization(
        original: np.ndarray,
        preprocessed: np.ndarray
    ) -> np.ndarray:
        """Create side-by-side comparison for debugging.

        Args:
            original: Original document image
            preprocessed: Preprocessed version

        Returns:
            Concatenated image showing before/after

        Example:
            >>> comparison = OCRPreprocessor.preprocess_for_visualization(
            ...     original, preprocessed
            ... )
            >>> cv2.imshow("Preprocessing", comparison)
        """
        # Convert to same format
        if len(original.shape) == 3:
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = original

        if len(preprocessed.shape) == 3:
            preprocessed_gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
        else:
            preprocessed_gray = preprocessed

        # Resize to same height if needed
        if original_gray.shape[0] != preprocessed_gray.shape[0]:
            h = min(original_gray.shape[0], preprocessed_gray.shape[0])
            original_gray = cv2.resize(original_gray, (int(original_gray.shape[1] * h / original_gray.shape[0]), h))
            preprocessed_gray = cv2.resize(preprocessed_gray, (int(preprocessed_gray.shape[1] * h / preprocessed_gray.shape[0]), h))

        # Concatenate horizontally
        return np.hstack([original_gray, preprocessed_gray])
