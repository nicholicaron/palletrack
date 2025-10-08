"""OCR processing modules for document text extraction.

This package provides comprehensive OCR capabilities including:
- Image preprocessing for improved OCR accuracy
- PaddleOCR wrapper with custom configuration
- Multi-frame result aggregation for reliability
- Post-processing for text cleaning and validation
"""

from .aggregator import MultiFrameOCRAggregator
from .document_ocr import DocumentOCR
from .post_processor import OCRPostProcessor
from .preprocessor import OCRPreprocessor

__all__ = [
    "OCRPreprocessor",
    "DocumentOCR",
    "MultiFrameOCRAggregator",
    "OCRPostProcessor",
]
