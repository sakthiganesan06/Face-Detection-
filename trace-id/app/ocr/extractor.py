"""
TRACE-ID — OCR & Watermark / Text Context Extractor
Detects visible text, watermarks, creator handles, and text overlays from input images.

Used as an additional contextual discovery signal to enhance candidate discovery
without replacing genuine reverse-image search.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app import config

logger = logging.getLogger(__name__)


class OCRExtractor:
    """
    Singleton wrapper around EasyOCR / Cloud Vision Text Detection.
    """

    _instance: Optional["OCRExtractor"] = None
    _reader = None

    def __new__(cls) -> "OCRExtractor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_loaded(self) -> None:
        """Lazy-load EasyOCR reader."""
        if self._reader is not None:
            return

        try:
            import easyocr
            # Load English / Latin reader on CPU or CUDA
            gpu = config.INSIGHTFACE_CTX_ID >= 0
            self._reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
            logger.info("EasyOCR model initialized successfully.")
        except Exception as exc:
            logger.warning("Could not initialize EasyOCR: %s. Using fallback.", exc)
            self._reader = None

    def extract_text(self, image_input: np.ndarray | str | Path) -> list[str]:
        """
        Extract visible text and watermark strings from an image.

        Args:
            image_input: BGR numpy image or path to image file.

        Returns:
            List of unique, cleaned text strings detected in the image.
        """
        if config.MOCK_MODE:
            return []

        # Load image if path
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                return []
            image = cv2.imread(str(path))
            if image is None:
                return []
        else:
            image = image_input

        if image is None or image.size == 0:
            return []

        detected_lines: list[str] = []

        # 1. Try EasyOCR
        try:
            self._ensure_loaded()
            if self._reader is not None:
                # EasyOCR expects RGB or image path/ndarray
                rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self._reader.readtext(rgb_img)
                for bbox, text, conf in results:
                    clean_text = str(text).strip()
                    # Filter out single-character noise or very low confidence
                    if len(clean_text) >= 2 and conf > 0.25:
                        detected_lines.append(clean_text)
        except Exception as exc:
            logger.debug("EasyOCR extraction error: %s", exc)

        # 2. Try Google Cloud Vision Text Detection if available and no text found
        if not detected_lines and config.GOOGLE_APPLICATION_CREDENTIALS:
            try:
                from google.cloud import vision
                client = vision.ImageAnnotatorClient()
                _, buffer = cv2.imencode(".jpg", image)
                content = buffer.tobytes()
                vision_image = vision.Image(content=content)
                response = client.text_detection(image=vision_image)
                if response.text_annotations:
                    full_text = response.text_annotations[0].description
                    for line in full_text.split("\n"):
                        clean_line = line.strip()
                        if len(clean_line) >= 2:
                            detected_lines.append(clean_line)
            except Exception as exc:
                logger.debug("Cloud Vision text detection error: %s", exc)

        # Deduplicate while preserving order and cleaning
        unique_texts: list[str] = []
        seen = set()
        for text in detected_lines:
            # Clean up whitespace and special non-alphanumeric noise
            t = text.strip()
            norm = t.lower()
            if norm and norm not in seen and len(t) >= 2:
                seen.add(norm)
                unique_texts.append(t)

        return unique_texts

    def build_context_queries(self, detected_texts: list[str]) -> list[str]:
        """
        Build focused web/social search queries from detected OCR text.
        Filters out common generic words and keeps watermarks/usernames/identifiers.
        """
        if not detected_texts:
            return []

        queries: list[str] = []
        # Filter stopwords
        stopwords = {"the", "and", "for", "with", "from", "this", "that", "image", "photo"}

        for text in detected_texts:
            cleaned = text.strip()
            # If contains handle (@name) or brand/watermark structure (e.g. NAMMA PAVOOR.360)
            if len(cleaned) >= 3 and cleaned.lower() not in stopwords:
                queries.append(cleaned)

        return queries[:3]
