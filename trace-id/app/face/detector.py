"""
TRACE-ID — Face Detector
Uses InsightFace buffalo_l model pack (ArcFace R100 recognition).

Responsibilities:
  - Load InsightFace FaceAnalysis model once (singleton pattern)
  - Detect all faces in an image
  - Return crops, bounding boxes, landmarks, and 512-dim ArcFace embeddings
  - Handle no-face and multi-face scenarios cleanly
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app import config

logger = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    """Single detected face result."""

    index: int
    """0-based index among all detected faces."""

    bbox: tuple[int, int, int, int]
    """Bounding box: (x1, y1, x2, y2) in pixels."""

    confidence: float
    """Detection confidence score."""

    embedding: np.ndarray
    """
    ArcFace embedding vector (512-dim for buffalo_l).
    L2-normalized by InsightFace before we receive it.
    NOT stored in blockchain or evidence JSON.
    """

    landmark: Optional[np.ndarray] = field(default=None)
    """5-point facial landmarks (optional)."""

    prominence_score: float = field(default=0.0)
    """Calculated prominence score combining bounding box area, centrality, and confidence."""

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def center(self) -> tuple[float, float]:
        """Return (cx, cy) pixel coordinates of face center."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Return the face crop from the source image."""
        x1, y1, x2, y2 = self.bbox
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return image[y1:y2, x1:x2].copy()


def score_face_prominence(
    face: DetectedFace,
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> float:
    """
    Compute a prominence score in [0, 1] for a detected face.
    Combines:
      - Bounding box area (foreground subjects weighted heavily)
      - Proximity to image center (central subjects preferred over edge/background)
      - Detection quality/confidence score
    """
    img_h, img_w = image_shape[:2]
    if img_h <= 0 or img_w <= 0:
        return 0.0

    img_area = float(img_h * img_w)
    area_ratio = min(1.0, face.area / img_area)

    cx, cy = face.center
    img_cx, img_cy = img_w / 2.0, img_h / 2.0
    # Normalized Euclidean distance from center in [0, ~0.71]
    norm_dist = np.sqrt(((cx - img_cx) / img_w) ** 2 + ((cy - img_cy) / img_h) ** 2)
    centrality = max(0.0, 1.0 - (norm_dist / 0.7071))

    quality = max(0.0, min(1.0, float(face.confidence)))

    # Weighting: 65% area (sqrt-scaled), 25% centrality, 10% quality
    area_scale = np.sqrt(area_ratio)
    prominence = (0.65 * area_scale) + (0.25 * centrality) + (0.10 * quality)
    return float(np.clip(prominence, 0.0, 1.0))


class FaceDetector:
    """
    Singleton wrapper around InsightFace FaceAnalysis.

    Usage:
        detector = FaceDetector()
        faces = detector.detect(image_bgr)
    """

    _instance: Optional["FaceDetector"] = None
    _app = None  # insightface.app.FaceAnalysis instance

    def __new__(cls) -> "FaceDetector":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_loaded(self) -> None:
        """Lazy-load InsightFace model on first use."""
        if self._app is not None:
            return

        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "InsightFace is not installed. Run: pip install insightface onnxruntime"
            ) from exc

        logger.info(
            "Loading InsightFace model '%s' (ctx=%d) — first run downloads ~200 MB",
            config.INSIGHTFACE_MODEL_NAME,
            config.INSIGHTFACE_CTX_ID,
        )
        app = FaceAnalysis(
            name=config.INSIGHTFACE_MODEL_NAME,
            allowed_modules=["detection", "recognition"],
        )
        app.prepare(ctx_id=config.INSIGHTFACE_CTX_ID, det_size=(640, 640))
        self._app = app
        logger.info("InsightFace model loaded successfully.")

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """
        Detect all faces in a BGR image (as returned by cv2.imread).

        Returns a list of DetectedFace objects, sorted by area descending
        so the largest face is always first.

        Raises:
            ValueError: If the image is invalid or empty.
        """
        if image is None or image.size == 0:
            raise ValueError("Empty or invalid image passed to FaceDetector.detect()")

        self._ensure_loaded()

        # InsightFace expects BGR
        faces_raw = self._app.get(image)

        if not faces_raw:
            return []

        results: list[DetectedFace] = []
        for idx, face in enumerate(faces_raw):
            bbox = tuple(int(v) for v in face.bbox)  # (x1, y1, x2, y2)
            confidence = float(face.det_score) if hasattr(face, "det_score") else 1.0
            embedding = np.array(face.embedding, dtype=np.float32)

            # Embeddings from InsightFace buffalo_l are already L2-normalized.
            # We normalize anyway to be safe (idempotent for already-normalized vectors).
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                embedding = embedding / norm

            landmark = np.array(face.kps, dtype=np.float32) if hasattr(face, "kps") and face.kps is not None else None

            face_obj = DetectedFace(
                index=idx,
                bbox=bbox,
                confidence=confidence,
                embedding=embedding,
                landmark=landmark,
            )
            face_obj.prominence_score = score_face_prominence(face_obj, image.shape)
            results.append(face_obj)

        # Sort by prominence score descending (largest central foreground face first)
        results.sort(key=lambda f: f.prominence_score, reverse=True)
        for new_idx, face_obj in enumerate(results):
            face_obj.index = new_idx

        return results

    def detect_from_path(self, image_path: Path | str) -> list[DetectedFace]:
        """
        Load an image from disk and detect faces.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be decoded as an image.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not decode image: {path}. Check format/corruption.")

        logger.debug("Loaded image %s — shape %s", path.name, image.shape)
        return self.detect(image)


def select_primary_face(
    faces: list[DetectedFace],
    selected_index: int | None = None,
) -> tuple[DetectedFace, int]:
    """
    Select the primary target face from a list of detected faces.

    Args:
        faces: List of DetectedFace objects (assumed sorted by prominence).
        selected_index: Optional manual face index override.

    Returns:
        tuple of (selected_face, selected_index)

    Raises:
        ValueError: If faces list is empty.
    """
    if not faces:
        raise ValueError("Cannot select primary face: no faces provided")

    if selected_index is not None and 0 <= selected_index < len(faces):
        chosen = faces[selected_index]
        return chosen, selected_index

    # Default to the most prominent face (first in sorted list)
    return faces[0], 0


def load_image(image_path: Path | str) -> np.ndarray:
    """
    Load an image from disk as a BGR NumPy array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be decoded.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
    if path.suffix.lower() not in supported:
        raise ValueError(
            f"Unsupported image format '{path.suffix}'. Supported: {sorted(supported)}"
        )
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def validate_image_path(image_path: Path | str) -> Path:
    """
    Validate that a path points to a readable, supported image file.

    Returns the resolved Path.
    Raises FileNotFoundError or ValueError with human-readable messages.
    """
    path = Path(image_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
    if path.suffix.lower() not in supported:
        raise ValueError(
            f"Unsupported image format '{path.suffix}'. "
            f"Supported formats: {', '.join(sorted(supported))}"
        )
    # Quick readability check
    try:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Image file exists but cannot be read/decoded: {path}")
    except Exception as exc:
        raise ValueError(f"Image read error: {exc}") from exc

    return path
